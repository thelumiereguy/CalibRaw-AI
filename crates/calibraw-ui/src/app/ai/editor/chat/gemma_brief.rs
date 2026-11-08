//! Gemma's requests (`Protocol::GemmaBrief`). A request is one reply, on a
//! fresh conversation that holds its system prompt and nothing else, a clone
//! of one Gemma read when the card came (`Platform::ai_editor_start_request`),
//! so a request costs the same however long the chat is. Gemma interprets
//! the photographer's words with the preview into a brief
//! (assets/gemma/interpret_system.md): items of tone, colour, detail and
//! effects, each a setting, up or down and a size word, what to keep, and a
//! sentence for the photographer. The message names the photo's genre and
//! role from its triage card (triage.rs) and says what the previous request
//! changed, as `name=before→after`, so "a bit more" and "undo that" work. The
//! app reads the brief into tool calls itself (brief.rs), which calls
//! `set_sliders`, the colour tools and `set_effect`: a second conversation
//! that wrote the values from the brief wrote some with the wrong sign or to
//! the wrong tool, and asked for the brief and the values at once, Gemma
//! planned a look and then wrote values that crushed the photo black. The
//! edit is one nudge, each value anywhere in its range.

use super::super::brief::read_brief;
use super::super::color::{grading_current, mixer_current};
use super::super::effects::effect_current;
use super::super::sliders::slider_value;
use super::super::tools::{change_tool, previous_value, record_changes, result_line, routed};
use super::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) const SYSTEM_PROMPT: &str =
    include_str!("../../../../../../../android/app/src/main/assets/gemma/interpret_system.md");
/// How a brief that answers a question instead starts.
const ANSWER: &str = "Answer:";
/// How the brief's sentence for the photographer starts.
const SUMMARY: &str = "Summary:";
/// How the app reports the brief's items it could not read.
const UNREAD: &str = "not understood:";
/// For the evals (research/gemma-evals): while this file exists in the app's
/// files, the app writes the preview that shows a Gemma request's edit and
/// what it set to the trace GemmaTrace keeps there. Only `adb run-as`, which
/// a release build refuses, can make it, as GemmaTrace's own lines need.
const TRACE_SWITCH: &str = "gemma-trace.on";
const TRACE_FILE: &str = "gemma-trace.txt";

impl CalibRawApp {
    /// Applies the brief Gemma wrote as one nudge, or ends the request with
    /// the answer to a question.
    pub(super) fn handle_gemma_brief(&mut self, text: &str, shown: usize, frame: &eframe::Frame) {
        if text.is_empty() {
            self.fail_ai_editor_request("Gemma wrote no brief for the request".to_owned());
            return;
        }
        if let Some(answer) = text.strip_prefix(ANSWER) {
            if let Some(message) = self.ai.editor.messages.get_mut(shown) {
                message.text = answer.trim().to_owned();
            }
            self.finish_ai_editor_request();
            return;
        }

        let previous = &self.ai.editor.last_changes;
        let brief = read_brief(text, &|name| self.ai_value(name), &|name| {
            previous_value(previous, name)
        });
        let calls = routed(brief.calls);
        let responses = self.apply_ai_editor_calls(&calls, frame);
        let mut results: Vec<String> = calls
            .iter()
            .zip(&responses)
            .map(|(call, response)| result_line(&call.name, response))
            .collect();
        if !brief.unread.is_empty() {
            results.push(format!("{UNREAD} {}", brief.unread.join("; ")));
        }
        let results = results.join("\n");

        let editor = &mut self.ai.editor;
        if let Some(message) = editor.messages.get_mut(shown) {
            message.role = ChatRole::Progress;
            message.text = without_summary(text);
        }
        if !results.is_empty() {
            editor.messages.push(ChatMessage {
                role: ChatRole::Progress,
                text: results.clone(),
            });
        }
        if let Some(summary) = brief.summary {
            editor.messages.push(ChatMessage {
                role: ChatRole::Model,
                text: summary,
            });
        }
        let Some(request) = editor.request.as_mut() else {
            return;
        };
        request.nudges += 1;
        for (call, response) in calls.iter().zip(&responses) {
            record_changes(&mut request.changes, &change_tool(call), response);
        }
        let tracing = self
            .platform
            .internal_data_dir()
            .is_some_and(|data_dir| data_dir.join(TRACE_SWITCH).is_file());
        if tracing {
            if let Some(request) = self.ai.editor.request.as_mut() {
                request.step = Step::Capturing(Outgoing::Applied { results });
            }
            self.egui_ctx.request_repaint();
        } else {
            self.finish_ai_editor_request();
        }
    }

    /// For the evals: once the preview shows Gemma's edit, writes it where
    /// the trace's previews are read and what the app set to the trace.
    pub(super) fn trace_ai_editor_result(&mut self, frame: &eframe::Frame) {
        let Some((capture, data_dir)) = self.ai_editor_preview(frame) else {
            return;
        };
        let Some(Request {
            step: Step::Capturing(Outgoing::Applied { results }),
            ..
        }) = self.ai.editor.request.as_ref()
        else {
            return;
        };
        let line = format!("APPLIED {results}");
        let jpeg_path = data_dir.join(EDITOR_IMAGE_FILE);
        let trace = data_dir.join(TRACE_FILE);
        let receiver = spawn_ui_worker(&self.egui_ctx, move || {
            capture.write(&capture.read()?, &jpeg_path)?;
            append_trace(&trace, &line)
        });
        if let Some(request) = self.ai.editor.request.as_mut() {
            request.step = Step::Tracing { receiver };
        }
    }

    /// The value `name` of a tool the brief reader calls, as the tool reads
    /// it.
    fn ai_value(&self, name: &str) -> Option<f32> {
        let exposure = &self.develop.exposure;
        slider_value(exposure, self.develop.loaded_raw.as_deref(), name)
            .or_else(|| mixer_current(exposure, name))
            .or_else(|| grading_current(exposure, name))
            .or_else(|| effect_current(&self.masks.stack.global_effects, name))
    }
}

/// `brief` without its summary, which the chat shows as Gemma's reply.
fn without_summary(brief: &str) -> String {
    brief
        .lines()
        .filter(|line| !line.trim_start().starts_with(SUMMARY))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Appends `line` to the eval trace at `path`, on one line, after a
/// timestamp as GemmaTrace writes its own.
fn append_trace(path: &Path, line: &str) -> Result<(), String> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_millis());
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("could not open the trace: {error}"))?;
    writeln!(file, "{millis} {}", line.replace('\n', "|"))
        .map_err(|error| format!("could not write the trace: {error}"))
}
