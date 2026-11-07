//! The AI editor, the second phase of on-device Gemma: a chat about the open
//! photo in which Gemma edits it through six tools: `set_sliders`
//! (sliders.rs), `set_color_mixer` and `set_color_grading` (color.rs),
//! `set_tone_curve` (tone_curve.rs), `set_effect` (effects.rs) and
//! `set_ai_denoise`. The brief reader (brief.rs) calls only the first four;
//! Gemma misused the tone curve and AI denoise, and the app still applies
//! them.
//!
//! A request is one Gemma reply, on a fresh conversation that holds its
//! system prompt and nothing else, a clone of one Gemma read when the card
//! came (`Platform::gemma_editor_start_request`), so a request costs the same
//! however long the chat is. Gemma interprets the photographer's words with
//! the preview into a brief (assets/gemma/interpret_system.md): items of
//! tone, colour, detail and effects, each a setting, up or down and a size
//! word, what to keep, and a sentence for the photographer. The message names
//! the photo's genre and role from its triage card (triage.rs) and says what
//! the previous request changed, as `name=before→after`, so "a bit more" and
//! "undo that" work. The app reads the brief into tool calls itself: a
//! second conversation that wrote the values from the brief wrote some with
//! the wrong sign or to the wrong tool, and asked for the brief and the
//! values at once, Gemma planned a look and then wrote values that crushed
//! the photo black.
//!
//! The edit is one nudge, each value anywhere in its range, and one undo
//! step (`observe_edit_history`).

use super::brief::read_brief;
use super::calls::{read_reply, Call};
use super::color::{
    apply_set_color_grading, apply_set_color_mixer, changed_grading, changed_mixer,
    grading_current, is_grading_value, is_mixer_value, mixer_current,
};
use super::effects::{changed_effects, effect_current, is_effect_value};
use super::sliders::{apply_set_sliders, changed_sliders, is_slider, slider_value};
use super::tone_curve::{apply_set_tone_curve, changed_curves, is_curve};
use super::tools::{describe, number, written};
use super::triage::{write_gemma_jpeg, TriageCard, GEMMA_IMAGE_EDGE};
use super::*;
use crate::app::preview_capture::PreviewSnapshot;
use calibraw_ai::AiFeature;
use serde_json::{json, Value};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const INTERPRET_PROMPT: &str =
    include_str!("../../../../../../android/app/src/main/assets/gemma/interpret_system.md");
/// How a brief that answers a question instead starts.
const ANSWER: &str = "Answer:";
/// How the brief's sentence for the photographer starts.
const SUMMARY: &str = "Summary:";
/// How the app reports the brief's items it could not read.
const UNREAD: &str = "not understood:";
const EDITOR_IMAGE_FILE: &str = "gemma-editor.jpg";
/// For the evals (research/gemma-evals): while this file exists in the app's
/// files, the app writes the preview that shows a request's edit and what it
/// set to the trace GemmaBridge keeps there. Only `adb run-as`, which a
/// release build refuses, can make it, as GemmaBridge's own trace needs.
const TRACE_SWITCH: &str = "gemma-trace.on";
const TRACE_FILE: &str = "gemma-trace.txt";
/// The tools the app applies.
pub(super) const SET_SLIDERS: &str = "set_sliders";
pub(super) const SET_COLOR_MIXER: &str = "set_color_mixer";
pub(super) const SET_COLOR_GRADING: &str = "set_color_grading";
const SET_TONE_CURVE: &str = "set_tone_curve";
pub(super) const SET_EFFECT: &str = "set_effect";
const SET_AI_DENOISE: &str = "set_ai_denoise";

/// Who wrote a chat message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChatRole {
    Photographer,
    Gemma,
    /// What a request set.
    Progress,
    /// Why a request stopped.
    Failure,
}

#[derive(Clone, Debug)]
pub(crate) struct ChatMessage {
    pub(crate) role: ChatRole,
    pub(crate) text: String,
}

/// The open photo's chat with the AI editor.
#[derive(Default)]
pub(crate) struct EditorState {
    /// Whether the chat is shown in place of the tools.
    pub(crate) open: bool,
    /// The photographer's message being typed.
    pub(crate) draft: String,
    messages: Vec<ChatMessage>,
    /// The document generation Gemma was given the system prompt for.
    prepared_for: Option<u64>,
    /// What the last request asked and changed, for the next one's message.
    last_change: Option<String>,
    /// What the last request changed, for the next one's `back` items.
    last_changes: Vec<Change>,
    request: Option<Request>,
    next_request_id: u64,
}

/// A request being worked on.
struct Request {
    generation: u64,
    /// The photographer's text.
    text: String,
    /// What it changed, in the order first set.
    changes: Vec<Change>,
    step: Step,
}

/// A value a request changed: from `was`, its value before the request, to
/// `now`.
struct Change {
    tool: String,
    name: String,
    was: Value,
    now: Value,
}

enum Step {
    /// Waiting for the preview to show every edit.
    Capturing(Outgoing),
    /// Gemma is writing the brief; `raw` is its text so far, which streams
    /// into `messages[shown]`.
    Replying {
        request_id: u64,
        shown: usize,
        raw: String,
        receiver: mpsc::Receiver<Result<Reply, String>>,
    },
    /// For the evals: writing the preview that shows the edit, and what the
    /// app set, to the trace.
    Tracing {
        receiver: mpsc::Receiver<Result<(), String>>,
    },
}

/// What the preview is captured for.
enum Outgoing {
    /// To go to Gemma with the photographer's words.
    Request,
    /// For the trace, with what the brief's calls set, a line per call.
    Applied { results: String },
}

/// Gemma's reply as `Platform::gemma_editor_send` returns it; the tool calls
/// it also carries are not read, since the brief asks for none.
#[derive(serde::Deserialize)]
struct Reply {
    text: String,
}

impl EditorState {
    pub(crate) fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Whether Gemma is working on a request.
    pub(crate) fn request_running(&self) -> bool {
        self.request.is_some()
    }

    fn next_request_id(&mut self) -> u64 {
        self.next_request_id = self.next_request_id.wrapping_add(1);
        self.next_request_id
    }
}

impl CalibRawApp {
    /// Opens or closes the AI editor's chat. It shows where the tools do, so
    /// opening it opens the tools' sheet or panel too.
    pub(crate) fn set_ai_editor_open(&mut self, open: bool) {
        self.ai.editor.open = open;
        if open {
            self.develop_ui.sidebar_open = true;
        }
    }

    /// Whether the photographer can send a message now: the photo's triage
    /// card is done and no request is running.
    pub(crate) fn ai_editor_can_send(&self) -> bool {
        self.ai.triage.card.is_some()
            && !self.ai.editor.request_running()
            && !self.ai.editor.draft.trim().is_empty()
    }

    /// Has Gemma read the open photo's editor prompt in the background once
    /// its triage card is done, so the first request does not wait for it.
    pub(super) fn prepare_ai_editor(&mut self) {
        let generation = self.persistence.document_generation;
        if self.ai.triage.card.is_none() || self.ai.editor.prepared_for == Some(generation) {
            return;
        }
        match self.platform.gemma_editor_prepare(INTERPRET_PROMPT) {
            Ok(()) => self.ai.editor.prepared_for = Some(generation),
            Err(error) => log::warn!("{error}"),
        }
    }

    /// Sends the typed message as a request.
    pub(crate) fn send_ai_editor_request(&mut self) {
        if !self.ai_editor_can_send() {
            return;
        }
        let text = std::mem::take(&mut self.ai.editor.draft).trim().to_owned();
        self.ai.editor.messages.push(ChatMessage {
            role: ChatRole::Photographer,
            text: text.clone(),
        });
        self.ai.editor.request = Some(Request {
            generation: self.persistence.document_generation,
            text,
            changes: Vec::new(),
            step: Step::Capturing(Outgoing::Request),
        });
        self.egui_ctx.request_repaint();
    }

    /// Ends the chat when another photo opens, cancelling a running reply.
    pub(in crate::app) fn forget_ai_editor(&mut self) {
        if self.ai.editor.prepared_for.is_some() {
            if let Err(error) = self.platform.gemma_editor_close() {
                log::warn!("{error}");
            }
        }
        self.ai.editor = EditorState::default();
    }

    /// Moves the running request on: sends the preview once it shows every
    /// edit, streams the brief Gemma writes and applies it.
    pub(in crate::app) fn poll_ai_editor(&mut self, frame: &eframe::Frame) {
        // Taken every frame, so text of an earlier reply is dropped.
        if let Some((request_id, text)) = self.platform.take_editor_text() {
            let editor = &mut self.ai.editor;
            if let Some(Request {
                step:
                    Step::Replying {
                        request_id: replying,
                        shown,
                        raw,
                        ..
                    },
                ..
            }) = &mut editor.request
            {
                if *replying == request_id {
                    raw.push_str(&text);
                    if let Some(message) = editor.messages.get_mut(*shown) {
                        message.text = read_reply(raw, is_value_name).0;
                    }
                }
            }
        }

        let Some(request) = self.ai.editor.request.as_ref() else {
            return;
        };
        if request.generation != self.persistence.document_generation {
            self.ai.editor.request = None;
            return;
        }
        match &request.step {
            Step::Capturing(Outgoing::Request) => self.send_to_ai_editor(frame),
            Step::Capturing(Outgoing::Applied { .. }) => self.trace_ai_editor_result(frame),
            Step::Replying { receiver, .. } => {
                let result = match receiver.try_recv() {
                    Ok(result) => result,
                    Err(mpsc::TryRecvError::Empty) => return,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        Err("the AI editor worker stopped".to_owned())
                    }
                };
                match result {
                    Ok(reply) => self.handle_ai_editor_reply(reply, frame),
                    Err(error) => self.fail_ai_editor_request(error),
                }
            }
            Step::Tracing { receiver } => {
                let result = match receiver.try_recv() {
                    Ok(result) => result,
                    Err(mpsc::TryRecvError::Empty) => return,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        Err("the trace worker stopped".to_owned())
                    }
                };
                if let Err(error) = result {
                    log::warn!("AI editor trace failed: {error}");
                }
                self.finish_ai_editor_request();
            }
        }
    }

    /// Once the preview shows every edit, a job that writes it as the JPEG
    /// Gemma reads, and the app's files where it goes. `None` while the
    /// preview renders, and when the request failed.
    fn ai_editor_preview(
        &mut self,
        frame: &eframe::Frame,
    ) -> Option<(
        impl FnOnce() -> Result<(), String> + Send + 'static,
        PathBuf,
    )> {
        // AI denoise holds the preview while it runs, and may first ask the
        // photographer to allow it.
        if self.ai_denoise_running() || self.ai_consent_is_for(AiFeature::Denoise) {
            return None;
        }
        let snapshot = match self.current_preview_snapshot(frame) {
            PreviewSnapshot::Ready(snapshot) => snapshot,
            PreviewSnapshot::Rendering => return None,
            PreviewSnapshot::NoGpu => {
                self.fail_ai_editor_request("the GPU preview is not available".to_owned());
                return None;
            }
        };
        let (Some(render_state), Some(data_dir)) =
            (frame.wgpu_render_state(), self.platform.internal_data_dir())
        else {
            self.fail_ai_editor_request("the AI editor cannot reach the photo".to_owned());
            return None;
        };
        let device = render_state.device.clone();
        let queue = render_state.queue.clone();
        let geometry = self.develop.geometry;
        let jpeg_path = data_dir.join(EDITOR_IMAGE_FILE);
        let write = move || {
            let preview = snapshot
                .read_thumbnail_blocking(&device, &queue, GEMMA_IMAGE_EDGE)
                .map_err(|error| format!("could not read the preview back: {error:#}"))?;
            let preview = crate::pipeline::transform_thumbnail_geometry(&preview, geometry);
            let preview = image::RgbaImage::from_raw(preview.width, preview.height, preview.rgba)
                .ok_or_else(|| "the preview read back with an invalid size".to_owned())?;
            write_gemma_jpeg(preview, &jpeg_path)
        };
        Some((write, data_dir))
    }

    /// Sends the request's message with the preview, once the preview shows
    /// every edit, on a fresh conversation.
    fn send_to_ai_editor(&mut self, frame: &eframe::Frame) {
        let Some(card) = self.ai.triage.card.as_ref() else {
            self.fail_ai_editor_request("the AI editor cannot reach the photo".to_owned());
            return;
        };
        let photo = photo_line(card);
        let Some((write_preview, data_dir)) = self.ai_editor_preview(frame) else {
            return;
        };
        let edit = self.ai_edit_summary();
        let editor = &mut self.ai.editor;
        let Some(request) = editor.request.as_ref() else {
            return;
        };
        let previous = editor
            .last_change
            .as_ref()
            .map(|change| format!("{change}\n"))
            .unwrap_or_default();
        let message = format!(
            "{photo}\n{previous}Request: {}\n\nCurrent values: {edit}",
            request.text
        );
        log::info!("AI editor message: {}", message.replace('\n', " | "));

        let jpeg_path = data_dir.join(EDITOR_IMAGE_FILE);
        let platform = Arc::clone(&self.platform);
        let request_id = editor.next_request_id();
        let receiver = spawn_ui_worker(&self.egui_ctx, move || {
            platform.gemma_editor_start_request(INTERPRET_PROMPT)?;
            write_preview()?;
            let body = platform.gemma_editor_send(request_id, &message, Some(&jpeg_path))?;
            serde_json::from_str::<Reply>(&body).map_err(|error| {
                format!("Gemma's reply is not the expected JSON ({error}): {body}")
            })
        });
        self.ai.editor.prepared_for = Some(self.persistence.document_generation);

        let editor = &mut self.ai.editor;
        editor.messages.push(ChatMessage {
            role: ChatRole::Gemma,
            text: String::new(),
        });
        let shown = editor.messages.len() - 1;
        if let Some(request) = editor.request.as_mut() {
            request.step = Step::Replying {
                request_id,
                shown,
                raw: String::new(),
                receiver,
            };
        }
    }

    /// For the evals: once the preview shows the request's edit, writes it
    /// where the trace's previews are read and what the app set to the
    /// trace.
    fn trace_ai_editor_result(&mut self, frame: &eframe::Frame) {
        let Some((write_preview, data_dir)) = self.ai_editor_preview(frame) else {
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
        let trace = data_dir.join(TRACE_FILE);
        let receiver = spawn_ui_worker(&self.egui_ctx, move || {
            write_preview()?;
            append_trace(&trace, &line)
        });
        if let Some(request) = self.ai.editor.request.as_mut() {
            request.step = Step::Tracing { receiver };
        }
    }

    /// The edit's values away from their defaults, by the tool that sets
    /// them, for a message to Gemma; `tools::describe` says why not in the
    /// call syntax.
    fn ai_edit_summary(&self) -> String {
        let exposure = &self.develop.exposure;
        let raw = self.develop.loaded_raw.as_deref();
        let mut groups: Vec<String> = [
            ("sliders", changed_sliders(exposure, raw)),
            ("colour mixer", changed_mixer(exposure)),
            ("colour grading", changed_grading(exposure)),
            ("tone curves", changed_curves(exposure)),
            ("effects", changed_effects(&self.masks.stack.global_effects)),
        ]
        .into_iter()
        .filter(|(_, values)| !values.is_empty())
        .map(|(group, values)| format!("{group}: {}", describe(&values)))
        .collect();
        if exposure.ai_denoise_enabled {
            groups.push(format!(
                "AI denoise: on, strength {}",
                (exposure.ai_denoise_amount * 100.0).round()
            ));
        }
        if groups.is_empty() {
            "every value at its default".to_owned()
        } else {
            groups.join("; ")
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

    /// Applies the brief Gemma wrote, shown in `messages[shown]`, or ends the
    /// request with the answer to a question.
    fn handle_ai_editor_reply(&mut self, reply: Reply, frame: &eframe::Frame) {
        let Some(Request {
            step: Step::Replying { shown, .. },
            ..
        }) = self.ai.editor.request
        else {
            return;
        };
        let text = reply.text.trim();
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
                role: ChatRole::Gemma,
                text: summary,
            });
        }
        let Some(request) = editor.request.as_mut() else {
            return;
        };
        for (call, response) in calls.iter().zip(&responses) {
            record_changes(&mut request.changes, &call.name, response);
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

    /// Applies `calls` as one nudge from the edit as the reply found it, and
    /// returns each call's response.
    fn apply_ai_editor_calls(&mut self, calls: &[Call], frame: &eframe::Frame) -> Vec<Value> {
        let baseline = self.develop.exposure;
        let baseline_effects = self.masks.stack.global_effects.clone();
        let mut responses = Vec::with_capacity(calls.len());
        let mut denoise_calls = Vec::new();
        for (index, call) in calls.iter().enumerate() {
            let raw = self.develop.loaded_raw.as_deref();
            let exposure = &mut self.develop.exposure;
            let arguments = &call.arguments;
            let response = match call.name.as_str() {
                SET_SLIDERS => apply_set_sliders(exposure, &baseline, raw, arguments),
                SET_COLOR_MIXER => apply_set_color_mixer(exposure, &baseline, arguments),
                SET_COLOR_GRADING => apply_set_color_grading(exposure, &baseline, arguments),
                SET_TONE_CURVE => apply_set_tone_curve(exposure, &baseline, arguments),
                SET_EFFECT => self.apply_set_effect(&baseline_effects, arguments),
                SET_AI_DENOISE => {
                    denoise_calls.push(index);
                    Value::Null
                }
                name => json!({ "error": format!("there is no tool named {name}") }),
            };
            responses.push(response);
        }
        if self.develop.exposure != baseline {
            self.mark_pipeline_dirty();
        }
        // Last, since turning AI denoise on hands the preview to its job.
        for index in denoise_calls {
            responses[index] = self.apply_set_ai_denoise(&calls[index].arguments, frame);
        }
        responses
    }

    /// Applies a `set_ai_denoise` call. Turning it on starts the RawNIND job,
    /// which may first ask the photographer to allow it; the response says
    /// which, since the app's notices never reach Gemma.
    fn apply_set_ai_denoise(&mut self, arguments: &Value, frame: &eframe::Frame) -> Value {
        let Some(enabled) = arguments.get("enabled").and_then(Value::as_bool) else {
            return json!({ "error": "enabled must be true or false" });
        };
        let Some(raw) = self.develop.loaded_raw.as_deref() else {
            return json!({ "error": "no photo is open" });
        };
        if raw.is_pre_demosaiced_raster() {
            return json!({
                "error": "AI denoise works only on RAW photos; use set_sliders' noise_luminance and noise_color"
            });
        }
        let was = json!({
            "enabled": self.develop.exposure.ai_denoise_enabled,
            "strength": (self.develop.exposure.ai_denoise_amount * 100.0).round(),
        });
        if let Some(strength) = arguments.get("strength").and_then(number) {
            let amount = strength.clamp(0.0, 100.0) / 100.0;
            if amount != self.develop.exposure.ai_denoise_amount {
                self.develop.exposure.ai_denoise_amount = amount;
                self.mark_pipeline_dirty();
            }
        }
        if enabled != self.develop.exposure.ai_denoise_enabled {
            if enabled && self.foreground_operation_active() {
                return json!({ "error": "another operation is running; try again once it finishes" });
            }
            self.set_ai_denoise_enabled(enabled, frame);
        }
        let status = if self.ai_denoise_running() {
            "running; it takes about a minute, then the app sends the new preview"
        } else if self.ai_consent_is_for(AiFeature::Denoise) {
            "waiting for the photographer to allow it"
        } else if self.develop.exposure.ai_denoise_enabled {
            "on"
        } else if enabled {
            let reason = self.ui.notice.clone();
            return json!({ "error": reason.unwrap_or_else(|| "AI denoise could not start".to_owned()) });
        } else {
            "off"
        };
        json!({
            "was": was,
            "now": {
                "enabled": self.develop.exposure.ai_denoise_enabled,
                "strength": (self.develop.exposure.ai_denoise_amount * 100.0).round(),
            },
            "status": status,
        })
    }

    /// Ends the running request, keeping what it changed for the next one,
    /// and closes its conversation, which reserves memory until closed.
    fn finish_ai_editor_request(&mut self) {
        let editor = &mut self.ai.editor;
        if let Some(request) = editor.request.take() {
            editor.last_change = Some(change_summary(&request.text, &request.changes));
            editor.last_changes = request.changes;
            if let Err(error) = self.platform.gemma_editor_close() {
                log::warn!("{error}");
            }
        }
    }

    fn fail_ai_editor_request(&mut self, error: String) {
        log::warn!("AI editor request failed: {error}");
        let editor = &mut self.ai.editor;
        if let Some(Request {
            step: Step::Replying { shown, .. },
            ..
        }) = editor.request
        {
            if editor
                .messages
                .get(shown)
                .is_some_and(|message| message.text.is_empty())
            {
                editor.messages.truncate(shown);
            }
        }
        // Its edit stays, so the next request hears of it.
        self.finish_ai_editor_request();
        self.ai.editor.messages.push(ChatMessage {
            role: ChatRole::Failure,
            text: error,
        });
    }
}

/// The tool that sets the value `name`; every value name belongs to one.
fn owner(name: &str) -> Option<&'static str> {
    [
        (SET_SLIDERS, is_slider as fn(&str) -> bool),
        (SET_COLOR_MIXER, is_mixer_value),
        (SET_COLOR_GRADING, is_grading_value),
        (SET_TONE_CURVE, is_curve),
        (SET_EFFECT, is_effect_value),
    ]
    .into_iter()
    .find_map(|(tool, owns)| owns(name).then_some(tool))
}

fn is_value_name(name: &str) -> bool {
    owner(name).is_some()
}

/// `calls` with each value moved to the call of the tool that sets it, in the
/// order the tools are first written: Gemma wrote `vignette_amount` in its
/// `set_sliders` line, which set nothing.
fn routed(calls: Vec<Call>) -> Vec<Call> {
    let mut routed: Vec<Call> = Vec::new();
    for call in calls {
        let arguments = match call.arguments {
            Value::Object(arguments) if call.name != SET_AI_DENOISE && !arguments.is_empty() => {
                arguments
            }
            arguments => {
                routed.push(Call {
                    name: call.name,
                    arguments,
                });
                continue;
            }
        };
        for (name, value) in arguments {
            let tool = owner(&name).map_or_else(|| call.name.clone(), str::to_owned);
            let index = match routed.iter().position(|routed| routed.name == tool) {
                Some(index) => index,
                None => {
                    routed.push(Call {
                        name: tool,
                        arguments: Value::Object(serde_json::Map::new()),
                    });
                    routed.len() - 1
                }
            };
            if let Value::Object(arguments) = &mut routed[index].arguments {
                arguments.insert(name, value);
            }
        }
    }
    routed
}

/// A tool's `response` to a call of `tool`, as a line in the call syntax:
/// `tool name=before→after …`, then what it kept in range or could not set.
fn result_line(tool: &str, response: &Value) -> String {
    if let Some(error) = response.get("error").and_then(Value::as_str) {
        return format!("{tool}: {error}");
    }
    let was = response.get("was").and_then(Value::as_object);
    let mut line = tool.to_owned();
    for (name, now) in response
        .get("now")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        match was.and_then(|was| was.get(name)) {
            Some(was) => line.push_str(&format!(" {name}={}→{}", written(was), written(now))),
            None => line.push_str(&format!(" {name}={}", written(now))),
        }
    }
    for (key, label) in [("limited", "kept within its range"), ("not_set", "not set")] {
        let names = response
            .get(key)
            .and_then(Value::as_array)
            .filter(|names| !names.is_empty());
        if let Some(names) = names {
            let names: Vec<String> = names.iter().map(written).collect();
            line.push_str(&format!("; {label}: {}", names.join(" ")));
        }
    }
    if let Some(status) = response.get("status").and_then(Value::as_str) {
        line.push_str(&format!("; {status}"));
    }
    line
}

/// Adds the values a call of `tool` set, by its `response`, to `changes`.
fn record_changes(changes: &mut Vec<Change>, tool: &str, response: &Value) {
    let (Some(was), Some(now)) = (
        response.get("was").and_then(Value::as_object),
        response.get("now").and_then(Value::as_object),
    ) else {
        return;
    };
    for (name, now) in now {
        let Some(was) = was.get(name) else {
            continue;
        };
        match changes
            .iter_mut()
            .find(|change| change.tool == tool && change.name == *name)
        {
            Some(change) => change.now = now.clone(),
            None => changes.push(Change {
                tool: tool.to_owned(),
                name: name.clone(),
                was: was.clone(),
                now: now.clone(),
            }),
        }
    }
}

/// The value `name` had before the request whose `changes` these are, when
/// it changed it.
fn previous_value(changes: &[Change], name: &str) -> Option<f32> {
    changes
        .iter()
        .find(|change| change.name == name && change.was != change.now)
        .and_then(|change| number(&change.was))
}

/// The request `text` and its `changes` for the next request's message, the
/// changes in the call syntax as `name=before→after`.
fn change_summary(text: &str, changes: &[Change]) -> String {
    let mut groups: Vec<(&str, Vec<String>)> = Vec::new();
    for change in changes.iter().filter(|change| change.was != change.now) {
        let pair = format!(
            "{}={}→{}",
            change.name,
            written(&change.was),
            written(&change.now)
        );
        match groups.iter_mut().find(|(tool, _)| *tool == change.tool) {
            Some((_, pairs)) => pairs.push(pair),
            None => groups.push((&change.tool, vec![pair])),
        }
    }
    if groups.is_empty() {
        return format!("Previous request: \"{text}\". It changed nothing.");
    }
    let changed = groups
        .iter()
        .map(|(tool, pairs)| format!("{tool} {}", pairs.join(" ")))
        .collect::<Vec<_>>()
        .join("\n");
    format!("Previous request: \"{text}\". It changed:\n{changed}")
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
/// timestamp as GemmaBridge writes its own.
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

/// The photo `card` describes, for the request's message. It is not in the
/// system prompt, so Gemma reads it once for every photo and opening a photo
/// costs only its triage.
fn photo_line(card: &TriageCard) -> String {
    format!("Photo: {}. {}", card.genre, card.role)
}
