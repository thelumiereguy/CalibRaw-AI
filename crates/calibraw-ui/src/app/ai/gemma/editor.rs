//! The AI editor, the second phase of on-device Gemma: a chat about the open
//! photo in which Gemma edits it through six tools: `set_sliders`
//! (sliders.rs), `set_color_mixer` and `set_color_grading` (color.rs),
//! `set_tone_curve` (tone_curve.rs), `set_effect` (effects.rs) and
//! `set_ai_denoise`. Gemma writes its calls as text lines, which the app reads
//! itself (calls.rs). editor_system.md no longer lists `set_tone_curve` or
//! `set_ai_denoise`, which Gemma misused; the app still applies them.
//!
//! A request runs in two phases, each on a fresh conversation that holds its
//! system prompt and nothing else, a clone of one Gemma read when the card
//! came (`Platform::gemma_editor_start_request`), so a request costs the same
//! however long the chat is. First Gemma interprets the photographer's words
//! with the preview into a brief of tone, colour, finish and what to keep
//! (assets/gemma/interpret_system.md), a prompt without tools. Then a second
//! conversation, whose prompt (assets/gemma/editor_system.md) lists the tools
//! and their syntax, carries out the brief without the photographer's words:
//! asked for both at once, Gemma planned a look and then wrote values that
//! crushed the photo black. The first phase's message names the photo's
//! genre and role from its triage card (triage.rs); the second gets the brief
//! without them or its Keep line (`without_keep`). Each says what the
//! previous request changed, as `name=before→after`, so "a bit more" and
//! "undo that" work.
//!
//! The edit is one nudge: the tool lines of the second phase's reply, each
//! applied to the edit, each value anywhere in its range. Once the preview
//! shows the nudge, it goes to Gemma with what the app set, and Gemma says
//! what it changed. A request, whatever its nudges, is one undo step
//! (`observe_edit_history`).
//!
//! Claude Code (`AiEditorModel::ClaudeCode`) needs no phases: it reads the
//! photographer's words and edits in the same reply, in one conversation per
//! photo that holds the whole chat (assets/claude/editor_system.md). After
//! each nudge it looks at the new preview and corrects it, up to
//! [`CLAUDE_MAX_NUDGES`] nudges a request. The conversation stays open
//! between requests and ends when another photo opens or a request fails.
//! Claude has three tools more: `set_mask` and `set_mask_part` (masks.rs)
//! and `set_crop` (crop.rs). After a nudge that set a mask, the preview
//! comes with an image of each such mask's selection.

use super::calls::{read_reply, Call};
use super::color::{
    apply_set_color_grading, apply_set_color_mixer, changed_grading, changed_mixer,
    is_grading_value, is_mixer_value,
};
use super::effects::{changed_effects, is_effect_value};
use super::masks::{mask_named, selection_image, PreviewPlacement, MAX_SELECTION_IMAGES};
use super::sliders::{apply_set_sliders, changed_sliders, is_slider};
use super::tone_curve::{apply_set_tone_curve, changed_curves, is_curve};
use super::tools::{describe, number, written};
use super::triage::{write_gemma_jpeg, TriageCard, GEMMA_IMAGE_EDGE};
use super::*;
use crate::app::preview_capture::PreviewSnapshot;
use crate::platform::AiEditorModel;
use calibraw_ai::AiFeature;
use serde_json::{json, Value};

const INTERPRET_PROMPT: &str =
    include_str!("../../../../../../android/app/src/main/assets/gemma/interpret_system.md");
const EDITOR_PROMPT: &str =
    include_str!("../../../../../../android/app/src/main/assets/gemma/editor_system.md");
const CLAUDE_EDITOR_PROMPT: &str =
    include_str!("../../../../../../android/app/src/main/assets/claude/editor_system.md");
/// How a brief that answers a question instead starts.
const ANSWER: &str = "Answer:";
const EDITOR_IMAGE_FILE: &str = "gemma-editor.jpg";
/// The tools the app applies.
const SET_SLIDERS: &str = "set_sliders";
const SET_COLOR_MIXER: &str = "set_color_mixer";
const SET_COLOR_GRADING: &str = "set_color_grading";
const SET_TONE_CURVE: &str = "set_tone_curve";
const SET_EFFECT: &str = "set_effect";
const SET_AI_DENOISE: &str = "set_ai_denoise";
/// Claude's tools besides those, which assets/claude/editor_system.md
/// describes.
const SET_MASK: &str = "set_mask";
const SET_MASK_PART: &str = "set_mask_part";
const SET_CROP: &str = "set_crop";
/// Nudges one request may make, as the editor prompt tells Gemma. With four,
/// every nudge after the first undid or overdid it (contrast 30→10, shadows
/// and highlights +100), so the photographer steers with follow-ups instead.
const MAX_NUDGES: u32 = 1;
/// Nudges one Claude request may make: its first, then corrections after
/// looking at the preview. Claude ends sooner with a reply without tool
/// lines once the preview shows what was asked.
const CLAUDE_MAX_NUDGES: u32 = 10;

/// Who wrote a chat message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChatRole {
    Photographer,
    Gemma,
    /// What one nudge set.
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
    /// What the last request asked and changed, for the next one's first
    /// message.
    last_change: Option<String>,
    /// Whether Claude's conversation about the photo is open, so a request
    /// continues it rather than starting one.
    claude_conversation: bool,
    request: Option<Request>,
    next_request_id: u64,
}

/// A request being worked on.
struct Request {
    generation: u64,
    /// The photographer's text.
    text: String,
    nudges: u32,
    /// What its nudges changed, in the order first set.
    changes: Vec<Change>,
    /// The masks the last nudge set, by name, whose selections go to Claude
    /// with the next preview.
    selections: Vec<String>,
    step: Step,
}

/// A value a request's nudges changed: from `was`, its value before the
/// first of them, to `now`.
struct Change {
    tool: String,
    name: String,
    was: Value,
    now: Value,
}

enum Step {
    /// Waiting for the preview to show every edit, then sending it.
    Capturing(Outgoing),
    /// Gemma is replying; `raw` is its text so far, of which what it says
    /// streams into `messages[shown]`. `interpreting` while it writes the
    /// brief.
    Replying {
        request_id: u64,
        shown: usize,
        raw: String,
        interpreting: bool,
        receiver: mpsc::Receiver<Result<Reply, String>>,
    },
}

/// What goes to Gemma with the preview.
enum Outgoing {
    /// The photographer's words, to the first phase.
    Request,
    /// The brief the first phase wrote, to the second.
    Brief { brief: String },
    /// The preview after a nudge, with what the nudge set, a line per call.
    Nudged { results: String },
}

/// Gemma's reply as `Platform::gemma_editor_send` returns it.
#[derive(serde::Deserialize)]
struct Reply {
    text: String,
    /// Calls Gemma wrote in its own call syntax that LiteRT-LM parsed.
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

#[derive(serde::Deserialize)]
struct ToolCall {
    name: String,
    #[serde(default)]
    arguments: Value,
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
        let prepared = match self.ai.editor_model {
            AiEditorModel::Gemma => self
                .platform
                .gemma_editor_prepare(INTERPRET_PROMPT)
                .and_then(|()| self.platform.gemma_editor_prepare(EDITOR_PROMPT)),
            AiEditorModel::ClaudeCode => self.platform.gemma_editor_prepare(CLAUDE_EDITOR_PROMPT),
        };
        match prepared {
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
            nudges: 0,
            changes: Vec::new(),
            selections: Vec::new(),
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
        // Request ids keep counting: a cancelled reply's text may still
        // stream in, and must not match the next photo's first request.
        self.ai.editor = EditorState {
            next_request_id: self.ai.editor.next_request_id,
            ..EditorState::default()
        };
    }

    /// Moves the running request on: streams what Gemma says, applies its
    /// nudges and sends the preview once it shows them.
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
            Step::Capturing(_) => self.send_to_ai_editor(frame),
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
        }
    }

    /// Sends the capturing request's message with the preview, once the
    /// preview shows every edit. Each phase's first message starts its
    /// conversation.
    fn send_to_ai_editor(&mut self, frame: &eframe::Frame) {
        // AI denoise holds the preview while it runs, and a model making a
        // mask's selection is not done; either may first ask the
        // photographer to allow it. A person mask made from the people
        // already found runs beside the foreground job (`composes`).
        if self.foreground_operation_active()
            || self.ai.consent.is_some()
            || !self.ai.people.composes.is_empty()
        {
            return;
        }
        let snapshot = match self.current_preview_snapshot(frame) {
            PreviewSnapshot::Ready(snapshot) => snapshot,
            PreviewSnapshot::Rendering => return,
            PreviewSnapshot::NoGpu => {
                self.fail_ai_editor_request("the GPU preview is not available".to_owned());
                return;
            }
        };
        let (Some(render_state), Some(data_dir), Some(card)) = (
            frame.wgpu_render_state(),
            self.platform.internal_data_dir(),
            self.ai.triage.card.as_ref(),
        ) else {
            self.fail_ai_editor_request("the AI editor cannot reach the photo".to_owned());
            return;
        };
        let edit = self.ai_edit_summary();
        let model = self.ai.editor_model;
        // Claude sees the preview as the photographer does, lens-corrected,
        // so its positions are where masks are drawn.
        let lens = match model {
            AiEditorModel::Gemma => None,
            AiEditorModel::ClaudeCode => self
                .develop
                .loaded_raw
                .as_ref()
                .and_then(|raw| raw.lens_geometry.clone()),
        };
        // The masks whose selection images follow the preview, by index.
        let selections: Vec<(String, usize)> = self
            .ai
            .editor
            .request
            .as_ref()
            .map(|request| {
                request
                    .selections
                    .iter()
                    .filter_map(|name| Some((name.clone(), mask_named(&self.masks.stack, name)?)))
                    .take(MAX_SELECTION_IMAGES)
                    .collect()
            })
            .unwrap_or_default();
        // The worker draws the selections, so it takes the masks; their
        // bitmaps are shared (`Arc`), so this copies only their parameters.
        let masks = (!selections.is_empty()).then(|| self.masks.stack.clone());
        let editor = &mut self.ai.editor;
        let Some(request) = editor.request.as_ref() else {
            return;
        };
        let Step::Capturing(outgoing) = &request.step else {
            return;
        };
        let photo = photo_line(card);
        let previous = editor
            .last_change
            .as_ref()
            .map(|change| format!("{change}\n"))
            .unwrap_or_default();
        // The conversation a message starts, by its system prompt, or None
        // to continue the open one.
        let (system_prompt, message) = match model {
            AiEditorModel::Gemma => {
                let system_prompt = match outgoing {
                    Outgoing::Request => Some(INTERPRET_PROMPT),
                    Outgoing::Brief { .. } => Some(EDITOR_PROMPT),
                    Outgoing::Nudged { .. } => None,
                };
                let message = match outgoing {
                    Outgoing::Request => format!(
                        "{photo}\n{previous}Request: {}\n\nCurrent values: {edit}",
                        request.text
                    ),
                    Outgoing::Brief { brief } => {
                        let brief = without_keep(brief);
                        format!("{previous}Brief:\n{brief}\n\nCurrent values: {edit}")
                    }
                    Outgoing::Nudged { results } => format!(
                        "The app set:\n{results}\n\nThe preview after your nudge. Current values: {edit}\nSay what you changed and why, without a tool line."
                    ),
                };
                (system_prompt, message)
            }
            AiEditorModel::ClaudeCode => {
                let starts = !editor.claude_conversation;
                let shown = if selections.is_empty() {
                    String::new()
                } else {
                    let names: Vec<&str> =
                        selections.iter().map(|(name, _)| name.as_str()).collect();
                    format!(
                        "\nAfter the preview, one image per mask you set shows its selection in red and the rest darkened, in this order: {}.",
                        names.join(", ")
                    )
                };
                let message = match outgoing {
                    Outgoing::Request if starts => format!(
                        "{photo}\n{previous}Request: {}\n\nCurrent values: {edit}",
                        request.text
                    ),
                    // Claude never writes a brief.
                    Outgoing::Request | Outgoing::Brief { .. } => {
                        format!("Request: {}\n\nCurrent values: {edit}", request.text)
                    }
                    Outgoing::Nudged { results } if request.nudges < CLAUDE_MAX_NUDGES => format!(
                        "The app set:\n{results}\n\nThe new preview.{shown} Current values: {edit}\nIf it shows what was asked, say what you changed in one or two sentences, without a tool line; otherwise write the tool lines that correct it."
                    ),
                    Outgoing::Nudged { results } => format!(
                        "The app set:\n{results}\n\nThe new preview.{shown} Current values: {edit}\nYou have no corrections left: say what you changed in one or two sentences, without a tool line."
                    ),
                };
                editor.claude_conversation = true;
                (starts.then_some(CLAUDE_EDITOR_PROMPT), message)
            }
        };
        let interpreting = model == AiEditorModel::Gemma && matches!(outgoing, Outgoing::Request);
        log::info!("AI editor message: {}", message.replace('\n', " | "));

        let device = render_state.device.clone();
        let queue = render_state.queue.clone();
        let geometry = self.develop.geometry;
        let jpeg_path = data_dir.join(EDITOR_IMAGE_FILE);
        let platform = Arc::clone(&self.platform);
        let request_id = self.ai.editor.next_request_id();
        let receiver = spawn_ui_worker(&self.egui_ctx, move || {
            if let Some(system_prompt) = system_prompt {
                platform.gemma_editor_start_request(system_prompt)?;
            }
            let preview = snapshot
                .read_thumbnail_blocking(&device, &queue, GEMMA_IMAGE_EDGE)
                .map_err(|error| format!("could not read the preview back: {error:#}"))?;
            let write = |image: &crate::pipeline::RawThumbnail, path: &Path| {
                let image = crate::pipeline::transform_thumbnail_geometry_with_lens(
                    image,
                    geometry,
                    lens.as_deref(),
                );
                let image = image::RgbaImage::from_raw(image.width, image.height, image.rgba)
                    .ok_or_else(|| "the preview read back with an invalid size".to_owned())?;
                write_gemma_jpeg(image, path)
            };
            write(&preview, &jpeg_path)?;
            let mut images = vec![jpeg_path];
            for (number, (_, layer)) in selections.iter().enumerate() {
                let Some(masks) = masks.as_ref() else {
                    break;
                };
                let path = data_dir.join(format!("claude-selection-{number}.jpg"));
                write(
                    &selection_image(masks, *layer, &preview, lens.as_deref()),
                    &path,
                )?;
                images.push(path);
            }
            let images: Vec<&Path> = images.iter().map(PathBuf::as_path).collect();
            let body = platform.gemma_editor_send(request_id, &message, &images)?;
            serde_json::from_str::<Reply>(&body).map_err(|error| {
                format!("the model's reply is not the expected JSON ({error}): {body}")
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
                interpreting,
                receiver,
            };
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
        if self.ai.editor_model == AiEditorModel::ClaudeCode {
            if let Some(masks) = self
                .preview_placement()
                .and_then(|placement| self.mask_summary(&placement))
            {
                groups.push(format!("masks: {masks}"));
            }
            if let Some(geometry) = self.geometry_summary() {
                groups.push(format!("crop: {geometry}"));
            }
        }
        if groups.is_empty() {
            "every value at its default".to_owned()
        } else {
            groups.join("; ")
        }
    }

    /// Applies the reply's tool lines as the next nudge, or finishes the
    /// request when it has none or no nudge is left.
    fn handle_ai_editor_reply(&mut self, reply: Reply, frame: &eframe::Frame) {
        let Some(Request {
            nudges,
            step:
                Step::Replying {
                    shown,
                    interpreting,
                    ..
                },
            ..
        }) = self.ai.editor.request
        else {
            return;
        };
        if interpreting {
            self.handle_ai_editor_brief(reply.text.trim(), shown);
            return;
        }
        let (said, mut calls) = read_reply(&reply.text, is_value_name);
        calls.extend(reply.tool_calls.into_iter().map(|call| Call {
            name: call.name,
            arguments: call.arguments,
        }));
        let calls = routed(calls);
        let max_nudges = match self.ai.editor_model {
            AiEditorModel::Gemma => MAX_NUDGES,
            AiEditorModel::ClaudeCode => CLAUDE_MAX_NUDGES,
        };

        if calls.is_empty() || nudges >= max_nudges {
            if !calls.is_empty() {
                log::info!("AI editor drops tool lines written after the last nudge");
            }
            let messages = &mut self.ai.editor.messages;
            if said.is_empty() {
                messages.truncate(shown);
            } else if let Some(message) = messages.get_mut(shown) {
                message.text = said;
            }
            self.finish_ai_editor_request();
            return;
        }

        let responses = self.apply_ai_editor_calls(&calls, frame);
        let mut selections: Vec<String> = Vec::new();
        for (call, response) in calls.iter().zip(&responses) {
            let set = matches!(call.name.as_str(), SET_MASK | SET_MASK_PART)
                && response.get("error").is_none()
                && response.get("status").and_then(Value::as_str) != Some("deleted");
            let name = call.arguments.get("name").and_then(Value::as_str);
            if let Some(name) =
                name.filter(|name| set && !selections.iter().any(|known| known == name))
            {
                selections.push(name.to_owned());
            }
        }
        let results = calls
            .iter()
            .zip(&responses)
            .map(|(call, response)| result_line(&call.name, response))
            .collect::<Vec<_>>()
            .join("\n");
        let editor = &mut self.ai.editor;
        if let Some(message) = editor.messages.get_mut(shown) {
            message.role = ChatRole::Progress;
            message.text = if said.is_empty() {
                results.clone()
            } else {
                format!("{said}\n{results}")
            };
        }
        let Some(request) = editor.request.as_mut() else {
            return;
        };
        request.nudges += 1;
        request.selections = selections;
        for (call, response) in calls.iter().zip(&responses) {
            record_changes(&mut request.changes, &change_tool(call), response);
        }
        request.step = Step::Capturing(Outgoing::Nudged { results });
        self.egui_ctx.request_repaint();
    }

    /// Takes the first phase's `brief`, shown in `messages[shown]`, to the
    /// second phase, or ends the request with the answer to a question.
    fn handle_ai_editor_brief(&mut self, brief: &str, shown: usize) {
        if brief.is_empty() {
            self.fail_ai_editor_request("Gemma wrote no brief for the request".to_owned());
            return;
        }
        let editor = &mut self.ai.editor;
        if let Some(answer) = brief.strip_prefix(ANSWER) {
            if let Some(message) = editor.messages.get_mut(shown) {
                message.text = answer.trim().to_owned();
            }
            self.finish_ai_editor_request();
            return;
        }
        if let Some(message) = editor.messages.get_mut(shown) {
            message.role = ChatRole::Progress;
            message.text = brief.to_owned();
        }
        if let Some(request) = editor.request.as_mut() {
            request.step = Step::Capturing(Outgoing::Brief {
                brief: brief.to_owned(),
            });
        }
        self.egui_ctx.request_repaint();
    }

    /// Where the fractions of the preview Claude saw lie in the photo.
    fn preview_placement(&self) -> Option<PreviewPlacement> {
        let raw = self.develop.loaded_raw.as_ref()?;
        Some(PreviewPlacement::new(
            self.develop.geometry,
            raw.lens_geometry.clone(),
            raw.width,
            raw.height,
        ))
    }

    /// Applies `calls` as one nudge from the edit as the reply found it, and
    /// returns each call's response.
    fn apply_ai_editor_calls(&mut self, calls: &[Call], frame: &eframe::Frame) -> Vec<Value> {
        let baseline = self.develop.exposure;
        let baseline_effects = self.masks.stack.global_effects.clone();
        // Mask positions are in the preview the reply saw, before this
        // nudge's crop.
        let placement = self.preview_placement();
        let claude = self.ai.editor_model == AiEditorModel::ClaudeCode;
        let mut responses = Vec::with_capacity(calls.len());
        let mut denoise_calls = Vec::new();
        for (index, call) in calls.iter().enumerate() {
            let raw = self.develop.loaded_raw.as_deref();
            let exposure = &mut self.develop.exposure;
            let arguments = &call.arguments;
            let response = match (call.name.as_str(), placement.as_ref()) {
                (SET_SLIDERS, _) => apply_set_sliders(exposure, &baseline, raw, arguments),
                (SET_COLOR_MIXER, _) => apply_set_color_mixer(exposure, &baseline, arguments),
                (SET_COLOR_GRADING, _) => apply_set_color_grading(exposure, &baseline, arguments),
                (SET_TONE_CURVE, _) => apply_set_tone_curve(exposure, &baseline, arguments),
                (SET_EFFECT, _) => self.apply_set_effect(&baseline_effects, arguments),
                (SET_AI_DENOISE, _) => {
                    denoise_calls.push(index);
                    Value::Null
                }
                (SET_MASK, Some(placement)) if claude => {
                    self.apply_set_mask(arguments, placement, frame)
                }
                (SET_MASK_PART, Some(placement)) if claude => {
                    self.apply_set_mask_part(arguments, placement, frame)
                }
                (SET_CROP, _) if claude => self.apply_set_crop(arguments),
                (name, _) => json!({ "error": format!("there is no tool named {name}") }),
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
    /// and closes Gemma's conversation, which reserves memory until closed.
    /// Claude's stays open for the next request.
    fn finish_ai_editor_request(&mut self) {
        let editor = &mut self.ai.editor;
        if let Some(request) = editor.request.take() {
            log::info!("AI editor request ended after {} nudges", request.nudges);
            editor.last_change = Some(change_summary(&request.text, &request.changes));
            if self.ai.editor_model == AiEditorModel::Gemma {
                self.close_ai_editor_conversation();
            }
        }
    }

    fn close_ai_editor_conversation(&mut self) {
        self.ai.editor.claude_conversation = false;
        if let Err(error) = self.platform.gemma_editor_close() {
            log::warn!("{error}");
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
        // Its nudges stay in the edit, so the next request hears of them.
        self.finish_ai_editor_request();
        // Claude's conversation may have lost a turn, so the next request
        // starts a new one.
        if self.ai.editor.claude_conversation {
            self.close_ai_editor_conversation();
        }
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
            // These tools' values are their own, even where another tool
            // has one of the same name, such as a mask's `exposure`.
            Value::Object(arguments)
                if ![SET_AI_DENOISE, SET_MASK, SET_MASK_PART, SET_CROP]
                    .contains(&call.name.as_str())
                    && !arguments.is_empty() =>
            {
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

/// A tool's `response` to a call of `tool`, as a line for Gemma in the call
/// syntax: `tool name=before→after …`, then what it kept in range or could
/// not set.
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

/// The tool `call`'s changes are recorded under: a mask tool's with the
/// mask and part it set, since every mask has an `exposure`.
fn change_tool(call: &Call) -> String {
    if !matches!(call.name.as_str(), SET_MASK | SET_MASK_PART) {
        return call.name.clone();
    }
    let mut tool = call.name.clone();
    for key in ["name", "part"] {
        if let Some(value) = call.arguments.get(key) {
            tool.push_str(&format!(" {key}={}", written(value)));
        }
    }
    tool
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

/// The request `text` and its `changes` for the next request's first
/// message, the changes in the call syntax as `name=before→after`.
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

/// `brief` without its Keep line, for the second phase. What to keep guides
/// the first phase's choice of changes; the second read it, and the genre's
/// role, as more changes: sharpening for "Keep: rotor blades sharp".
fn without_keep(brief: &str) -> String {
    brief
        .lines()
        .filter(|line| !line.trim_start().to_lowercase().starts_with("keep"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The photo `card` describes, for the first phase's message. It is not in
/// the system prompts, so Gemma reads them once for every photo and opening a
/// photo costs only its triage.
fn photo_line(card: &TriageCard) -> String {
    format!("Photo: {}. {}", card.genre, card.role)
}
