//! The AI editor, the second phase after the triage: a chat about the open
//! photo in which the model edits it through its tools: `set_sliders`
//! (sliders.rs), `set_color_mixer` and `set_color_grading` (color.rs),
//! `set_tone_curve` (tone_curve.rs), `set_effect` (effects.rs) and
//! `set_ai_denoise` (denoise.rs), and for Claude Code `set_mask` and
//! `set_mask_part` (masks.rs) and `set_crop` (crop.rs).
//!
//! This module runs a request whatever the model: it waits until the
//! preview shows every edit, sends it with a message, streams the reply into
//! the chat and applies the reply's calls as one nudge. Where the two models
//! differ, it asks the request's [`Protocol`] (protocol.rs): Gemma writes a
//! brief on a conversation of its own per request (gemma_brief.rs), and
//! Claude Code keeps one conversation per photo, in which it corrects its
//! edit after each nudge (claude_conversation.rs).
//!
//! A request, whatever its nudges, is one undo step (`observe_edit_history`).

mod claude_conversation;
mod gemma_brief;
mod protocol;

use super::calls::{read_reply, Call};
use super::color::{
    apply_set_color_grading, apply_set_color_mixer, changed_grading, changed_mixer,
};
use super::effects::changed_effects;
use super::masks::{mask_named, selection_image, PreviewPlacement, MAX_SELECTION_IMAGES};
use super::sliders::{apply_set_sliders, changed_sliders};
use super::tone_curve::{apply_set_tone_curve, changed_curves};
use super::tools::{
    change_summary, describe, is_value_name, Change, SET_AI_DENOISE, SET_COLOR_GRADING,
    SET_COLOR_MIXER, SET_CROP, SET_EFFECT, SET_MASK, SET_MASK_PART, SET_SLIDERS, SET_TONE_CURVE,
};
use super::triage::{write_model_jpeg, TriageCard, MODEL_IMAGE_EDGE};
use super::*;
use crate::app::preview_capture::PreviewSnapshot;
use crate::i18n::tr_fmt;
use crate::pipeline::{GpuOutputSnapshot, LensGeometryMap, RawThumbnail};
use protocol::{MessageParts, Protocol};
use serde_json::{json, Value};

const EDITOR_IMAGE_FILE: &str = "ai-editor.jpg";

/// Who wrote a chat message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChatRole {
    Photographer,
    /// The selected model, Gemma or Claude Code.
    Model,
    /// What a nudge set.
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
    /// The document generation the model was given the system prompt for.
    prepared_for: Option<u64>,
    /// What the last request asked and changed, for the next one's first
    /// message.
    last_change: Option<String>,
    /// What the last request changed, for the next Gemma brief's `back`
    /// items.
    last_changes: Vec<Change>,
    /// Whether the model's conversation about the photo is open, so a
    /// request continues it rather than starting one
    /// ([`Protocol::keeps_conversation`]).
    conversation_open: bool,
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
    /// The masks the last nudge set, by name, whose selections go to the
    /// model with the next preview.
    selections: Vec<String>,
    step: Step,
}

enum Step {
    /// Waiting for the preview to show every edit.
    Capturing(Outgoing),
    /// The model is replying; `raw` is its text so far, of which what it says
    /// streams into `messages[shown]`.
    Replying {
        request_id: u64,
        shown: usize,
        raw: String,
        receiver: mpsc::Receiver<Result<Reply, String>>,
    },
    /// For the evals: writing the preview that shows Gemma's edit, and what
    /// the app set, to the trace.
    Tracing {
        receiver: mpsc::Receiver<Result<(), String>>,
    },
}

/// What the preview is captured for.
enum Outgoing {
    /// To go to the model with the photographer's words.
    Request,
    /// To go to Claude after a nudge, with what the nudge set, a line per
    /// call.
    Nudged { results: String },
    /// For the trace, with what Gemma's brief set, a line per call.
    Applied { results: String },
}

/// The model's reply as `Platform::ai_editor_send` returns it.
#[derive(serde::Deserialize)]
struct Reply {
    text: String,
    /// Calls Gemma wrote in its own call syntax that LiteRT-LM parsed; the
    /// brief asks for none, so only Claude's are applied.
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

#[derive(serde::Deserialize)]
struct ToolCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

/// The preview as the model sees it, read back on a worker and written as
/// the JPEGs the model reads.
struct PreviewCapture {
    snapshot: GpuOutputSnapshot,
    device: wgpu::Device,
    queue: wgpu::Queue,
    geometry: GeometryTransform,
    /// The lens correction Develop shows, which the preview has when the
    /// model places masks, so its positions are where masks are drawn.
    lens: Option<Arc<LensGeometryMap>>,
}

impl PreviewCapture {
    /// The preview, read back at the size the model reads.
    fn read(&self) -> Result<RawThumbnail, String> {
        self.snapshot
            .read_thumbnail_blocking(&self.device, &self.queue, MODEL_IMAGE_EDGE)
            .map_err(|error| format!("could not read the preview back: {error:#}"))
    }

    /// Writes `image`, read back by [`Self::read`], as the model sees it.
    fn write(&self, image: &RawThumbnail, path: &Path) -> Result<(), String> {
        let image = crate::pipeline::transform_thumbnail_geometry_with_lens(
            image,
            self.geometry,
            self.lens.as_deref(),
        );
        let image = image::RgbaImage::from_raw(image.width, image.height, image.rgba)
            .ok_or_else(|| "the preview read back with an invalid size".to_owned())?;
        write_model_jpeg(image, path)
    }
}

impl EditorState {
    pub(crate) fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Whether the model is working on a request.
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

    /// How requests talk to the selected model.
    fn ai_editor_protocol(&self) -> Protocol {
        Protocol::of(self.ai.editor_model)
    }

    /// Has the model read the open photo's editor prompt in the background
    /// once its triage card is done, so the first request does not wait for it.
    pub(super) fn prepare_ai_editor(&mut self) {
        let generation = self.persistence.document_generation;
        if self.ai.triage.card.is_none() || self.ai.editor.prepared_for == Some(generation) {
            return;
        }
        let system_prompt = self.ai_editor_protocol().system_prompt();
        match self.platform.ai_editor_prepare(system_prompt) {
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
            if let Err(error) = self.platform.ai_editor_close() {
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

    /// Moves the running request on: sends the preview once it shows every
    /// edit, streams what the model writes and applies it.
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
            Step::Capturing(Outgoing::Applied { .. }) => self.trace_ai_editor_result(frame),
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

    /// Once the preview shows every edit, what a worker needs to read it back
    /// as the model sees it, and the app's files where it goes. `None` while
    /// the preview renders, and when the request failed.
    fn ai_editor_preview(&mut self, frame: &eframe::Frame) -> Option<(PreviewCapture, PathBuf)> {
        // AI denoise holds the preview while it runs, and a model making a
        // mask's selection is not done; either may first ask the
        // photographer to allow it. A person mask made from the people
        // already found runs beside the foreground job (`composes`).
        if self.foreground_operation_active()
            || self.ai.consent.is_some()
            || !self.ai.people.composes.is_empty()
        {
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
        let lens = if self.ai_editor_protocol().places_masks_and_crop() {
            self.develop
                .loaded_raw
                .as_ref()
                .and_then(|raw| raw.lens_geometry.clone())
        } else {
            None
        };
        let capture = PreviewCapture {
            snapshot,
            device: render_state.device.clone(),
            queue: render_state.queue.clone(),
            geometry: self.develop.geometry,
            lens,
        };
        Some((capture, data_dir))
    }

    /// Sends the capturing request's message with the preview, once the
    /// preview shows every edit, starting a conversation or continuing the
    /// open one as the protocol says.
    fn send_to_ai_editor(&mut self, frame: &eframe::Frame) {
        let Some(card) = self.ai.triage.card.as_ref() else {
            self.fail_ai_editor_request("the AI editor cannot reach the photo".to_owned());
            return;
        };
        let photo = photo_line(card);
        let Some((capture, data_dir)) = self.ai_editor_preview(frame) else {
            return;
        };
        let edit = self.ai_edit_summary();
        let protocol = self.ai_editor_protocol();
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
        let previous = editor
            .last_change
            .as_ref()
            .map(|change| format!("{change}\n"))
            .unwrap_or_default();
        let parts = MessageParts {
            first: format!(
                "{photo}\n{previous}Request: {}\n\nCurrent values: {edit}",
                request.text
            ),
            edit: &edit,
            selections: selections.iter().map(|(name, _)| name.as_str()).collect(),
        };
        let (system_prompt, message) =
            protocol.message(&mut editor.conversation_open, request, outgoing, parts);
        log::info!("AI editor message: {}", message.replace('\n', " | "));

        let jpeg_path = data_dir.join(EDITOR_IMAGE_FILE);
        let platform = Arc::clone(&self.platform);
        let request_id = editor.next_request_id();
        let receiver = spawn_ui_worker(&self.egui_ctx, move || {
            if let Some(system_prompt) = system_prompt {
                platform.ai_editor_start_request(system_prompt)?;
            }
            let preview = capture.read()?;
            capture.write(&preview, &jpeg_path)?;
            let mut images = vec![jpeg_path];
            for (number, (_, layer)) in selections.iter().enumerate() {
                let Some(masks) = masks.as_ref() else {
                    break;
                };
                let path = data_dir.join(format!("claude-selection-{number}.jpg"));
                capture.write(
                    &selection_image(masks, *layer, &preview, capture.lens.as_deref()),
                    &path,
                )?;
                images.push(path);
            }
            let images: Vec<&Path> = images.iter().map(PathBuf::as_path).collect();
            let body = platform.ai_editor_send(request_id, &message, &images)?;
            serde_json::from_str::<Reply>(&body).map_err(|error| {
                format!("the model's reply is not the expected JSON ({error}): {body}")
            })
        });
        self.ai.editor.prepared_for = Some(self.persistence.document_generation);

        let editor = &mut self.ai.editor;
        editor.messages.push(ChatMessage {
            role: ChatRole::Model,
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

    /// The edit's values away from their defaults, by the tool that sets
    /// them, for a message to the model; `tools::describe` says why not in the
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
        if self.ai_editor_protocol().places_masks_and_crop() {
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

    /// Takes the model's reply, shown in `messages[shown]`.
    fn handle_ai_editor_reply(&mut self, reply: Reply, frame: &eframe::Frame) {
        let Some(Request {
            nudges,
            step: Step::Replying { shown, .. },
            ..
        }) = self.ai.editor.request
        else {
            return;
        };
        let protocol = self.ai_editor_protocol();
        protocol.handle_reply(self, reply, shown, nudges, frame);
    }

    /// Where the fractions of the preview the model saw lie in the photo.
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
        let masks_and_crop = self.ai_editor_protocol().places_masks_and_crop();
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
                (SET_MASK, Some(placement)) if masks_and_crop => {
                    self.apply_set_mask(arguments, placement, frame)
                }
                (SET_MASK_PART, Some(placement)) if masks_and_crop => {
                    self.apply_set_mask_part(arguments, placement, frame)
                }
                (SET_CROP, _) if masks_and_crop => self.apply_set_crop(arguments),
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

    /// Ends the running request, keeping what it changed for the next one,
    /// and closes the model's conversation unless it outlives the request.
    fn finish_ai_editor_request(&mut self) {
        let editor = &mut self.ai.editor;
        if let Some(request) = editor.request.take() {
            log::info!("AI editor request ended after {} nudges", request.nudges);
            editor.last_change = Some(change_summary(&request.text, &request.changes));
            editor.last_changes = request.changes;
            if !self.ai_editor_protocol().keeps_conversation() {
                self.close_ai_editor_conversation();
            }
        }
    }

    fn close_ai_editor_conversation(&mut self) {
        self.ai.editor.conversation_open = false;
        if let Err(error) = self.platform.ai_editor_close() {
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
        // A conversation that outlives requests may have lost a turn, so the
        // next request starts a new one.
        if self.ai.editor.conversation_open {
            self.close_ai_editor_conversation();
        }
        self.ai.editor.messages.push(ChatMessage {
            role: ChatRole::Failure,
            text: tr_fmt("The request stopped: {error}", &[("error", &error)]),
        });
    }
}

/// The photo `card` describes, for the request's first message. It is not
/// in the system prompts, so the model reads them once for every photo and
/// opening a photo costs only its triage.
fn photo_line(card: &TriageCard) -> String {
    format!("Photo: {}. {}", card.genre, card.role)
}
