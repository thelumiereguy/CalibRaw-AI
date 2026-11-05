//! The AI editor, the second phase of on-device Gemma: a chat about the open
//! photo in which Gemma moves the sliders through its `set_sliders` tool
//! (sliders.rs). Its conversation starts with
//! android/app/src/main/assets/gemma/editor_system.md filled in from the
//! photo's triage card (triage.rs).
//!
//! A request runs as a loop of nudges. The photographer's text goes to Gemma
//! with the preview as it is. A `set_sliders` call is applied to the edit and
//! answered with the sliders as they now are, and Gemma says in a few words
//! what it changed. Once the preview shows the nudge, it goes to Gemma as a
//! message of its own, since Gemma 4's chat template renders only text in a
//! tool response, and Gemma nudges again or finishes. The nudges of one
//! request make one undo step (`observe_edit_history`).

use super::sliders::{apply_set_sliders, describe, slider_values};
use super::triage::{write_gemma_jpeg, TriageCard, GEMMA_IMAGE_EDGE};
use super::*;
use crate::app::preview_capture::PreviewSnapshot;
use serde_json::{json, Value};

const EDITOR_PROMPT: &str =
    include_str!("../../../../../android/app/src/main/assets/gemma/editor_system.md");
const EDITOR_IMAGE_FILE: &str = "gemma-editor.jpg";
const SET_SLIDERS: &str = "set_sliders";
/// Nudges one request may make, as the editor prompt tells Gemma.
const MAX_NUDGES: u32 = 4;

/// Who wrote a chat message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChatRole {
    Photographer,
    Gemma,
    /// Gemma's few words after a nudge.
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
    /// The document generation Gemma's conversation was started for.
    started_for: Option<u64>,
    request: Option<Request>,
    next_request_id: u64,
}

/// A request being worked on.
struct Request {
    generation: u64,
    nudges: u32,
    /// A nudge was applied that Gemma has not seen in a preview yet.
    unseen_nudge: bool,
    step: Step,
}

enum Step {
    /// Waiting for the preview to show every edit, then sending it.
    Capturing(Outgoing),
    /// Gemma is replying; its text streams into `messages[shown]`.
    Replying {
        request_id: u64,
        after_nudge: bool,
        shown: usize,
        receiver: mpsc::Receiver<Result<Reply, String>>,
    },
}

/// What goes to Gemma with the preview.
enum Outgoing {
    Request(String),
    /// The preview after a nudge.
    Nudged,
}

/// Gemma's reply as `Platform::gemma_editor_send` returns it.
#[derive(serde::Deserialize)]
struct Reply {
    text: String,
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
            nudges: 0,
            unseen_nudge: false,
            step: Step::Capturing(Outgoing::Request(text)),
        });
        self.egui_ctx.request_repaint();
    }

    /// Ends the chat when another photo opens, cancelling a running reply.
    pub(in crate::app) fn forget_ai_editor(&mut self) {
        if self.ai.editor.started_for.is_some() {
            if let Err(error) = self.platform.gemma_editor_close() {
                log::warn!("{error}");
            }
        }
        self.ai.editor = EditorState::default();
    }

    /// Moves the running request on: streams Gemma's text, applies its
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
                        ..
                    },
                ..
            }) = &editor.request
            {
                if *replying == request_id {
                    if let Some(message) = editor.messages.get_mut(*shown) {
                        message.text.push_str(&text);
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
                    Ok(reply) => self.handle_ai_editor_reply(reply),
                    Err(error) => self.fail_ai_editor_request(error),
                }
            }
        }
    }

    /// Sends the capturing request's message with the preview, once the
    /// preview shows every edit.
    fn send_to_ai_editor(&mut self, frame: &eframe::Frame) {
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
        let generation = self.persistence.document_generation;
        let system_prompt =
            (self.ai.editor.started_for != Some(generation)).then(|| editor_prompt(card));
        let Some(request) = self.ai.editor.request.as_mut() else {
            return;
        };
        let Step::Capturing(outgoing) = &request.step else {
            return;
        };
        let sliders = describe(&slider_values(
            &self.develop.exposure,
            self.develop.loaded_raw.as_deref(),
        ));
        let message = match outgoing {
            Outgoing::Request(text) => format!("{text}\n\nSliders now: {sliders}."),
            Outgoing::Nudged if request.nudges >= MAX_NUDGES => {
                format!("The preview after your last nudge. Sliders now: {sliders}. Finish now.")
            }
            Outgoing::Nudged => format!(
                "The preview after nudge {}. Sliders now: {sliders}. Nudge again, or finish.",
                request.nudges
            ),
        };
        request.unseen_nudge = false;

        let device = render_state.device.clone();
        let queue = render_state.queue.clone();
        let geometry = self.develop.geometry;
        let jpeg_path = data_dir.join(EDITOR_IMAGE_FILE);
        let platform = Arc::clone(&self.platform);
        let request_id = self.ai.editor.next_request_id();
        let receiver = spawn_ui_worker(&self.egui_ctx, move || {
            if let Some(system_prompt) = system_prompt {
                platform.gemma_editor_start(&system_prompt)?;
            }
            let preview = snapshot
                .read_thumbnail_blocking(&device, &queue, GEMMA_IMAGE_EDGE)
                .map_err(|error| format!("could not read the preview back: {error:#}"))?;
            let preview = crate::pipeline::transform_thumbnail_geometry(&preview, geometry);
            let preview = image::RgbaImage::from_raw(preview.width, preview.height, preview.rgba)
                .ok_or_else(|| "the preview read back with an invalid size".to_owned())?;
            write_gemma_jpeg(preview, &jpeg_path)?;
            let body = platform.gemma_editor_send(request_id, &message, Some(&jpeg_path))?;
            parse_reply(&body)
        });
        self.ai.editor.started_for = Some(generation);
        self.start_ai_editor_reply(request_id, false, receiver);
    }

    /// Shows the reply `receiver` brings as the next message.
    fn start_ai_editor_reply(
        &mut self,
        request_id: u64,
        after_nudge: bool,
        receiver: mpsc::Receiver<Result<Reply, String>>,
    ) {
        let editor = &mut self.ai.editor;
        editor.messages.push(ChatMessage {
            role: if after_nudge {
                ChatRole::Progress
            } else {
                ChatRole::Gemma
            },
            text: String::new(),
        });
        let shown = editor.messages.len() - 1;
        if let Some(request) = editor.request.as_mut() {
            request.step = Step::Replying {
                request_id,
                after_nudge,
                shown,
                receiver,
            };
        }
    }

    fn handle_ai_editor_reply(&mut self, reply: Reply) {
        let Some(request) = self.ai.editor.request.as_mut() else {
            return;
        };
        let Step::Replying {
            after_nudge, shown, ..
        } = request.step
        else {
            return;
        };
        let text = reply.text.trim();
        if text.is_empty() {
            self.ai.editor.messages.truncate(shown);
        } else if let Some(message) = self.ai.editor.messages.get_mut(shown) {
            message.text = text.to_owned();
        }

        if reply.tool_calls.is_empty() {
            if after_nudge && request.unseen_nudge {
                request.step = Step::Capturing(Outgoing::Nudged);
                self.egui_ctx.request_repaint();
            } else {
                self.ai.editor.request = None;
            }
            return;
        }

        let nudge_allowed = request.nudges < MAX_NUDGES;
        let mut exposure = self.develop.exposure;
        let responses: Vec<Value> = reply
            .tool_calls
            .iter()
            .map(|call| {
                let response = if call.name != SET_SLIDERS {
                    json!({ "error": format!("there is no tool named {}", call.name) })
                } else if !nudge_allowed {
                    json!({ "error": "no nudges are left; finish now without a tool call" })
                } else {
                    apply_set_sliders(
                        &mut exposure,
                        self.develop.loaded_raw.as_deref(),
                        &call.arguments,
                    )
                };
                json!({ "name": call.name, "response": response })
            })
            .collect();
        if nudge_allowed {
            request.nudges += 1;
            request.unseen_nudge = true;
        }
        if exposure != self.develop.exposure {
            self.develop.exposure = exposure;
            self.mark_pipeline_dirty();
        }

        let responses = Value::Array(responses).to_string();
        let platform = Arc::clone(&self.platform);
        let request_id = self.ai.editor.next_request_id();
        let receiver = spawn_ui_worker(&self.egui_ctx, move || {
            parse_reply(&platform.gemma_editor_tool_responses(request_id, &responses)?)
        });
        self.start_ai_editor_reply(request_id, true, receiver);
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
        editor.request = None;
        editor.messages.push(ChatMessage {
            role: ChatRole::Failure,
            text: error,
        });
    }
}

fn parse_reply(body: &str) -> Result<Reply, String> {
    serde_json::from_str(body)
        .map_err(|error| format!("Gemma's reply is not the expected JSON ({error}): {body}"))
}

/// The editor's system prompt for the photo `card` describes.
fn editor_prompt(card: &TriageCard) -> String {
    let bullets = |items: &[String]| {
        items
            .iter()
            .map(|item| format!("- {item}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    EDITOR_PROMPT
        .replace("{{genre}}", &card.genre)
        .replace("{{role}}", &card.role)
        .replace("{{observations}}", &bullets(&card.observations))
        .replace("{{priorities}}", &bullets(&card.editing_priorities))
}
