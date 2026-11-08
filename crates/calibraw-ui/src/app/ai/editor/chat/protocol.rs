//! Where the two models' requests differ, decided in one place. Gemma reads
//! one message on a fresh conversation and writes a brief, which the app
//! reads into tool calls (gemma_brief.rs). Claude Code keeps one
//! conversation per photo, in which it writes tool lines, looks at the new
//! preview and corrects the edit (claude_conversation.rs). The rest of a
//! request is the same for both (chat.rs).

use super::{claude_conversation, gemma_brief, Outgoing, Reply, Request};
use crate::app::CalibRawApp;
use crate::platform::AiEditorModel;

/// How a request talks to the selected model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Protocol {
    /// Gemma: a brief per request, on a conversation of its own.
    GemmaBrief,
    /// Claude Code: tool lines and corrections, in the photo's conversation.
    ClaudeConversation,
}

/// What a message to the model says besides the photographer's words.
pub(super) struct MessageParts<'a> {
    /// A request's first message: the photo, what the last request changed,
    /// the request and the edit's values.
    pub(super) first: String,
    /// The edit's values away from their defaults.
    pub(super) edit: &'a str,
    /// The masks whose selection images follow the preview, in order.
    pub(super) selections: Vec<&'a str>,
}

impl Protocol {
    pub(super) fn of(model: AiEditorModel) -> Self {
        match model {
            AiEditorModel::Gemma => Self::GemmaBrief,
            AiEditorModel::ClaudeCode => Self::ClaudeConversation,
        }
    }

    /// The system prompt of the model's editor conversations.
    pub(super) fn system_prompt(self) -> &'static str {
        match self {
            Self::GemmaBrief => gemma_brief::SYSTEM_PROMPT,
            Self::ClaudeConversation => claude_conversation::SYSTEM_PROMPT,
        }
    }

    /// Whether the model places masks and sets the crop. It then sees the
    /// preview with Develop's lens correction, where masks are drawn, and
    /// each message names the edit's masks and crop.
    pub(super) fn places_masks_and_crop(self) -> bool {
        self == Self::ClaudeConversation
    }

    /// Whether the model's conversation outlives a request. Gemma's reserves
    /// memory until closed, so it ends with each request; Claude's holds the
    /// photo's whole chat.
    pub(super) fn keeps_conversation(self) -> bool {
        self == Self::ClaudeConversation
    }

    /// The message that sends `request`'s preview for `outgoing`, and the
    /// system prompt of the conversation it starts, or `None` when it
    /// continues the open one. A conversation that outlives the request is
    /// marked open in `conversation_open`.
    pub(super) fn message(
        self,
        conversation_open: &mut bool,
        request: &Request,
        outgoing: &Outgoing,
        parts: MessageParts<'_>,
    ) -> (Option<&'static str>, String) {
        match self {
            Self::GemmaBrief => (Some(gemma_brief::SYSTEM_PROMPT), parts.first),
            Self::ClaudeConversation => {
                let starts = !*conversation_open;
                *conversation_open = true;
                (
                    starts.then_some(claude_conversation::SYSTEM_PROMPT),
                    claude_conversation::message(starts, request, outgoing, parts),
                )
            }
        }
    }

    /// Takes the model's reply to the running request, shown in the chat's
    /// message `shown`, after the request's `nudges`.
    pub(super) fn handle_reply(
        self,
        app: &mut CalibRawApp,
        reply: Reply,
        shown: usize,
        nudges: u32,
        frame: &eframe::Frame,
    ) {
        match self {
            Self::GemmaBrief => app.handle_gemma_brief(reply.text.trim(), shown, frame),
            Self::ClaudeConversation => app.handle_claude_reply(reply, shown, nudges, frame),
        }
    }
}
