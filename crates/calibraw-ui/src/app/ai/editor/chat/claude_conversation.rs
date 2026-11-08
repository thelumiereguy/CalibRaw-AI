//! Claude Code's requests (`Protocol::ClaudeConversation`). Claude writes no
//! brief: it reads the photographer's words and writes its calls as text
//! lines in the same reply, which the app reads itself (calls.rs), in one
//! conversation per photo that holds the whole chat
//! (assets/claude/editor_system.md). After each nudge it looks at the new
//! preview and corrects it, up to [`MAX_NUDGES`] nudges a request. The
//! conversation stays open between requests and ends when another photo
//! opens or a request fails. Claude has three tools more than Gemma:
//! `set_mask` and `set_mask_part` (masks.rs) and `set_crop` (crop.rs). After
//! a nudge that set a mask, the preview comes with an image of each such
//! mask's selection.

use super::super::calls::{read_reply, Call};
use super::super::tools::{
    change_tool, is_value_name, record_changes, result_line, routed, SET_MASK, SET_MASK_PART,
};
use super::protocol::MessageParts;
use super::*;

pub(super) const SYSTEM_PROMPT: &str =
    include_str!("../../../../../../../android/app/src/main/assets/claude/editor_system.md");
/// Nudges one Claude request may make: its first, then corrections after
/// looking at the preview. Claude ends sooner with a reply without tool
/// lines once the preview shows what was asked.
const MAX_NUDGES: u32 = 10;

/// The message that sends `request`'s preview for `outgoing`: the request's
/// first message when it `starts` the conversation, the request alone when
/// it continues it, or what the last nudge set.
pub(super) fn message(
    starts: bool,
    request: &Request,
    outgoing: &Outgoing,
    parts: MessageParts<'_>,
) -> String {
    let edit = parts.edit;
    let shown = if parts.selections.is_empty() {
        String::new()
    } else {
        format!(
            "\nAfter the preview, one image per mask you set shows its selection in red and the rest darkened, in this order: {}.",
            parts.selections.join(", ")
        )
    };
    // `Applied` is the Gemma trace's, which `poll_ai_editor` hands to
    // `trace_ai_editor_result` instead.
    match outgoing {
        Outgoing::Request if starts => parts.first,
        Outgoing::Request | Outgoing::Applied { .. } => {
            format!("Request: {}\n\nCurrent values: {edit}", request.text)
        }
        // The system prompt says how to answer a nudge; only the last one
        // says more.
        Outgoing::Nudged { results } if request.nudges < MAX_NUDGES => {
            format!("The app set:\n{results}\n\nThe new preview.{shown} Current values: {edit}")
        }
        Outgoing::Nudged { results } => format!(
            "The app set:\n{results}\n\nThe new preview.{shown} Current values: {edit}\nYou have no corrections left, so reply without a tool line."
        ),
    }
}

impl CalibRawApp {
    /// Applies the tool lines of Claude's reply as the next nudge, or
    /// finishes the request when it has none or no nudge is left.
    pub(super) fn handle_claude_reply(
        &mut self,
        reply: Reply,
        shown: usize,
        nudges: u32,
        frame: &eframe::Frame,
    ) {
        let (said, mut calls) = read_reply(&reply.text, is_value_name);
        calls.extend(reply.tool_calls.into_iter().map(|call| Call {
            name: call.name,
            arguments: call.arguments,
        }));
        let calls = routed(calls);

        if calls.is_empty() || nudges >= MAX_NUDGES {
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
}
