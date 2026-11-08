//! The AI editor's edit of the open photo (Android), by on-device Gemma or by
//! Claude Code in Termux. The model never touches pixels: it reads the
//! preview and sets the edit's values.
//!
//! - `model` — which model it is, chosen in Settings, and Claude Code's setup.
//! - `triage` — when a photo opens, the model names its genre and fills the
//!   triage card.
//! - `chat` — the AI editor's chat, in which the model edits the photo. Its
//!   `protocol` decides where the two models' requests differ:
//!   `gemma_brief` for Gemma, `claude_conversation` for Claude Code. Its
//!   readers turn a reply into tool calls: `brief` reads the brief Gemma
//!   writes, `calls` the tool lines Claude writes.
//! - `tools` — the tools, which read and write the edit, keeping each value
//!   in its control's range: `sliders`, `color` (mixer and grading),
//!   `tone_curve` and `effects`, with no app state, and `denoise`, which
//!   turns AI denoise on or off. Claude Code has two more, which make masks
//!   and their selections (`masks`) and set the crop and geometry (`crop`).
//!   The module itself holds the call they are given, routes each value to
//!   its tool and records what a request changed.
//!
//! Prompts live in android/app/src/main/assets/gemma/ and, for Claude's
//! editor, assets/claude/.

use super::*;

mod chat;
mod model;
mod tools;
mod triage;

pub(crate) use chat::{ChatRole, EditorState};
pub(crate) use model::ClaudeSetupState;
pub(crate) use triage::TriageState;
