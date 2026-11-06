//! The AI editor's edit of the open photo (Android), by on-device Gemma or by
//! Claude Code in Termux. The model never touches pixels: it reads the
//! preview and sets the edit's values.
//!
//! - `model` — which model it is, chosen in Settings, and Claude Code's setup.
//! - `triage` — when a photo opens, the model names its genre and fills the
//!   triage card.
//! - `chat` — the AI editor's chat, in which the model edits the photo.
//!
//! Beneath them, with no app state: `calls` reads the tool lines the model
//! writes, and the tools, `sliders`, `color` (mixer and grading),
//! `tone_curve` and `effects`, read and write the edit, keeping each value in
//! its control's range (`tools`). Claude Code has two more, which make masks
//! and their selections (`masks`) and set the crop and geometry (`crop`).
//!
//! Prompts live in android/app/src/main/assets/gemma/ and, for Claude's
//! editor, assets/claude/.

use super::*;

mod calls;
mod chat;
mod color;
mod crop;
mod effects;
mod masks;
mod model;
mod sliders;
mod tone_curve;
mod tools;
mod triage;

pub(crate) use chat::{ChatRole, EditorState};
pub(crate) use model::ClaudeSetupState;
pub(crate) use triage::TriageState;
