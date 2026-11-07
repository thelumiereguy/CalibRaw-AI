//! On-device Gemma's edit of the open photo (Android). Gemma never touches
//! pixels: it reads the preview and sets the edit's values.
//!
//! - `triage` — when a photo opens, Gemma names its genre and fills the
//!   triage card.
//! - `editor` — the AI editor's chat, in which Gemma edits the photo.
//!
//! Beneath them, with no app state: `brief` reads the brief Gemma writes for
//! a request into tool calls, `calls` reads tool lines Gemma writes on its
//! own, and the tools, `sliders`, `color` (mixer and grading), `tone_curve`
//! and `effects`, read and write the edit, keeping each value in its
//! control's range (`tools`).
//!
//! Prompts live in android/app/src/main/assets/gemma/.

use super::*;

mod brief;
mod calls;
mod color;
mod editor;
mod effects;
mod sliders;
mod tone_curve;
mod tools;
mod triage;

pub(crate) use editor::{ChatRole, EditorState};
pub(crate) use triage::TriageState;
