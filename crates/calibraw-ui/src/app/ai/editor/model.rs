//! The model behind the triage and the AI editor, which the photographer
//! chooses in Settings: Gemma on the device or Claude Code in Termux, with
//! the Claude model it runs. Also Claude Code's setup, which Settings checks
//! and runs step by step.

use super::*;
use crate::platform::{AiEditorModel, ClaudeCodeSetup, ClaudeCodeStatus, ClaudeModel};

/// How far Claude Code is set up, as last checked.
#[derive(Default)]
pub(crate) struct ClaudeSetupState {
    pub(crate) status: Option<ClaudeCodeStatus>,
    /// The running check, which takes seconds while Termux starts.
    checking: Option<mpsc::Receiver<Result<ClaudeCodeStatus, String>>>,
    /// Why the last check or setup step failed.
    pub(crate) failure: Option<String>,
}

impl ClaudeSetupState {
    pub(crate) fn checking(&self) -> bool {
        self.checking.is_some()
    }
}

impl AiState {
    /// The model chosen in Settings, which the performance settings keep:
    /// Claude Code while it is set aside for want of internet.
    pub(crate) fn chosen_editor_model(&self) -> AiEditorModel {
        if self.offline_fallback {
            AiEditorModel::ClaudeCode
        } else {
            self.editor_model
        }
    }
}

impl CalibRawApp {
    /// Hands the triage and the AI editor to the model the app starts with:
    /// Gemma when Claude Code is chosen but the device has no internet, which
    /// Claude Code needs. Only the start decides; the internet coming back
    /// later does not switch, as switching ends the open photo's chat.
    pub(in crate::app) fn select_initial_ai_model(&mut self) {
        if !self.caps().ai_editor {
            return;
        }
        if self.ai.editor_model == AiEditorModel::ClaudeCode {
            // A failed check keeps the chosen model; Claude's own failure
            // then says why.
            let online = self.platform.internet_available().unwrap_or_else(|error| {
                log::warn!("{error}");
                true
            });
            if !online {
                log::info!("No internet at start: the AI editor uses Gemma instead of Claude Code");
                self.ai.editor_model = AiEditorModel::Gemma;
                self.ai.offline_fallback = true;
            }
        }
        if let Err(error) = self
            .platform
            .select_ai_model(self.ai.editor_model, &self.ai.claude_model)
        {
            log::warn!("{error}");
        }
        if self.ai.editor_model == AiEditorModel::ClaudeCode {
            self.check_claude_code();
        }
    }

    /// Makes `model` the one behind the triage and the AI editor. The open
    /// photo's chat and card are forgotten, and the photo is triaged again by
    /// the new model when Develop next shows it (Claude Code: when the chat
    /// next opens). Choosing a model ends a start without internet's use of
    /// Gemma; choosing Gemma then keeps the chat and saves Gemma as the
    /// choice.
    pub(crate) fn set_ai_editor_model(&mut self, model: AiEditorModel) {
        if self.ai.editor_model == model {
            if self.ai.offline_fallback {
                self.ai.offline_fallback = false;
                self.persist_performance_settings();
            }
            return;
        }
        self.ai.offline_fallback = false;
        self.ai.editor_model = model;
        self.reselect_ai_model();
        self.persist_performance_settings();
        if model == AiEditorModel::ClaudeCode {
            self.check_claude_code();
        }
    }

    /// Makes `model` the one Claude Code runs. While Claude Code is in use,
    /// the open photo's chat and card are forgotten, as for a change of
    /// model, since both were made by the model before; while Gemma stands in
    /// for it after a start without internet, the choice is only saved.
    pub(crate) fn set_claude_model(&mut self, model: ClaudeModel) {
        if self.ai.claude_model == model {
            return;
        }
        self.ai.claude_model = model;
        if self.ai.editor_model == AiEditorModel::ClaudeCode {
            self.reselect_ai_model();
        }
        self.persist_performance_settings();
    }

    /// Hands the triage and the AI editor to the model and Claude model now
    /// chosen: forgets the open photo's chat and card, and triages the photo
    /// again when Develop next shows it (Claude Code: when the chat next
    /// opens). Selecting again ends every conversation the model before
    /// started ahead of time.
    fn reselect_ai_model(&mut self) {
        self.forget_ai_editor();
        self.ai.triage.forget();
        if let Err(error) = self
            .platform
            .select_ai_model(self.ai.editor_model, &self.ai.claude_model)
        {
            log::warn!("{error}");
        }
        self.ai.triage.waiting = self.develop.loaded_raw.is_some();
    }

    /// Checks again how far Claude Code is set up, on a worker.
    pub(crate) fn check_claude_code(&mut self) {
        if self.ai.claude_setup.checking() {
            return;
        }
        let platform = Arc::clone(&self.platform);
        self.ai.claude_setup.failure = None;
        self.ai.claude_setup.checking = Some(spawn_ui_worker(&self.egui_ctx, move || {
            platform.claude_code_status()
        }));
    }

    /// Runs a Claude Code setup `step`; the photographer checks again after
    /// finishing it in Termux or the permission dialog.
    pub(crate) fn run_claude_code_setup(&mut self, step: ClaudeCodeSetup) {
        self.ai.claude_setup.failure = self.platform.claude_code_setup(step).err();
    }

    /// Takes the finished Claude Code check.
    pub(in crate::app) fn poll_claude_code_check(&mut self) {
        let setup = &mut self.ai.claude_setup;
        let Some(receiver) = &setup.checking else {
            return;
        };
        let result = match receiver.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("the Claude Code check stopped".to_owned())
            }
        };
        setup.checking = None;
        match result {
            Ok(status) => {
                log::info!("Claude Code status: {status:?}");
                setup.status = Some(status);
            }
            Err(error) => {
                log::warn!("{error}");
                setup.status = None;
                setup.failure = Some(error);
            }
        }
    }
}
