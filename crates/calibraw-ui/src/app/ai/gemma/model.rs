//! The model behind the triage and the AI editor, which the photographer
//! chooses in Settings: Gemma on the device or Claude Code in Termux. Also
//! Claude Code's setup, which Settings checks and runs step by step.

use super::*;
use crate::platform::{AiEditorModel, ClaudeCodeSetup, ClaudeCodeStatus};

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

impl CalibRawApp {
    /// Hands the triage and the AI editor to the model the app starts with.
    pub(in crate::app) fn select_initial_ai_model(&mut self) {
        if !self.caps().on_device_gemma {
            return;
        }
        if let Err(error) = self.platform.select_ai_model(self.ai.editor_model) {
            log::warn!("{error}");
        }
        if self.ai.editor_model == AiEditorModel::ClaudeCode {
            self.check_claude_code();
        }
    }

    /// Makes `model` the one behind the triage and the AI editor. The open
    /// photo's chat and card are forgotten, and the photo is triaged again by
    /// the new model when Develop next shows it.
    pub(crate) fn set_ai_editor_model(&mut self, model: AiEditorModel) {
        if self.ai.editor_model == model {
            return;
        }
        self.forget_ai_editor();
        self.ai.triage.forget();
        self.ai.editor_model = model;
        if let Err(error) = self.platform.select_ai_model(model) {
            log::warn!("{error}");
        }
        self.ai.triage.restart_pending = self.develop.loaded_raw.is_some();
        self.persist_performance_settings();
        if model == AiEditorModel::ClaudeCode {
            self.check_claude_code();
        }
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
