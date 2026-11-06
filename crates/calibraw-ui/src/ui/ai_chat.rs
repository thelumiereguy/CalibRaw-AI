//! The AI editor's chat with the selected model (Gemma or Claude) about the
//! open photo, shown in place of the Develop tools while it is open
//! (`EditorState::open`).

use crate::app::{CalibRawApp, ChatRole, EditorState, TriageState};
use crate::i18n::tr;
use crate::platform::AiEditorModel;
use eframe::egui::{self, Ui};

/// What the photographer did in the chat.
enum ChatAction {
    None,
    Send,
    Close,
}

/// The first emoji of Gemma's `emoji`, for the AI editor's badge: egui draws
/// emoji one glyph at a time and in monochrome, so presentation selectors
/// and the rest of a joined sequence would draw as boxes.
pub(crate) fn badge_glyph(emoji: &str) -> String {
    emoji
        .chars()
        .find(|glyph| !glyph.is_whitespace() && *glyph != '\u{FE0F}')
        .map_or_else(|| "📷".to_owned(), String::from)
}

/// The photo's genre badge at `size` points: the colour emoji where the
/// platform drew one, else egui's glyph.
pub(crate) fn show_badge(ui: &mut Ui, triage: &TriageState, emoji: &str, size: f32) {
    match &triage.badge {
        Some(texture) => {
            ui.add(egui::Image::new((texture.id(), egui::vec2(size, size))));
        }
        None => {
            ui.label(egui::RichText::new(badge_glyph(emoji)).size(size * 0.8));
        }
    }
}

/// The triage badge's tooltip while `model` looks at the photo.
pub(crate) fn looking_at_photo(model: AiEditorModel) -> &'static str {
    match model {
        AiEditorModel::Gemma => tr("Gemma is looking at the photo…"),
        AiEditorModel::ClaudeCode => tr("Claude is looking at the photo…"),
    }
}

pub(crate) fn show(ui: &mut Ui, app: &mut CalibRawApp) {
    let can_send = app.ai_editor_can_send();
    let model = app.ai.editor_model;
    match chat(ui, model, &app.ai.triage, &mut app.ai.editor, can_send) {
        ChatAction::None => {}
        ChatAction::Send => app.send_ai_editor_request(),
        ChatAction::Close => app.set_ai_editor_open(false),
    }
}

fn chat(
    ui: &mut Ui,
    model: AiEditorModel,
    triage: &TriageState,
    editor: &mut EditorState,
    can_send: bool,
) -> ChatAction {
    let mut action = ChatAction::None;
    let margin = moduwu_design::SPACE_SM;
    let inner = ui.available_rect_before_wrap().shrink(margin);
    let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(inner));
    let ui = &mut ui;

    ui.horizontal(|ui| {
        triage_heading(ui, model, triage);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if moduwu_design::ghost_icon_button(
                ui,
                egui_phosphor::regular::X,
                moduwu_design::toolbar_icon_size(),
                tr("Close the AI editor"),
            )
            .clicked()
            {
                action = ChatAction::Close;
            }
        });
    });
    ui.separator();

    let input_height = moduwu_design::CONTROL_HEIGHT + moduwu_design::SPACE_SM * 2.0;
    egui::ScrollArea::vertical()
        .id_salt("ai-editor-chat")
        .max_height((ui.available_height() - input_height).max(0.0))
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for message in editor.messages() {
                chat_message(ui, message.role, &message.text);
            }
        });

    ui.add_space(moduwu_design::SPACE_SM);
    let hint = match model {
        AiEditorModel::Gemma => tr("Message Gemma"),
        AiEditorModel::ClaudeCode => tr("Message Claude"),
    };
    ui.horizontal(|ui| {
        let send_width = moduwu_design::toolbar_icon_size().x;
        let field = ui.add_sized(
            [
                (ui.available_width() - send_width - ui.spacing().item_spacing.x).max(1.0),
                moduwu_design::CONTROL_HEIGHT,
            ],
            moduwu_design::singleline_text_edit(&mut editor.draft).hint_text(hint),
        );
        let entered = field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
        if editor.request_running() {
            ui.add(egui::Spinner::new().size(send_width * 0.6));
        } else {
            let sent = moduwu_design::icon_button_enabled(
                ui,
                can_send,
                egui_phosphor::regular::PAPER_PLANE_RIGHT,
                moduwu_design::toolbar_icon_size(),
                tr("Send"),
            )
            .clicked();
            if (sent || entered) && can_send {
                action = ChatAction::Send;
            }
        }
    });
    action
}

/// The photo's genre, opening its triage card, or what `model` is doing
/// before the card is done.
fn triage_heading(ui: &mut Ui, model: AiEditorModel, triage: &TriageState) {
    let Some((genre, emoji)) = triage.headline() else {
        ui.label(if triage.running() {
            looking_at_photo(model)
        } else {
            match model {
                AiEditorModel::Gemma => tr("Gemma has not triaged this photo."),
                AiEditorModel::ClaudeCode => tr("Claude has not triaged this photo."),
            }
        });
        return;
    };
    show_badge(ui, triage, emoji, 18.0);
    let Some(card) = &triage.card else {
        ui.label(egui::RichText::new(genre).strong());
        ui.add(egui::Spinner::new());
        return;
    };
    egui::CollapsingHeader::new(egui::RichText::new(genre).strong())
        .id_salt("ai-editor-triage-card")
        .default_open(false)
        .show(ui, |ui| {
            ui.add(egui::Label::new(&card.role).wrap());
        });
}

fn chat_message(ui: &mut Ui, role: ChatRole, text: &str) {
    let visuals = ui.visuals();
    let (layout, fill, text) = match role {
        ChatRole::Photographer => (
            egui::Layout::right_to_left(egui::Align::Min),
            visuals.selection.bg_fill.gamma_multiply(0.35),
            egui::RichText::new(text),
        ),
        ChatRole::Gemma => (
            egui::Layout::left_to_right(egui::Align::Min),
            visuals.faint_bg_color,
            egui::RichText::new(text),
        ),
        ChatRole::Progress => (
            egui::Layout::left_to_right(egui::Align::Min),
            egui::Color32::TRANSPARENT,
            egui::RichText::new(text).weak().italics(),
        ),
        ChatRole::Failure => (
            egui::Layout::left_to_right(egui::Align::Min),
            egui::Color32::TRANSPARENT,
            egui::RichText::new(text).color(visuals.error_fg_color),
        ),
    };
    if text.text().is_empty() {
        // A reply Gemma has not started writing.
        ui.with_layout(layout, |ui| ui.add(egui::Spinner::new()));
        return;
    }
    let max_width = ui.available_width() * 0.85;
    ui.with_layout(layout, |ui| {
        egui::Frame::new()
            .fill(fill)
            .corner_radius(8.0)
            .inner_margin(egui::Margin::symmetric(10, 6))
            .show(ui, |ui| {
                ui.set_max_width(max_width);
                ui.add(egui::Label::new(text).wrap());
            });
    });
    ui.add_space(moduwu_design::SPACE_XS);
}
