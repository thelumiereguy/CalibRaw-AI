//! Gemma's triage of the open photo, the first phase of the AI editor. When a
//! photo opens, its rendered preview goes to on-device Gemma in a session of
//! its own (`Platform::gemma_triage`), which names the photo's genre and
//! fills the triage card that the AI editor's system prompt is built from
//! (android/app/src/main/assets/gemma/editor_system.md). The genre and emoji
//! come first, so the badge shows them (`Platform::take_triage_headline`)
//! while Gemma still writes the rest of the card. Claude Code triages a photo
//! only once the AI editor's chat opens, so a photo goes to Anthropic only
//! when the photographer asks Claude about it.

use super::*;
use crate::app::preview_capture::PreviewSnapshot;
use crate::platform::AiEditorModel;
use serde::Deserialize;
use std::path::Path;

/// Longest edge of the image Gemma sees; its vision encoder scales it down
/// further, so a larger one only costs encode time.
pub(super) const GEMMA_IMAGE_EDGE: u32 = 1024;
const GEMMA_JPEG_QUALITY: u8 = 90;
const TRIAGE_IMAGE_FILE: &str = "gemma-triage.jpg";
/// Shown when Gemma's emoji is missing.
const FALLBACK_EMOJI: &str = "📷";
/// Pixel size of the colour badge the platform draws; the toolbar shows it
/// at about 14 points, so this stays sharp up to 4.5 pixels per point.
const BADGE_PX: u32 = 64;

/// A finished triage: the card and its emoji drawn in colour by the
/// platform, when it can.
type TriageResult = Result<(TriageCard, Option<Vec<u8>>), String>;

/// What Gemma made of the photo. The limits mirror
/// assets/gemma/triage_schema.json; [`TriageCard::clamped`] enforces them,
/// since constrained decoding may not.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct TriageCard {
    /// The photography genre in a working photographer's words.
    pub(crate) genre: String,
    pub(crate) emoji: String,
    /// What a specialist in the genre knows and protects, in the second person.
    /// The card asks for no editing priorities or observations: given either,
    /// Gemma edited towards them instead of the photographer's request.
    pub(crate) role: String,
}

impl TriageCard {
    fn clamped(self) -> Result<Self, String> {
        let genre = clamp_text(&self.genre, 40);
        if genre.is_empty() {
            return Err("Gemma's triage card names no genre".to_owned());
        }
        Ok(Self {
            genre,
            emoji: clamp_emoji(&self.emoji),
            role: clamp_text(&self.role, 300),
        })
    }
}

fn clamp_text(text: &str, max_chars: usize) -> String {
    text.trim().chars().take(max_chars).collect()
}

fn clamp_emoji(emoji: &str) -> String {
    let emoji = clamp_text(emoji, 8);
    if emoji.is_empty() {
        FALLBACK_EMOJI.to_owned()
    } else {
        emoji
    }
}

/// The open photo's triage card and the triage that makes it.
#[derive(Default)]
pub(crate) struct TriageState {
    pub(crate) card: Option<TriageCard>,
    /// The genre and emoji Gemma wrote first, while the card is not done.
    headline: Option<(String, String)>,
    /// The card's emoji in colour; `None` where the platform cannot draw it,
    /// and the badge falls back to egui's monochrome glyph.
    pub(crate) badge: Option<egui::TextureHandle>,
    /// The document generation the running triage is for, and its result.
    pending: Option<(u64, mpsc::Receiver<TriageResult>)>,
    /// Why the open photo has no card. The AI editor's chat shows it with a
    /// retry ([`CalibRawApp::retry_photo_triage`]), never an error dialog:
    /// the photo stays editable by hand without one.
    pub(crate) failure: Option<String>,
    /// The open photo is triaged once Develop shows it, and with Claude Code
    /// once the AI editor's chat is open too: Claude Code waits for the chat,
    /// and a photo is triaged again after a model switch.
    pub(in crate::app) waiting: bool,
}

impl TriageState {
    /// Forgets the card when another photo opens.
    pub(in crate::app) fn forget(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn running(&self) -> bool {
        self.pending.is_some()
    }

    /// Whether the triage runs or waits to start. In the AI editor's chat,
    /// which shows with Develop and lets Claude Code see the photo, a
    /// waiting triage starts once the preview shows every edit.
    pub(crate) fn looking(&self) -> bool {
        self.pending.is_some() || self.waiting
    }

    /// The photo's genre and emoji, from the card or, while it is not done,
    /// from what Gemma wrote first.
    pub(crate) fn headline(&self) -> Option<(&str, &str)> {
        match (&self.card, &self.headline) {
            (Some(card), _) => Some((&card.genre, &card.emoji)),
            (None, Some((genre, emoji))) => Some((genre, emoji)),
            (None, None) => None,
        }
    }
}

impl CalibRawApp {
    /// Has the photo of `document_generation`, which just opened, triaged
    /// once Develop shows its preview with every edit; with Claude Code once
    /// the AI editor's chat is open too.
    pub(in crate::app) fn start_photo_triage(
        &mut self,
        frame: &eframe::Frame,
        document_generation: u64,
    ) {
        if document_generation != self.persistence.document_generation {
            return;
        }
        self.ai.triage.failure = None;
        self.ai.triage.waiting = true;
        self.start_waiting_photo_triage(frame);
    }

    /// Triages the open photo again after its triage failed, as the
    /// photographer asked in the AI editor's chat. The chat is open, so
    /// Claude Code may see the photo.
    pub(crate) fn retry_photo_triage(&mut self) {
        if self.ai.triage.running() || self.develop.loaded_raw.is_none() {
            return;
        }
        self.ai.triage.failure = None;
        self.ai.triage.waiting = true;
        self.egui_ctx.request_repaint();
    }

    /// Starts the waiting triage of the open photo once Develop shows its
    /// preview with every edit, and with Claude Code once the AI editor's
    /// chat is open. The preview is read back, cropped and sent on a worker,
    /// as the AI editor sends it (editor.rs).
    pub(in crate::app) fn start_waiting_photo_triage(&mut self, frame: &eframe::Frame) {
        if !self.ai.triage.waiting
            || self.ui.active_tab != AppTab::Develop
            || self.develop.loaded_raw.is_none()
            || self.document_load_is_background()
            || (self.ai.editor_model == AiEditorModel::ClaudeCode && !self.ai.editor.open)
        {
            return;
        }
        let snapshot = match self.current_preview_snapshot(frame) {
            PreviewSnapshot::Ready(snapshot) => snapshot,
            // A repaint is requested to look again.
            PreviewSnapshot::Rendering => return,
            PreviewSnapshot::NoGpu => {
                self.fail_photo_triage("the GPU preview is not available".to_owned());
                return;
            }
        };
        let Some(render_state) = frame.wgpu_render_state() else {
            self.fail_photo_triage("the GPU preview is not available".to_owned());
            return;
        };
        let Some(data_dir) = self.platform.internal_data_dir() else {
            self.fail_photo_triage("the app has no private data folder".to_owned());
            return;
        };
        self.ai.triage.waiting = false;
        let device = render_state.device.clone();
        let queue = render_state.queue.clone();
        let geometry = self.develop.geometry;
        // As the AI editor sends the preview: lens-corrected for Claude,
        // which sees it as the photographer does.
        let lens = match self.ai.editor_model {
            AiEditorModel::Gemma => None,
            AiEditorModel::ClaudeCode => self
                .develop
                .loaded_raw
                .as_ref()
                .and_then(|raw| raw.lens_geometry.clone()),
        };
        let jpeg_path = data_dir.join(TRIAGE_IMAGE_FILE);
        let platform = Arc::clone(&self.platform);
        let document_generation = self.persistence.document_generation;
        let receiver = spawn_ui_worker(&self.egui_ctx, move || {
            let preview = snapshot
                .read_thumbnail_blocking(&device, &queue, GEMMA_IMAGE_EDGE)
                .map_err(|error| format!("could not read the preview back: {error:#}"))?;
            let preview = crate::pipeline::transform_thumbnail_geometry_with_lens(
                &preview,
                geometry,
                lens.as_deref(),
            );
            let image = image::RgbaImage::from_raw(preview.width, preview.height, preview.rgba)
                .ok_or_else(|| "the preview read back with an invalid size".to_owned())?;
            triage_photo(platform.as_ref(), image, &jpeg_path, document_generation)
        });
        self.ai.triage.pending = Some((document_generation, receiver));
    }

    /// Records why the open photo has no card. The genre written before the
    /// failure goes too, since no card follows it.
    fn fail_photo_triage(&mut self, error: String) {
        log::warn!("AI triage failed: {error}");
        let triage = &mut self.ai.triage;
        triage.waiting = false;
        triage.headline = None;
        triage.badge = None;
        triage.failure = Some(error);
    }

    /// Takes the running triage's headline, then its finished card, when
    /// they are for the open photo.
    pub(in crate::app) fn poll_photo_triage(&mut self) {
        // Taken every frame, so a headline of an earlier photo is dropped.
        if let Some(headline) = self.platform.take_triage_headline() {
            let triage = &mut self.ai.triage;
            let current = triage
                .pending
                .as_ref()
                .is_some_and(|(generation, _)| *generation == headline.request_id);
            if current && triage.card.is_none() {
                log::info!(
                    "Gemma named the genre {} {}",
                    headline.emoji,
                    headline.genre
                );
                triage.badge = badge_texture(&self.egui_ctx, headline.badge);
                triage.headline = Some((
                    clamp_text(&headline.genre, 40),
                    clamp_emoji(&headline.emoji),
                ));
            }
        }

        let Some((generation, receiver)) = &self.ai.triage.pending else {
            return;
        };
        let result = match receiver.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("the Gemma triage worker stopped".to_owned())
            }
        };
        let current = *generation == self.persistence.document_generation;
        self.ai.triage.pending = None;
        if !current {
            return;
        }
        match result {
            Ok((card, badge)) => {
                log::info!("Gemma triaged the photo as {} {}", card.emoji, card.genre);
                if self.ai.triage.badge.is_none() {
                    self.ai.triage.badge = badge_texture(&self.egui_ctx, badge);
                }
                self.ai.triage.headline = None;
                self.ai.triage.card = Some(card);
                self.prepare_ai_editor();
            }
            Err(error) => self.fail_photo_triage(error),
        }
    }
}

/// The badge texture for `rgba`, an emoji the platform drew in a
/// [`BADGE_PX`] square; `None` when there is none or its size is wrong.
fn badge_texture(ctx: &egui::Context, rgba: Option<Vec<u8>>) -> Option<egui::TextureHandle> {
    let side = BADGE_PX as usize;
    rgba.filter(|rgba| rgba.len() == side * side * 4)
        .map(|rgba| {
            ctx.load_texture(
                "gemma-triage-badge",
                egui::ColorImage::from_rgba_premultiplied([side, side], &rgba),
                egui::TextureOptions::LINEAR,
            )
        })
}

/// Runs on a worker: writes `image` as a JPEG to `jpeg_path`, where Gemma
/// reads it, and returns the card Gemma fills for it with its emoji drawn.
fn triage_photo(
    platform: &dyn Platform,
    image: image::RgbaImage,
    jpeg_path: &Path,
    request_id: u64,
) -> TriageResult {
    let card = fill_triage_card(platform, image, jpeg_path, request_id)?;
    let badge = platform
        .render_emoji(&card.emoji, BADGE_PX)
        .unwrap_or_else(|error| {
            log::warn!("{error}");
            None
        });
    Ok((card, badge))
}

fn fill_triage_card(
    platform: &dyn Platform,
    image: image::RgbaImage,
    jpeg_path: &Path,
    request_id: u64,
) -> Result<TriageCard, String> {
    write_gemma_jpeg(image, jpeg_path)?;
    let body = platform.gemma_triage(jpeg_path, "", request_id, BADGE_PX)?;
    serde_json::from_str::<TriageCard>(&body)
        .map_err(|error| format!("Gemma's triage card is not the expected JSON ({error}): {body}"))?
        .clamped()
}

/// Writes `image` to `path` as the JPEG Gemma reads, scaled down to
/// [`GEMMA_IMAGE_EDGE`].
pub(super) fn write_gemma_jpeg(image: image::RgbaImage, path: &Path) -> Result<(), String> {
    let mut image = image::DynamicImage::ImageRgba8(image);
    if image.width().max(image.height()) > GEMMA_IMAGE_EDGE {
        image = image.resize(
            GEMMA_IMAGE_EDGE,
            GEMMA_IMAGE_EDGE,
            image::imageops::FilterType::Triangle,
        );
    }
    let image = image::DynamicImage::ImageRgb8(image.to_rgb8());
    let mut encoded = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, GEMMA_JPEG_QUALITY)
        .encode_image(&image)
        .map_err(|error| format!("could not encode the image for Gemma: {error}"))?;
    calibraw_core::file_ops::write_bytes_atomically(path, &encoded).map_err(|error| {
        format!(
            "could not write the image for Gemma {}: {error}",
            path.display()
        )
    })
}
