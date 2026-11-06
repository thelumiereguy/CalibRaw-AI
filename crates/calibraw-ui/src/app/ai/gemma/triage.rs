//! Gemma's triage of the open photo, the first phase of the AI editor. When a
//! photo opens, its rendered preview goes to on-device Gemma in a session of
//! its own (`Platform::gemma_triage`), which names the photo's genre and
//! fills the triage card that the AI editor's system prompt is built from
//! (android/app/src/main/assets/gemma/editor_system.md). The genre and emoji
//! come first, so the badge shows them (`Platform::take_triage_headline`)
//! while Gemma still writes the rest of the card.

use super::*;
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
    /// Why the open photo has no card. Logged, never shown as an error: the
    /// photo stays editable without one.
    pub(crate) failure: Option<String>,
    /// The open photo is triaged again once Develop shows it, since the
    /// photographer chose another model in Settings.
    pub(in crate::app) restart_pending: bool,
}

impl TriageState {
    /// Forgets the card when another photo opens.
    pub(in crate::app) fn forget(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn running(&self) -> bool {
        self.pending.is_some()
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
    /// Starts Gemma's triage of the photo of `document_generation`, which
    /// just opened, from its rendered preview.
    pub(in crate::app) fn start_photo_triage(
        &mut self,
        frame: &eframe::Frame,
        document_generation: u64,
    ) {
        let started = self
            .read_triage_image(frame)
            .and_then(|image| {
                let data_dir = self
                    .platform
                    .internal_data_dir()
                    .ok_or_else(|| "the app has no private data folder".to_owned())?;
                Ok((image, data_dir.join(TRIAGE_IMAGE_FILE)))
            })
            .map(|(image, jpeg_path)| {
                let platform = Arc::clone(&self.platform);
                spawn_ui_worker(&self.egui_ctx, move || {
                    triage_photo(platform.as_ref(), image, &jpeg_path, document_generation)
                })
            });
        match started {
            Ok(receiver) => self.ai.triage.pending = Some((document_generation, receiver)),
            Err(error) => {
                log::warn!("Gemma triage could not start: {error}");
                self.ai.triage.failure = Some(error);
            }
        }
    }

    /// Triages the open photo again with the newly chosen model once Develop
    /// shows it with its preview.
    pub(in crate::app) fn restart_photo_triage(&mut self, frame: &eframe::Frame) {
        if !self.ai.triage.restart_pending
            || self.ui.active_tab != AppTab::Develop
            || self.develop.loaded_raw.is_none()
            || self.preview.pipeline().is_none()
            || self.document_load_is_background()
        {
            return;
        }
        self.ai.triage.restart_pending = false;
        self.start_photo_triage(frame, self.persistence.document_generation);
    }

    /// The preview as shown, read back from the GPU.
    fn read_triage_image(&self, frame: &eframe::Frame) -> Result<image::RgbaImage, String> {
        let render_state = frame
            .wgpu_render_state()
            .ok_or_else(|| "the GPU preview is not available".to_owned())?;
        let pipeline = self
            .preview
            .pipeline()
            .ok_or_else(|| "the photo has no preview yet".to_owned())?;
        let rgba = pipeline
            .read_output_region_blocking(
                &render_state.device,
                &render_state.queue,
                0,
                0,
                pipeline.width,
                pipeline.height,
            )
            .map_err(|error| format!("could not read the preview back: {error:#}"))?;
        image::RgbaImage::from_raw(pipeline.width, pipeline.height, rgba)
            .ok_or_else(|| "the preview read back with an invalid size".to_owned())
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
            Err(error) => {
                log::warn!("Gemma triage failed: {error}");
                self.ai.triage.failure = Some(error);
            }
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
