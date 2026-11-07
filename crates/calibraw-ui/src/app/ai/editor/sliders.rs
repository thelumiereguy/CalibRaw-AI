//! The sliders the `set_sliders` tool moves, read from and written to the
//! edit. For Gemma, the brief reader (brief.rs) moves those a brief asks for;
//! Claude's editor_system.md gives it every name and range.

use super::tools::{number, rounded, Limits, Outcome};
use crate::pipeline::{ExposureParams, LoadedRaw, HUE_ROTATION_LIMIT_DEGREES};
use serde_json::{json, Map, Value};

/// Mired the RAW white balance moves away from as shot per step of warmth:
/// ±100 warmth is about 3,500–12,000 K for a daylight shot.
const MIRED_PER_WARMTH: f32 = 1.0;
const MIRED_PER_KELVIN: f32 = 1_000_000.0;

#[derive(Clone, Copy)]
enum Slider {
    Exposure,
    Contrast,
    Highlights,
    Shadows,
    Whites,
    Blacks,
    Warmth,
    Tint,
    Hue,
    Vibrance,
    Saturation,
    Texture,
    Clarity,
    Dehaze,
    SharpenAmount,
    SharpenRadius,
    SharpenDetail,
    SharpenMasking,
    NoiseLuminance,
    NoiseColor,
    NoiseDetail,
}

const SLIDERS: [Slider; 21] = [
    Slider::Exposure,
    Slider::Contrast,
    Slider::Highlights,
    Slider::Shadows,
    Slider::Whites,
    Slider::Blacks,
    Slider::Warmth,
    Slider::Tint,
    Slider::Hue,
    Slider::Vibrance,
    Slider::Saturation,
    Slider::Texture,
    Slider::Clarity,
    Slider::Dehaze,
    Slider::SharpenAmount,
    Slider::SharpenRadius,
    Slider::SharpenDetail,
    Slider::SharpenMasking,
    Slider::NoiseLuminance,
    Slider::NoiseColor,
    Slider::NoiseDetail,
];

impl Slider {
    fn name(self) -> &'static str {
        match self {
            Self::Exposure => "exposure",
            Self::Contrast => "contrast",
            Self::Highlights => "highlights",
            Self::Shadows => "shadows",
            Self::Whites => "whites",
            Self::Blacks => "blacks",
            Self::Warmth => "warmth",
            Self::Tint => "tint",
            Self::Hue => "hue",
            Self::Vibrance => "vibrance",
            Self::Saturation => "saturation",
            Self::Texture => "texture",
            Self::Clarity => "clarity",
            Self::Dehaze => "dehaze",
            Self::SharpenAmount => "sharpen_amount",
            Self::SharpenRadius => "sharpen_radius",
            Self::SharpenDetail => "sharpen_detail",
            Self::SharpenMasking => "sharpen_masking",
            Self::NoiseLuminance => "noise_luminance",
            Self::NoiseColor => "noise_color",
            Self::NoiseDetail => "noise_detail",
        }
    }

    /// The slider's range and the decimals it shows: exposure in stops and
    /// sharpen_radius in pixels keep hundredths, the rest are whole.
    fn limits(self) -> Limits {
        match self {
            Self::Exposure => Limits::new(-5.0, 5.0, 2),
            Self::Hue => Limits::new(-HUE_ROTATION_LIMIT_DEGREES, HUE_ROTATION_LIMIT_DEGREES, 0),
            Self::SharpenAmount => Limits::new(0.0, 150.0, 0),
            Self::SharpenRadius => Limits::new(0.5, 3.0, 2),
            Self::SharpenDetail
            | Self::SharpenMasking
            | Self::NoiseLuminance
            | Self::NoiseColor
            | Self::NoiseDetail => Limits::new(0.0, 100.0, 0),
            _ => Limits::new(-100.0, 100.0, 0),
        }
    }

    /// The slider's value, or `None` for warmth and tint where the photo has
    /// no white balance they can move.
    fn value(self, exposure: &ExposureParams, raw: Option<&LoadedRaw>) -> Option<f32> {
        let value = match self {
            Self::Exposure => exposure.exposure,
            Self::Contrast => exposure.contrast,
            Self::Highlights => exposure.highlights,
            Self::Shadows => exposure.shadows,
            Self::Whites => exposure.whites,
            Self::Blacks => exposure.blacks,
            Self::Warmth => return warmth(exposure, raw?),
            Self::Tint => return tint(exposure, raw?),
            Self::Hue => exposure.hue,
            Self::Vibrance => exposure.vibrance,
            Self::Saturation => exposure.saturation,
            Self::Texture => exposure.texture,
            Self::Clarity => exposure.clarity,
            Self::Dehaze => exposure.dehaze,
            Self::SharpenAmount => exposure.sharpen_amount,
            Self::SharpenRadius => exposure.sharpen_radius,
            Self::SharpenDetail => exposure.sharpen_detail,
            Self::SharpenMasking => exposure.sharpen_masking,
            Self::NoiseLuminance => exposure.luminance_denoise,
            // Stored as a fraction, shown in percent.
            Self::NoiseColor => exposure.chroma_denoise * 100.0,
            Self::NoiseDetail => exposure.denoise_detail,
        };
        Some(rounded(value, self.limits().decimals))
    }

    /// Sets the slider; `false` when warmth or tint cannot be set on this
    /// photo.
    fn set(self, exposure: &mut ExposureParams, raw: Option<&LoadedRaw>, value: f32) -> bool {
        let field = match self {
            Self::Exposure => &mut exposure.exposure,
            Self::Contrast => &mut exposure.contrast,
            Self::Highlights => &mut exposure.highlights,
            Self::Shadows => &mut exposure.shadows,
            Self::Whites => &mut exposure.whites,
            Self::Blacks => &mut exposure.blacks,
            Self::Warmth => return raw.is_some_and(|raw| set_warmth(exposure, raw, value)),
            Self::Tint => return raw.is_some_and(|raw| set_tint(exposure, raw, value)),
            Self::Hue => &mut exposure.hue,
            Self::Vibrance => &mut exposure.vibrance,
            Self::Saturation => &mut exposure.saturation,
            Self::Texture => &mut exposure.texture,
            Self::Clarity => &mut exposure.clarity,
            Self::Dehaze => &mut exposure.dehaze,
            Self::SharpenAmount => &mut exposure.sharpen_amount,
            Self::SharpenRadius => &mut exposure.sharpen_radius,
            Self::SharpenDetail => &mut exposure.sharpen_detail,
            Self::SharpenMasking => &mut exposure.sharpen_masking,
            Self::NoiseLuminance => &mut exposure.luminance_denoise,
            Self::NoiseColor => {
                exposure.chroma_denoise = value / 100.0;
                return true;
            }
            Self::NoiseDetail => &mut exposure.denoise_detail,
        };
        *field = value;
        true
    }
}

/// Warmth of a rendered photo is its own temperature slider; a RAW's is its
/// white balance's distance from as shot, in [`MIRED_PER_WARMTH`] steps.
fn warmth(exposure: &ExposureParams, raw: &LoadedRaw) -> Option<f32> {
    if raw.is_display_referred_raster() {
        return Some(exposure.raster_temperature.round());
    }
    let as_shot = raw.as_shot_temperature_kelvin()?;
    let (kelvin, _) = raw.shown_white_balance(exposure.white_balance)?;
    let warmth = (MIRED_PER_KELVIN / as_shot - MIRED_PER_KELVIN / kelvin) / MIRED_PER_WARMTH;
    Some(warmth.round())
}

fn set_warmth(exposure: &mut ExposureParams, raw: &LoadedRaw, warmth: f32) -> bool {
    if raw.is_display_referred_raster() {
        exposure.raster_temperature = warmth;
        return true;
    }
    let (Some(as_shot), Some((_, tint)), Some(ranges)) = (
        raw.as_shot_temperature_kelvin(),
        raw.shown_white_balance(exposure.white_balance),
        raw.white_balance_ranges(),
    ) else {
        return false;
    };
    let mired = MIRED_PER_KELVIN / as_shot - warmth * MIRED_PER_WARMTH;
    // A warmth past the warmest temperature the photo allows takes that one.
    let kelvin = if mired > 0.0 {
        MIRED_PER_KELVIN / mired
    } else {
        *ranges.temperature.end()
    };
    set_white_balance(exposure, raw, kelvin, tint)
}

/// A RAW's tint as its slider shows it, less the as-shot tint. A rendered
/// photo has no tint.
fn tint(exposure: &ExposureParams, raw: &LoadedRaw) -> Option<f32> {
    if raw.is_display_referred_raster() {
        return None;
    }
    let (_, as_shot) = raw.as_shot_white_balance()?;
    let (_, tint) = raw.shown_white_balance(exposure.white_balance)?;
    Some((tint - as_shot).round())
}

fn set_tint(exposure: &mut ExposureParams, raw: &LoadedRaw, tint: f32) -> bool {
    if raw.is_display_referred_raster() {
        return false;
    }
    let (Some((_, as_shot)), Some((kelvin, _))) = (
        raw.as_shot_white_balance(),
        raw.shown_white_balance(exposure.white_balance),
    ) else {
        return false;
    };
    set_white_balance(exposure, raw, kelvin, as_shot + tint)
}

fn set_white_balance(
    exposure: &mut ExposureParams,
    raw: &LoadedRaw,
    kelvin: f32,
    tint: f32,
) -> bool {
    match raw.custom_white_balance(kelvin, tint) {
        Some(white_balance) => {
            exposure.white_balance = white_balance;
            true
        }
        None => false,
    }
}

/// Whether `name` is one of the sliders.
pub(super) fn is_slider(name: &str) -> bool {
    SLIDERS.iter().any(|slider| slider.name() == name)
}

/// The slider `name`'s value as `set_sliders` reads it, or `None` for another
/// name and for warmth and tint where the photo has no white balance they
/// can move.
pub(super) fn slider_value(
    exposure: &ExposureParams,
    raw: Option<&LoadedRaw>,
    name: &str,
) -> Option<f32> {
    SLIDERS
        .iter()
        .find(|slider| slider.name() == name)?
        .value(exposure, raw)
}

/// The sliders away from their defaults, as the model reads them.
pub(super) fn changed_sliders(
    exposure: &ExposureParams,
    raw: Option<&LoadedRaw>,
) -> Map<String, Value> {
    let defaults = ExposureParams::default();
    SLIDERS
        .iter()
        .filter_map(|slider| {
            let value = slider.value(exposure, raw)?;
            (slider.value(&defaults, raw) != Some(value))
                .then(|| (slider.name().to_owned(), json!(value)))
        })
        .collect()
}

/// Applies one `set_sliders` call's `arguments` to `exposure`, each value
/// within its slider's range. `baseline`, the edit when the reply came, gives
/// the values they were.
pub(super) fn apply_set_sliders(
    exposure: &mut ExposureParams,
    baseline: &ExposureParams,
    raw: Option<&LoadedRaw>,
    arguments: &Value,
) -> Value {
    let mut outcome = Outcome::default();
    for (name, requested) in arguments.as_object().into_iter().flatten() {
        let slider = SLIDERS.iter().copied().find(|slider| slider.name() == name);
        let (Some(slider), Some(requested)) = (slider, number(requested)) else {
            outcome.not_set(name);
            continue;
        };
        let Some(from) = slider.value(baseline, raw) else {
            outcome.not_set(name);
            continue;
        };
        let limits = slider.limits();
        let value = limits.limit(requested);
        if slider.set(exposure, raw, value) {
            // Read back: the white balance keeps to the camera's range.
            let now = slider.value(exposure, raw).unwrap_or(value);
            outcome.set_number(name, requested, from, now, limits.decimals);
        } else {
            outcome.not_set(name);
        }
    }
    outcome.into_response()
}
