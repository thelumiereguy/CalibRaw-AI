//! The sliders Gemma's `set_sliders` tool moves, read from and written to the
//! edit. Their names and ranges are those of
//! android/app/src/main/assets/gemma/set_sliders_tool.json, and their nudge
//! limits those the editor prompt (assets/gemma/editor_system.md) states.

use crate::pipeline::{ExposureParams, LoadedRaw};
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
    Vibrance,
    Saturation,
    Texture,
    Clarity,
    Dehaze,
}

const SLIDERS: [Slider; 12] = [
    Slider::Exposure,
    Slider::Contrast,
    Slider::Highlights,
    Slider::Shadows,
    Slider::Whites,
    Slider::Blacks,
    Slider::Warmth,
    Slider::Vibrance,
    Slider::Saturation,
    Slider::Texture,
    Slider::Clarity,
    Slider::Dehaze,
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
            Self::Vibrance => "vibrance",
            Self::Saturation => "saturation",
            Self::Texture => "texture",
            Self::Clarity => "clarity",
            Self::Dehaze => "dehaze",
        }
    }

    /// The slider's range: stops for exposure, else -100..=100.
    fn limit(self) -> f32 {
        match self {
            Self::Exposure => 5.0,
            _ => 100.0,
        }
    }

    /// The most one nudge moves the slider.
    fn nudge(self) -> f32 {
        match self {
            Self::Exposure => 0.5,
            _ => 20.0,
        }
    }

    /// Exposure keeps the hundredths its slider shows; the rest are whole.
    fn rounded(self, value: f32) -> f32 {
        match self {
            Self::Exposure => (value * 100.0).round() / 100.0,
            _ => value.round(),
        }
    }

    /// The slider's value, or `None` for warmth where the photo has no white
    /// balance it can move.
    fn value(self, exposure: &ExposureParams, raw: Option<&LoadedRaw>) -> Option<f32> {
        let value = match self {
            Self::Exposure => exposure.exposure,
            Self::Contrast => exposure.contrast,
            Self::Highlights => exposure.highlights,
            Self::Shadows => exposure.shadows,
            Self::Whites => exposure.whites,
            Self::Blacks => exposure.blacks,
            Self::Warmth => return warmth(exposure, raw?),
            Self::Vibrance => exposure.vibrance,
            Self::Saturation => exposure.saturation,
            Self::Texture => exposure.texture,
            Self::Clarity => exposure.clarity,
            Self::Dehaze => exposure.dehaze,
        };
        Some(self.rounded(value))
    }

    /// Sets the slider; `false` when warmth cannot be set on this photo.
    fn set(self, exposure: &mut ExposureParams, raw: Option<&LoadedRaw>, value: f32) -> bool {
        let field = match self {
            Self::Exposure => &mut exposure.exposure,
            Self::Contrast => &mut exposure.contrast,
            Self::Highlights => &mut exposure.highlights,
            Self::Shadows => &mut exposure.shadows,
            Self::Whites => &mut exposure.whites,
            Self::Blacks => &mut exposure.blacks,
            Self::Warmth => return raw.is_some_and(|raw| set_warmth(exposure, raw, value)),
            Self::Vibrance => &mut exposure.vibrance,
            Self::Saturation => &mut exposure.saturation,
            Self::Texture => &mut exposure.texture,
            Self::Clarity => &mut exposure.clarity,
            Self::Dehaze => &mut exposure.dehaze,
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
    match raw.custom_white_balance(kelvin, tint) {
        Some(white_balance) => {
            exposure.white_balance = white_balance;
            true
        }
        None => false,
    }
}

/// The sliders' current values, as Gemma reads them. Warmth is left out where
/// the photo has no white balance to move.
pub(super) fn slider_values(
    exposure: &ExposureParams,
    raw: Option<&LoadedRaw>,
) -> Map<String, Value> {
    SLIDERS
        .iter()
        .filter_map(|slider| {
            let value = slider.value(exposure, raw)?;
            Some((slider.name().to_owned(), json!(value)))
        })
        .collect()
}

/// Applies one `set_sliders` call's `arguments` to `exposure`. Each value is
/// kept within its slider's range and one nudge of where the slider was.
/// Returns the tool's response for Gemma: the sliders as they now are, and
/// which values were limited or could not be set.
pub(super) fn apply_set_sliders(
    exposure: &mut ExposureParams,
    raw: Option<&LoadedRaw>,
    arguments: &Value,
) -> Value {
    let mut limited = Vec::new();
    let mut not_set = Vec::new();
    for (name, requested) in arguments.as_object().into_iter().flatten() {
        let slider = SLIDERS.iter().copied().find(|slider| slider.name() == name);
        let requested = requested
            .as_f64()
            .or_else(|| requested.as_str()?.trim().parse().ok())
            .map(|value| value as f32)
            .filter(|value| value.is_finite());
        let (Some(slider), Some(requested)) = (slider, requested) else {
            not_set.push(name.clone());
            continue;
        };
        let Some(current) = slider.value(exposure, raw) else {
            not_set.push(name.clone());
            continue;
        };
        let value = slider.rounded(
            requested
                .clamp(current - slider.nudge(), current + slider.nudge())
                .clamp(-slider.limit(), slider.limit()),
        );
        if (value - requested).abs() > f32::EPSILON {
            limited.push(name.clone());
        }
        if !slider.set(exposure, raw, value) {
            not_set.push(name.clone());
        }
    }
    let mut response = Map::new();
    response.insert(
        "sliders".to_owned(),
        Value::Object(slider_values(exposure, raw)),
    );
    if !limited.is_empty() {
        response.insert("limited_to_one_nudge_or_range".to_owned(), json!(limited));
    }
    if !not_set.is_empty() {
        response.insert("not_set".to_owned(), json!(not_set));
    }
    Value::Object(response)
}

/// `values` as one line of `name value` pairs for a message to Gemma.
pub(super) fn describe(values: &Map<String, Value>) -> String {
    values
        .iter()
        .map(|(name, value)| format!("{name} {value}"))
        .collect::<Vec<_>>()
        .join(", ")
}
