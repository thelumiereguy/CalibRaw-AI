//! The Color Mixer and Color Grading values Gemma's `set_color_mixer` and
//! `set_color_grading` tools move. Their names and ranges are those
//! editor_system.md gives Gemma.

use super::tools::{number, rounded, Limits, Outcome};
use crate::pipeline::{ColorGradeWheel, ColorGrading, ExposureParams};
use serde_json::{json, Map, Value};

/// The Color Mixer's ranges, in the order of `ExposureParams::hsl_*`.
const MIXER_COLORS: [&str; 8] = [
    "red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta",
];
const MIXER_LIMITS: Limits = Limits::new(-100.0, 100.0, 0);
/// Stored mixer hue per unit Gemma reads, which are Lightroom's: the stored
/// hue spans ±`HSL_HUE_LIMIT` (200), as presets/xmp.rs reads Lightroom's.
const MIXER_HUE_SCALE: f32 = 2.0;

#[derive(Clone, Copy)]
enum Channel {
    Hue,
    Saturation,
    Luminance,
}

const CHANNELS: [(Channel, &str); 3] = [
    (Channel::Hue, "hue"),
    (Channel::Saturation, "saturation"),
    (Channel::Luminance, "luminance"),
];

fn channel(name: &str) -> Option<Channel> {
    CHANNELS
        .iter()
        .find(|(_, channel_name)| *channel_name == name)
        .map(|(channel, _)| *channel)
}

/// The mixer value `color_channel`, such as `blue_saturation`.
fn mixer_value<'a>(exposure: &'a mut ExposureParams, name: &str) -> Option<(&'a mut f32, f32)> {
    let (color, channel_name) = name.split_once('_')?;
    let index = MIXER_COLORS.iter().position(|known| *known == color)?;
    Some(match channel(channel_name)? {
        Channel::Hue => (&mut exposure.hsl_hue[index], MIXER_HUE_SCALE),
        Channel::Saturation => (&mut exposure.hsl_saturation[index], 1.0),
        Channel::Luminance => (&mut exposure.hsl_luminance[index], 1.0),
    })
}

/// Whether `name` is a mixer value.
pub(super) fn is_mixer_value(name: &str) -> bool {
    mixer_value(&mut ExposureParams::default(), name).is_some()
}

/// Whether `name` is a grading value.
pub(super) fn is_grading_value(name: &str) -> bool {
    grading_value(&mut ColorGrading::default(), name).is_some()
}

/// The mixer values away from zero, as Gemma reads them.
pub(super) fn changed_mixer(exposure: &ExposureParams) -> Map<String, Value> {
    let mut exposure = *exposure;
    let mut values = Map::new();
    for color in MIXER_COLORS {
        for (_, channel_name) in CHANNELS {
            let name = format!("{color}_{channel_name}");
            let Some((stored, scale)) = mixer_value(&mut exposure, &name) else {
                continue;
            };
            let value = rounded(*stored / scale, 0);
            if value != 0.0 {
                values.insert(name, json!(value));
            }
        }
    }
    values
}

/// Applies one `set_color_mixer` call's `arguments` to `exposure`, each value
/// within its range. `baseline`, the edit when the reply came, gives the
/// values they were.
pub(super) fn apply_set_color_mixer(
    exposure: &mut ExposureParams,
    baseline: &ExposureParams,
    arguments: &Value,
) -> Value {
    let mut outcome = Outcome::default();
    let mut baseline = *baseline;
    for (name, requested) in arguments.as_object().into_iter().flatten() {
        let (Some(requested), Some((from, _))) =
            (number(requested), mixer_value(&mut baseline, name))
        else {
            outcome.not_set(name);
            continue;
        };
        let from = *from;
        let Some((field, scale)) = mixer_value(exposure, name) else {
            outcome.not_set(name);
            continue;
        };
        let was = rounded(from / scale, 0);
        let value = MIXER_LIMITS.limit(requested);
        *field = value * scale;
        outcome.set_number(name, requested, was, value, MIXER_LIMITS.decimals);
    }
    outcome.into_response()
}

fn wheel<'a>(grading: &'a mut ColorGrading, name: &str) -> Option<&'a mut ColorGradeWheel> {
    Some(match name {
        "shadows" => &mut grading.shadows,
        "midtones" => &mut grading.midtones,
        "highlights" => &mut grading.highlights,
        "global" => &mut grading.global,
        _ => return None,
    })
}

/// The grading value `name` and its limits.
fn grading_value<'a>(grading: &'a mut ColorGrading, name: &str) -> Option<(&'a mut f32, Limits)> {
    match name {
        "blending" => return Some((&mut grading.blending, Limits::new(0.0, 100.0, 0))),
        "balance" => return Some((&mut grading.balance, Limits::new(-100.0, 100.0, 0))),
        _ => {}
    }
    let (wheel_name, channel_name) = name.split_once('_')?;
    let wheel = wheel(grading, wheel_name)?;
    Some(match channel(channel_name)? {
        Channel::Hue => (&mut wheel.hue, Limits::new(0.0, 360.0, 0)),
        Channel::Saturation => (&mut wheel.saturation, Limits::new(0.0, 100.0, 0)),
        Channel::Luminance => (&mut wheel.luminance, Limits::new(-100.0, 100.0, 0)),
    })
}

/// The grading values away from their defaults, as Gemma reads them; a
/// wheel's hue only while it has saturation to show it.
pub(super) fn changed_grading(exposure: &ExposureParams) -> Map<String, Value> {
    let grading = exposure.color_grading;
    let defaults = ColorGrading::default();
    let mut values = Map::new();
    let wheels = [
        ("shadows", grading.shadows),
        ("midtones", grading.midtones),
        ("highlights", grading.highlights),
        ("global", grading.global),
    ];
    for (wheel_name, wheel) in wheels {
        let saturation = rounded(wheel.saturation, 0);
        let luminance = rounded(wheel.luminance, 0);
        if saturation != 0.0 {
            values.insert(format!("{wheel_name}_hue"), json!(rounded(wheel.hue, 0)));
            values.insert(format!("{wheel_name}_saturation"), json!(saturation));
        }
        if luminance != 0.0 {
            values.insert(format!("{wheel_name}_luminance"), json!(luminance));
        }
    }
    if rounded(grading.blending, 0) != defaults.blending {
        values.insert("blending".to_owned(), json!(rounded(grading.blending, 0)));
    }
    if rounded(grading.balance, 0) != defaults.balance {
        values.insert("balance".to_owned(), json!(rounded(grading.balance, 0)));
    }
    values
}

/// Applies one `set_color_grading` call's `arguments` to `exposure`, each
/// value within its range. `baseline`, the edit when the reply came, gives
/// the values they were.
pub(super) fn apply_set_color_grading(
    exposure: &mut ExposureParams,
    baseline: &ExposureParams,
    arguments: &Value,
) -> Value {
    let mut outcome = Outcome::default();
    let mut baseline = baseline.color_grading;
    for (name, requested) in arguments.as_object().into_iter().flatten() {
        let (Some(requested), Some((from, _))) =
            (number(requested), grading_value(&mut baseline, name))
        else {
            outcome.not_set(name);
            continue;
        };
        let from = *from;
        // A wheel without a tint has no hue chosen, and its hue 0 is red:
        // Gemma set saturations alone and cast the photo red.
        if let Some(wheel_name) = name.strip_suffix("_saturation") {
            let hue = format!("{wheel_name}_hue");
            if from == 0.0 && arguments.get(&hue).is_none() {
                outcome.not_set(&format!("{name} (set {hue} with it)"));
                continue;
            }
        }
        let Some((field, limits)) = grading_value(&mut exposure.color_grading, name) else {
            outcome.not_set(name);
            continue;
        };
        let was = rounded(from, 0);
        let value = limits.limit(requested);
        *field = value;
        outcome.set_number(name, requested, was, value, limits.decimals);
    }
    outcome.into_response()
}
