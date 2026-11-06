//! The tone curves Gemma's `set_tone_curve` tool shapes, as five outputs
//! (0–255) at fixed inputs. editor_system.md no longer lists the tool: the
//! curves Gemma wrote clipped or cast the photo. Gemma still reads the curves
//! set by hand, and the tool stays settable.

use super::tools::{number, rounded, Limits, Outcome};
use crate::pipeline::{ExposureParams, PointCurve, MAX_POINT_CURVE_POINTS};
use serde_json::{json, Map, Value};

/// The inputs Gemma sets the outputs at, in 0–255.
const INPUTS: [f32; 5] = [0.0, 64.0, 128.0, 192.0, 255.0];
const LEVELS: f32 = 255.0;
const OUTPUT_LIMITS: Limits = Limits::new(0.0, LEVELS, 0);

const CURVES: [&str; 4] = ["rgb", "red", "green", "blue"];

fn curve<'a>(exposure: &'a mut ExposureParams, name: &str) -> Option<&'a mut PointCurve> {
    Some(match name {
        "rgb" => &mut exposure.tone_curve,
        "red" => &mut exposure.tone_curve_red,
        "green" => &mut exposure.tone_curve_green,
        "blue" => &mut exposure.tone_curve_blue,
        _ => return None,
    })
}

/// The outputs of `curve` at [`INPUTS`], between its points linearly. A curve
/// with points elsewhere reads close to how it renders, not exactly.
fn outputs(curve: &PointCurve) -> [f32; 5] {
    INPUTS.map(|input| rounded(curve.polyline_value(input / LEVELS) * LEVELS, 0))
}

/// Whether `name` is one of the curves.
pub(super) fn is_curve(name: &str) -> bool {
    CURVES.contains(&name)
}

/// The curves that are not straight, as Gemma reads them.
pub(super) fn changed_curves(exposure: &ExposureParams) -> Map<String, Value> {
    let mut exposure = *exposure;
    CURVES
        .iter()
        .filter_map(|name| {
            let curve = curve(&mut exposure, name)?;
            (!curve.is_identity()).then(|| ((*name).to_owned(), json!(outputs(curve))))
        })
        .collect()
}

/// Applies one `set_tone_curve` call's `arguments` to `exposure`. A curve
/// given becomes the five points; `baseline`, the edit when the reply came,
/// gives the outputs they were.
pub(super) fn apply_set_tone_curve(
    exposure: &mut ExposureParams,
    baseline: &ExposureParams,
    arguments: &Value,
) -> Value {
    let mut outcome = Outcome::default();
    let mut baseline = *baseline;
    for (name, requested) in arguments.as_object().into_iter().flatten() {
        // Outputs that do not rise are a garbled curve, such as 5/70/2/6/25,
        // which Gemma wrote for 15/70/120/167/255; setting it cast the photo
        // yellow.
        let requested: Option<Vec<f32>> = requested
            .as_array()
            .filter(|values| values.len() == INPUTS.len())
            .and_then(|values| values.iter().map(number).collect())
            .filter(|outputs: &Vec<f32>| outputs.windows(2).all(|pair| pair[0] <= pair[1]));
        let (Some(requested), Some(from)) = (requested, curve(&mut baseline, name)) else {
            outcome.not_set(name);
            continue;
        };
        let from = outputs(from);
        let Some(target) = curve(exposure, name) else {
            outcome.not_set(name);
            continue;
        };
        let mut points = [[1.0, 1.0]; MAX_POINT_CURVE_POINTS];
        let mut limited = false;
        for (index, point) in points.iter_mut().take(INPUTS.len()).enumerate() {
            let output = OUTPUT_LIMITS.limit(requested[index]);
            limited |= output != rounded(requested[index], 0);
            *point = [INPUTS[index] / LEVELS, output / LEVELS];
        }
        *target = PointCurve {
            points,
            len: INPUTS.len() as u32,
        };
        target.sanitize();
        if limited {
            outcome.limited(name);
        }
        outcome.set(name, json!(from), json!(outputs(target)));
    }
    outcome.into_response()
}
