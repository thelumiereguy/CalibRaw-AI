//! What Gemma's editing tools share: reading a requested value, keeping it
//! within its control's range, answering the call, and how
//! values are written to Gemma. Each tool is described to Gemma by
//! android/app/src/main/assets/gemma/editor_system.md, whose value names and
//! ranges its module follows.

use serde_json::{json, Map, Value};
use std::ops::RangeInclusive;

/// A requested value: a number, or a number Gemma wrote as a string.
pub(super) fn number(value: &Value) -> Option<f32> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
}

/// Two numbers written `a/b`, such as a position.
pub(super) fn pair(value: &Value) -> Option<[f32; 2]> {
    match value.as_array()?.as_slice() {
        [a, b] => Some([number(a)?, number(b)?]),
        _ => None,
    }
}

/// `value` with `decimals` decimal places, as its slider shows it.
pub(super) fn rounded(value: f32, decimals: u32) -> f32 {
    let scale = 10_f32.powi(decimals as i32);
    (value * scale).round() / scale
}

/// One setting a tool changes. A nudge may move it anywhere in its range: a
/// cap per nudge kept Gemma from reaching black and white or a strong effect
/// within the nudges it has.
#[derive(Clone)]
pub(super) struct Limits {
    pub(super) range: RangeInclusive<f32>,
    pub(super) decimals: u32,
}

impl Limits {
    pub(super) const fn new(min: f32, max: f32, decimals: u32) -> Self {
        Self {
            range: min..=max,
            decimals,
        }
    }

    /// `requested` within the range, as its control shows it.
    pub(super) fn limit(&self, requested: f32) -> f32 {
        rounded(
            requested.clamp(*self.range.start(), *self.range.end()),
            self.decimals,
        )
    }
}

/// A tool's response to one call: the values it set as they were and as
/// they now are, and which it kept in range or could not set.
#[derive(Default)]
pub(super) struct Outcome {
    was: Map<String, Value>,
    now: Map<String, Value>,
    limited: Vec<String>,
    not_set: Vec<String>,
}

impl Outcome {
    /// Notes that `name` went from `was`, its value when the reply came, to
    /// `now`.
    pub(super) fn set(&mut self, name: &str, was: Value, now: Value) {
        self.was.insert(name.to_owned(), was);
        self.now.insert(name.to_owned(), now);
    }

    /// Notes that `name` is now `now`, where it had no value before, such as
    /// a new mask's shape.
    pub(super) fn now(&mut self, name: &str, now: Value) {
        self.now.insert(name.to_owned(), now);
    }

    /// Notes that `name` went from `was` to `value`, and when `value` is not
    /// `requested`.
    pub(super) fn set_number(
        &mut self,
        name: &str,
        requested: f32,
        was: f32,
        value: f32,
        decimals: u32,
    ) {
        if (value - rounded(requested, decimals)).abs() > f32::EPSILON {
            self.limited(name);
        }
        self.set(name, json!(was), json!(value));
    }

    pub(super) fn limited(&mut self, name: &str) {
        self.limited.push(name.to_owned());
    }

    pub(super) fn not_set(&mut self, name: &str) {
        self.not_set.push(name.to_owned());
    }

    /// `{"was": {...}, "now": {...}}`, with `"limited"` and `"not_set"` name
    /// lists when there are any.
    pub(super) fn into_response(self) -> Value {
        let mut response = Map::new();
        response.insert("was".to_owned(), Value::Object(self.was));
        response.insert("now".to_owned(), Value::Object(self.now));
        if !self.limited.is_empty() {
            response.insert("limited".to_owned(), json!(self.limited));
        }
        if !self.not_set.is_empty() {
            response.insert("not_set".to_owned(), json!(self.not_set));
        }
        Value::Object(response)
    }
}

/// `value` as Gemma writes it: a number, `true`, or a curve's outputs joined
/// by `/`.
pub(super) fn written(value: &Value) -> String {
    match value {
        Value::Array(values) => values.iter().map(written).collect::<Vec<_>>().join("/"),
        Value::String(text) => text.clone(),
        // The values are f32s, so 0.3 is written 0.3, not 0.30000001192092896.
        Value::Number(number) if number.is_f64() => number
            .as_f64()
            .map_or_else(|| number.to_string(), |value| (value as f32).to_string()),
        other => other.to_string(),
    }
}

/// `values` as `name value` pairs, a curve's outputs apart, for the edit's
/// current values in a message to Gemma. Not the call syntax: Gemma copied
/// values shown in it back as calls, and garbled the curves it copied.
pub(super) fn describe(values: &Map<String, Value>) -> String {
    values
        .iter()
        .map(|(name, value)| match value {
            Value::Array(values) => {
                let outputs: Vec<String> = values.iter().map(written).collect();
                format!("{name} {}", outputs.join(" "))
            }
            value => format!("{name} {}", written(value)),
        })
        .collect::<Vec<_>>()
        .join(", ")
}
