//! What the editing tools share: their names, which tool sets each value,
//! reading a requested value, keeping it within its control's range,
//! answering the call, how values are written to the model, and what a
//! request's calls changed. Each tool is described to Gemma by
//! android/app/src/main/assets/gemma/editor_system.md, whose value names and
//! ranges its module follows.

use super::calls::Call;
use super::color::{is_grading_value, is_mixer_value};
use super::effects::is_effect_value;
use super::sliders::is_slider;
use super::tone_curve::is_curve;
use serde_json::{json, Map, Value};
use std::ops::RangeInclusive;

/// The tools the app applies.
pub(super) const SET_SLIDERS: &str = "set_sliders";
pub(super) const SET_COLOR_MIXER: &str = "set_color_mixer";
pub(super) const SET_COLOR_GRADING: &str = "set_color_grading";
pub(super) const SET_TONE_CURVE: &str = "set_tone_curve";
pub(super) const SET_EFFECT: &str = "set_effect";
pub(super) const SET_AI_DENOISE: &str = "set_ai_denoise";
/// Claude's tools besides those, which assets/claude/editor_system.md
/// describes.
pub(super) const SET_MASK: &str = "set_mask";
pub(super) const SET_MASK_PART: &str = "set_mask_part";
pub(super) const SET_CROP: &str = "set_crop";

/// A requested value: a number, or a number the model wrote as a string.
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

/// `value` as the model writes it: a number, `true`, or a curve's outputs joined
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
/// current values in a message to the model. Not the call syntax: Gemma copied
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

/// The tool that sets the value `name`; every value name belongs to one.
fn owner(name: &str) -> Option<&'static str> {
    [
        (SET_SLIDERS, is_slider as fn(&str) -> bool),
        (SET_COLOR_MIXER, is_mixer_value),
        (SET_COLOR_GRADING, is_grading_value),
        (SET_TONE_CURVE, is_curve),
        (SET_EFFECT, is_effect_value),
    ]
    .into_iter()
    .find_map(|(tool, owns)| owns(name).then_some(tool))
}

pub(super) fn is_value_name(name: &str) -> bool {
    owner(name).is_some()
}

/// `calls` with each value moved to the call of the tool that sets it, in the
/// order the tools are first written: Gemma wrote `vignette_amount` in its
/// `set_sliders` line, which set nothing.
pub(super) fn routed(calls: Vec<Call>) -> Vec<Call> {
    let mut routed: Vec<Call> = Vec::new();
    for call in calls {
        let arguments = match call.arguments {
            // These tools' values are their own, even where another tool
            // has one of the same name, such as a mask's `exposure`.
            Value::Object(arguments)
                if ![SET_AI_DENOISE, SET_MASK, SET_MASK_PART, SET_CROP]
                    .contains(&call.name.as_str())
                    && !arguments.is_empty() =>
            {
                arguments
            }
            arguments => {
                routed.push(Call {
                    name: call.name,
                    arguments,
                });
                continue;
            }
        };
        for (name, value) in arguments {
            let tool = owner(&name).map_or_else(|| call.name.clone(), str::to_owned);
            let index = match routed.iter().position(|routed| routed.name == tool) {
                Some(index) => index,
                None => {
                    routed.push(Call {
                        name: tool,
                        arguments: Value::Object(serde_json::Map::new()),
                    });
                    routed.len() - 1
                }
            };
            if let Value::Object(arguments) = &mut routed[index].arguments {
                arguments.insert(name, value);
            }
        }
    }
    routed
}

/// A tool's `response` to a call of `tool`, as a line for the model in the call
/// syntax: `tool name=before→after …`, then what it kept in range or could
/// not set.
pub(super) fn result_line(tool: &str, response: &Value) -> String {
    if let Some(error) = response.get("error").and_then(Value::as_str) {
        return format!("{tool}: {error}");
    }
    let was = response.get("was").and_then(Value::as_object);
    let mut line = tool.to_owned();
    for (name, now) in response
        .get("now")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        match was.and_then(|was| was.get(name)) {
            Some(was) => line.push_str(&format!(" {name}={}→{}", written(was), written(now))),
            None => line.push_str(&format!(" {name}={}", written(now))),
        }
    }
    for (key, label) in [("limited", "kept within its range"), ("not_set", "not set")] {
        let names = response
            .get(key)
            .and_then(Value::as_array)
            .filter(|names| !names.is_empty());
        if let Some(names) = names {
            let names: Vec<String> = names.iter().map(written).collect();
            line.push_str(&format!("; {label}: {}", names.join(" ")));
        }
    }
    if let Some(status) = response.get("status").and_then(Value::as_str) {
        line.push_str(&format!("; {status}"));
    }
    line
}

/// A value a request's nudges changed: from `was`, its value before the
/// first of them, to `now`.
pub(super) struct Change {
    tool: String,
    name: String,
    was: Value,
    now: Value,
}

/// The tool `call`'s changes are recorded under: a mask tool's with the
/// mask and part it set, since every mask has an `exposure`.
pub(super) fn change_tool(call: &Call) -> String {
    if !matches!(call.name.as_str(), SET_MASK | SET_MASK_PART) {
        return call.name.clone();
    }
    let mut tool = call.name.clone();
    for key in ["name", "part"] {
        if let Some(value) = call.arguments.get(key) {
            tool.push_str(&format!(" {key}={}", written(value)));
        }
    }
    tool
}

/// Adds the values a call of `tool` set, by its `response`, to `changes`.
pub(super) fn record_changes(changes: &mut Vec<Change>, tool: &str, response: &Value) {
    let (Some(was), Some(now)) = (
        response.get("was").and_then(Value::as_object),
        response.get("now").and_then(Value::as_object),
    ) else {
        return;
    };
    for (name, now) in now {
        let Some(was) = was.get(name) else {
            continue;
        };
        match changes
            .iter_mut()
            .find(|change| change.tool == tool && change.name == *name)
        {
            Some(change) => change.now = now.clone(),
            None => changes.push(Change {
                tool: tool.to_owned(),
                name: name.clone(),
                was: was.clone(),
                now: now.clone(),
            }),
        }
    }
}

/// The request `text` and its `changes` for the next request's first
/// message, the changes in the call syntax as `name=before→after`.
pub(super) fn change_summary(text: &str, changes: &[Change]) -> String {
    let mut groups: Vec<(&str, Vec<String>)> = Vec::new();
    for change in changes.iter().filter(|change| change.was != change.now) {
        let pair = format!(
            "{}={}→{}",
            change.name,
            written(&change.was),
            written(&change.now)
        );
        match groups.iter_mut().find(|(tool, _)| *tool == change.tool) {
            Some((_, pairs)) => pairs.push(pair),
            None => groups.push((&change.tool, vec![pair])),
        }
    }
    if groups.is_empty() {
        return format!("Previous request: \"{text}\". It changed nothing.");
    }
    let changed = groups
        .iter()
        .map(|(tool, pairs)| format!("{tool} {}", pairs.join(" ")))
        .collect::<Vec<_>>()
        .join("\n");
    format!("Previous request: \"{text}\". It changed:\n{changed}")
}
