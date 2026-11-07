//! The brief Gemma writes for a request (assets/gemma/interpret_system.md),
//! read into the tool calls that carry it out. Each item names a setting, up
//! or down and a size word, and the app turns the size into a step added to
//! the setting's current value. A second Gemma conversation used to write the
//! values from the brief, and wrote some with the wrong sign or to the wrong
//! tool (`shadows up` as a shadow tint).
//!
//! Items are read as Gemma writes them: a label repeated on several lines,
//! several items on one line between commas, and no direction for a setting
//! that only goes one way (`vignette strong`). An item that could mean two
//! things, such as `tint blue slight` (the tint slider or a tint's colour),
//! is reported as unread, not guessed.

use super::calls::Call;
use super::color::MIXER_COLORS;
use super::tools::{rounded, SET_COLOR_GRADING, SET_COLOR_MIXER, SET_EFFECT, SET_SLIDERS};
use serde_json::{json, Map, Value};

/// A size word's step, smallest first: a touch, slight, moderate, strong and
/// very strong.
type Steps = [f32; 5];
const SLIDER_STEPS: Steps = [5.0, 10.0, 25.0, 50.0, 75.0];
/// The photographer's own edits move highlights about -62 and shadows about
/// +55 (research/fine-tuning.md), so strong recovers more than elsewhere.
const RECOVERY_STEPS: Steps = [5.0, 10.0, 25.0, 70.0, 90.0];
/// In stops.
const EXPOSURE_STEPS: Steps = [0.15, 0.3, 0.7, 1.5, 2.5];
/// From the default 40, slight, moderate and strong reach 60, 80 and 110.
const SHARPEN_STEPS: Steps = [10.0, 20.0, 40.0, 70.0, 100.0];
/// From off, slight, moderate and strong reach 20, 40 and 60.
const NOISE_STEPS: Steps = [10.0, 20.0, 40.0, 60.0, 80.0];

/// A shadow or highlight tint's colours and their hues on the grading wheel.
const TINT_HUES: [(&str, f32); 10] = [
    ("red", 0.0),
    ("orange", 30.0),
    ("amber", 45.0),
    ("yellow", 60.0),
    ("green", 120.0),
    ("aqua", 180.0),
    ("teal", 195.0),
    ("blue", 240.0),
    ("purple", 270.0),
    ("magenta", 300.0),
];

/// A colour range's qualities, by the word for each and its mixer value.
const MIXER_CHANNELS: [(&str, &str); 4] = [
    ("saturation", "saturation"),
    ("brightness", "luminance"),
    ("luminance", "luminance"),
    ("hue", "hue"),
];

/// The settings an item names in one word, or two for noise reduction.
const SETTINGS: [StepSetting; 19] = [
    StepSetting::new(&["exposure"], SET_SLIDERS, &["exposure"], EXPOSURE_STEPS),
    StepSetting::new(&["contrast"], SET_SLIDERS, &["contrast"], SLIDER_STEPS),
    StepSetting::new(
        &["highlights", "highlight"],
        SET_SLIDERS,
        &["highlights"],
        RECOVERY_STEPS,
    ),
    StepSetting::new(
        &["shadows", "shadow"],
        SET_SLIDERS,
        &["shadows"],
        RECOVERY_STEPS,
    ),
    StepSetting::new(&["whites"], SET_SLIDERS, &["whites"], SLIDER_STEPS),
    StepSetting::new(&["blacks"], SET_SLIDERS, &["blacks"], SLIDER_STEPS),
    StepSetting::new(&["warmth"], SET_SLIDERS, &["warmth"], SLIDER_STEPS),
    StepSetting::new(&["tint"], SET_SLIDERS, &["tint"], SLIDER_STEPS),
    StepSetting::new(&["saturation"], SET_SLIDERS, &["saturation"], SLIDER_STEPS),
    StepSetting::new(&["vibrance"], SET_SLIDERS, &["vibrance"], SLIDER_STEPS),
    StepSetting::new(&["texture"], SET_SLIDERS, &["texture"], SLIDER_STEPS),
    StepSetting::new(&["clarity"], SET_SLIDERS, &["clarity"], SLIDER_STEPS),
    StepSetting::new(&["dehaze"], SET_SLIDERS, &["dehaze"], SLIDER_STEPS),
    StepSetting::new(
        &["sharpening", "sharpen", "sharpness"],
        SET_SLIDERS,
        &["sharpen_amount"],
        SHARPEN_STEPS,
    )
    .one_way(),
    StepSetting::new(
        &["noise"],
        SET_SLIDERS,
        &["noise_luminance", "noise_color"],
        NOISE_STEPS,
    )
    .one_way(),
    // Up is a stronger vignette, a more negative amount.
    StepSetting::new(
        &["vignette"],
        SET_EFFECT,
        &["vignette_amount"],
        SLIDER_STEPS,
    )
    .one_way()
    .inverted(),
    StepSetting::new(&["grain"], SET_EFFECT, &["grain_amount"], SLIDER_STEPS).one_way(),
    StepSetting::new(&["glow"], SET_EFFECT, &["glow_amount"], SLIDER_STEPS).one_way(),
    StepSetting::new(
        &["halation"],
        SET_EFFECT,
        &["halation_amount"],
        SLIDER_STEPS,
    )
    .one_way(),
];

/// A setting whose values an item moves by a step.
#[derive(Clone, Copy)]
struct StepSetting {
    /// The words that name it.
    words: &'static [&'static str],
    tool: &'static str,
    values: &'static [&'static str],
    steps: Steps,
    /// Whether it goes only up from off, so an item without a direction
    /// raises it.
    one_way: bool,
    /// Whether up lowers its values.
    inverted: bool,
}

impl StepSetting {
    const fn new(
        words: &'static [&'static str],
        tool: &'static str,
        values: &'static [&'static str],
        steps: Steps,
    ) -> Self {
        Self {
            words,
            tool,
            values,
            steps,
            one_way: false,
            inverted: false,
        }
    }

    const fn one_way(self) -> Self {
        Self {
            one_way: true,
            ..self
        }
    }

    const fn inverted(self) -> Self {
        Self {
            inverted: true,
            ..self
        }
    }
}

/// What an item names.
#[derive(Clone, Copy)]
enum Setting {
    Steps(StepSetting),
    /// Saturation at -100.
    BlackAndWhite,
    /// The shadow or highlight wheel's tint: its hue from a colour's name and
    /// its saturation by steps.
    Tint {
        wheel: &'static str,
    },
    /// One colour range's hue, saturation or luminance.
    Mixer {
        range: &'static str,
        channel: &'static str,
    },
}

/// What the app makes of a brief.
pub(super) struct Brief {
    /// The calls that carry out its items, one per tool.
    pub(super) calls: Vec<Call>,
    /// The sentence for the photographer, from its `Summary:` line.
    pub(super) summary: Option<String>,
    /// The items it could not read.
    pub(super) unread: Vec<String>,
}

/// The values the brief's items set so far, so an item that moves a value
/// another already moved steps from there.
struct Plan<'a> {
    /// Each value's tool, name and new value, in the order first set.
    values: Vec<(&'static str, String, f32)>,
    current: &'a dyn Fn(&str) -> Option<f32>,
    previous: &'a dyn Fn(&str) -> Option<f32>,
}

impl Plan<'_> {
    fn value(&self, name: &str) -> f32 {
        self.values
            .iter()
            .rev()
            .find(|(_, planned, _)| planned == name)
            .map(|(_, _, value)| *value)
            .or_else(|| (self.current)(name))
            // A value the photo cannot read, such as a JPEG's tint, which its
            // tool then reports as not set.
            .unwrap_or(0.0)
    }

    fn set(&mut self, tool: &'static str, name: &str, value: f32) {
        let decimals = if name == "exposure" { 2 } else { 0 };
        let value = rounded(value, decimals);
        match self
            .values
            .iter_mut()
            .find(|(_, planned, _)| planned == name)
        {
            Some((_, _, planned)) => *planned = value,
            None => self.values.push((tool, name.to_owned(), value)),
        }
    }

    fn step(&mut self, tool: &'static str, name: &str, delta: f32) {
        let value = self.value(name) + delta;
        self.set(tool, name, value);
    }

    /// Sets each of `names` back to its value before the previous request;
    /// `false` when that request changed none of them.
    fn back(&mut self, tool: &'static str, names: &[String]) -> bool {
        let mut any = false;
        for name in names {
            if let Some(was) = (self.previous)(name) {
                self.set(tool, name, was);
                any = true;
            }
        }
        any
    }

    fn into_calls(self) -> Vec<Call> {
        let mut calls: Vec<Call> = Vec::new();
        for (tool, name, value) in self.values {
            let index = match calls.iter().position(|call| call.name == tool) {
                Some(index) => index,
                None => {
                    calls.push(Call {
                        name: tool.to_owned(),
                        arguments: Value::Object(Map::new()),
                    });
                    calls.len() - 1
                }
            };
            if let Value::Object(arguments) = &mut calls[index].arguments {
                arguments.insert(name, json!(value));
            }
        }
        calls
    }
}

/// Reads `text`, a brief, into the calls that carry it out. `current` gives a
/// value as its tool reads it, and `previous` its value before the previous
/// request, for `back`.
pub(super) fn read_brief(
    text: &str,
    current: &dyn Fn(&str) -> Option<f32>,
    previous: &dyn Fn(&str) -> Option<f32>,
) -> Brief {
    let mut plan = Plan {
        values: Vec::new(),
        current,
        previous,
    };
    let mut summary = None;
    let mut unread = Vec::new();
    for line in text.lines() {
        let Some((label, items)) = line.split_once(':') else {
            continue;
        };
        let label = label
            .trim_matches(|c: char| c == '*' || c == '-' || c == '#' || c.is_whitespace())
            .to_lowercase();
        let items = items.trim_matches(|c: char| c == '*' || c.is_whitespace());
        match label.as_str() {
            "summary" => summary = Some(items.to_owned()).filter(|text| !text.is_empty()),
            // What to keep guided Gemma's choice of items; other lines, such
            // as a genre Gemma wrote first, ask for nothing.
            "tone" | "colour" | "color" | "detail" | "effects" => {
                for item in items.split([',', ';']) {
                    let item = item.trim().trim_end_matches('.');
                    if item.is_empty() || item.eq_ignore_ascii_case("no change") {
                        continue;
                    }
                    if read_item(&mut plan, item).is_none() {
                        unread.push(item.to_owned());
                    }
                }
            }
            _ => {}
        }
    }
    Brief {
        calls: plan.into_calls(),
        summary,
        unread,
    }
}

/// Adds `item` to `plan`; `None` when the item is not one the brief's
/// grammar allows or could mean two things.
fn read_item(plan: &mut Plan<'_>, item: &str) -> Option<()> {
    let words: Vec<String> = item
        .to_lowercase()
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| {
                !(c.is_alphanumeric() || c == '.' || c == '-' || c == '+' || c == '_')
            })
            .trim_end_matches('.')
            .to_owned()
        })
        .filter(|word| !word.is_empty())
        .collect();
    let has = |wanted: &str| words.iter().any(|word| word == wanted);
    let setting = setting(&words)?;
    let back = has("back");
    let direction = direction(&words)?;
    let size = size(&words);
    let number = words.iter().find_map(|word| word.parse::<f32>().ok());

    match setting {
        Setting::BlackAndWhite => {
            if back {
                return plan
                    .back(SET_SLIDERS, &["saturation".to_owned()])
                    .then_some(());
            }
            plan.set(SET_SLIDERS, "saturation", -100.0);
        }
        Setting::Steps(setting) => {
            let names: Vec<String> = setting
                .values
                .iter()
                .map(|name| (*name).to_owned())
                .collect();
            if back {
                return plan.back(setting.tool, &names).then_some(());
            }
            let delta = match (number, size) {
                // Only exposure may give its change as a number, in stops.
                (Some(stops), None) if matches!(setting.values, ["exposure"]) => match direction {
                    Some(direction) => direction * stops.abs(),
                    None => stops,
                },
                (None, Some(size)) => {
                    let direction = direction.or(setting.one_way.then_some(1.0))?;
                    let sign = if setting.inverted { -1.0 } else { 1.0 };
                    direction * sign * setting.steps[size]
                }
                _ => return None,
            };
            for name in &names {
                plan.step(setting.tool, name, delta);
            }
        }
        Setting::Tint { wheel } => {
            let hue_name = format!("{wheel}_hue");
            let saturation_name = format!("{wheel}_saturation");
            if back {
                return plan
                    .back(SET_COLOR_GRADING, &[hue_name, saturation_name])
                    .then_some(());
            }
            let hue = TINT_HUES
                .iter()
                .find(|(colour, _)| has(colour))
                .map(|(_, hue)| *hue)?;
            let step = SLIDER_STEPS[size?];
            if number.is_some() {
                return None;
            }
            plan.set(SET_COLOR_GRADING, &hue_name, hue);
            plan.step(
                SET_COLOR_GRADING,
                &saturation_name,
                direction.unwrap_or(1.0) * step,
            );
        }
        Setting::Mixer { range, channel } => {
            let name = format!("{range}_{channel}");
            if back {
                return plan.back(SET_COLOR_MIXER, &[name]).then_some(());
            }
            if number.is_some() {
                return None;
            }
            let delta = direction? * SLIDER_STEPS[size?];
            plan.step(SET_COLOR_MIXER, &name, delta);
        }
    }
    Some(())
}

/// The setting `words` name, or `None` when they name none or more than one.
fn setting(words: &[String]) -> Option<Setting> {
    let has = |wanted: &str| words.iter().any(|word| word == wanted);
    if words
        .windows(3)
        .any(|three| three[0] == "black" && three[1] == "and" && three[2] == "white")
        || has("black-and-white")
    {
        return Some(Setting::BlackAndWhite);
    }
    if has("tint") {
        let shadows = has("shadow") || has("shadows");
        let highlights = has("highlight") || has("highlights");
        match (shadows, highlights) {
            (true, false) => return Some(Setting::Tint { wheel: "shadows" }),
            (false, true) => {
                return Some(Setting::Tint {
                    wheel: "highlights",
                })
            }
            (true, true) => return None,
            (false, false) => {}
        }
    }
    let colours: Vec<&str> = MIXER_COLORS
        .iter()
        .copied()
        .filter(|colour| has(colour))
        .collect();
    let channels: Vec<&str> = MIXER_CHANNELS
        .iter()
        .filter(|(word, _)| has(word))
        .map(|(_, channel)| *channel)
        .collect();
    match (colours.as_slice(), channels.as_slice()) {
        ([range], [channel]) => return Some(Setting::Mixer { range, channel }),
        ([], _) => {}
        _ => return None,
    }
    // A colour without a range's quality, as in `tint blue`, is a tint or a
    // colour range the item does not say.
    if TINT_HUES.iter().any(|(colour, _)| has(colour)) {
        return None;
    }
    let mut named = SETTINGS
        .iter()
        .filter(|setting| setting.words.iter().any(|word| has(word)));
    let setting = named.next()?;
    named.next().is_none().then_some(Setting::Steps(*setting))
}

/// +1 for up, -1 for down, `None` inside when the item gives no direction,
/// and `None` outside when it gives both.
fn direction(words: &[String]) -> Option<Option<f32>> {
    let up = words.iter().any(|word| {
        matches!(
            word.as_str(),
            "up" | "more" | "increase" | "raise" | "boost"
        )
    });
    let down = words.iter().any(|word| {
        matches!(
            word.as_str(),
            "down" | "less" | "decrease" | "reduce" | "lower"
        )
    });
    match (up, down) {
        (true, true) => None,
        (true, false) => Some(Some(1.0)),
        (false, true) => Some(Some(-1.0)),
        (false, false) => Some(None),
    }
}

/// The index of the item's size word in a [`Steps`], or `None` without one.
fn size(words: &[String]) -> Option<usize> {
    words.iter().enumerate().find_map(|(index, word)| {
        let very = index > 0 && words[index - 1] == "very";
        if word == "touch" {
            Some(0)
        } else if word.starts_with("slight") {
            Some(1)
        } else if word.starts_with("moderate") {
            Some(2)
        } else if word.starts_with("strong") {
            Some(if very { 4 } else { 3 })
        } else {
            None
        }
    })
}
