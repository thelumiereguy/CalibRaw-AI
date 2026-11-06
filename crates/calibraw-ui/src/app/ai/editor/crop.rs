//! Claude's `set_crop` tool (Claude Code only,
//! assets/claude/editor_system.md): the crop and its aspect ratio, the
//! straighten angle, quarter turns, flips, perspective and skew, set as the
//! Crop panel sets them (ui/sidebar/crop.rs). The crop box is given in
//! fractions of the uncropped photo as it is displayed, after the rotation,
//! flips and quarter turns, x to the right and y down.

use super::tools::{number, rounded, Outcome};
use super::*;
use crate::pipeline::{CropAspectRatio, GeometryMapping, PerspectiveCorrection};
use serde_json::{json, Map, Value};

/// The fixed ratios the Crop panel names; any other is a custom one.
const NAMED_RATIOS: [CropAspectRatio; 7] = [
    CropAspectRatio::Square,
    CropAspectRatio::FourThree,
    CropAspectRatio::ThreeFour,
    CropAspectRatio::ThreeTwo,
    CropAspectRatio::TwoThree,
    CropAspectRatio::SixteenNine,
    CropAspectRatio::NineSixteen,
];

/// The values the tool sets, as `geometry_values` writes them.
const VALUES: [&str; 10] = [
    "aspect",
    "crop",
    "straighten",
    "turns",
    "flip_horizontal",
    "flip_vertical",
    "vertical",
    "horizontal",
    "horizontal_skew",
    "vertical_skew",
];

fn aspect_value(ratio: CropAspectRatio) -> Value {
    match ratio {
        CropAspectRatio::Free => json!("free"),
        CropAspectRatio::Original => json!("original"),
        fixed => fixed
            .terms()
            .map_or(json!("free"), |(width, height)| json!([width, height])),
    }
}

/// The ratio `value` names: `original`, `free` or `width/height`.
fn aspect_named(value: &Value) -> Option<CropAspectRatio> {
    match value {
        Value::String(name) => match name.as_str() {
            "free" => Some(CropAspectRatio::Free),
            "original" => Some(CropAspectRatio::Original),
            _ => None,
        },
        Value::Array(terms) => {
            let [width, height] = match terms.as_slice() {
                [width, height] => [number(width)?, number(height)?],
                _ => return None,
            };
            let term = |value: f32| {
                (value.round() >= 1.0)
                    .then(|| value.round().min(f32::from(CropAspectRatio::MAX_TERM)) as u16)
            };
            let terms = (term(width)?, term(height)?);
            Some(
                NAMED_RATIOS
                    .into_iter()
                    .find(|ratio| ratio.terms() == Some(terms))
                    .unwrap_or(CropAspectRatio::Custom {
                        width: terms.0,
                        height: terms.1,
                    }),
            )
        }
        _ => None,
    }
}

/// The stored crop of `geometry` as fractions of the uncropped photo as
/// displayed: `[left, top, right, bottom]`.
fn displayed_crop(geometry: GeometryTransform, width: u32, height: u32) -> [f32; 4] {
    let mapping = GeometryMapping::new(geometry, width, height);
    let crop = mapping.workspace_crop(geometry.sanitized().crop);
    corners_box(
        mapping.canvas_from_workspace([crop[0], crop[1]]),
        mapping.canvas_from_workspace([crop[2], crop[3]]),
    )
}

/// The inverse of [`displayed_crop`].
fn stored_crop(geometry: GeometryTransform, width: u32, height: u32, shown: [f32; 4]) -> [f32; 4] {
    let mapping = GeometryMapping::new(geometry, width, height);
    mapping.stored_crop(corners_box(
        mapping.workspace_from_canvas([shown[0], shown[1]]),
        mapping.workspace_from_canvas([shown[2], shown[3]]),
    ))
}

/// The box with opposite corners `a` and `b`.
fn corners_box(a: [f32; 2], b: [f32; 2]) -> [f32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[0].max(b[0]),
        a[1].max(b[1]),
    ]
}

/// Every value of `geometry` the tool sets, as Claude writes them.
fn geometry_values(geometry: GeometryTransform, width: u32, height: u32) -> Map<String, Value> {
    let crop = displayed_crop(geometry, width, height).map(|value| rounded(value, 3));
    let [vertical, horizontal] = geometry.displayed_perspective();
    let degrees = |value: f32| json!(rounded(value, 1));
    [
        ("aspect", aspect_value(geometry.aspect_ratio)),
        ("crop", json!(crop)),
        ("straighten", degrees(geometry.rotation_degrees)),
        ("turns", json!(geometry.quarter_turns % 4)),
        ("flip_horizontal", json!(geometry.flip_horizontal)),
        ("flip_vertical", json!(geometry.flip_vertical)),
        ("vertical", degrees(vertical)),
        ("horizontal", degrees(horizontal)),
        ("horizontal_skew", degrees(geometry.horizontal_skew_degrees)),
        ("vertical_skew", degrees(geometry.vertical_skew_degrees)),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value))
    .collect()
}

impl CalibRawApp {
    /// Applies a `set_crop` call to the geometry and refits the crop inside
    /// the transformed photo, as the Crop panel does.
    pub(super) fn apply_set_crop(&mut self, arguments: &Value) -> Value {
        let Some((width, height)) = self.source_dimensions() else {
            return json!({ "error": "no photo is open" });
        };
        let Some(arguments) = arguments.as_object() else {
            return json!({ "error": "set_crop needs values" });
        };
        let before = self.develop.geometry;
        let was = geometry_values(before, width, height);
        let focal_length = self.perspective_focal_length();
        let mut outcome = Outcome::default();
        let geometry = &mut self.develop.geometry;
        // A quarter turn transposes the aspect ratio, so the turns come
        // first and an aspect written with them is the one displayed; the
        // arguments themselves are sorted by name.
        if let Some(value) = arguments.get("turns") {
            match number(value) {
                Some(turns) => {
                    let turns = (turns.round() as i32).rem_euclid(4) as u8;
                    while geometry.quarter_turns % 4 != turns {
                        geometry.rotate_quarter_turn(true);
                    }
                }
                None => outcome.not_set("turns"),
            }
        }
        for (name, value) in arguments {
            let degrees = |limit: f32| number(value).map(|value| value.clamp(-limit, limit));
            let applied = match name.as_str() {
                "aspect" => aspect_named(value).map(|ratio| {
                    geometry.aspect_ratio = ratio;
                    geometry.conform_crop_to_aspect(width, height);
                    self.develop_ui.crop_constraint_reference = Some(geometry.crop);
                }),
                "flip_horizontal" => value.as_bool().map(|flip| geometry.flip_horizontal = flip),
                "flip_vertical" => value.as_bool().map(|flip| geometry.flip_vertical = flip),
                "straighten" => degrees(GeometryTransform::MAX_ROTATION_DEGREES)
                    .map(|degrees| geometry.rotation_degrees = degrees),
                "vertical" | "horizontal" => {
                    degrees(PerspectiveCorrection::MAX_DEGREES).map(|degrees| {
                        let mut displayed = geometry.displayed_perspective();
                        displayed[usize::from(name == "horizontal")] = degrees;
                        // A new correction is made with this photo's lens; an
                        // existing one keeps its own.
                        if geometry.perspective.is_identity() {
                            geometry.perspective.focal_length_35mm = focal_length;
                        }
                        geometry.set_displayed_perspective(displayed);
                    })
                }
                "horizontal_skew" => degrees(GeometryTransform::MAX_SKEW_DEGREES)
                    .map(|degrees| geometry.horizontal_skew_degrees = degrees),
                "vertical_skew" => degrees(GeometryTransform::MAX_SKEW_DEGREES)
                    .map(|degrees| geometry.vertical_skew_degrees = degrees),
                // Set before, and last, inside the geometry the others make.
                "turns" | "crop" => Some(()),
                _ => None,
            };
            if applied.is_none() {
                outcome.not_set(name);
            }
        }

        self.fit_crop_after_geometry_edit(before, width, height);
        if let Some(requested) = arguments.get("crop") {
            let shown = requested
                .as_array()
                .and_then(|values| match values.as_slice() {
                    [left, top, right, bottom] => {
                        Some([number(left)?, number(top)?, number(right)?, number(bottom)?])
                    }
                    _ => None,
                });
            match shown {
                Some(shown) => {
                    let shown = corners_box(
                        [shown[0], shown[1]].map(|value| value.clamp(0.0, 1.0)),
                        [shown[2], shown[3]].map(|value| value.clamp(0.0, 1.0)),
                    );
                    let geometry = &mut self.develop.geometry;
                    geometry.crop = stored_crop(*geometry, width, height, shown);
                    *geometry = geometry.sanitized();
                    geometry.conform_crop_to_aspect(width, height);
                    geometry.fit_crop_inside_transformed_source(width, height);
                    self.develop_ui.crop_constraint_reference = Some(geometry.crop);
                }
                None => outcome.not_set("crop (write it left/top/right/bottom)"),
            }
        }
        if self.develop.geometry != before {
            self.note_geometry_changed();
        }

        let now = geometry_values(self.develop.geometry, width, height);
        for name in VALUES {
            let (Some(was), Some(now)) = (was.get(name), now.get(name)) else {
                continue;
            };
            // The crop is also reported when fitting it to the rest moved it.
            if arguments.contains_key(name) || (name == "crop" && was != now) {
                outcome.set(name, was.clone(), now.clone());
                let requested = arguments.get(name).and_then(number);
                let set = now.as_f64().map(|value| value as f32);
                if let (Some(requested), Some(set)) = (requested, set) {
                    if (requested - set).abs() > 0.05 {
                        outcome.limited(name);
                    }
                }
            }
        }
        outcome.into_response()
    }

    /// The geometry's values away from their defaults, and always the crop,
    /// for a message to Claude.
    pub(super) fn geometry_summary(&self) -> Option<String> {
        let (width, height) = self.source_dimensions()?;
        let defaults = geometry_values(GeometryTransform::default(), width, height);
        let values: Map<String, Value> = geometry_values(self.develop.geometry, width, height)
            .into_iter()
            .filter(|(name, value)| name == "crop" || defaults.get(name) != Some(value))
            .collect();
        Some(super::tools::describe(&values))
    }
}
