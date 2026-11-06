//! Claude's mask tools (Claude Code only, assets/claude/editor_system.md):
//! `set_mask` makes, edits or deletes a mask Claude names, a group of the
//! Masks panel, with its adjustments (the mask's own sliders, Color Mixer and
//! Color Grading) and its first part; `set_mask_part` adds, edits or deletes
//! the group's other parts, its sub-masks. Positions are fractions of the
//! preview Claude sees, x to the right and y down ([`PreviewPlacement`]).
//!
//! A part made by a model (sky, subject, background, person, object, depth)
//! starts its job or the dialog that asks to allow it; the request waits for
//! it before sending the next preview, so one such part is made per reply.

use super::color::{
    apply_grading, apply_mixer, changed_grading_values, changed_mixer_values, is_grading_value,
    is_mixer_value, Mixer,
};
use super::tools::{describe, number, pair, rounded, Limits, Outcome};
use super::*;
use crate::app::mask_tool::sample_color_range;
use crate::pipeline::effect_params::{adjustment, FloatParamSpec};
use crate::pipeline::{
    lightness_to_luminance, luminance_to_lightness, GeometryMapping, Homography, LensGeometryMap,
    LocalAdjustments, MaskCombineMode, MaskComponent, ObjectStroke, PersonPart, RawThumbnail,
    LUMINANCE_FEATHER_MAX, LUMINANCE_FEATHER_WIDTH, MAX_LOCAL_MASKS,
};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

/// The kinds of part Claude names, and the mask kind each makes. Brushes and
/// paths are left to the photographer.
const KINDS: [(&str, MaskKind); 11] = [
    ("full", MaskKind::Fullscreen),
    ("radial", MaskKind::Radial),
    ("linear", MaskKind::Linear),
    ("luminance", MaskKind::LuminanceRange),
    ("color", MaskKind::ColorRange),
    ("sky", MaskKind::Sky),
    ("subject", MaskKind::Subject),
    ("background", MaskKind::Background),
    ("person", MaskKind::Person),
    ("object", MaskKind::Object),
    ("depth", MaskKind::DepthRange),
];

const PERSON_PARTS: [(&str, PersonPart); 8] = [
    ("entire", PersonPart::EntirePerson),
    ("face", PersonPart::FaceNeck),
    ("skin", PersonPart::BodySkin),
    ("hair", PersonPart::Hair),
    ("clothes", PersonPart::Clothes),
    ("lips", PersonPart::Lips),
    ("teeth", PersonPart::Teeth),
    ("glasses", PersonPart::Eyeglasses),
];

const COMBINES: [(&str, MaskCombineMode); 3] = [
    ("add", MaskCombineMode::Add),
    ("subtract", MaskCombineMode::Subtract),
    ("intersect", MaskCombineMode::Intersect),
];

/// The values of a part's shape; `at` is read apart, as it samples the photo.
const SHAPE_VALUES: [&str; 9] = [
    "centre",
    "radius",
    "angle",
    "feather",
    "from",
    "to",
    "tolerance",
    "range",
    "parts",
];

/// A mask's sliders, by the names Claude writes.
const SLIDERS: [&str; 13] = [
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "temperature",
    "tint",
    "hue",
    "saturation",
    "texture",
    "clarity",
    "dehaze",
];

/// The masks whose selections one nudge sends at most, an image each.
pub(super) const MAX_SELECTION_IMAGES: usize = 4;

fn slider<'a>(
    adjustments: &'a mut LocalAdjustments,
    name: &str,
) -> Option<(&'a mut f32, FloatParamSpec)> {
    Some(match name {
        "exposure" => (&mut adjustments.exposure, adjustment::EXPOSURE),
        "contrast" => (&mut adjustments.contrast, adjustment::CONTRAST),
        "highlights" => (&mut adjustments.highlights, adjustment::HIGHLIGHTS),
        "shadows" => (&mut adjustments.shadows, adjustment::SHADOWS),
        "whites" => (&mut adjustments.whites, adjustment::WHITES),
        "blacks" => (&mut adjustments.blacks, adjustment::BLACKS),
        "temperature" => (&mut adjustments.temperature, adjustment::TEMPERATURE),
        "tint" => (&mut adjustments.tint, adjustment::TINT),
        "hue" => (&mut adjustments.hue, adjustment::HUE),
        "saturation" => (&mut adjustments.saturation, adjustment::SATURATION),
        "texture" => (&mut adjustments.texture, adjustment::TEXTURE),
        "clarity" => (&mut adjustments.clarity, adjustment::CLARITY),
        "dehaze" => (&mut adjustments.dehaze, adjustment::DEHAZE),
        _ => return None,
    })
}

fn kind_named(name: &str) -> Option<MaskKind> {
    KINDS
        .iter()
        .find(|(known, _)| *known == name)
        .map(|(_, kind)| *kind)
}

fn kind_name(kind: MaskKind) -> &'static str {
    KINDS
        .iter()
        .find(|(_, known)| *known == kind)
        .map_or("brush or path", |(name, _)| name)
}

fn combine_name(combine: MaskCombineMode) -> &'static str {
    COMBINES
        .iter()
        .find(|(_, known)| *known == combine)
        .map_or("add", |(name, _)| name)
}

/// Whether making a part of `kind` runs a model.
const fn runs_model(kind: MaskKind) -> bool {
    matches!(
        kind,
        MaskKind::Sky
            | MaskKind::Subject
            | MaskKind::Background
            | MaskKind::DepthRange
            | MaskKind::Person
            | MaskKind::Object
    )
}

/// A mask's name as Claude writes it, which cannot hold a space.
fn written_name(name: &str) -> String {
    name.replace(' ', "_")
}

/// The index of the mask Claude calls `name`.
pub(super) fn mask_named(stack: &MaskStack, name: &str) -> Option<usize> {
    stack
        .masks
        .iter()
        .position(|mask| mask.name == name || written_name(&mask.name) == name)
}

fn text(value: &Value) -> Option<&str> {
    value
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

fn flag(value: &Value) -> Option<bool> {
    value.as_bool()
}

fn point_value(point: [f32; 2]) -> Value {
    json!([rounded(point[0], 3), rounded(point[1], 3)])
}

fn error(message: impl Into<String>) -> Value {
    json!({ "error": message.into() })
}

/// Where a fraction of Claude's preview lies in the photo. The preview is the
/// edit's output, cropped and transformed by the geometry
/// ([`GeometryMapping`]), so a position in it maps back through the geometry
/// to lens-corrected source coordinates, where radial and linear masks live,
/// and through the lens correction to the native ones the other masks use.
pub(super) struct PreviewPlacement {
    output_from_source: Homography,
    source_from_output: Homography,
    /// The output's width and height in pixels.
    output_size: [f32; 2],
    /// The source's width and height in pixels.
    source_size: [u32; 2],
    lens: Option<Arc<LensGeometryMap>>,
}

impl PreviewPlacement {
    pub(super) fn new(
        geometry: GeometryTransform,
        lens: Option<Arc<LensGeometryMap>>,
        source_width: u32,
        source_height: u32,
    ) -> Self {
        let mapping = GeometryMapping::new(geometry, source_width, source_height);
        let output_from_source = mapping.output_uv_from_source_uv();
        Self {
            output_from_source,
            source_from_output: output_from_source.inverse(),
            output_size: mapping.output_size().map(|size| size.max(1.0)),
            source_size: [source_width.max(1), source_height.max(1)],
            lens,
        }
    }

    /// The lens-corrected source position of the preview position `preview`.
    fn corrected(&self, preview: [f32; 2]) -> [f32; 2] {
        self.source_from_output.apply(preview)
    }

    /// The preview position of the lens-corrected source position `corrected`.
    fn preview(&self, corrected: [f32; 2]) -> [f32; 2] {
        self.output_from_source.apply(corrected)
    }

    /// The native source position of the preview position `preview`, as the
    /// preview's pointer tools map it (`corrected_uv_to_native_source`).
    fn native(&self, preview: [f32; 2]) -> [f32; 2] {
        let corrected = self.corrected(preview);
        let [width, height] = self.source_size;
        self.lens.as_ref().map_or(corrected, |lens| {
            lens.source_uv_for_raster(corrected, width, height)
        })
    }

    fn source_pixels(&self) -> [f32; 2] {
        self.source_size.map(|size| size as f32)
    }

    /// A radial mask's `center`, `radius` and `rotation` as stored (lens-corrected
    /// source fractions and radians, `MaskGeometry::Radial`) for an ellipse of
    /// the preview: its centre, its half-axes along `angle_degrees` (clockwise
    /// from the x axis) as a fraction of the preview's width and across it as a
    /// fraction of its height.
    fn radial_from_preview(
        &self,
        centre: [f32; 2],
        radius: [f32; 2],
        angle_degrees: f32,
    ) -> ([f32; 2], [f32; 2], f32) {
        let [width, height] = self.output_size;
        let (sin, cos) = angle_degrees.to_radians().sin_cos();
        let major = [
            centre[0] + cos * radius[0],
            centre[1] + sin * radius[0] * width / height,
        ];
        let minor = [
            centre[0] - sin * radius[1] * height / width,
            centre[1] + cos * radius[1],
        ];
        let center = self.corrected(centre);
        let [source_width, source_height] = self.source_pixels();
        let offset = |point: [f32; 2]| {
            let point = self.corrected(point);
            [
                (point[0] - center[0]) * source_width,
                (point[1] - center[1]) * source_height,
            ]
        };
        let [major, minor] = [offset(major), offset(minor)];
        (
            center,
            [
                major[0].hypot(major[1]) / source_width,
                minor[0].hypot(minor[1]) / source_height,
            ],
            major[1].atan2(major[0]),
        )
    }

    /// The inverse of [`Self::radial_from_preview`].
    fn radial_to_preview(
        &self,
        center: [f32; 2],
        radius: [f32; 2],
        rotation: f32,
    ) -> ([f32; 2], [f32; 2], f32) {
        let [source_width, source_height] = self.source_pixels();
        let (sin, cos) = rotation.sin_cos();
        let major = [
            center[0] + cos * radius[0],
            center[1] + sin * radius[0] * source_width / source_height,
        ];
        let minor = [
            center[0] - sin * radius[1] * source_height / source_width,
            center[1] + cos * radius[1],
        ];
        let centre = self.preview(center);
        let [width, height] = self.output_size;
        let offset = |point: [f32; 2]| {
            let point = self.preview(point);
            [
                (point[0] - centre[0]) * width,
                (point[1] - centre[1]) * height,
            ]
        };
        let [major, minor] = [offset(major), offset(minor)];
        (
            centre,
            [
                major[0].hypot(major[1]) / width,
                minor[0].hypot(minor[1]) / height,
            ],
            major[1].atan2(major[0]).to_degrees(),
        )
    }
}

/// The shape values of `component` as Claude writes them.
fn shape_values(component: &MaskComponent, placement: &PreviewPlacement) -> Map<String, Value> {
    let mut values = Map::new();
    let mut insert = |name: &str, value: Value| {
        values.insert(name.to_owned(), value);
    };
    match &component.geometry {
        MaskGeometry::Radial {
            center,
            radius,
            rotation,
            feather,
            ..
        } => {
            let (centre, radius, angle) = placement.radial_to_preview(*center, *radius, *rotation);
            insert("centre", point_value(centre));
            insert("radius", point_value(radius));
            insert("angle", json!(rounded(angle, 0)));
            insert("feather", json!(rounded(feather * 100.0, 0)));
        }
        MaskGeometry::Linear {
            start,
            end,
            feather,
            ..
        } => {
            insert("from", point_value(placement.preview(*start)));
            insert("to", point_value(placement.preview(*end)));
            insert("feather", json!(rounded(feather * 100.0, 0)));
        }
        MaskGeometry::LuminanceRange {
            low, high, feather, ..
        } => {
            insert(
                "range",
                json!([
                    rounded(luminance_to_lightness(*low) * 100.0, 0),
                    rounded(luminance_to_lightness(*high) * 100.0, 0)
                ]),
            );
            insert(
                "feather",
                json!(rounded(feather * LUMINANCE_FEATHER_WIDTH * 100.0, 0)),
            );
        }
        MaskGeometry::ColorRange { tolerance, .. } => {
            insert("tolerance", json!(rounded(tolerance * 100.0, 0)));
        }
        MaskGeometry::DepthRange { range, .. } => {
            insert(
                "range",
                json!([
                    rounded(range.near * 100.0, 0),
                    rounded(range.far * 100.0, 0)
                ]),
            );
            insert("feather", json!(rounded(range.near_feather * 100.0, 0)));
        }
        MaskGeometry::Person { parts, .. } => {
            let names: Vec<&str> = PERSON_PARTS
                .iter()
                .filter(|(_, part)| parts.contains(part))
                .map(|(name, _)| *name)
                .collect();
            insert("parts", json!(names));
        }
        _ => {}
    }
    values
}

/// Sets the shape values `values` names on `component`, noting them in
/// `outcome`. Returns whether the shape changed.
fn apply_shape(
    component: &mut MaskComponent,
    values: &Map<String, Value>,
    placement: &PreviewPlacement,
    outcome: &mut Outcome,
) -> bool {
    let before = shape_values(component, placement);
    let percent = |name: &str| {
        values
            .get(name)
            .and_then(number)
            .map(|value| value.clamp(0.0, 100.0) / 100.0)
    };
    let range = || values.get("range").and_then(pair);
    match &mut component.geometry {
        MaskGeometry::Radial {
            center,
            radius,
            rotation,
            feather,
            initialized,
        } => {
            if ["centre", "radius", "angle"]
                .iter()
                .any(|name| values.contains_key(*name))
            {
                let (mut centre, mut half_axes, mut angle) =
                    placement.radial_to_preview(*center, *radius, *rotation);
                if let Some(point) = values.get("centre").and_then(pair) {
                    centre = point.map(|value| value.clamp(-1.0, 2.0));
                }
                if let Some(size) = values.get("radius").and_then(pair) {
                    half_axes = size.map(|value| value.clamp(0.01, 2.0));
                }
                if let Some(degrees) = values.get("angle").and_then(number) {
                    angle = degrees;
                }
                (*center, *radius, *rotation) =
                    placement.radial_from_preview(centre, half_axes, angle);
            }
            if let Some(value) = percent("feather") {
                *feather = value;
            }
            *initialized = true;
        }
        MaskGeometry::Linear {
            start,
            end,
            feather,
            initialized,
        } => {
            for (name, point) in [("from", start), ("to", end)] {
                if let Some(preview) = values.get(name).and_then(pair) {
                    *point = placement.corrected(preview.map(|value| value.clamp(-1.0, 2.0)));
                }
            }
            if let Some(value) = percent("feather") {
                *feather = value.max(0.02);
            }
            *initialized = true;
        }
        MaskGeometry::LuminanceRange {
            low,
            high,
            feather,
            high_feather,
            ..
        } => {
            if let Some([from, to]) = range() {
                let [from, to] = [from.min(to), from.max(to)]
                    .map(|lightness| lightness_to_luminance(lightness.clamp(0.0, 100.0) / 100.0));
                (*low, *high) = (from, to);
            }
            if let Some(value) = percent("feather") {
                *feather = (value / LUMINANCE_FEATHER_WIDTH).min(LUMINANCE_FEATHER_MAX);
                *high_feather = Some(*feather);
            }
        }
        MaskGeometry::ColorRange { tolerance, .. } => {
            if let Some(value) = percent("tolerance") {
                *tolerance = value.max(0.005);
            }
        }
        MaskGeometry::DepthRange { range: depth, .. } => {
            if let Some([near, far]) = range() {
                let [near, far] =
                    [near.min(far), near.max(far)].map(|value| value.clamp(0.0, 100.0) / 100.0);
                (depth.near, depth.far) = (near, far);
            }
            if let Some(value) = percent("feather") {
                (depth.near_feather, depth.far_feather) = (value, value);
            }
        }
        MaskGeometry::Person { parts, .. } => {
            if let Some(requested) = values.get("parts") {
                let names: Vec<&str> = match requested {
                    Value::Array(names) => names.iter().filter_map(text).collect(),
                    other => text(other).into_iter().collect(),
                };
                let chosen: BTreeSet<PersonPart> = PERSON_PARTS
                    .iter()
                    .filter(|(name, _)| names.contains(name))
                    .map(|(_, part)| *part)
                    .collect();
                // The entire person covers the other parts.
                if chosen.contains(&PersonPart::EntirePerson) {
                    *parts = BTreeSet::from([PersonPart::EntirePerson]);
                } else if !chosen.is_empty() {
                    *parts = chosen;
                }
            }
        }
        _ => {}
    }
    let after = shape_values(component, placement);
    let mut changed = false;
    for (name, value) in values {
        // A value written in another form sets nothing above.
        let unread = match name.as_str() {
            "centre" | "radius" | "from" | "to" | "range" => pair(value)
                .is_none()
                .then_some("write two numbers, such as 0.5/0.4"),
            "angle" | "feather" | "tolerance" => {
                number(value).is_none().then_some("write a number")
            }
            _ => None,
        };
        if let Some(hint) = unread {
            outcome.not_set(&format!("{name} ({hint})"));
            continue;
        }
        let Some(now) = after.get(name) else {
            outcome.not_set(&format!(
                "{name} (not a value of a {} part)",
                kind_name(component.kind)
            ));
            continue;
        };
        match before.get(name) {
            Some(was) => {
                changed |= was != now;
                outcome.set(name, was.clone(), now.clone());
            }
            None => {
                changed = true;
                outcome.now(name, now.clone());
            }
        }
    }
    changed
}

/// Sets the sliders, mixer and grading values among `arguments` on
/// `adjustments`, noting them in `outcome`. Returns the names it took.
fn apply_adjustments(
    adjustments: &mut LocalAdjustments,
    arguments: &Map<String, Value>,
    outcome: &mut Outcome,
) -> Vec<String> {
    let mut taken = Vec::new();
    for (name, requested) in arguments {
        let Some((field, spec)) = slider(adjustments, name) else {
            continue;
        };
        taken.push(name.clone());
        let Some(requested) = number(requested) else {
            outcome.not_set(name);
            continue;
        };
        let decimals = spec.decimals as u32;
        let limits = Limits::new(spec.min, spec.max, decimals);
        let was = rounded(*field, decimals);
        let value = limits.limit(requested);
        *field = value;
        outcome.set_number(name, requested, was, value, decimals);
    }
    let subset = |owns: fn(&str) -> bool| -> Map<String, Value> {
        arguments
            .iter()
            .filter(|(name, _)| owns(name))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    };
    let mixer_values = subset(is_mixer_value);
    if !mixer_values.is_empty() {
        let mut mixer: Mixer = [
            adjustments.hsl_hue,
            adjustments.hsl_saturation,
            adjustments.hsl_luminance,
        ];
        let baseline = mixer;
        apply_mixer(
            &mut mixer,
            baseline,
            &Value::Object(mixer_values.clone()),
            outcome,
        );
        [
            adjustments.hsl_hue,
            adjustments.hsl_saturation,
            adjustments.hsl_luminance,
        ] = mixer;
        taken.extend(mixer_values.into_iter().map(|(name, _)| name));
    }
    let grading_values = subset(is_grading_value);
    if !grading_values.is_empty() {
        let baseline = adjustments.color_grading;
        apply_grading(
            &mut adjustments.color_grading,
            baseline,
            &Value::Object(grading_values.clone()),
            outcome,
        );
        taken.extend(grading_values.into_iter().map(|(name, _)| name));
    }
    taken
}

/// The adjustments of a mask away from their defaults, as Claude writes them.
fn changed_adjustments(adjustments: &LocalAdjustments) -> Map<String, Value> {
    let mut adjustments = *adjustments;
    let mut values = Map::new();
    for name in SLIDERS {
        let Some((field, spec)) = slider(&mut adjustments, name) else {
            continue;
        };
        let value = rounded(*field, spec.decimals as u32);
        if value != rounded(spec.default, spec.decimals as u32) {
            values.insert(name.to_owned(), json!(value));
        }
    }
    values.extend(changed_mixer_values([
        adjustments.hsl_hue,
        adjustments.hsl_saturation,
        adjustments.hsl_luminance,
    ]));
    values.extend(changed_grading_values(adjustments.color_grading));
    values
}

/// `response` with `status` added.
fn with_status(mut response: Value, status: Option<String>) -> Value {
    if let (Some(status), Value::Object(response)) = (status, &mut response) {
        response.insert("status".to_owned(), json!(status));
    }
    response
}

/// `preview` (before the geometry, as read back) with the coverage of mask
/// `layer` tinted red and the rest darkened, so Claude can check what the
/// mask selects.
pub(super) fn selection_image(
    stack: &MaskStack,
    layer: usize,
    preview: &RawThumbnail,
    lens: Option<&LensGeometryMap>,
) -> RawThumbnail {
    let size = [preview.width, preview.height];
    let coverage = stack.rasterize_layer_region(layer, size, [0, 0, size[0], size[1]], size, lens);
    let mut rgba = preview.rgba.clone();
    for (pixel, alpha) in rgba.chunks_exact_mut(4).zip(coverage) {
        let selected = f32::from(alpha) / 255.0;
        for (channel, red) in pixel.iter_mut().zip([230.0, 40.0, 40.0]) {
            let base = f32::from(*channel);
            let value = base * 0.4 * (1.0 - selected) + (base * 0.45 + red * 0.55) * selected;
            *channel = value.round().clamp(0.0, 255.0) as u8;
        }
    }
    RawThumbnail {
        width: preview.width,
        height: preview.height,
        rgba,
    }
}

impl CalibRawApp {
    /// The error for making a part of `kind` now, when that runs a model and
    /// another model or the dialog to allow one is running.
    fn model_busy(&self, kind: MaskKind) -> Option<Value> {
        (runs_model(kind) && (self.foreground_operation_active() || self.ai.consent.is_some()))
            .then(|| {
                error("another selection model is running; make this part in your next tool lines")
            })
    }

    /// What became of the model a part needs, for the response.
    fn model_status(&self, location: (usize, usize)) -> Option<String> {
        if self.ai.consent.is_some() {
            Some(
                "waiting for the photographer to allow the selection model; the app sends the preview once it is decided"
                    .to_owned(),
            )
        } else if self.foreground_operation_active() || !self.ai.people.composes.is_empty() {
            Some(
                "the model is making the selection; the app sends the preview once it is done"
                    .to_owned(),
            )
        } else if self
            .mask_part(location)
            .is_some_and(|component| !component.geometry.is_initialized())
        {
            Some("the selection is empty: the model could not run".to_owned())
        } else {
            None
        }
    }

    fn mask_part(&self, (mask_index, component_index): (usize, usize)) -> Option<&MaskComponent> {
        self.masks
            .stack
            .masks
            .get(mask_index)?
            .components
            .get(component_index)
    }

    fn mask_part_mut(
        &mut self,
        (mask_index, component_index): (usize, usize),
    ) -> Option<&mut MaskComponent> {
        self.masks
            .stack
            .masks
            .get_mut(mask_index)?
            .components
            .get_mut(component_index)
    }

    /// Selects `previous` again after a tool selected the mask it edited, or
    /// what is left when it is gone.
    fn restore_mask_selection(&mut self, previous: (Option<usize>, Option<usize>)) {
        let stack = &mut self.masks.stack;
        let restored = match previous {
            (Some(mask_index), Some(component_index)) => {
                stack.select_component(mask_index, component_index)
            }
            (Some(mask_index), None) => stack.select_mask(mask_index),
            (None, _) => {
                stack.selected_mask = None;
                stack.selected_component = None;
                true
            }
        };
        if !restored {
            stack.ensure_selection();
        }
        self.masks.thumbnail_component_mask = None;
    }

    /// Applies the shape values and `at` of a call to the part at `location`,
    /// and makes its selection when it is `new` or what it selects changed.
    /// Returns the part's status for the response.
    fn apply_part(
        &mut self,
        location: (usize, usize),
        arguments: &Map<String, Value>,
        new: bool,
        placement: &PreviewPlacement,
        outcome: &mut Outcome,
        frame: &eframe::Frame,
    ) -> Option<String> {
        let shape: Map<String, Value> = arguments
            .iter()
            .filter(|(name, _)| SHAPE_VALUES.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let at = match arguments.get("at") {
            Some(value) => {
                let point = pair(value);
                if point.is_none() {
                    outcome.not_set("at (write it x/y)");
                }
                point
            }
            None => None,
        };
        let component = self.mask_part_mut(location)?;
        let kind = component.kind;
        let changed = apply_shape(component, &shape, placement, outcome);
        let (mask_index, component_index) = location;
        let mut status = None;
        match kind {
            MaskKind::Sky | MaskKind::Subject | MaskKind::Background | MaskKind::DepthRange
                if new =>
            {
                match kind {
                    MaskKind::Sky => self.request_sky_mask(frame),
                    MaskKind::DepthRange => self.request_depth_mask(frame),
                    _ => self.request_subject_mask(frame),
                }
                status = self.model_status(location);
            }
            MaskKind::Person if new || (changed && shape.contains_key("parts")) => {
                if !new && self.model_busy(kind).is_some() {
                    outcome.not_set("parts (another selection model is running)");
                } else {
                    self.request_people_masks(vec![location], frame);
                    status = self.model_status(location);
                }
            }
            MaskKind::Object => match at {
                Some(_) if self.model_busy(kind).is_some() => {
                    outcome.not_set("at (another selection model is running)");
                }
                Some(preview) => {
                    if let Err(message) = self.capture_mask_source(frame) {
                        return Some(format!("the selection is empty: {message}"));
                    }
                    let point = placement.native(preview);
                    if let Some(MaskGeometry::Object { mask, strokes, .. }) =
                        self.mask_part_mut(location).map(|part| &mut part.geometry)
                    {
                        *mask = None;
                        *strokes = vec![ObjectStroke {
                            points: vec![point],
                            positive: true,
                            brush_size: 0.0,
                        }];
                    }
                    outcome.now("at", point_value(preview));
                    self.request_object_mask(mask_index, component_index);
                    status = self.model_status(location);
                }
                None if new => {
                    status = Some("the selection is empty: give at=x/y on the object".to_owned());
                }
                None => {}
            },
            MaskKind::LuminanceRange | MaskKind::ColorRange if new || at.is_some() => {
                if let Err(message) = self.capture_mask_source(frame) {
                    return Some(format!("the selection is empty: {message}"));
                }
                let source = self.masks.source_cache.clone();
                let point = at.map(|preview| placement.native(preview));
                if let Some(component) = self.mask_part_mut(location) {
                    match &mut component.geometry {
                        MaskGeometry::LuminanceRange { source: target, .. } => *target = source,
                        geometry @ MaskGeometry::ColorRange { .. } => {
                            if let MaskGeometry::ColorRange { source: target, .. } = &mut *geometry
                            {
                                *target = source;
                            }
                            if let Some(point) = point {
                                sample_color_range(geometry, point);
                            }
                        }
                        _ => {}
                    }
                }
                match (kind, at) {
                    (MaskKind::ColorRange, Some(preview)) => {
                        outcome.now("at", point_value(preview))
                    }
                    (MaskKind::ColorRange, None) => {
                        status =
                            Some("the selection is empty: give at=x/y on the colour".to_owned());
                    }
                    (_, Some(_)) => outcome.not_set("at (not a value of a luminance part)"),
                    _ => {}
                }
            }
            _ if at.is_some() => {
                outcome.not_set(&format!("at (not a value of a {} part)", kind_name(kind)));
            }
            _ => {}
        }
        if new || changed || at.is_some() {
            self.mark_mask_geometry_dirty(mask_index);
        }
        status
    }

    /// Applies a `set_mask` call: makes the mask `name` when there is none,
    /// with a part of `kind`, or replaces its first part when `kind` differs,
    /// sets its first part's shape, its opacity and inversion and its
    /// adjustments, or deletes it.
    pub(super) fn apply_set_mask(
        &mut self,
        arguments: &Value,
        placement: &PreviewPlacement,
        frame: &eframe::Frame,
    ) -> Value {
        let Some(arguments) = arguments.as_object() else {
            return error("set_mask needs name=");
        };
        let Some(name) = arguments.get("name").and_then(text) else {
            return error("set_mask needs name=, the mask's name");
        };
        let existing = mask_named(&self.masks.stack, name);
        let kind = match arguments.get("kind") {
            Some(value) => match text(value).and_then(kind_named) {
                Some(kind) => Some(kind),
                None => return error(kind_error()),
            },
            None => None,
        };
        if arguments.get("delete").and_then(flag) == Some(true) {
            let Some(mask_index) = existing else {
                return error(format!("there is no mask named {name}"));
            };
            let previous = (
                self.masks.stack.selected_mask,
                self.masks.stack.selected_component,
            );
            self.masks.stack.delete_mask(mask_index);
            self.restore_mask_selection(previous);
            self.mark_all_mask_layers_dirty();
            return json!({ "now": { "name": name }, "status": "deleted" });
        }

        let previous = (
            self.masks.stack.selected_mask,
            self.masks.stack.selected_component,
        );
        let mut outcome = Outcome::default();
        // Names the mask in the result line, as a reply may set several.
        outcome.now("name", json!(name));
        let (mask_index, new) = match (existing, kind) {
            (Some(mask_index), Some(kind))
                if self
                    .mask_part((mask_index, 0))
                    .is_some_and(|part| part.kind != kind) =>
            {
                if let Some(busy) = self.model_busy(kind) {
                    return busy;
                }
                let was = self
                    .mask_part((mask_index, 0))
                    .map(|part| kind_name(part.kind));
                if let Some(part) = self.mask_part_mut((mask_index, 0)) {
                    replace_kind(part, kind);
                }
                outcome.set("kind", json!(was), json!(kind_name(kind)));
                (mask_index, true)
            }
            (Some(mask_index), _) => (mask_index, false),
            (None, Some(kind)) => {
                if let Some(busy) = self.model_busy(kind) {
                    return busy;
                }
                let Some((mask_index, _)) = self.masks.stack.add_mask(kind) else {
                    return error(format!(
                        "the photo holds {MAX_LOCAL_MASKS} masks at most; delete one first"
                    ));
                };
                self.masks.stack.masks[mask_index].common.name = name.to_owned();
                outcome.now("kind", json!(kind_name(kind)));
                (mask_index, true)
            }
            (None, None) => {
                return error(format!(
                    "there is no mask named {name}; give it a kind= to make it"
                ));
            }
        };
        let status = self.apply_part(
            (mask_index, 0),
            arguments,
            new,
            placement,
            &mut outcome,
            frame,
        );

        let mut taken: Vec<String> = ["name", "kind", "delete", "at"]
            .into_iter()
            .chain(SHAPE_VALUES)
            .map(str::to_owned)
            .collect();
        let mask = &mut self.masks.stack.masks[mask_index];
        let mut changed = false;
        if let Some(requested) = arguments.get("opacity") {
            taken.push("opacity".to_owned());
            match number(requested) {
                Some(requested) => {
                    let limits = Limits::new(0.0, 100.0, 0);
                    let was = rounded(mask.opacity * 100.0, 0);
                    let value = limits.limit(requested);
                    changed |= mask.set_opacity(value / 100.0);
                    outcome.set_number("opacity", requested, was, value, 0);
                }
                None => outcome.not_set("opacity"),
            }
        }
        if let Some(requested) = arguments.get("invert") {
            taken.push("invert".to_owned());
            match flag(requested) {
                Some(invert) => {
                    outcome.set("invert", json!(mask.invert), json!(invert));
                    changed |= mask.invert != invert;
                    mask.common.invert = invert;
                }
                None => outcome.not_set("invert (true or false)"),
            }
        }
        let adjustments_before = mask.adjustments;
        taken.extend(apply_adjustments(
            &mut mask.adjustments,
            arguments,
            &mut outcome,
        ));
        let adjusted = mask.adjustments != adjustments_before;
        for name in arguments.keys().filter(|name| !taken.contains(name)) {
            outcome.not_set(name);
        }

        self.restore_mask_selection(previous);
        if changed {
            self.mark_mask_geometry_dirty(mask_index);
        } else if adjusted {
            self.mark_mask_adjustments_dirty();
        }
        with_status(outcome.into_response(), status)
    }

    /// Applies a `set_mask_part` call: adds a part of `kind` to the mask
    /// `name`, combined by `combine`, or with `part` edits or deletes that
    /// part.
    pub(super) fn apply_set_mask_part(
        &mut self,
        arguments: &Value,
        placement: &PreviewPlacement,
        frame: &eframe::Frame,
    ) -> Value {
        let Some(arguments) = arguments.as_object() else {
            return error("set_mask_part needs name=");
        };
        let Some(name) = arguments.get("name").and_then(text) else {
            return error("set_mask_part needs name=, the mask's name");
        };
        let Some(mask_index) = mask_named(&self.masks.stack, name) else {
            return error(format!(
                "there is no mask named {name}; make it with set_mask first"
            ));
        };
        let kind = match arguments.get("kind") {
            Some(value) => match text(value).and_then(kind_named) {
                Some(kind) => Some(kind),
                None => return error(kind_error()),
            },
            None => None,
        };
        let combine = match arguments.get("combine") {
            Some(value) => match text(value).and_then(|written| {
                COMBINES
                    .iter()
                    .find(|(name, _)| *name == written)
                    .map(|(_, combine)| *combine)
            }) {
                Some(combine) => Some(combine),
                None => return error("combine must be add, subtract or intersect"),
            },
            None => None,
        };
        let part_count = self.masks.stack.masks[mask_index].components.len();
        let part = match arguments.get("part") {
            Some(value) => match number(value).map(|part| part.round() as usize) {
                Some(part @ 1..) if part <= part_count => Some(part - 1),
                _ => return error(format!("{name} has parts 1 to {part_count}")),
            },
            None => None,
        };

        let previous = (
            self.masks.stack.selected_mask,
            self.masks.stack.selected_component,
        );
        if arguments.get("delete").and_then(flag) == Some(true) {
            let Some(component_index) = part else {
                return error("name the part to delete with part=");
            };
            if !self
                .masks
                .stack
                .delete_component(mask_index, component_index)
            {
                return error(
                    "a mask keeps at least one part; delete the mask with set_mask instead",
                );
            }
            self.restore_mask_selection(previous);
            self.mark_mask_geometry_dirty(mask_index);
            return json!({ "now": { "name": name, "part": component_index + 1 }, "status": "deleted" });
        }

        let mut outcome = Outcome::default();
        outcome.now("name", json!(name));
        if let Some(component_index) = part {
            outcome.now("part", json!(component_index + 1));
        }
        let (component_index, new) = match (part, kind) {
            (Some(component_index), Some(kind))
                if self
                    .mask_part((mask_index, component_index))
                    .is_some_and(|part| part.kind != kind) =>
            {
                if let Some(busy) = self.model_busy(kind) {
                    return busy;
                }
                let location = (mask_index, component_index);
                let was = self.mask_part(location).map(|part| kind_name(part.kind));
                if let Some(part) = self.mask_part_mut(location) {
                    replace_kind(part, kind);
                }
                outcome.set("kind", json!(was), json!(kind_name(kind)));
                (component_index, true)
            }
            (Some(component_index), _) => (component_index, false),
            (None, Some(kind)) => {
                if let Some(busy) = self.model_busy(kind) {
                    return busy;
                }
                self.masks.stack.select_mask(mask_index);
                let combine = combine.unwrap_or(MaskCombineMode::Add);
                let Some((_, component_index)) = self.masks.stack.add_component(kind, combine)
                else {
                    self.restore_mask_selection(previous);
                    return error(format!("{name} holds the most parts it can"));
                };
                outcome.now("part", json!(component_index + 1));
                outcome.now("kind", json!(kind_name(kind)));
                outcome.now("combine", json!(combine_name(combine)));
                (component_index, true)
            }
            (None, None) => {
                return error("give kind= to add a part, or part= to edit one");
            }
        };
        let location = (mask_index, component_index);
        let status = self.apply_part(location, arguments, new, placement, &mut outcome, frame);

        let taken: Vec<&str> = ["name", "kind", "delete", "at", "part", "combine", "invert"]
            .into_iter()
            .chain(SHAPE_VALUES)
            .collect();
        let mut changed = false;
        if let Some(component) = self.mask_part_mut(location) {
            if let (Some(combine), false) = (combine, new && part.is_none()) {
                if component_index == 0 {
                    outcome.not_set("combine (the first part always adds)");
                } else {
                    outcome.set(
                        "combine",
                        json!(combine_name(component.combine)),
                        json!(combine_name(combine)),
                    );
                    changed |= component.set_combine(combine);
                }
            }
            if let Some(requested) = arguments.get("invert") {
                match flag(requested) {
                    Some(invert) => {
                        outcome.set("invert", json!(component.invert), json!(invert));
                        changed |= component.invert != invert;
                        component.common.invert = invert;
                    }
                    None => outcome.not_set("invert (true or false)"),
                }
            }
        }
        for name in arguments
            .keys()
            .filter(|name| !taken.contains(&name.as_str()))
        {
            outcome.not_set(name);
        }

        self.restore_mask_selection(previous);
        if changed {
            self.mark_mask_geometry_dirty(mask_index);
        }
        with_status(outcome.into_response(), status)
    }

    /// The masks, their parts and adjustments, for a message to Claude, or
    /// `None` without masks.
    pub(super) fn mask_summary(&self, placement: &PreviewPlacement) -> Option<String> {
        let masks = &self.masks.stack.masks;
        if masks.is_empty() {
            return None;
        }
        let described = masks
            .iter()
            .map(|mask| {
                let mut details: Vec<String> = mask
                    .components
                    .iter()
                    .enumerate()
                    .map(|(index, component)| {
                        let mut part = format!("part {}", index + 1);
                        if index > 0 {
                            part.push(' ');
                            part.push_str(combine_name(component.combine));
                        }
                        part.push(' ');
                        part.push_str(kind_name(component.kind));
                        let shape = shape_values(component, placement);
                        if !shape.is_empty() {
                            part.push(' ');
                            part.push_str(&describe(&shape));
                        }
                        if component.invert {
                            part.push_str(", inverted");
                        }
                        if !component.enabled {
                            part.push_str(", off");
                        } else if !component.geometry.is_initialized() {
                            part.push_str(", selection empty");
                        }
                        part
                    })
                    .collect();
                if mask.opacity < 1.0 {
                    details.push(format!("opacity {}", rounded(mask.opacity * 100.0, 0)));
                }
                if mask.invert {
                    details.push("inverted".to_owned());
                }
                if !mask.enabled {
                    details.push("off".to_owned());
                }
                let adjustments = changed_adjustments(&mask.adjustments);
                if !adjustments.is_empty() {
                    details.push(describe(&adjustments));
                }
                format!("{} ({})", written_name(&mask.name), details.join("; "))
            })
            .collect::<Vec<_>>()
            .join(", ");
        Some(described)
    }
}

/// `component` as a new part of `kind`, keeping how it combines and whether
/// it is inverted.
fn replace_kind(component: &mut MaskComponent, kind: MaskKind) {
    let invert = component.invert;
    *component = MaskComponent::new(kind, component.combine);
    component.common.invert = invert;
}

fn kind_error() -> String {
    let names: Vec<&str> = KINDS.iter().map(|(name, _)| *name).collect();
    format!("kind must be one of {}", names.join(", "))
}
