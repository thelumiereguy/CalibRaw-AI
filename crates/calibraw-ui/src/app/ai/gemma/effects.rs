//! The finish effects Gemma's `set_effect` tool moves: the sliders of the
//! global Vignette, Grain, Glow and Halation, named `effect_setting` after
//! each slider's label (Preserve highlights is `vignette_preserve_highlights`).
//! editor_system.md lists only each effect's amount; Gemma left the rest at
//! odd values, such as a vignette with no feather. The rest stay settable.
//! Colours, positions and pattern seeds are left to the photographer.

use super::tools::{number, rounded, Limits, Outcome};
use super::*;
use crate::pipeline::effect_params::FloatParamSpec;
use crate::pipeline::{
    EffectComponent, EffectControl, EffectControls, MaskEffect, MaskEffectSettings,
};
use serde_json::{json, Map, Value};

/// The effects the tool reaches, by the name their values start with.
const EFFECTS: [(MaskEffect, &str); 4] = [
    (MaskEffect::Vignette, "vignette"),
    (MaskEffect::Grain, "grain"),
    (MaskEffect::Glow, "glow"),
    (MaskEffect::Halation, "halation"),
];

/// The tool's name for a slider labelled `label`.
fn setting_name(label: &str) -> String {
    label.to_lowercase().replace(' ', "_")
}

/// The slider `setting` of `controls`.
fn slider<'a>(
    controls: &'a mut EffectControls<'_>,
    setting: &str,
) -> Option<(&'a mut f32, FloatParamSpec)> {
    controls
        .shown
        .iter_mut()
        .chain(
            controls
                .details
                .iter_mut()
                .flat_map(|details| details.controls.iter_mut()),
        )
        .find_map(|control| match control {
            EffectControl::Slider(value, spec) if setting_name(spec.label) == setting => {
                Some((&mut **value, *spec))
            }
            _ => None,
        })
}

fn controls(settings: &mut MaskEffectSettings, effect: MaskEffect) -> Option<EffectControls<'_>> {
    settings
        .all_controls()
        .find(|(controls_effect, _)| *controls_effect == effect)
        .map(|(_, controls)| controls)
}

fn slider_value(settings: &MaskEffectSettings, effect: MaskEffect, setting: &str) -> Option<f32> {
    let mut settings = *settings;
    let mut controls = controls(&mut settings, effect)?;
    slider(&mut controls, setting).map(|(value, _)| *value)
}

/// `effect`'s sliders in `settings`, by setting name, with their values.
fn slider_values(settings: &MaskEffectSettings, effect: MaskEffect) -> Vec<(String, f32)> {
    let mut settings = *settings;
    let Some(controls) = controls(&mut settings, effect) else {
        return Vec::new();
    };
    controls
        .all()
        .filter_map(|control| match control {
            EffectControl::Slider(value, spec) => Some((
                setting_name(spec.label),
                rounded(**value, spec.decimals as u32),
            )),
            _ => None,
        })
        .collect()
}

/// The effect and setting a value name such as `vignette_amount` names.
fn parse_name(name: &str) -> Option<(MaskEffect, &str)> {
    let (effect_name, setting) = name.split_once('_')?;
    let (effect, _) = EFFECTS.iter().find(|(_, known)| *known == effect_name)?;
    Some((*effect, setting))
}

/// Whether `name` is an effect's slider.
pub(super) fn is_effect_value(name: &str) -> bool {
    parse_name(name).is_some_and(|(effect, setting)| {
        slider_value(&MaskEffectSettings::initial(effect), effect, setting).is_some()
    })
}

/// The enabled effects' amounts and their other sliders away from where a
/// new effect starts, as Gemma reads them.
pub(super) fn changed_effects(effects: &[EffectComponent]) -> Map<String, Value> {
    let mut values = Map::new();
    for (effect, effect_name) in EFFECTS {
        let Some(component) = effects
            .iter()
            .find(|component| component.effect == effect && component.enabled)
        else {
            continue;
        };
        let initial = MaskEffectSettings::initial(effect);
        for (setting, value) in slider_values(&component.settings, effect) {
            if setting == "amount" || slider_value(&initial, effect, &setting) != Some(value) {
                values.insert(format!("{effect_name}_{setting}"), json!(value));
            }
        }
    }
    values
}

impl CalibRawApp {
    /// Applies one `set_effect` call's `arguments` to the global effects,
    /// adding (or turning on) an effect a value is set for. Each value stays
    /// within its slider's range. `baseline`, the effects when the reply
    /// came, or where a new effect starts, gives the values they were.
    pub(super) fn apply_set_effect(
        &mut self,
        baseline: &[EffectComponent],
        arguments: &Value,
    ) -> Value {
        let layout = ScreenLayout::from_size(self.egui_ctx.content_rect().size());
        let before = self.masks.stack.global_effects.clone();
        let mut outcome = Outcome::default();
        for (name, requested) in arguments.as_object().into_iter().flatten() {
            let (Some((effect, setting)), Some(requested)) = (parse_name(name), number(requested))
            else {
                outcome.not_set(name);
                continue;
            };
            let shown = baseline
                .iter()
                .any(|component| component.effect == effect && component.enabled);
            let from_settings = baseline
                .iter()
                .find(|component| component.effect == effect)
                .map_or_else(
                    || MaskEffectSettings::initial(effect),
                    |component| component.settings,
                );
            let Some(from) = slider_value(&from_settings, effect, setting) else {
                outcome.not_set(name);
                continue;
            };
            let present = self
                .masks
                .stack
                .global_effects
                .iter()
                .any(|component| component.effect == effect);
            // Fails when the edit holds as many effects as it can.
            if !present && !self.add_global_effect(effect, layout) {
                outcome.not_set(name);
                continue;
            }
            let Some(component) = self
                .masks
                .stack
                .global_effects
                .iter_mut()
                .find(|component| component.effect == effect)
            else {
                outcome.not_set(name);
                continue;
            };
            component.enabled = true;
            let Some(mut controls) = controls(&mut component.settings, effect) else {
                outcome.not_set(name);
                continue;
            };
            let Some((field, spec)) = slider(&mut controls, setting) else {
                outcome.not_set(name);
                continue;
            };
            let limits = Limits::new(spec.min, spec.max, spec.decimals as u32);
            let value = limits.limit(requested);
            *field = value;
            // An effect that was off showed nothing, so its amount was 0, as
            // the prompt says; its slider's value made a new vignette read
            // -25→-25, and Gemma, seeing no change, doubled it.
            let was = if setting == "amount" && !shown {
                0.0
            } else {
                rounded(from, limits.decimals)
            };
            outcome.set_number(name, requested, was, value, limits.decimals);
        }
        if self.masks.stack.global_effects != before {
            self.mark_mask_adjustments_dirty();
        }
        outcome.into_response()
    }
}
