//! The WGSL the effect registry determines (calibraw-core
//! `pipeline/masks/effects`): each effect's shader ID, its lanes, the per-slot
//! switches and the creative pass's dispatch. It is appended to the creative
//! effects module, so the effects read these names as if declared beside them:
//!
//! - `MASK_EFFECT_<NAME>_ID` for every effect;
//! - `<NAME>_<LANE>_LANE` for every lane an effect packs, and `<OPTION>_OPTION`;
//! - `is_creative_slot_effect(id)` and `apply_creative_slot_effect(id, pos, rgb,
//!   params)`, which calls `apply_<name>(pos, rgb, params)` of each effect that
//!   runs in [`EffectPass::Creative`] while its override `CREATIVE_<NAME>` is
//!   true ([`creative_variant_constants`]).
//!
//! `<NAME>` is [`MaskEffect::shader_name`] in upper case.

use crate::pipeline::{effect_lanes::options, EffectPass, MaskEffect};

/// The effects of [`EffectPass::Creative`], in the order of their bits in a
/// creative variant (`GpuParams::creative_slot_effects`).
pub(super) fn creative_slot_effects() -> impl Iterator<Item = MaskEffect> {
    MaskEffect::ALL
        .into_iter()
        .filter(|effect| *effect != MaskEffect::Adjustment && effect.runs_in(EffectPass::Creative))
}

/// Creative variants a program can hold: one per subset of
/// [`creative_slot_effects`].
pub(super) fn creative_variant_count() -> usize {
    1 << creative_slot_effects().count()
}

/// The override constants that compile in only the creative effects whose
/// bits `effects` sets. Effects compiled into the creative pass cost every
/// pixel whether or not a slot uses them, which on Adreno outlasts the driver's
/// GPU timeout on a zoomed view.
pub(super) fn creative_variant_constants(effects: u32) -> Vec<(String, f64)> {
    creative_slot_effects()
        .enumerate()
        .map(|(bit, effect)| {
            (
                creative_override_name(effect),
                f64::from(u8::from(effects & (1 << bit) != 0)),
            )
        })
        .collect()
}

fn creative_override_name(effect: MaskEffect) -> String {
    format!("CREATIVE_{}", effect.shader_name().to_uppercase())
}

/// The registry's WGSL.
pub(super) fn registry_wgsl() -> String {
    let effects = || {
        MaskEffect::ALL
            .into_iter()
            .filter(|effect| *effect != MaskEffect::Adjustment)
    };
    let mut wgsl = String::from("// Generated from the effect registry (effect_wgsl.rs).\n");
    for effect in effects() {
        let name = effect.shader_name().to_uppercase();
        wgsl.push_str(&format!(
            "const MASK_EFFECT_{name}_ID: u32 = {}u;\n",
            effect.shader_id()
        ));
        for lane in effect.lanes() {
            wgsl.push_str(&format!(
                "const {name}_{}_LANE: u32 = {}u;\n",
                lane.name, lane.index
            ));
        }
    }
    for (name, index) in options::ALL {
        wgsl.push_str(&format!("const {name}_OPTION: u32 = {index}u;\n"));
    }

    let creative: Vec<MaskEffect> = creative_slot_effects().collect();
    for effect in &creative {
        wgsl.push_str(&format!(
            "override {}: bool = true;\n",
            creative_override_name(*effect)
        ));
    }
    let ids = creative
        .iter()
        .map(|effect| format!("{}u", effect.shader_id()))
        .collect::<Vec<_>>()
        .join(", ");
    wgsl.push_str("fn is_creative_slot_effect(effect_id: u32) -> bool {\n    switch effect_id {\n");
    if !creative.is_empty() {
        wgsl.push_str(&format!("        case {ids}: {{ return true; }}\n"));
    }
    wgsl.push_str("        default: { return false; }\n    }\n}\n");
    wgsl.push_str(
        "fn apply_creative_slot_effect(\n    effect_id: u32,\n    pos: vec2<i32>,\n    \
         rgb: vec3<f32>,\n    params: MaskEffectParams,\n) -> vec3<f32> {\n    switch effect_id {\n",
    );
    for effect in &creative {
        wgsl.push_str(&format!(
            "        case {}u: {{ if {} {{ return apply_{}(pos, rgb, params); }} return rgb; }}\n",
            effect.shader_id(),
            creative_override_name(*effect),
            effect.shader_name()
        ));
    }
    wgsl.push_str("        default: { return rgb; }\n    }\n}\n");
    wgsl
}
