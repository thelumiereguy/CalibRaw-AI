//! The `set_ai_denoise` tool, which turns the RawNIND job on or off and sets
//! its strength. Unlike the other tools it starts app work, so it runs last
//! in a nudge (`apply_ai_editor_calls`).

use super::tools::number;
use super::*;
use calibraw_ai::AiFeature;
use serde_json::{json, Value};

impl CalibRawApp {
    /// Applies a `set_ai_denoise` call. Turning it on starts the RawNIND job,
    /// which may first ask the photographer to allow it; the response says
    /// which, since the app's notices never reach the model.
    pub(super) fn apply_set_ai_denoise(
        &mut self,
        arguments: &Value,
        frame: &eframe::Frame,
    ) -> Value {
        let Some(enabled) = arguments.get("enabled").and_then(Value::as_bool) else {
            return json!({ "error": "enabled must be true or false" });
        };
        let Some(raw) = self.develop.loaded_raw.as_deref() else {
            return json!({ "error": "no photo is open" });
        };
        if raw.is_pre_demosaiced_raster() {
            return json!({
                "error": "AI denoise works only on RAW photos; use set_sliders' noise_luminance and noise_color"
            });
        }
        let was = json!({
            "enabled": self.develop.exposure.ai_denoise_enabled,
            "strength": (self.develop.exposure.ai_denoise_amount * 100.0).round(),
        });
        if let Some(strength) = arguments.get("strength").and_then(number) {
            let amount = strength.clamp(0.0, 100.0) / 100.0;
            if amount != self.develop.exposure.ai_denoise_amount {
                self.develop.exposure.ai_denoise_amount = amount;
                self.mark_pipeline_dirty();
            }
        }
        if enabled != self.develop.exposure.ai_denoise_enabled {
            if enabled && self.foreground_operation_active() {
                return json!({ "error": "another operation is running; try again once it finishes" });
            }
            self.set_ai_denoise_enabled(enabled, frame);
        }
        let status = if self.ai_denoise_running() {
            "running; it takes about a minute, then the app sends the new preview"
        } else if self.ai_consent_is_for(AiFeature::Denoise) {
            "waiting for the photographer to allow it"
        } else if self.develop.exposure.ai_denoise_enabled {
            "on"
        } else if enabled {
            let reason = self.ui.notice.clone();
            return json!({ "error": reason.unwrap_or_else(|| "AI denoise could not start".to_owned()) });
        } else {
            "off"
        };
        json!({
            "was": was,
            "now": {
                "enabled": self.develop.exposure.ai_denoise_enabled,
                "strength": (self.develop.exposure.ai_denoise_amount * 100.0).round(),
            },
            "status": status,
        })
    }
}
