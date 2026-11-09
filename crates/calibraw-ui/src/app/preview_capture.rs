//! Capturing the edited preview once it shows the current edits, for the
//! library's developed thumbnail and for what the AI editor's model sees.

use super::CalibRawApp;
use crate::pipeline::GpuOutputSnapshot;

/// What [`CalibRawApp::current_preview_snapshot`] found.
pub(in crate::app) enum PreviewSnapshot {
    /// The output as it shows the current edits, to read back on a worker.
    Ready(GpuOutputSnapshot),
    /// The preview is still rendering; a repaint is requested to look again.
    Rendering,
    /// There is no wgpu backend to read the preview from.
    NoGpu,
}

/// Whether the edited (not the original) preview shows `preview_revision`.
fn edited_preview_is_current(
    original_requested: bool,
    original_rendered_state: Option<(bool, u64)>,
    preview_revision: u64,
) -> bool {
    !original_requested && original_rendered_state == Some((false, preview_revision))
}

impl CalibRawApp {
    /// A snapshot of the preview's output once it shows every edit made so
    /// far: the full preview when it is rendered, else the navigation preview.
    pub(in crate::app) fn current_preview_snapshot(
        &self,
        frame: &eframe::Frame,
    ) -> PreviewSnapshot {
        if self.preview.quality_dirty
            || self.develop.lens_correction_dirty
            || self.lens_correction_busy()
            || !edited_preview_is_current(
                self.preview.original_requested,
                self.preview.original_rendered_state,
                self.preview.revision,
            )
        {
            self.egui_ctx.request_repaint();
            return PreviewSnapshot::Rendering;
        }
        let Some(render_state) = frame.wgpu_render_state() else {
            return PreviewSnapshot::NoGpu;
        };
        let snapshot = if self.preview.pending_stage.is_none() {
            self.preview
                .pipeline()
                .map(|pipeline| pipeline.output_snapshot(&render_state.device, &render_state.queue))
        } else if self.preview.navigation_pending_stage.is_none() {
            self.preview.navigation.as_ref().map(|preview| {
                preview
                    .pipeline
                    .gpu()
                    .output_snapshot(&render_state.device, &render_state.queue)
            })
        } else {
            None
        };
        match snapshot {
            Some(snapshot) => PreviewSnapshot::Ready(snapshot),
            None => {
                self.egui_ctx.request_repaint();
                PreviewSnapshot::Rendering
            }
        }
    }
}
