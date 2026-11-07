# Architecture

This document records what each part of CalibRaw owns, which dependencies are
allowed, how the allowed graph is enforced, and the threading contracts that
background work follows. Update it in the change that alters a boundary.

## Crates and ownership

| Crate | Owns | May depend on | Must not depend on |
|---|---|---|---|
| `calibraw-core` | Image and edit models, colour mathematics, geometry, parameter validation, RAW/rendered loading, sidecars, presets, the LUT library and `.cube` parsing (`luts`), thumbnail cache | std, decoding and serde crates | any other CalibRaw crate, egui/eframe |
| `calibraw-gpu` | Headless devices (`device`), adapter memory queries (`memory/`), GPU resource planning, WGSL composition, processing passes, readback, tiled export and file encoding | `calibraw-core`, wgpu, naga_oil; in `memory/`, the platform APIs under wgpu (`windows` DXGI, `ash`, `objc2-metal`) | UI crates, egui/eframe |
| `calibraw-ai` | AI configuration and device capabilities (`config`), ONNX Runtime loading, its WebGPU plugin and the automatic packages (`runtime/`), where sessions run (GPU if they fit in free memory, else CPU), CPU memory admission, CPU fallback, GPU failure handling and platform and provider quirks (`placement/`), the model table, downloads and installation (`registry/`), the session slot (`slot`), the job worker (`jobs`), model-specific pre/postprocessing (`models/`) | `calibraw-core`, `calibraw-gpu` (denoise/remove render paths) | UI crates, egui/eframe |
| `calibraw-ffi` | C ABI and the Android JNI bridge: storage, pickers, export publishing, notifications, replay encoder, soft-keyboard text input, the AI editor's models | `calibraw-core` (sidecar types), jni, android-activity | processing crates, UI crates, egui/eframe |
| `calibraw-ui` (lib `calibraw`) | Application coordination (`app/`), views (`ui/`), presentation-independent services (`services/`), preview presentation, platform capabilities and OS services (`platform/`) | all crates above, eframe/egui, `moduwu-design`; in `platform/`, the OS crates (`calibraw-ffi`, `rfd`, `trash`, `zbus`, `windows-sys`, `objc2` AppKit bindings, Discord IPC, loggers) | — |
| `calibraw-cli` | Headless develop export and white-balance diagnostics | `calibraw-core`, `calibraw-gpu` | UI crates, egui/eframe |
| `xtask` | Repository tooling: architecture checks, line counts, baselines, Android build helpers, icons | std and tooling crates | product crates |
| `moduwu-design` (separate repository) | Reusable egui tokens, themes, controls, frames and responsive helpers | egui | any CalibRaw crate |

Application services start as modules (`calibraw-ui/src/services/`) and become a
crate only when a second real caller or dependency ownership justifies it.

### Enforced boundaries

`cargo xtask arch-check` runs in CI. Each rule inspects the package's resolved
normal and build dependency graph for every target platform
(`cargo tree -e normal,build --target all`) and fails if `cargo tree` itself
fails, so a lockfile or network error is never read as "dependency absent".

| Package | Forbidden in its graph | Since |
|---|---|---|
| `calibraw-core` | `calibraw-gpu`, `calibraw-ai`, `calibraw-ui`, `calibraw-ffi`, egui family | baseline |
| `calibraw-gpu` | `calibraw-ai`, `calibraw-ui`, `calibraw-ffi`, egui family | rework step 4 |
| `calibraw-ai` | `calibraw-ui`, `calibraw-ffi`, egui family | rework step 4 |
| `calibraw-ffi` | `calibraw-gpu`, `calibraw-ai`, `calibraw-ui`, egui family | rework step 4 |
| `calibraw-cli` | `calibraw-ui`, `calibraw-ffi`, egui family | rework step 4 |
| `moduwu-design` | any `calibraw*` package | baseline |

Source rules (in code; comments and strings are ignored):

- `crates/calibraw-ui/src/services/` must not use `egui`, `eframe` or
  `egui_wgpu`, since a module cannot be held to a narrower dependency set than
  its crate by Cargo.
- `calibraw-ai`'s `models/`, `registry/`, `jobs.rs` and `slot.rs` must not use
  `target_os`: they read `config::AiCapabilities` (what the device can do) and
  `placement::quirks` (platform bugs) instead.
- `crates/calibraw-ui/src/` outside `platform/` must not use `target_os`,
  `calibraw_ffi`, `rfd`, `zbus`, `windows_sys`, `objc2*`, `env_logger` or
  `android_logger`: UI code reads `PlatformCaps` and calls `Platform`.
  `cargo xtask cfg-ratchet` does not count `platform/` either.

"egui family" is egui, egui-wgpu, eframe and egui-winit. Add a rule in the
change that establishes a boundary, not in advance.

### Decoupling decisions

- GPU processing is headless. `RawGpuPipeline::output_view` exposes the display
  output; `calibraw-ui` registers it with egui (`app::preview_texture`).
- The Android bridge wakes the UI through a plain callback (`attach_ui`), not
  an `egui::Context`.
- Export destinations are explicit: `ExportTarget::File` is written beside the
  path and renamed into place; `ExportTarget::NewFile` (desktop batch and
  carousel exports) is moved into place without replacing anything, taking the
  next free `{stem}-N` name; `ExportTarget::Descriptor` (Android MediaStore)
  is written in place with intermediates in a caller-chosen staging directory.
  The exporter no longer asks the platform bridge which kind a path is.
- A carousel export (`ExportFiles::Carousel`) renders a window of the output
  raster once, with the photo's geometry unchanged, and cuts its encoded rows
  into one file per slide (`SlideWriter`). Sharpening runs on the whole window,
  so slides continue each other pixel for pixel; a watermark is placed on every
  slide, each as its own frame. Its switch and placement belong to the open
  photo and are not saved in the sidecar.
- A batch export may apply a preset and a LUT to every photo
  (`ExportOverride`). They change only the edit snapshot each item is rendered
  from; sidecars are never written.
- A camera profile changes only how the samples' colour is read
  (`ProfileColor`), never the samples. A decode keeps what it read the colour
  from (`ProfileColorSource`), so another profile is applied to the decode in
  memory (`LoadedRaw::with_camera_profile_selection`), sharing its
  `SensorSamples`, exactly as decoding with that profile would apply it. One
  `DecodedRawCache` entry therefore serves every camera-profile request; only
  a decode with the embedded matrix alone, which did not read the RAW's
  calibration signature, cannot serve requests that match DCPs.
- On-device Gemma (Android, `app/ai/editor/`) never touches pixels; it reads the
  preview and sets the edit's values. Opening a photo triages it
  (`triage.rs`): Gemma names its genre, an emoji and a role (the triage
  card); it asks for no editing priorities or observations, since
  Gemma edited towards them instead of the photographer's request. The AI
  editor (`chat.rs`) is a chat in which each request is one Gemma reply,
  whose message names that card's genre and role; the system prompt names no
  photo, so Gemma reads it once per process and keeps it across photos, so
  opening a photo costs only its triage. Gemma interprets the photographer's
  words and the preview into a brief (`assets/gemma/interpret_system.md`,
  which names no tools): lines of tone, colour, detail and effects items, each
  a setting, up or down and a size word (a touch to very strong, or stops for
  exposure), what to keep, and a sentence for the photographer. The app reads
  the brief into tool calls itself (`brief.rs`), each size a step added to the
  setting's current value. A second conversation that wrote the values from
  the brief wrote some with the wrong sign or to the wrong tool, and asked to
  plan a look and write its values in one reply, Gemma crushed a photo to
  black. An item the reader cannot read unambiguously is reported in the chat,
  not guessed. The tools are `set_sliders`
  (`sliders.rs`; on a RAW, Warmth is the white balance in mired around the
  as-shot value and Tint the shown tint less the as-shot one),
  `set_color_mixer` and `set_color_grading` (`color.rs`), `set_tone_curve`
  (`tone_curve.rs`, five outputs at fixed inputs), `set_effect` (`effects.rs`,
  the global Vignette, Grain, Glow and Halation, by slider label) and
  `set_ai_denoise` (`denoise.rs`); the brief reaches `set_sliders`, the colour
  tools and `set_effect`, and each value goes where the reader puts it within
  its range (`tools.rs`). Tool lines Gemma writes on its own are kept out of
  the chat's text (`calls.rs`). A request is one nudge, so the photographer
  steers with follow-ups: with four nudges every one after the first undid or
  overdid it. Each message to Gemma carries the edit's values away from their
  defaults and what the previous request changed, as `name=before→after`, so
  a follow-up such as "a bit more" or "undo that" works; a `back` item
  returns a value to its `before`. A message waits until
  `current_preview_snapshot` reports the render current and no foreground job
  (AI denoise, an AI mask), consent dialog or people mask from the analysis
  holds it. The nudge is one undo step (`observe_edit_history`), and undo and
  redo are off while a request runs (`can_undo_edit`); an explicit save
  meanwhile commits what is there. The chat is kept in the app only: each
  request runs on a clone of a conversation that holds just the system prompt
  (`GemmaBridge.startEditorRequest`), read once, when the first card comes
  (`prepareEditor`), since a clone takes milliseconds and a new
  conversation's first prefill took 16–23 s after the process's first. A
  request's conversation is closed when it ends, since an open one keeps its
  memory. The engine holds 4,096 tokens per conversation
  (`GemmaBridge.MAX_NUM_TOKENS`): LiteRT-LM's GPU backend reserves the whole
  context for every layer up front (LiteRT-LM issue 4096), the largest
  conversation used about 2,700, and 8,192 left the phone swapping. For the
  evals (`research/gemma-evals/`), while `files/gemma-trace.on` exists the app
  writes the preview that shows a request's edit and what it set to the
  trace GemmaBridge keeps.
- Claude Code in Termux (Android) is the other model behind the same triage
  and AI editor, chosen in Settings (`AiEditorModel`, `app/ai/editor/model.rs`).
  Java holds one `AiModel` at a time, `GemmaBridge` or `ClaudeBridge`, which
  Rust reaches through `AiModels` (`AiModels.select`). Each Claude
  conversation is one `claude -p` process that Termux starts through its
  `RUN_COMMAND` intent (`assets/claude/connect.sh`); it connects back to a
  loopback port of the app, proves itself with a one-time token and speaks
  stream-json over that connection. The triage uses Gemma's prompt and
  schema (`--json-schema`), and waits until the AI editor's chat opens
  (`TriageState::waiting`), so a photo goes to Anthropic only when the
  photographer asks Claude about it. The editor writes no brief: one
  conversation per photo holds the whole chat, plans and writes its tool
  lines in the same reply (`assets/claude/editor_system.md`), which the app
  reads itself (`calls.rs`), and looks at the preview after each nudge, with
  what the app set as `name=before→after`, to correct it, up to ten nudges a
  request; it ends sooner with a reply without tool lines. Claude also sets
  the tone curves, which the brief does not reach, and has three tools Gemma
  is not given: `set_mask`
  and `set_mask_part` (`masks.rs`) make and edit masks by a name Claude gives
  them, with their parts (radial, linear, luminance, colour and the model
  kinds; brushes and paths stay the photographer's) and adjustments, and
  `set_crop` (`crop.rs`) sets the crop, straightening, turns, flips,
  perspective and skew as the Crop panel does. Mask positions are fractions
  of the preview Claude sees, mapped back through the geometry and the lens
  correction (`PreviewPlacement`), so Claude's preview is transformed with
  the lens as Develop shows it; the crop is in fractions of the uncropped
  photo as displayed. A model-made part starts its foreground job or consent
  dialog, so one is made per reply and the nudge waits for it. After a nudge
  that set masks, each such mask's selection goes to Claude as an image of
  its own after the preview, at most four (`selection_image`); the image
  paths travel to Java one per line in `editorSend`'s path string. Each
  message also lists the masks and the crop. Switching models forgets the open
  photo's chat and card, and the photo is triaged again by the new model when
  Develop next shows it (with Claude, when the chat next opens).
  When the app starts with Claude chosen but Android reports no validated
  internet (`AiModels.online`), Gemma runs instead until the app restarts or
  a model is chosen in Settings (`AiState::offline_fallback`); the settings
  keep Claude. Only the start decides, since a switch ends the open photo's
  chat. Settings checks how far Claude Code is set up (`claude auth status`) and
  runs each missing step.
- Calibration (the hue and saturation of the primaries) is folded into the
  camera transform whenever a stage's camera uniforms are packed
  (`pack_camera_params`), after white balance, so like a camera profile it
  changes how the samples' colour is read and nothing cached from the decode.
  Its changes rerun the pipeline from the Tone stage, the first that applies
  the camera transform.

## Inside `calibraw-ui`

- `app/` — `CalibRawApp` and its state structs, command handling (`AppAction`),
  document lifecycle, worker startup and polling, persistence scheduling.
- `ui/` — views. They read state, keep view-local drafts, selection, focus and
  drag state, and return actions; application handlers perform mutation,
  persistence and background work.
- `services/` — presentation-independent workflows (no egui/eframe), currently
  edit replay. Progress is reported through callbacks and failures through
  typed errors; the app turns them into repaints and displayed state.
- `platform/` — the only module that names an OS (see
  [Platform boundaries](#platform-boundaries)).

## State ownership

| Kind | Owner | Examples |
|---|---|---|
| Document state | `DevelopState`, `MaskState::stack`, `InpaintState::edits`, `PersistenceState::history` | exposure, geometry, masks, remove/retouch edits, undo history |
| Interaction state | `DevelopUiState`, `MaskState` drag fields, view-local `egui` memory | crop/mask drags, pickers, open sections |
| Active work | `ForegroundOperation`, `ExportTask`, `PreviewState` receivers (including `full_resolution`), `AiState::update`, `InpaintState::receiver`, and on Android `TriageState`, `EditorState`'s request and `ClaudeSetupState`'s check | AI masks, export, preview rebuilds, the idle full-resolution render, the photo triage, AI editor requests, the Claude Code setup check |
| AI editor (Android) | `AiState::triage` (the open photo's triage card), `AiState::editor` (`EditorState`: the chat, its draft and what the last request changed), `AiState::claude_setup`; kept in the app only, forgotten when another photo opens or the model is switched | the triage card, chat messages, Claude Code's setup status |
| Caches | `DevelopState::decoded_raws` (shared with workers), `PreviewState::program_template`, `MaskState` caches, `AiState::people` (kept beside the RAW as `.calibraw-people`), library thumbnail caches, `calibraw-ai` model runtime, `calibraw-gpu` shared relighting surface | decoded RAWs, compiled GPU programs, AI inference results, the people found in the photo and their body parts, the surface derived from the current scene depth while a pipeline shows it |

`PersistenceState::document_generation` identifies the open document: it
increases whenever a different document is installed, and every document-bound
job records the value it started with.

## Threads and concurrency contracts

### Thread affinity

| Resource | Allowed threads |
|---|---|
| `CalibRawApp` state, egui UI state | the eframe UI thread only |
| `egui::Context` | any thread, for `request_repaint` only |
| `wgpu::Device` / `wgpu::Queue` | any thread; cloned into workers |
| egui-wgpu renderer (texture registration and release) | UI thread, through `frame.wgpu_render_state()`. `PreviewPipeline` owns a registration; dropping it queues the texture in `TextureRetirement`, which the UI thread releases at the start of the next frame, after the current frame can no longer paint it |
| JNI environment | the calling thread; `with_activity` attaches it for the call's duration. Java→Rust callbacks run on Java threads and only enqueue results and request a repaint |
| ONNX Runtime sessions | one process-wide slot in `calibraw-ai::slot`, locked by the running job; every AI job runs on its own `jobs::spawn_job` thread |
| LiteRT-LM engine and Gemma conversations (Android) | one process-wide engine in `GemmaBridge`; triage and AI editor turns run on UI worker threads (`spawn_ui_worker`) and hold `GemmaBridge`'s class lock, so one turn runs at a time. Streamed text arrives on LiteRT's callback thread (`AiModels.nativeOnTriageHeadline`, `nativeOnEditorText`), which only fills an FFI slot and requests a repaint |
| Claude Code conversations (Android) | `ClaudeBridge`; each conversation's connection is read on a daemon thread of its own, which streams text through the same `AiModels` callbacks. Triage and AI editor turns run on UI worker threads, one triage at a time (`triageLock`) and one editor turn at a time (`editorLock`); a triage may run beside an editor turn |

### Background jobs

| Job | Queue | Cancellation | Document change | Failure / disconnect |
|---|---|---|---|---|
| Document load | single receiver; a new load replaces it | the replaced receiver is dropped and its result discarded | the load installs the new document and bumps `document_generation` | notice or unsupported-file dialog; a disconnected worker is reported |
| Neighbour prefetch (desktop) | one thread after each interactive load; prepares the library neighbours into `DecodedRawCache` | `AtomicBool` checked between steps; a new load or a profile-settings change cancels it | results only fill caches; an open of a photo being prefetched waits for that decode | ignored; the open reports its own failure |
| Preview rebuild / detail | single receiver each; latest request wins | replaced receiver discards the stale result; `PreviewState::revision` rejects outdated detail renders. The zoomed crop's worker (`DetailRebuild`) also holds an `AtomicBool`, set when the view moves (`note_preview_motion`): it checks it every few rows it resamples (`build_region_proxy_until`) and ends with `Cancelled`; the next crop starts once that is reported and the view has settled. A finished crop is not installed while a drag is followed by the reduced preview | receivers cleared with the preview (`drop_detail_rebuild` also sets the flag) | disconnect clears the pending state |
| Reduced preview warm-up | one receiver in `InteractiveState`, started when the editor is quiet and the GPU is slow | none; short | the result is used only while its source is still the fitted proxy or the zoomed crop's raw (`Arc::ptr_eq`); the graph is dropped when that changes (the fitted proxy, or the zoomed crop's raw), when an export starts, or when the GPU is no longer slow | a dead worker is an error dialog; a graph the budget cannot hold is skipped with a diagnostics line; neither is retried for that proxy |
| Full-resolution preview (desktop) | one `RunningRender` in `PreviewState::full_resolution`, started after the editor has been idle; no second one starts until its result arrives | `AtomicBool`, set by `FullResolutionState::observe` when the fitted preview shows another `FittedOutput` (texture and `output_revision`), and at once by an edit, a moved view or a pressed pointer. The worker renders one stage at a time and waits for the GPU between them (`wait_for_gpu`), so a cancel is noticed before the next stage | the result is installed only when `FittedOutput` and `document_generation` still match | a photo whose graph does not fit the adapter's free memory or texture limit is skipped with a diagnostics line; a failed render is an error dialog; neither is retried for the document |
| Foreground AI / lens correction / auto straighten and perspective | one `ForegroundOperation` slot; a second request is refused | shared `AtomicBool`; the worker stops at its next safe point | results are applied only if `document_id` equals the current `document_generation`; an automatic geometry estimate also needs the flips and skews it was made for (and, for straightening, the perspective correction) | error dialog or notice; slot cleared |
| People mask from the analysis | `AiState::people_composes`, one per people component; a newer change of the component replaces its entry. Runs no model, so it never takes the foreground slot | none; the replaced receiver is dropped and its result discarded | `forget_people` and history restores clear the entries; a result is applied only while its component still has the person and parts it had when it started | error dialog |
| Photo triage (Android) | one receiver in `TriageState`, started when an opened photo's load is installed; it waits (`waiting`) until Develop shows the photo and its render is current (`current_preview_snapshot`), and with Claude Code until the AI editor's chat is open too. A worker reads the preview back and applies the geometry | none; it ends with the model's reply or after 120 s with Gemma, and with Claude Code after up to 35 s for its process to connect plus 120 s for the reply | the receiver is dropped when another photo opens or the model is switched (`forget`); a headline or card is taken only for the generation it was started with | logged and kept as `TriageState::failure`, which the chat shows with a retry (`retry_photo_triage`); the photo stays editable and the editor cannot send |
| AI editor request (Android) | one `Request` in `EditorState`; the chat sends nothing while one runs. It steps between capturing the preview and the model replying, one receiver per turn, and, for the evals, to writing Gemma's edited preview to the trace | another photo opening or a model switch closes the conversation (`forget_ai_editor`; LiteRT's `cancelProcess`, or `Session.end`, which closes Claude Code's connection: its process exits once the running turn's API call, already billed, has finished) | the request carries `document_generation` and is dropped when it no longer matches; streamed text is taken only for the turn's request id, which keeps counting across photos | the request ends with a failure message in the chat, and Claude's conversation is closed; a turn times out after 120 s, and with Claude Code a request first waits up to 35 s for its process to connect |
| Claude Code setup check (Android) | one receiver in `ClaudeSetupState`; no second check starts while one runs | none; it ends when `claude auth status` exits or after 60 s | none; the setup is an app setting | the failure is shown in Settings; Termux not running the check reads as blocked |
| Remove / retouch | one receiver in `InpaintState` | `AtomicBool` | `reset_for_document` sets the flag and drops the receiver, so late results are discarded | notice; the pending stroke is kept only when the user must re-consent to a download |
| Export / batch export / replay | one `ExportTask`; batch items run sequentially | `AtomicBool`; partial files are removed and never reported as success | export works on an immutable snapshot of the edit | notice; temporary output removed |
| Export watermark image | one receiver in `WatermarkState`: the saved image read at start-up, or a newly chosen one decoded and kept as `watermark.png` beside the settings | none; the card cannot choose another image while a load runs, and a picker result that arrives meanwhile replaces it | none; the watermark is an app setting, not part of a document, and each export takes the image it holds when it starts | error dialog; a saved image that cannot be read turns the watermark off |
| Sidecar save | `VecDeque` of requests, one write in flight | none; writes are short | requests carry `generation` and `revision`; stale completions are ignored | failure keeps a recovery request and shows a dialog |
| Heap purge (Android, Linux glibc) | one `calibraw-heap-purge` thread in `calibraw_core::system_memory`; requests made while a purge is queued or running are served by one more | none; a purge takes up to a few hundred milliseconds | none; it only returns pages the allocator already freed | a thread that cannot start leaves memory with the allocator; an Android without `M_PURGE` changes nothing |
| Library ratings and flags | one `calibraw-review-writer` thread per library, started by the first change; saves run in the order they were made | none; a later change to a photo queues behind the earlier one, and the grid shows a saved review only once its photo has no save waiting | the open photo changes through its document: on Android its review is saved with the edits, on desktop with its editing time | each save reports the review its sidecar then holds; failures show the error dialog; library photo actions, opening a photo and exit wait up to two seconds for queued saves, as a media-library save rewrites the whole sidecar |
| Linked folder tree | one `calibraw-library-folders` thread at a time, beside the photo scan; started by a refresh, not by selecting a folder within the tree; a refresh during a walk walks again after it | none; bounded by the folder limits | its folders carry the linked folder walked and are shown only while it is; switching linked folders abandons the walk | logged; the photo scan reaches the same folder and shows the error dialog |
| Linked folder AI results | one thread per finished AI job, polled through `AndroidState::companion_publishes` | none | copies to the photo the job finished for | error dialog; the result stays usable until the photo closes |
| Library thumbnails | shared work queue for a worker pool | `AtomicU64` generation; workers exit when it changes | generation bump on folder change | each item's result, success or error, is sent with its generation; older generations are ignored |

Job channels are unbounded `std::sync::mpsc` channels: each job sends a
bounded number of progress events and one result, drained each frame by
`drain_worker_events`, which stops at a terminal event and reports a
disconnected worker so the app never waits on a dead job. The library
thumbnail pool is the exception: its request and result channels are bounded
(`MAX_PENDING_THUMBNAILS`, `MAX_PENDING_THUMBNAIL_RESULTS`), so a large folder
applies back-pressure to the scanner and workers instead of growing memory.
GPU readbacks wait on one-slot channels.

### Synchronization

| Lock | Owner | Holders | Purpose |
|---|---|---|---|
| decode gate (`RwLock<()>`) | `LibraryState`, shared by `decode_gate()` | write: document load and batch-export decodes, and the UI thread while clearing the thumbnail cache; read: library thumbnail, reference-preview and neighbour-prefetch decodes | an interactive decode runs alone; background decodes share the gate and yield; the cache is never cleared under a reader |
| `REFERENCE_PREVIEW_SERIAL` | `ui::develop` | the reference-preview worker | one reference decode at a time |
| `DEVELOPED_THUMBNAIL_GPU` | `ui::library::thumbnails::developed` | thumbnail workers | one headless device for developed thumbnails, used by one render at a time |
| thumbnail work queue and request receiver | `ThumbnailWorkerContext` | thumbnail workers | short critical sections; never held while decoding |
| rendered-thumbnail permits (counting semaphore) | `calibraw_core::thumbnail_cache` | thumbnail workers rendering a full RAW (no embedded preview, or edited) | at most four such renders, fewer with fewer workers, bounding indexing memory |
| `ExtractionPermits` (counting semaphore) | one per media-library scan | its thumbnail workers while reading a photo (preview or display metadata) | at most the thumbnail-worker setting of photo reads at once; the scan has more workers, which serve cached previews without a permit |
| model runtime slot | `calibraw_ai::slot` | the running AI job | one ONNX session at a time; evicted models unload once released |
| `RUNTIME_INIT_LOCK`, runtime probe cache, runtime install lock | `calibraw_ai::runtime` | AI jobs, the Settings runtime picker | serialize ONNX Runtime initialization, probes and automatic runtime installs |
| provider statuses | `calibraw_ai::placement` | AI jobs, Settings | the provider each model's latest session uses |
| out-of-memory records | `calibraw_ai::placement::out_of_memory` | AI jobs, also under the model runtime slot; Settings when GPU AI is switched | which models ran out of GPU memory and when they may retry; a leaf lock, never held while a session is created or run |
| AI settings (`RwLock`) | `calibraw_ai::config` | the UI thread writes at startup and on Settings changes; AI jobs read | runtime source and model root; never held across a load or download |
| effect inputs (`EffectInputs`) | each `RawGpuPipeline` | threads encoding that pipeline | allocates the image-light map, haze map, contour grid or scene emission textures, and the scene bake and its light bake (`calibraw_gpu::pipeline::gpu::scene`, shared by `Arc` with views of part of the image), the first time an effect reads them, and the LUT texture when a LUT is uploaded, and rebinds the passes that read them; encoding holds it only to clone a bind group, never while waiting on the GPU |
| LUT cache | `calibraw_core::luts::LutStore`, shared by its clones | the UI thread, document loads, batch export and thumbnail workers | the recently parsed LUTs; a leaf lock, never held while a file is read or parsed |
| `SIDECAR_SAVE_LOCK` | `calibraw_core::sidecar::files` | sidecar writers | one sidecar write at a time |
| decoded-RAW cache state and condvar | `DecodedRawCache` | document loads and the prefetch worker | short critical sections; decodes run unlocked, and a second request for a decoding entry waits on the condvar |
| per-entry lens-correction slot | `DecodedRaw` | document loads and the prefetch worker | held while correcting, so one correction per selection runs at a time |
| per-entry recoloured-copy slot | `DecodedRaw` | document loads and the prefetch worker | held while another camera profile is applied to the decode (reading its DCP); never held with the lens-correction slot |
| serialized-read gate (HDD mode) | `calibraw_core::serialized_reads` | thumbnail, display-metadata and RAW loads, only while HDD mode is on | one photo file read at a time; decoding stays outside; a leaf lock, re-entered inline on the holding thread |
| `TextureRetirement` | `PreviewState` | any thread dropping a `PreviewPipeline`; drained by the UI thread | short pushes and one drain per frame |
| FFI result queues, `REPAINT_NOTIFIER`, the AI models' `TRIAGE_HEADLINE` and `EDITOR_TEXT` slots, the soft keyboard's `TYPED_INPUT` queue | `calibraw_ffi::android` | Java callback threads push; the UI thread pops | short critical sections; no JNI call while held |

Apart from the serialized-read gate, only these nestings exist, each in a fixed
order: the reference-preview serial lock is taken before the decode gate; a
thumbnail worker takes `DEVELOPED_THUMBNAIL_GPU` or an `ExtractionPermits`
permit while holding the decode gate for reading; and a decode that owns a decoded-RAW cache entry takes the decode
gate without holding the cache lock, so waiting for an entry never holds the
gate. No code takes the decode gate while holding any other lock in this
table. The serialized-read gate may be taken while holding any of them, but none
of them is taken while holding it. Poisoned locks are recovered
(`PoisonError::into_inner`) where the protected data stays valid, and reported
as errors where it may not.

### Shutdown

`eframe::App::on_exit` empties the warm AI feature set (models unload once
their job releases them), stops Discord presence, clears the Android task
notification, persists performance settings and flushes the pending sidecar
synchronously. Worker threads are detached; GPU and model resources are
released when their owners drop.

## GPU resources

`RawGpuPipeline` owns all textures and buffers of one processing graph and a
`GpuBudgetReservation` (RAII) against the process-wide GPU working-set limit.
The LUT texture is sized by its LUT, so it is not part of that plan: it takes
a reservation of its own when a LUT is first uploaded, and pipelines without a
LUT reserve nothing for it.
A graph is built for one `PixelLayout` of its source. A mosaic is
reconstructed and demosaiced from the CFA textures. A preview proxy that
reduces a mosaic is binned to co-sited RGB instead (`BinnedSensor`, built by
`build_region_proxy`), so detail near the reduced resolution cannot demosaic
into false colour: its graph keeps the pixels in the RGB source texture and
replaces highlight reconstruction and demosaicing with
`binned_highlight_reconstruct` and the raster finish program. A raster takes
the raster finish alone. Graphs reused for new pixels (`upload_raw_tile`)
take only sources of their own layout.
The graphs that show a photo with camera profiles to choose from (fitted,
navigation and zoomed crop) size their profile buffer for
`PREVIEWED_PROFILE_BYTES`, so a hovered profile renders on them
(`GpuParams::new_with_profile_color`) without a rebuild; the preview budget
already counts that room.
Presenting its output is a UI concern: `PreviewPipeline` pairs a pipeline with
its egui registration, so a displayed pipeline cannot outlive or leak its
texture. On Android a replaced detail preview frees its texture immediately
(`free_now`) to return memory before the replacement is allocated.
The editor's previews (fitted, zoomed crop, navigation proxy) render in
batches: each frame submits every pending stage of each preview, at most one
batch is in flight (`PreviewDispatchGate`, opened by `on_submitted_work_done`),
and edits made meanwhile only update the target parameters, so the next batch
renders the newest ones. While a control is dragged zoomed in and only the
Output stage changes, the fitted preview and the navigation proxy wait until
the pointer lifts (`defer_background_preview_processing`). Where the dispatch
gate's timings show the GPU takes more than about 45 ms for a batch of the
preview that is shown (a device with little memory is assumed slow until it
has timings) and the setting `interactive_preview_fallback` is on (on by
default where memory is constrained, off on desktop), the drag is followed by a second, smaller graph at half the edge
instead (`interactive_preview.rs`): the whole photo at half the fitted edge,
or, zoomed in, the sharp crop's part of the photo at half the crop's edge
(`interactive_crop.rs`), drawn with the crop's coordinates. The sharp preview
keeps merging the stage it owes and renders it when the drag ends or pauses;
the reduced picture stays until that is done. The sharp preview does not start
at the release or pause itself: the reduced preview keeps the edits for another
`CATCH_UP_DELAY`, because the GPU cannot take back a batch it was given and a
control grabbed again within that finds the GPU free. While a zoomed drag is followed,
the navigation proxy still renders Raw and Tone changes, because the reduced
crop takes its tone statistics from it. There is one reduced graph at a time,
built from the sharp preview's source (the fitted proxy, or the crop's raw);
it counts against the same GPU budget and is dropped when that source changes,
when an export starts, or when the GPU is no longer slow.
The fitted preview graph is built with mask layers for its masks plus spare layers in steps of four (`fitted_pipeline_options`, `preview_mask_layer_capacity`), not every layer the editor allows; one with fewer layers than masks is rebuilt (`fitted_mask_capacity_stale`) and keeps showing its last frame until then. On Android the fitted and zoomed graphs are sized against shares of the RAM-scaled working-set limit that remain after an export tile and the navigation graph (`MobilePreviewRole`, `bounded_mobile_preview_edge`); a Preview graph accounts only for its RGBA8 readback, since RGBA32F readbacks need `ProcessingQuality::High`. Where the platform allows (`PlatformCaps::full_resolution_preview`), an idle editor renders the whole photo from the native RAW on a worker thread with its own graph and shows the scaled result (`app/full_resolution.rs`) in place of the fitted texture while the fitted output is unchanged.
Each mask effect is described once, in the effect registry of `calibraw-core`
(`pipeline/masks/effects/`): its shader ID, the lanes its settings pack, the passes it
runs in, how far it reads and which scene inputs it needs. Validation, export-tile
halos and scene depth generation read it there; `calibraw-gpu` writes the IDs, lanes
and the creative pass's per-slot dispatch into the creative module (`effect_wgsl.rs`)
and decides from it which passes and maps to prepare. Fog, Smoke, Grain and Vignette
read the photo and the scene through `mask_effects/effect_api.wgsl` and `medium.wgsl`;
Relight, Rain and Wet Surfaces shade the scene's surfaces through the scene bake
(`shaders/scene/geometry.wgsl`).

Every mask effect and the scene it reads compile into one module
(`creative_effects.wgsl` with `mask_effects/`), and every effect pass binds the same
scene, so an effect may read any of it whichever pass it runs in. An effect names what
it reads in its registry row (`scene: [Geometry, LightBake, AirLight, Image]`), and
`GpuParams::needs_*` build only that:

- **The scene bake** (`calibraw-gpu/src/pipeline/gpu/scene/`, `shaders/scene/bake_*.wgsl`)
  holds, once per depth result and unedited scene, up to two surfaces per pixel (depth,
  normal, colour and how much the near one covers) and the working image with a mip
  chain, in the full-image frame at the working size, so every crop and export tile reads
  the same values. The preview or export pipeline that renders the whole image owns it,
  filling its working image itself; views of part of it share it. Without a depth result
  it holds the working image alone. `geometry.wgsl` is how effects read it:
  `scene_pixel` gives the two surfaces, `scene_split` and `scene_blend` shade each on
  its own, `scene_image` reads the photo with a footprint, `scene_trace` follows a ray
  through the visible surfaces and `scene_ground_*` give the ground plane fitted to the
  photo. The camera (`camera.wgsl`) maps the depth model's disparity to distance and
  defines the control depth sliders use.
- **The light bake** (`scene/lights.rs`, `lights_common.wgsl`) holds a shadow cube per
  light, each surface's visibility of it and, for the air, the share of each light that
  reaches a view ray. `scene_lights.wgsl` turns Relight's lights and Light Rays' sources
  into lights in that frame (a flat plane without depth) and receivers read them:
  `medium.wgsl` is what Fog, Smoke and Rain's mist share to scatter them.
- `scene_environment.wgsl` is the scene's light as effects receive it: its ambient
  level, the photo's coarse colour and the light sources in the photo.
- `scene_emission.wgsl` makes effects that emit over an area (a self-illuminating
  Glow, Neon's lines) lights too. Each emitter gives its radiance at a full-image
  point from full-image inputs only (its parameters, its mask in the full-image mask
  atlas, the contour grid of the tone stage), so every export tile and zoomed view
  builds the same coarse emitter grid at the start of its output stage. Two passes
  then spread that grid with inverse-square falloff over the scene's surfaces and
  along view rays through the media's depth cells, and receivers read only those
  maps, whatever the number of emitters. A new emitter is one case in
  `scene_emission_slot_at` and `scene_emission_present`, and one in
  `MaskEffect::reads_full_image_mask`. Emitters cast no shadows.

The bake depends on the scene and the depth result, the light bake on the lights'
placement and the bake, not on the rest of the edit, and each is rebuilt when any of
them changes. While a control is dragged the light bake is built in a cheaper tier and
rebuilt in the full one afterwards.

Rain and Wet Surfaces do not run inside the creative pass: `apply_wet_surfaces`
runs before it and `apply_weather_air` after it, so their result does not depend on
the order of their cards and edits without them run the same passes as before. Each
pass reads the image from one work texture and writes the other, and
`encode_output_stage` tracks which one holds it.
Compiled programs are shared through `RawGpuProgramTemplate` and the persistent
pipeline cache. Rust/WGSL layout contracts are tested in
`calibraw-gpu/src/pipeline/gpu/layout_contract_tests.rs` against the modules the
production `ShaderManager` composes.

## Platform boundaries

- Android: `calibraw-ffi` exposes content-URI storage, document pickers,
  MediaStore publishing, background-task notifications, the MediaCodec replay
  encoder and the AI editor's models behind `AiModels` (`android/ai_models.rs`):
  on-device Gemma through LiteRT-LM (`GemmaBridge`) and Claude Code in Termux
  (`ClaudeBridge`).
  Java classes live in `android/app/src/main/java/de/duecki/calibraw`.
- `calibraw-ui/src/platform/` decides platform differences once. Everything
  else compiles on every target and branches at run time:
  - `PlatformCaps` (`form_factor`, `storage`, `memory`,
    `tasks_need_foreground`, `configurable_ai_runtime`, `video_encoder`,
    `ai_editor`, `linked_folders`) is what views and handlers read. Views take
    it, or
    `app.caps()`, and return actions; leaf widgets without the app use
    `PlatformCaps::CURRENT`. Data that already names its kind is matched
    instead: `LibraryLocator`, `SidecarTarget` and `ExportDestination`.
  - `Platform` (`Arc<dyn Platform>`, in `CalibRawApp::platform` and
    `LibraryState`) is the service behind actions: the Android bridge's
    pickers, media library, sidecars, thumbnails, publishing, clipboard,
    notifications, system bars, soft keyboard (`set_soft_keyboard`, whose
    typed text egui receives through `take_typed_input`, so swipe typing and
    dictation reach text fields) and replay encoder, Android's end after a lost
    GPU device, and the desktop's settings and cache folders and GPU-loss
    restart. Its default methods answer
    "unavailable" or do nothing, so a backend implements only what it has;
    `release_free_memory` defaults to the heap purge every backend shares,
    which tests replace to observe it.
    `platform/android.rs` wraps `calibraw-ffi`; `platform/desktop/` holds the
    window entry point, the Flatpak portal, Discord IPC, on macOS the menu
    bar (`macos_menu`, whose Settings… reaches the app through
    `take_settings_request`) and, on Windows, the NVIDIA driver profile that
    keeps G-SYNC off for the window.
  - Stateless desktop services are free functions: `FileDialog` (`rfd`),
    `trash`, `show_fatal_message`, camera-profile search folders.
  - `platform::fake::FakePlatform` (tests) takes any `PlatformCaps` and
    scripted media-library contents and records the services called.
- Library storage lives in `ui/library/folders/` (folders on disk) and
  `ui/library/media_library/` (the Android library); `ui/library/sources.rs`
  dispatches on `PlatformCaps::storage`.
- The Android library has one `LibraryStore` per place photos live:
  `FileLibraryStore` (`.library`, by path) and one `TreeLibraryStore` per linked
  folder (the Storage Access Framework, no manifest permission).
  `StorageManager` picks the store from the photo's URI, so Rust keeps treating
  URIs as opaque. Linked photos are `calibraw-tree://{key}/{path}/{name}`, as a
  document URI gives no parent and can change on rename. Android's persisted
  grants are the list of linked folders. `TreeLibraryStore` replaces a file by
  writing beside it and renaming, as SAF has no atomic replace, and the next read
  restores a sidecar a crash left moved aside. Native code reads and writes AI
  companions by path, so a linked photo's live in the cache and are copied with
  `fetch_photo_companion` (document load) and `publish_photo_companion` (after the
  job).
- Media-library previews: Java owns the cache directory (`ThumbnailCache`:
  trimming, size, clearing); `calibraw-ffi` names, reads and writes the files in
  it, so a cached preview needs no JNI call. The listing reports each photo's
  sidecar size and time. The folder cache (`calibraw_core::display_metadata_cache`,
  one file per folder in the same directory) keeps each photo's dimensions and a
  summary of its sidecar, so a scan shows final tile shapes and ratings at once
  and reads a sidecar once per change (`ui/library/media_library/catalog.rs`,
  `ui/library/deferred_metadata.rs`). A scan runs `MEDIA_LIBRARY_PREVIEW_WORKERS`
  workers; `ExtractionPermits` keeps photo reads within the worker setting.
- Edit replay picks FFmpeg or the platform encoder from
  `PlatformCaps::video_encoder` (`services/replay/`).
