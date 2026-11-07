# Fine-tuning on-device Gemma 4 for better photo edits

Research only, 2026-11-05. No code changed, nothing trained, no dataset downloaded.
Status marks: ✅ verified on a primary source or real data · ⚠️ your call, or unverified ·
❌ not possible (with reason) · ℹ️ FYI.

## Summary

**Can it be done?** Yes for training, maybe for getting the result back onto the phone. The phone
is the risky part, so test it first.

**Latest (2026-11-06): you're short on time, so use the [short path](#short-path-rtx-3070)**:
E2B on your RTX 3070, your own edits only, no teacher. About 2–3 days end to end, ~2–3 h of your
own time. The two-Mac plan below is kept for reference.

0. ⚠️ **Everything runs on your two Macs, $0** (decided 2026-11-06, no rented GPUs). The
   **M5 Pro (24 GB)** trains and runs the teacher; its GPU's neural accelerators make those
   compute-heavy steps several times faster. The **M1 Pro (32 GB)** has more memory, so it does
   the phone export, previews and triage cards. Details: [Local plan](#local-plan-two-macs).
1. ⚠️ **Do the phone check before anything else (0.5–1 day, $0).** Convert Google's *stock*
   Gemma 4 weights yourself with `litert-torch export_hf` (with vision) on the M1 Pro, run the
   file on the phone with LiteRT-LM 0.19.0, and compare speed and answers with Google's
   prebuilt file.
   People who did this report problems: crashes, garbled vision, ~10% bigger files, no
   multi-token drafter, about half the decode speed (§5). No one reports running it on 0.19.0.
   If this check fails, a fine-tune can only ship by switching the app to llama.cpp, a
   separate, large change.
2. ✅ **Training runs on a Mac with mlx-vlm** (LoRA SFT, Gemma 4 support, loss on the answer
   only). Unsloth needs an NVIDIA GPU, so it's out. ⚠️ No measured Mac training speed exists.
   Estimated ~2–7 h on the M5 Pro for ~1,000 examples × 2 epochs on E4B (8–22 h on the M1 Pro);
   measure 20 examples first.
3. ✅ **Your own edits are the best data and need no teacher** for the editor phase:
   832 RAW + XMP pairs, ~250–400 of them different after removing pasted looks.
   CalibRaw's Lightroom importer already maps them (§1).
4. ✅ **Teacher: an open Gemma 4 model run locally**, for looks you never edited yourself
   (Apache 2.0, takes images). ⚠️ You picked 31B when a rented GPU was assumed. 31B doesn't fit
   the M5 Pro's 24 GB and needs ~3 nights per 500 examples on the M1 Pro. 26B A4B fits the
   M5 Pro and needs ~3–5 h (§3, estimates). Claude was ruled out: the Usage Policy forbids
   training a model on its outputs without Anthropic's prior authorization.
5. ⚠️ **Public expert data is non-commercial only.** PPR10K ships real expert XMPs (11,161
   portraits × 3 experts) for "non-commercial research purposes only". FiveK is research-only
   and uses old PV2010 sliders (§2).
6. ❌ **No custom model on the NPU for now.** Google's public tools can't turn your own
   weights into the uncompiled NPU file, so a fine-tune runs on the GPU (§5). The sibling
   session's NPU result still helps choose E2B vs E4B, but not NPU vs GPU for a fine-tune.

**Total if the phone check passes:** about **4–5 working days plus 1–2 overnight runs**, **$0**
(table in §6).

---

## Prompt tuning from your XMPs (no training, 2026-11-06)

Fine-tuning is parked. This section is about using the same 853 edits to tune `editor_system.md`
instead.

### Your house style ✅ (measured, all 853 edits, read-only)

| Value | How often you set it | Typical size (median when set) | `editor_system.md` scale today |
|---|---|---|---|
| highlights | 734, **all but 2 negative** | **−62** (p25–p75: 37–62) | strong ≈ 50 |
| shadows | 701, 603 positive | **+55** (p75: 100) | strong ≈ 50 |
| highlights − *and* shadows + together | **594 (70%)** | — | not mentioned |
| clarity | 592, **549 negative** | **−24** | — (generic edits usually add clarity) |
| blacks | 627, split evenly | ±25–39 | — |
| whites | 690, mostly + | +21 | — |
| exposure | 536, split | **±0.5 stops** | slight 0.3 / moderate 0.7 |
| contrast | 401, almost all + | **+5** | moderate ≈ 25 |
| vibrance / saturation | 363 / 495 | **+20 / +3** | — |
| texture | 358 | +12 | — |
| HSL saturation (102 distinct looks) | 605 moves, **555 negative** | −15 | — |
| point curve, faded blacks | 318 (37%) | black point +19 | "a higher first output fades the blacks" |
| grain / vignette | 192 / 66 | 17 / −12 | new effect starts at 25 / −25 |

In short: strong highlight recovery and shadow lift, negative clarity, light contrast, vibrance
over saturation, individual colours muted, and faded blacks in about a third of your edits. It's
a soft, flat, film-leaning look, consistently different from the prompt's generic scale.

⚠️ Same caveats as §1: 43% of these edits also used masks, and Lightroom's "+55 shadows" is not
yet checked against CalibRaw's (~1 h on ~20 photos).

### Options

1. ✅ **Calibrate the prompt from these numbers (~1–2 h, recommended).** Replace "slight 10 /
   moderate 25 / strong 50" with your sizes per value, and add 2–3 lines of tendencies
   (highlights down with shadows up, clarity down, vibrance over saturation). About +100 tokens.
   - It moves the average edit towards yours.
   - It can't teach per-photo judgment.
2. ⚠️ **Automatic prompt search, scored against your edits (~1–2 days).** Run Gemma on held-out
   photos, measure how far its values land from yours, and keep the prompt that scores best.
   It needs the XMP → values exporter (§6 step 3) and Gemma running headlessly on the Mac. The
   gain is uncertain, because E4B follows prompts loosely.
3. ❌ **Your edits as worked examples in the prompt:** the model already copies the prompt's
   examples, and each example costs tokens out of the 4,096.

### What you do depends on the photo ✅ (looked at, not just counted)

Method: 73 photos sampled across all 23 folders, spread over different tone settings. Each RAW
was rendered as shot (macOS `sips`) and I labelled its scene and light from the picture
**before** reading its settings. Then I joined the labels with the settings, and checked 8 of
them against your exported JPEGs. Scenes: 27 landscapes, 12 concerts, 12 city/street, 7
interiors, 6 people, 5 night skies, 2 cars, 2 wildlife.

The overall averages above hide that most of your moves depend on what is in the photo:

| Move | When you make it (sample numbers) | When you don't |
|---|---|---|
| **Exposure** follows how dark the RAW is | dark night streets, dusk, stage, a dim interior: **+1.2 to +3.1 stops** (38, 58, 59, 69, 73, 3, 18) | bright sea, beach or stage blowout: **−0.9 to −1.2** (65, 66, 52). The prompt's "strong = 1.5 stops" is too small for night RAWs. |
| **Highlights −60 to −100** | a bright sky, snow, shiny paint or lamps in the frame (6, 25, 28, 31, 34, 22) | night skies with nothing bright: −12 (9, 10) |
| **Shadows +50 to +100** | a dark foreground under a bright or dull sky: dusk mountains, overcast valleys, a tunnel (31, 33, 40, 47, 50, 62) | dark areas that frame the photo stay dark or go darker: a beach seen through foliage, a temple against the sky, a concert crowd (66, 71, 68: −28 to 0; concerts ~+15) |
| **Clarity −15 to −60, dehaze −10 to −30** | people, concerts, interiors, night streets, cars: softer midtones and a glow around lights (14, 37, 53, 62, 70, 73) | landscapes and night skies: clarity 0 or slightly + (9, 32, 49) |
| **Texture +10 to +47** | with that negative clarity on concerts and night shots (14, 15, 53, 73) | — |
| **White balance changed** | under artificial light: blue stage light warmed to 6,600–12,100 K, warm street light cooled to 3,300–5,100 K (52, 54, 56, 69, 73) | daylight: as shot |
| **Grain, faded blacks** | per shoot, pasted across a folder (Thailand, City Walks, Munich Oct 2024) | not tied to the scene |

**Your style also changed over time** (by the year each edit was saved): 2023 edits are
"punchy" (blacks down, vibrance up, clarity untouched: 11 of 11 styled ones). 2025 edits are
mostly "soft" (clarity down, blacks up: 207 vs 31 punchy). So the recent soft look is the one to
teach. Subject and year overlap, though (the punchy edits are mostly the snowy Munich 2023 Xmas
landscapes), so I can't fully separate them.

**What the before/after pairs show** (8, 21, 28, 36, 47, 66, 69, 70):
- Warmer, brighter people and street scenes.
- A teal/cyan shift with faded blacks on night streets (70).
- Punchy, saturated greens on the 2023 landscape (36).
- ⚠️ **Masks can override the global values.** Photo 47's sliders say brighter (shadows +87,
  whites +52), but the export is darker and moodier, because a mask did that. Edits with masks
  (43%) shouldn't be used to tune values.
- ℹ️ **Crop is part of the look** (36, 66, 69). The AI editor has no crop tool.

ℹ️ 10 of the 73 sampled "edits" change no global value at all (crop or masks only). Caveats: 73
photos labelled from small thumbnails; Lightroom ≠ CalibRaw (not yet compared).

**What this means for the prompt:** write *conditions*, not fixed numbers. For example:
- "Correct a dark RAW by 1–3 stops."
- "Recover highlights hard when the sky or lights are bright."
- "Lift a dark foreground, but keep dark framing and silhouettes dark."
- "Under stage or street light, correct the white balance, lower clarity and dehaze, and raise
  texture."
- "Leave clarity alone on landscapes."

That is closer to your judgment than the per-genre presets you rejected.

⚠️ **Your call:** should the house style apply to **every** edit, or only to vague requests like
"edit it my way" or "cinematic"? On every edit it starts to act like a default preset, which you
rejected before. On vague requests only, precise ones ("lift the shadows a bit") stay literal.

---

## Short path (RTX 3070)

**What the 3070 changes**
- ✅ It runs Unsloth (NVIDIA CUDA). Unsloth's guide: "Gemma 4 E2B trains on **8GB VRAM**. E4B
  requires 10GB VRAM" ([guide](https://unsloth.ai/docs/models/gemma-4/train)).
  - So **E2B only**. ❌ E4B doesn't fit in the 3070's 8 GB.
  - ⚠️ 8 GB is the floor for E2B, and our examples are long (~4k tokens). Use batch size 1 and
    gradient checkpointing. Not tested.
- ✅ Unsloth saves merged 16-bit HF weights itself, so the hand-written MLX → HF merge script and
  its risk go away.
- ⚠️ Speed is estimated, not measured: ~2–5 s per example, so 1,000 examples × 2 epochs take
  **~1–3 h**.
- ⚠️ Run it on Linux or Windows WSL2. Unsloth documents installs for those; I haven't checked
  native Windows training.
- ❌ No teacher here: 26B A4B and 31B don't fit in 8 GB of VRAM.

**What it doesn't change**
- Most of the time goes to building and checking, not to the GPU: the phone export check, the
  XMP → training data exporter, and testing on the phone.

**The short path** (your own edits only, no teacher, no PPR10K):

| # | Step | Where | Elapsed | Your time |
|---|---|---|---|---|
| 1 | Phone check: export stock E2B with `litert-torch` and run it on the phone. If it fails, stop here. | M1 Pro (or the PC under WSL2) | ~0.5 day | ~15 min |
| 2 | Exporter + data: RAW + XMP → preview + triage card + templated brief + tool lines, de-duplicated | M1 Pro | ~1 day | ~30–60 min naming your top looks |
| 3 | Train E2B (QLoRA, settings in §4) | 3070 | ~1–3 h | — |
| 4 | Export the merged weights, test on the phone | M1 Pro + phone | ~0.5 day | ~30 min |
| 5 | Evaluate: stock vs fine-tuned on ~30 requests | phone | ~1 h | ~1 h |

**≈ 2–3 days elapsed, ~2–3 h of your own time, $0.** ⚠️ What you give up:
- Looks you never edited yourself get no examples. Words like "cinematic" are learned only from
  the looks you name in step 2.
- E4B quality: you train and ship E2B.
- The Lightroom vs CalibRaw look check is folded into step 5 instead of run first.

---

## Local plan: two Macs

| | M1 Pro (this Mac) | M5 Pro (your second Mac) |
|---|---|---|
| Memory | ✅ 32 GB (measured) | 24 GB (from you) |
| GPU memory limit (macOS default, user reports: ~65–75% of RAM) | ≈ 21–24 GB | ≈ 16–18 GB |
| Memory bandwidth (sets decode speed) | 200 GB/s (Apple spec, from memory ⚠️) | 307 GB/s ([MacRumors](https://www.macrumors.com/2026/03/03/apple-debuts-m5-pro-and-m5-max-chips/)) |
| Matrix compute (sets prefill and training speed) | GPU without matrix units | GPU with a Neural Accelerator in each core: Apple measured **3.3–4× faster prompt processing** in MLX for base M5 vs M4 ([Apple ML research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)). Needs **macOS 26.2+**. |
| Free disk | ✅ 76 GB (measured) | ⚠️ not checked |

ℹ️ M1 Pro facts measured 2026-11-06: 8P + 2E CPU cores, `iogpu.wired_limit_mb` = 0 (the default),
no MLX, llama.cpp or Ollama installed. ⚠️ I can't see the M5 Pro: its disk, macOS version and GPU
core count are unchecked. No M5 Pro MLX training benchmark exists, so the M5 times below scale my
M1 Pro estimates by ~3–4× for compute-bound work and ~1.5× for decode.

| Step | Which Mac | Fits? | Time (⚠️ estimates unless marked) |
|---|---|---|---|
| Teacher, 26B A4B | **M5 Pro**, [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) server, `mlx-community` 4-bit | ⚠️ 14.3 GB peak on an M1 Pro ([oMLX](https://omlx.ai/benchmarks/6ihwol7v), quantization not stated) against a ~16–18 GB limit: tight, so close other apps | ~20–30 s per example → **500 in ~3–5 h**. (M1 Pro: ~1 min each, one night.) |
| Teacher, 31B | **M1 Pro** only | ❌ on the M5 Pro (~17–18 GB of weights). ⚠️ Tight on the M1 Pro | ≤10 tok/s decode → ~3–4 min per example → **500 in ~3 nights** |
| Triage cards + previews | **M1 Pro**: CalibRaw `calibraw-develop-export` + mlx-vlm E4B | ✅ | ~10–20 s per photo → 800 photos in ~3–4 h |
| Training, E4B (or E2B) | **M5 Pro**: mlx-vlm LoRA SFT (`train_on_completions`, 4-bit base, vision frozen by default) ([training docs](https://github.com/Blaizzy/mlx-vlm/blob/main/docs/training.md)) | ⚠️ should fit in ~16–18 GB; not measured | ~4–12 s per example (~4k tokens each, mostly the system prompt) → 1,000 × 2 epochs ≈ **2–7 h**. (M1 Pro: 8–22 h.) E2B about half. |
| Merge LoRA → HF weights | **M1 Pro**: a ~50-line script of our own adds the adapter deltas into Google's bf16 HF checkpoint on the CPU | ✅ 16 GB bf16 fits in 32 GB RAM | minutes. ⚠️ mlx-vlm has `--dequantize` but no adapter merge, so key names must be mapped by hand |
| Export to `.litertlm` | **M1 Pro**: `litert-torch export_hf` on the CPU | ⚠️ the wheels exist for macOS arm64 (`litert-converter`, `ai-edge-litert`; `litert-torch` is pure Python, checked on PyPI) but I haven't installed or run them. One community report says E4B needs ~48 GB RAM, so expect heavy swapping on 32 GB, and worse on the 24 GB M5 Pro. Try E2B first. | ~1–3 h |

ℹ️ The two Macs can work at the same time: the M1 Pro exports and renders while the M5 Pro runs
the teacher or training. Files move between them over AirDrop or a shared folder: previews, JSONL
and the ~0.1 GB adapter.

ℹ️ Facts that drive these numbers:
- ✅ **The phone runs Gemma without thinking** (`ThinkingConfig(false, 0)` and greedy sampling in
  `GemmaBridge.java`, read only), so the teacher also writes plain answers: a few hundred tokens per
  example, not thousands.
- ⚠️ Whether mlx-vlm's server reuses the ~3k-token system prompt between requests is not checked.
  If it does, the teacher's prefill time drops by most of its share.
- ⚠️ Disk. M1 Pro: Google's E4B HF weights (16 GB), the 4-bit E4B (~5 GB), the merged copy
  (16 GB) and the export (~4 GB + temporary files), about 45–55 GB of its 76 GB. M5 Pro: the
  4-bit E4B (~5 GB) and the 4-bit 26B A4B teacher (~15 GB), about 20 GB. Download `mlx-community`
  4-bit builds only, never the bf16 26B/31B (57–62 GB). PPR10K's 313 GB of RAWs can't fit.
- ⚠️ Each Mac is busy (and hot) during its long steps; run them with `caffeinate`.
- ⚠️ Optional speed-up (your call): trained models need less instruction. Training with a short
  system prompt, and shipping that prompt in the app, cuts each example from ~4k to ~1.5k tokens,
  making training ~2.5× faster. It is an app change, and it makes before/after comparisons less
  fair.

---

## 1. Your own edits (Lightroom XMP)

### What is there ✅ (read from `~/photos`, read-only)

| | Count |
|---|---|
| XMP files | 1,190 in 52 folders (316 GB folder) |
| XMPs with develop settings (`crs:ProcessVersion` present) | **853** (the other 337 hold only ratings/metadata) |
| …with the photo next to them (same file stem) | **832** |
| Cameras | Sony A7 III (502), Sony RX100 VII (229), Fuji X100S (122): `.ARW` and `.RAF` |
| Process version | 15.4 for all 853 (current Lightroom sliders, no legacy PV2010 values) |
| Written by | Lightroom 7.1–8.5 |
| Folders with edits | 23; biggest: Munich Oct 2024 (228), Thailand 2025 (179), Munich 2023 Xmas (112), Comet (78), Opeth (55) |

A real sample (`Thailand 2025/DSC05134.xmp`): exposure −0.86, highlights −32, shadows +20,
whites +13, blacks +15, clarity −21, dehaze −8, saturation −5, a full HSL mix (8 hues,
saturations −15…−45), point curve `0→19, 74→76, 127→131, 186→190, 255→245` (faded blacks,
rolled-off whites), grain 36, plus camera calibration (red hue +30, green hue +50, …).
That is a deliberate film-like look, the kind of taste the fine-tune should learn.

### How varied the edits are ✅ (measured)

- **375 distinct tone-slider sets** across 853 edits: tone is mostly set per photo.
- **240 distinct colour looks** (HSL + calibration + grading + grain + profile + curve), and
  **one look covers 212 photos**; the top 4 looks cover 412 (48%). Colour was often synced
  across a whole shoot.
- ⚠️ So the honest size of this dataset is **~250–400 genuinely different edits**, not 853.
  Training on all 853 as-is would over-teach your 4 most-pasted looks, which comes close to
  the "presets" you rejected. Recommendation: de-duplicate (keep at most ~5 photos per
  identical colour look, picked for different scenes).

### CalibRaw already imports Lightroom XMP ✅: reuse it, don't invent a mapping

`crates/calibraw-core/src/presets/xmp.rs` (`import_xmp`, `read_xmp_preset_file`) translates `crs:`
settings into CalibRaw's `ExposureParams` + global effects, with fixtures in
`data/formats/xmp/` (checked by `crates/calibraw-core/tests/formats.rs`). The AI editor's tools
already turn `ExposureParams` into Gemma's names and ranges: `changed_sliders` (`sliders.rs`),
`changed_mixer` / `changed_grading` (`color.rs`), `changed_curves` (`tone_curve.rs`) and
`changed_effects` (`effects.rs`). So a training target is

```
XMP bytes → import_xmp → ExposureParams + effects → changed_* → tool lines
```

with no new mapping table. ⚠️ Not yet checked: the fixtures are *presets*. The importer reads
`crs:*` attributes on a top-level `rdf:Description`, which is how photo sidecars store them too.
Its doc comment also mentions `crs:Preset` "embedded in a photo's sidecar", but I have not run
it on one of your 853 files (a 10-minute check once an exporter exists). The `changed_*`
functions are `pub(super)` inside `calibraw-ui`, so an exporter binary would need access to them.
That is a code change for later, not done here.

### Field coverage: what maps to the AI editor's tools

"Used in" is the share of your 853 edits that set it (measured).

| Lightroom (`crs:`) | AI editor tool / value | Used in | Notes |
|---|---|---|---|
| Exposure2012, Contrast2012, Highlights2012, Shadows2012, Whites2012, Blacks2012 | `set_sliders` exposure … blacks | 88% | ✅ 1:1, same ranges |
| Texture, Clarity2012, Dehaze, Vibrance, Saturation | `set_sliders` | 87% | ✅ 1:1 |
| Temperature, Tint | `set_sliders` warmth, tint | 47% changed | ✅ maps. ⚠️ Gemma's warmth is **mired away from the RAW's as-shot white balance** and tint is **relative to as-shot** (`sliders.rs`). XMP stores absolute Kelvin, so the exporter must load the RAW. 37 files use "Auto" white balance, whose value may not be stored. |
| Hue/Saturation/LuminanceAdjustment × 8 colours | `set_color_mixer` | 50% | ✅ 1:1 (the importer doubles hue to CalibRaw's ±200; `color.rs` halves it back for Gemma) |
| ColorGrade* (shadow/midtone/highlight/global, blending), SplitToning* | `set_color_grading` | 15% | ✅ 1:1 |
| ToneCurvePV2012 (+Red/Green/Blue) | `set_tone_curve` | 83% | ✅ maps. ⚠️ Lossy: Gemma writes 5 outputs at inputs 0/64/128/192/255, so curves with other points are resampled (`tone_curve.rs` `outputs`). |
| Parametric curve | folded into the RGB point curve (`curves.rs`) | 2% | ✅, then resampled as above |
| Sharpness, SharpenRadius/Detail/EdgeMasking, LuminanceSmoothing, ColorNoiseReduction | `set_sliders` sharpen_*, noise_* | ~95% | ✅ 1:1 (mostly near defaults; low value as a target) |
| GrainAmount/Size/Frequency, PostCropVignette* | `set_effect` grain_*, vignette_* | 24% | ✅ maps (grain size rescaled by the importer) |
| — | `set_effect` glow_*, halation_* | — | ℹ️ Lightroom has neither, so your edits never teach these |
| Camera calibration (Red/Green/Blue Hue & Saturation) | **none** | **42%** | ❌ no tool, and the importer skips it (with a warning). Your colour looks lean on it heavily. |
| Masks (MaskGroupBasedCorrections) | **none** | **43%** | ❌ the AI editor sets global values only |
| Spot heal / remove | none | 37% | ❌ not a grading task |
| Crop | none | 67% | ℹ️ ignore |
| Camera profile (Camera Standard, Deep, PROVIA, …) | none | 44% use one other than Adobe Standard | ❌ not imported; part of the look lives in the profile |

**Bottom line** ✅: every *tool* the editor has (except glow and halation) is covered by your
edits. ⚠️ But only **179 of 853 edits (20%) are global-only** (no masks, no calibration). In the
other 80%, the global values were chosen *together with* masks or calibration, so on their own
they reproduce only part of the look (the sample's exposure −0.86 came with a +0.4 subject
mask). Training on those still teaches tone and colour choices, but some will come out darker
or less coloured than you meant.

### Pairs can be built ✅, with two risks ⚠️

- Inputs: 832 RAWs sit next to their XMPs. The "before" image must be **CalibRaw's default
  render** at the 1024 px edge (`GEMMA_IMAGE_EDGE`, `write_gemma_jpeg` in `triage.rs`), not
  Lightroom's, because on the phone Gemma sees CalibRaw's preview. `calibraw-develop-export`
  already renders RAWs headlessly on the Mac.
- ⚠️ Unverified: whether Lightroom values *look the same* in CalibRaw. Demosaic, profile and
  tone mapping differ, so "shadows +20" may not mean the same in both. Check ~20 photos (render
  the imported XMP in CalibRaw next to Lightroom's export) before training on any of it (~1 h).
- ⚠️ Missing: your edits carry no request text, and each photo needs a triage card. §6 covers
  both.

---

## 2. Public expert-edit datasets

| Dataset | Ships slider values? | Size | License | Fit |
|---|---|---|---|---|
| **MIT-Adobe FiveK** ([site](https://data.csail.mit.edu/graphics/fivek/)) | ✅ Yes, inside the archive: "An Adobe Lightroom catalog with renditions by 5 experts", including "values of individual sliders and the full history of adjustments for each photo". The catalog is SQLite, with settings as Lua text in `Adobe_imageDevelopSettings` ([lrcat-extractor](https://github.com/hfiguiere/lrcat-extractor), Rust, MPL-2.0). | 5,000 DNGs, ~50 GB tar | ⚠️ research only: "solely for your own research purposes … not … intended for or directed toward commercial advantage" ([legal](https://data.csail.mit.edu/graphics/fivek/legal/)) | ⚠️ The edits (2011) almost certainly use **PV2010** sliders (Recovery/Fill Light/Brightness, no Highlights/Shadows/Whites); unverified, the site doesn't say. CalibRaw's importer warns on process versions below 11. Poor fit. |
| **PPR10K** ([repo](https://github.com/csjliang/PPR10K)) | ✅ Yes: `xmp_target_a/b/c`, "CameraRaw meta-file of the raw photos recoding the full adjustments by expert a", 11,161 each (130 MB each), plus `xmp_source` defaults | 11,161 portrait RAWs in 1,681 groups; 406 GB total, RAWs 313 GB | ⚠️ "available for non-commercial research purposes only"; must not "exploit for any commercial purposes" | Best public match: modern Camera Raw XMPs that CalibRaw's importer reads directly. ⚠️ Portraits only. Groups were edited consistently (the same pasted-look problem). Process version not checked. |
| **MMArt-PPR10k** (JarvisArt) ([HF](https://huggingface.co/datasets/JarvisArt/MMArt-PPR10k)) | ✅ `config.xmp` + before/after + instructions, ~350k rows | 3.3 GB | ⚠️ tagged Apache-2.0, but built on PPR10K's non-commercial images | ❌ for taste: Qwen2.5-VL-72B *picked presets* from a library and a human approved them, so these are chosen presets, not expert slider choices ([paper](https://arxiv.org/abs/2506.17612)). Instructions could still seed request wording. |
| **MonetGPT** ([paper](https://arxiv.org/abs/2505.06176), [repo](https://github.com/niladridutt/monetgpt)) | builds ~25k "puzzle" samples from ~8k PPR10K edits with scripts; no dataset download | — | code + weights MIT; data inherits PPR10K | ℹ️ useful precedent: a 7B VLM (Qwen2-VL) fine-tuned with DoRA to output exposure/contrast/…/HSL ops |
| **VeraRetouch** ([repo](https://github.com/OpenVeraTeam/VeraRetouch)) | its AetherRetouch-1M+ data is **not released** | — | no license | ℹ️ precedent: on-device-sized (FastVLM-0.5B) model predicting PV2012 sliders |
| **INRetouch RTD** ([HF](https://huggingface.co/datasets/omaralezaby/Retouch_Transfer_Dataset)) | ❌ images only (FiveK RAWs × 167 presets), no values on the card | ~100k images | CC BY-NC-SA 4.0 | ❌ |
| **Preset collections** | [peva3/Lightroom-Presets](https://github.com/peva3/Lightroom-Presets): 451 XMPs, PV 15.4, **MIT**; [synaptic](https://github.com/xsynaptic/synaptic-lightroom-presets): GPLv3 | no paired photos | MIT / GPL | ❌ for taste: presets, exactly what you rejected. ℹ️ Could serve as "named look" vocabulary only. |

ℹ️ Facts above come from the dataset pages (FiveK and PPR10K re-checked by me). JarvisArt,
MonetGPT and VeraRetouch were read by a research agent on their pages. Not opened: a PPR10K
XMP's process version, MMArt-Bench's license.

⚠️ **Your call:** is the published demo "non-commercial research"? If yes, PPR10K is usable and
is the only source of real expert values beyond your own. If the demo or app is ever sold or
used for a business, it is not. The weights would count as "derived data".

---

## 3. A big model as teacher (distillation)

### ❌⚠️ Terms: Anthropic's Usage Policy forbids this without prior authorization

> **"Use outputs to train an AI model (e.g., "model scraping" or "model distillation") without
> prior authorization from Anthropic."**
> Anthropic Usage Policy, *Universal Usage Standards → Do Not Abuse Our Platform*,
> [anthropic.com/legal/aup](https://www.anthropic.com/legal/aup)

- ⚠️ Dates: the version on the page is marked **effective Nov 12, 2026, one week from today**.
  I could not open the version history to read the one in force today. A secondary source
  quotes the earlier wording as "Utilization of inputs and outputs to train an AI model (e.g.,
  "model scraping" or "model distillation") without prior authorization from Anthropic"
  ([conductatlas](https://conductatlas.com/platform/anthropic/anthropic-api-usage-policy/prohibition-on-training-ai-models-using-outputs/)).
  Both versions say the same thing for this project.
- ✅ Commercial Terms (API) §D.4(a): may not "access the Services to build a competing product or
  service, including to train competing AI models"
  ([commercial terms](https://www.anthropic.com/legal/commercial-terms), effective Jun 17, 2025).
- ✅ Consumer Terms §3.2 (Claude.ai, Pro, Max): may not use the Services "to develop any products
  or services that compete with our Services, including to develop or train any artificial …
  or machine learning algorithms or models"
  ([consumer terms](https://www.anthropic.com/legal/consumer-terms), effective Oct 8, 2025).

ℹ️ Reading: the Usage Policy rule covers **any** AI model, not only competing ones. A Gemma
photo editor trained on Claude-written edits therefore needs Anthropic's prior authorization.
Owning your outputs (Commercial Terms §B) does not override it. ⚠️ Whether to ask for
authorization is a legal call for you (and your employer, if this is a work project), not mine.
Without authorization, Claude can still be used where its output is **not training data**, for
example to judge held-out edits during evaluation.

**Gemini, for comparison** ✅: "You may not use the Services to develop models that compete with
the Services", and no reverse-engineering or replicating "any component of the Services"
([aistudio.google.com/docs/terms](https://aistudio.google.com/docs/terms), effective Mar 23,
2026). That is narrower (competing models only). Whether an on-device photo editor "competes"
is again a legal judgment ⚠️.

**Cleanest option: an open-weight teacher.** ✅ Gemma 4 is plain Apache 2.0, with no extra
use policy on the license page ([license](https://ai.google.dev/gemma/docs/gemma_4_license),
[model card](https://huggingface.co/google/gemma-4-E4B-it)). Its larger sibling Gemma 4 31B can
write teacher examples with no output restriction, on a rented GPU (A100 80 GB, ~$1.19–1.79/h
on RunPod). ⚠️ Unverified: whether 31B's taste is good enough. It is a stronger model from the
same family that made the current mistakes.

### ✅ Decision (2026-11-06): a local Gemma 4 teacher, ⚠️ 31B or 26B A4B open

- ✅ Model cards: [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) is 30.7B
  dense; [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) is a 25.2B
  mixture of experts with 3.8B active. Both take "Text, Image" input, have 256K context and are
  Apache 2.0.
- **Run it on a Mac** with mlx-vlm and a 4-bit build (speeds and fit in the
  [Local plan](#local-plan-two-macs)). You picked 31B when a rented A100 was assumed. It doesn't
  fit the M5 Pro's 24 GB, fits the M1 Pro only tightly, and is ~10× slower there than 26B A4B on
  the M5 Pro. ⚠️ Your call.
  ⚠️ Don't call either one through Google's Gemini API / AI Studio: those Services carry the
  "models that compete" clause above, while local Apache 2.0 weights don't.
- **Job:** only looks you never edited yourself. On your photos it writes the interpret brief
  for a named look ("make it cinematic") *and* the tool lines, through the same two prompts
  the phone uses. CalibRaw renders each result, and you accept or reject it in a viewer.
  Rejected examples are dropped, so the fine-tune learns your filtered taste, not 31B's raw output.
- **Cost: $0.** Time (⚠️ estimated): the phone runs without thinking, so the teacher writes a
  few hundred tokens per example. That makes 500 examples ~3–5 h on 26B A4B (M5 Pro), or about
  three nights on 31B (M1 Pro). Your review is ~20 s per example: **~3 h per 500**.
- ⚠️ Risk: 31B may share the small model's habits (generic values, copying the prompt's examples).
  Check it first on ~20 requests before generating hundreds (~1 h, step 6 in §6).

### Cost if authorized: Claude Opus 5.5 per 1,000 examples (✅ prices, ⚠️ token counts estimated)

Prices ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)): $4 / MTok input,
$20 / MTok output, cache hits $0.20, **Batch API −50%** ($2 / $10). Image cost
([vision](https://platform.claude.com/docs/en/build-with-claude/vision)):
⌈width/28⌉ × ⌈height/28⌉ visual tokens, so a 1024×683 preview ≈ 37 × 25 = **925 tokens**.

| Per example (both phases: brief + tool lines) | Tokens | Batch cost |
|---|---|---|
| System prompts + tool list (~3,000, cached) | 3,000 × $0.20/M | $0.0006 |
| Image + message, twice | ~2,500 × $2/M | $0.005 |
| Output incl. thinking at medium effort (estimate) | 2,000–5,000 × $10/M | $0.02–0.05 |
| Optional self-check: render the edit, send it back, fix | +~1,500 in, ~1,000 out | +$0.013 |

**≈ $25–65 per 1,000 examples with the Batch API** (≈ $50–130 without it). Your review time on
top: at ~20 s per example in an accept/reject viewer, **~5–6 h per 1,000**. ℹ️ Kept for
reference only: the teacher is Gemma 4 31B (decision above).

---

## 4. Training

ℹ️ Sources for §4–5: I re-read litert-torch#998, the Gemma 4 license page, Unsloth's Gemma 4
guide, and Anthropic's pricing and vision docs myself. The other ✅ marks in §4–5 (the NPU
colabs, PR #3497, the Kotlin API code search, the notebooks' VRAM outputs, the Google guides)
come from a research agent that read those primary pages; I did not re-open each one.

| Option | Vision fine-tune? | GPU | Notes |
|---|---|---|---|
| **Unsloth** ([guide](https://unsloth.ai/docs/models/gemma-4/train), notebooks [E4B Vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Vision.ipynb), [E2B Vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Vision.ipynb)) | ✅ E2B and E4B, image + text (`FastVisionModel`; the guide says keep `finetune_vision_layers = False` at first) | ✅ free Colab T4: E4B 4-bit QLoRA r32 peaked at 10.8 GB (240 examples in 5.9 min); E2B 16-bit LoRA 10.8 GB (from the notebooks' saved outputs, read by the agent) | ⚠️ the guide itself says "E4B LoRA requires 17GB" in one place and 10 GB in another. Exports GGUF / merged safetensors; no LiteRT export. |
| **HF Transformers + TRL + PEFT** ([Google's vision QLoRA guide](https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora), [HF blog](https://huggingface.co/blog/gemma4)) | ✅ (guide uses E2B; class `Gemma4ForConditionalGeneration`, loaded with `AutoModelForMultimodalLM`) | bf16 GPU (L4/A100, not T4), >16 GB VRAM, >30 GB RAM to merge | ℹ️ Google's own run: 3 epochs, 3,630 steps, 57 min. Community reports PEFT trips on Gemma 4's `ClippableLinear` wrapper. |
| Keras / KerasHub LoRA ([tune index](https://ai.google.dev/gemma/docs/tune)) | ⚠️ no Gemma 4 vision guide found | — | skip |
| Vertex AI managed tuning ([docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/open-model-tuning)) | ❌ Gemma 4 E2B/E4B are **text-only** there | — | can't use photos |
| **mlx-vlm** ([repo](https://github.com/Blaizzy/mlx-vlm), [training docs](https://github.com/Blaizzy/mlx-vlm/blob/main/docs/training.md)) | ✅ has a `gemma4` model and "VLM SFT … with LoRA, QLoRA, DoRA"; adapters train the language layers with vision frozen unless `--train-vision` is set; `train_on_completions` gives answer-only loss | ✅ **your Macs** (Apple Silicon, MLX) | ⚠️ the docs don't list Gemma 4 as a tested fine-tune target. Saves adapters as safetensors, with no merge-into-base step. |
| mlx-tune ([docs](https://arahim.dev/mlx-tune/vlm.html)) | ⚠️ Unsloth-style wrapper over mlx-vlm (community, not Unsloth); claims Gemma 4 and has `save_pretrained_merged` | Mac | output format of the merge not documented |

ℹ️ Unsloth doesn't run on macOS: it depends on Triton, which needs NVIDIA (per mlx-tune's
description, not Unsloth's docs).

**Recommended, on the M5 Pro:** mlx-vlm, LoRA on a 4-bit base (`quantization_bits=4`), rank 16–32,
vision frozen, `train_on_completions`, 2 epochs, learning rate ~2e-4. Hold out 10% of your edits
for evaluation. Merge the adapter into Google's bf16 HF weights at the end, because the phone
cannot load adapters (§5). ⚠️ First run a 10-example dry run all the way through merge and export,
to prove that the MLX adapter maps onto the HF weights.

**Time and cost on a GPU, kept for reference** (estimates from the measured rates of
~0.9 s/example for E2B and ~1.5 s for E4B on a T4; the Macs' estimates are in the
[Local plan](#local-plan-two-macs)):

| Run | Free T4 | L4 (Colab Pro ~$0.17/h, RunPod $0.44–0.59/h) | A100 80 GB (RunPod $1.19–1.79/h) |
|---|---|---|---|
| 2k examples × 2 epochs | ~1–1.5 h, $0 | ~0.5–1 h, <$1 | ~15–30 min, <$1 |
| 5k examples × 3 epochs | ~4–6 h, $0 (Colab may cut the session) | ~1.5–3 h, ~$1–2 | ~0.5–1.5 h, ~$1–3 |

([RunPod prices](https://www.runpod.io/pricing), Colab units per the agent's sources, the A100
rate there unverified.) **Training is not the cost or the risk.**

---

## 5. Back onto the phone

### GPU path: `litert-torch export_hf` ⚠️ (exists, rough, untested on 0.19.0)

- ✅ Official converter: `litert-torch export_hf` with `--task=image_text_to_text
  --export_vision_encoder` lists `Gemma4ForConditionalGeneration`
  ([docs](https://developers.google.com/edge/litert/conversion/pytorch/genai)). Gemma 4 export
  arrived in v0.9.4. [v0.10.0](https://github.com/google-ai-edge/litert-torch/releases/tag/v0.10.0)
  (2026-10-20) adds the audio encoder.
- ✅ From Google contributor `sirakiin` in
  [litert-torch#998](https://github.com/google-ai-edge/litert-torch/issues/998) (re-read by me):
  - The prebuilt files come from `gemma-4-*-qat-mobile-transformers` checkpoints, whose
    pre-quantized safetensors "litert-torch doesn't support now". So your file cannot match
    Google's exactly.
  - For your own weights, use the post-training quantization recipes `dynamic_wi4c_hr_afp32` or
    `dynamic_wi4b32_afp32` (int4).
  - `--jinja_chat_template_override litert-community/gemma-4-E4B-it-litert-lm` "is indeed
    needed".
  - The vision encoder export works.
  - The multi-token drafter and audio were still being open-sourced.
- ⚠️ Community reports (not verified):
  - Merged checkpoints need their `ClippableLinear` keys renamed.
  - Exports have segfaulted or produced empty output, and one user got gibberish vision
    after an Unsloth fine-tune.
  - Self-exported files are 4.0 GB vs 3.66 GB prebuilt.
  - On a Pixel 6 GPU a merged int4 export decoded at 7.6–9 tok/s vs 14.5 stock
    ([LiteRT-LM#3173](https://github.com/google-ai-edge/LiteRT-LM/issues/3173)).
  - Every report used LiteRT-LM 0.10–0.18; **none ran on 0.19.0**.
- ⚠️ Machine: one report says the E4B export needed ~48 GB+ RAM. This Mac has **32 GB RAM and
  76 GB free disk** (M1 Pro, re-measured 2026-11-06). The converter's packages have macOS arm64
  wheels (✅ PyPI), so try it here with swap, E2B first. ⚠️ If it runs out of memory, the only
  options are a 64 GB machine, or freeing RAM by exporting the parts one at a time (not
  checked whether `export_hf` allows that).
- ✅ int4 quantization: via the recipes above. The vision encoder survives (exported as its own
  part).

### LoRA adapters at runtime ❌ (merge instead)

- LiteRT-LM's C and Python APIs have `supported_lora_ranks`; the Kotlin API the app uses
  doesn't (per the agent's code search).
- The stock Gemma 4 bundles have no LoRA inputs: an adapter "loads with no effect", and the
  `-gpu` file fails with `supported_lora_ranks: []` (#3173, community).
- [PR #3497](https://github.com/google-ai-edge/LiteRT-LM/pull/3497) "Fix Gemma-4 GPU LoRA" was
  merged 2026-10-08, but no LoRA-capable bundle has been published.
- **Merge the LoRA into the weights before export.** ℹ️ Cost of merging: one 3.7–4 GB file per
  fine-tune, swapped in place of the stock model.

### NPU (Hexagon, SM8850) ❌ for a fine-tune, today

- ✅ Correction to the starting premise: `gemma-4-E2B-it_NPU_Uncompiled.litertlm` is documented
  as compiled **ahead of time on a host**, with Google's
  [NPU compiler colab](https://github.com/google-ai-edge/litert-torch/blob/main/litert_torch/generative/export_hf/experimental/litert_lm_npu_compiler/colab/litert_lm_npu_compiler.ipynb)
  targeting `SM8850`. It compiles the vision encoder too, so the NPU model has vision. SM8850
  supports both ahead-of-time and on-device compilation in general
  ([Qualcomm docs](https://developers.google.com/edge/litert/next/qualcomm)).
- ❌ What's missing is producing that *uncompiled* file from your own weights. Google's
  bring-your-own-model NPU export
  ([colab](https://github.com/google-ai-edge/litert-torch/blob/main/litert_torch/generative/export_hf/experimental/npu_export/litert_lm_npu_export.ipynb))
  ships configs only for Gemma 3 270M/1B and Qwen3-0.6B. The docs add that the Qualcomm NPU
  supports "only full float or SRQ models". A LoRA-tuned Gemma 4 gave garbage on a Pixel 10's
  NPU ([#2497](https://github.com/google-ai-edge/LiteRT-LM/issues/2497), community).
- ℹ️ Qualcomm AI Hub has [Gemma-4-E2B-it](https://huggingface.co/qualcomm/Gemma-4-E2B-it) for the
  llama.cpp-based GenieX runtime (q4_0, ~22–34 tok/s on the 8 Elite Gen 5), and its card says the
  export accepts custom weights (not tested). That is a different runtime from LiteRT-LM.

### Fallback runtime if the GPU export fails

✅ llama.cpp supports Gemma 4 vision via `--mmproj`
([PR #21309](https://github.com/ggml-org/llama.cpp/pull/21309),
[multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)), and Unsloth
exports GGUF directly. This is the most direct fine-tune path, but it means replacing LiteRT-LM in
`GemmaBridge` (the file the other session is editing). ⚠️ That is a big, separate change. ❌ MLC
LLM has no Gemma 4 vision yet ([PR #3559](https://github.com/mlc-ai/mlc-llm/pull/3559)).

---

## 6. Recommended plan

### Data: split by the app's two phases

The editor asks Gemma twice (`editor.rs`). Each phase gets its own examples:

| Phase | Learns | Source | Teacher needed? | Count |
|---|---|---|---|---|
| **Editor** (`editor_system.md`: brief + preview → tool lines) | your taste in *numbers*: how strong "moderate" is on this photo, which values go together. Addresses the "generic values" and "copying the prompt's examples" problems. | your de-duplicated XMPs, with the brief **written by a template from the values** (slight ≈10, moderate ≈25, strong ≈50; 0.3/0.7/1.5 stops, the scale `editor_system.md` already states) | **No** | ~300–400 (+ ~1,500 PPR10K if you accept its license) |
| **Editor**, precise requests | "lift the shadows" changes one value and nothing else | synthetic: one-line briefs ↔ one value, by template | No | ~300–500 |
| **Interpret** (`interpret_system.md`: words + preview → Tone/Colour/Finish/Keep brief) | what a vague word means *for this photo* ("cinematic", "moody") | (a) your edits with a generic request ("edit this the way I would") and the templated brief; (b) **you name your ~20–30 biggest colour looks once** ("moody film", "clean daylight"…), turning those photos into "make it moody film" examples | No | ~400 + ~300–600 |
| **Both phases**, looks you never edited | brief + tool lines for words outside your own edits | ✅ **Gemma 4 31B teacher** (decided 2026-11-06), filtered by your accept/reject | Yes | ~300–500 kept |

Total **~1.8–3.5k examples**, mostly from your own edits plus templates. ⚠️ Still your call:
include PPR10K (portraits, non-commercial).

**Example format** (Hugging Face chat JSONL as Unsloth takes it; the text matches what
`editor.rs` sends, so the fine-tune sees exactly the phone's messages):

```json
{"messages": [
  {"role": "system", "content": [{"type": "text", "text": "<editor_system.md verbatim>"}]},
  {"role": "user", "content": [
    {"type": "image", "image": "previews/DSC05134.jpg"},
    {"type": "text", "text": "Photo: tropical travel. You know that …\nBrief:\nTone: slightly darker (moderate), recover highlights moderately, lift shadows slightly, faded blacks.\nColour: slightly less saturated; greens, aquas and purples muted moderately.\nFinish: moderate grain.\nKeep: the detail in the shadows.\n\nCurrent values: every value at its default"}]},
  {"role": "assistant", "content": [{"type": "text", "text": "set_sliders exposure=-0.86 highlights=-32 shadows=20 whites=13 blacks=15 clarity=-21 dehaze=-8 saturation=-5\nset_color_mixer red_hue=25 orange_hue=-5 green_saturation=-30 aqua_saturation=-35 purple_saturation=-45 …\nset_tone_curve rgb=19/68/132/195/245\nset_effect grain_amount=36 grain_roughness=85"}]}
]}
```

ℹ️ The values are the real sample from §1. The curve is resampled by hand from its 5 points,
and the brief wording is illustrative. The "Photo:" line needs each photo's **triage card**, so
run `triage_system.md` on every training photo first (on the M1 Pro with mlx-vlm E4B), or the
fine-tune trains on cards the phone never writes. ℹ️ MLX's 4-bit E4B is not the phone's int4
file, so a few cards may differ slightly. The third turn ("The app set: …" →
one sentence) can be templated or left out.

### Training

mlx-vlm LoRA on the M5 Pro, E4B or E2B (pick after the sibling session's speed result),
settings in §4, a few hours. Keep the current system prompts unchanged at first so a before/after
comparison is fair. ℹ️ A later option is to shorten them, since a fine-tune
needs fewer instructions (the tool list costs ~1,550 tokens of prefill).

### Evaluation (what "better" means)

1. Held-out 10% of your edits: render CalibRaw with Gemma's values vs your values and compare
   per-value error and colour difference (ΔE).
2. A fixed set of ~30 requests ("cinematic", "moody", "fix the colour cast", "lift the shadows")
   on ~10 photos, stock vs fine-tuned, judged blind by you (~1 h).

### Steps, time, cost

| # | Step | Time | Cost |
|---|---|---|---|
Both Macs, $0.

| # | Step | Mac | Time |
|---|---|---|---|
| 1 | **Phone check**: export stock E2B (then E4B) with `litert-torch` (vision, int4), run on the phone with 0.19.0, compare with Google's file | M1 Pro | 0.5–1 day (the export itself 1–3 h) |
| 2 | Lightroom vs CalibRaw look check on ~20 photos | M1 Pro | ~1 h |
| 3 | Exporter: RAW + XMP → CalibRaw preview JPEG + tool lines (reuses `import_xmp` + `changed_*`) | M1 Pro | ~0.5–1 day (code change) |
| 4 | Build data: de-duplicate, triage cards (mlx-vlm E4B), templated briefs; you name your top looks | M1 Pro | ~0.5 day + ~30–60 min of yours; cards ~3–4 h unattended |
| 5 | Dry run: 10 examples through training (M5), merge and export (M1), to prove the format chain | both | ~2–3 h |
| 6 | Teacher: test on ~20 requests, then generate ~500 and review them | M5 Pro (26B A4B) or M1 Pro (31B) | 3–5 h, or ~3 nights for 31B; + ~3 h of yours |
| 7 | Train: time 20 examples, then the full run (1–2 runs) | M5 Pro | 2–7 h each |
| 8 | Merge, export, test on the phone | M1 Pro | ~0.5 day |
| 9 | Evaluate | — | ~2 h of yours |

**≈ 4–5 working days plus 1–2 overnight runs, $0** (with 26B A4B; add ~3 nights for 31B).
ℹ️ Steps 6–7 on the M5 Pro can run while steps 3–4 continue on the M1 Pro.

### Blockers

- ⚠️ Fine-tune → `.litertlm` with vision is undocumented end to end and untested on 0.19.0
  (step 1 settles it).
- ❌ No runtime LoRA on the stock Gemma 4 bundles (merge instead).
- ❌ No custom Gemma 4 on the NPU with public tools.
- ✅ Teacher terms solved: self-hosted Gemma 4 31B (Apache 2.0) instead of Claude, whose outputs
  need Anthropic's prior authorization to be training data.
- ⚠️ PPR10K / FiveK are non-commercial / research-only.
- ⚠️ 80% of your edits rely on masks or calibration the editor can't set; only ~250–400 are
  distinct.
- ⚠️ Even the bigger Mac (M1 Pro, 32 GB RAM, 76 GB free disk) may be too small for the export
  (one report: ~48 GB RAM), and is too small for PPR10K's 313 GB of RAWs.
- ⚠️ The M5 Pro's macOS must be 26.2+ for MLX to use its Neural Accelerators; not checked.
- ⚠️ The MLX adapter → HF weights → `litert-torch` chain is unproven (step 5 settles it).
