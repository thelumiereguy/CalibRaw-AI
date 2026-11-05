//! On-device Gemma through `GemmaBridge` (LiteRT-LM).

use super::*;
use jni::objects::JByteArray;

/// A triage's genre and emoji, reported while Gemma still writes the rest of
/// its card.
#[derive(Debug)]
pub struct TriageHeadline {
    /// The `request_id` the triage was started with.
    pub request_id: u64,
    pub genre: String,
    pub emoji: String,
    /// The emoji drawn in colour as premultiplied RGBA rows, `None` when the
    /// emoji font lacks it.
    pub badge: Option<Vec<u8>>,
}

/// Only the newest headline matters, so this is a slot rather than a queue.
static TRIAGE_HEADLINE: Mutex<Option<TriageHeadline>> = Mutex::new(None);

/// Triage card JSON for the JPEG at `image_path`. `hint` is the photographer's
/// own genre, or empty. Blocks for the whole model call, so run it on a
/// worker; the card's genre and emoji arrive earlier through
/// [`take_triage_headline`], tagged with `request_id` and with the emoji
/// drawn in a `badge_px` square.
pub fn gemma_triage(
    app: &AndroidApp,
    image_path: &Path,
    hint: &str,
    request_id: u64,
    badge_px: u32,
) -> Result<String, String> {
    let image_path = image_path.to_string_lossy().into_owned();
    let badge_px = i32::try_from(badge_px).map_err(|_| "badge size is too large".to_owned())?;
    with_gemma_bridge(app, |env, gemma_bridge| {
        let image_path = env.new_string(&image_path)?;
        let hint = env.new_string(hint)?;
        let value = env.call_method(
            gemma_bridge,
            jni::jni_str!("triage"),
            jni::jni_sig!((JString, JString, i64, i32) -> JString),
            &[
                JValue::Object(&image_path),
                JValue::Object(&hint),
                // Java has no unsigned long; the id comes back bit for bit.
                JValue::Long(request_id as i64),
                JValue::Int(badge_px),
            ],
        )?;
        java_string(env, value)
    })
    .map_err(|error| format!("Gemma could not triage the photo: {error:#}"))
}

/// The newest triage headline reported since the last call.
pub fn take_triage_headline() -> Option<TriageHeadline> {
    TRIAGE_HEADLINE.lock().ok()?.take()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_de_duecki_calibraw_GemmaBridge_nativeOnTriageHeadline<'local>(
    mut unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
    request_id: jni::sys::jlong,
    genre: JString<'local>,
    emoji: JString<'local>,
    badge: JByteArray<'local>,
) {
    unowned_env
        .with_env(|env| -> jni::errors::Result<()> {
            let badge = env.convert_byte_array(&badge)?;
            let headline = TriageHeadline {
                request_id: request_id as u64,
                genre: genre.to_string(),
                emoji: emoji.to_string(),
                badge: (!badge.is_empty()).then_some(badge),
            };
            if let Ok(mut slot) = TRIAGE_HEADLINE.lock() {
                *slot = Some(headline);
            }
            request_repaint();
            Ok(())
        })
        .resolve_with::<LogContextErrorAndDefault, _>(|| {
            "GemmaBridge.nativeOnTriageHeadline".to_owned()
        });
}

/// The text of the editor reply Gemma is writing, so far.
#[derive(Debug)]
pub struct EditorText {
    /// The `request_id` of the call the reply answers.
    pub request_id: u64,
    pub text: String,
}

/// The pieces of the newest reply since the last take, joined.
static EDITOR_TEXT: Mutex<Option<EditorText>> = Mutex::new(None);

/// Starts the AI editor's chat with `system_prompt`, replacing any earlier
/// one. Blocks for seconds, so run it on a worker.
pub fn gemma_editor_start(app: &AndroidApp, system_prompt: &str) -> Result<(), String> {
    with_gemma_bridge(app, |env, gemma_bridge| {
        let system_prompt = env.new_string(system_prompt)?;
        env.call_method(
            gemma_bridge,
            jni::jni_str!("startEditor"),
            jni::jni_sig!((JString) -> void),
            &[JValue::Object(&system_prompt)],
        )?;
        Ok(())
    })
    .map_err(|error| format!("Gemma could not start the editor chat: {error:#}"))
}

/// Sends the photographer's `text`, after the JPEG at `image_path` when there
/// is one, and returns Gemma's reply as JSON (`GemmaBridge.editorSend`).
/// Blocks for the whole reply, whose text arrives earlier through
/// [`take_editor_text`], tagged with `request_id`.
pub fn gemma_editor_send(
    app: &AndroidApp,
    request_id: u64,
    text: &str,
    image_path: Option<&Path>,
) -> Result<String, String> {
    let image_path = image_path
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();
    with_gemma_bridge(app, |env, gemma_bridge| {
        let text = env.new_string(text)?;
        let image_path = env.new_string(&image_path)?;
        let value = env.call_method(
            gemma_bridge,
            jni::jni_str!("editorSend"),
            jni::jni_sig!((i64, JString, JString) -> JString),
            &[
                // Java has no unsigned long; the id comes back bit for bit.
                JValue::Long(request_id as i64),
                JValue::Object(&text),
                JValue::Object(&image_path),
            ],
        )?;
        java_string(env, value)
    })
    .map_err(|error| format!("Gemma could not reply: {error:#}"))
}

/// Answers Gemma's tool calls with `responses_json`
/// (`GemmaBridge.editorToolResponses`) and returns its reply as
/// [`gemma_editor_send`] does.
pub fn gemma_editor_tool_responses(
    app: &AndroidApp,
    request_id: u64,
    responses_json: &str,
) -> Result<String, String> {
    with_gemma_bridge(app, |env, gemma_bridge| {
        let responses_json = env.new_string(responses_json)?;
        let value = env.call_method(
            gemma_bridge,
            jni::jni_str!("editorToolResponses"),
            jni::jni_sig!((i64, JString) -> JString),
            &[
                JValue::Long(request_id as i64),
                JValue::Object(&responses_json),
            ],
        )?;
        java_string(env, value)
    })
    .map_err(|error| format!("Gemma could not reply: {error:#}"))
}

/// Ends the editor chat and cancels its running reply. Returns at once.
pub fn gemma_editor_close(app: &AndroidApp) -> Result<(), String> {
    with_gemma_bridge(app, |env, gemma_bridge| {
        env.call_method(
            gemma_bridge,
            jni::jni_str!("closeEditor"),
            jni::jni_sig!(() -> void),
            &[],
        )?;
        Ok(())
    })
    .map_err(|error| format!("Gemma could not close the editor chat: {error:#}"))
}

/// The newest editor reply's text written since the last call.
pub fn take_editor_text() -> Option<EditorText> {
    EDITOR_TEXT.lock().ok()?.take()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_de_duecki_calibraw_GemmaBridge_nativeOnEditorText<'local>(
    mut unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
    request_id: jni::sys::jlong,
    text: JString<'local>,
) {
    unowned_env
        .with_env(|_env| -> jni::errors::Result<()> {
            let request_id = request_id as u64;
            let text = text.to_string();
            if let Ok(mut slot) = EDITOR_TEXT.lock() {
                match slot.as_mut() {
                    Some(pending) if pending.request_id == request_id => {
                        pending.text.push_str(&text);
                    }
                    _ => *slot = Some(EditorText { request_id, text }),
                }
            }
            request_repaint();
            Ok(())
        })
        .resolve_with::<LogContextErrorAndDefault, _>(|| {
            "GemmaBridge.nativeOnEditorText".to_owned()
        });
}

/// `emoji` drawn by Android's colour emoji font in a `size_px` square, as
/// premultiplied RGBA rows, or `None` when the font has no glyph for it.
pub fn render_emoji(
    app: &AndroidApp,
    emoji: &str,
    size_px: u32,
) -> Result<Option<Vec<u8>>, String> {
    let size = i32::try_from(size_px).map_err(|_| "emoji size is too large".to_owned())?;
    let rgba = with_gemma_bridge(app, |env, gemma_bridge| {
        let emoji = env.new_string(emoji)?;
        let value = env
            .call_method(
                gemma_bridge,
                jni::jni_str!("renderEmoji"),
                jni::jni_sig!((JString, i32) -> byte[]),
                &[JValue::Object(&emoji), JValue::Int(size)],
            )?
            .l()?;
        let pixels = env.cast_local::<JByteArray>(value)?;
        env.convert_byte_array(&pixels)
    })
    .map_err(|error| format!("Android could not draw the emoji: {error:#}"))?;
    Ok((!rgba.is_empty()).then_some(rgba))
}

fn with_gemma_bridge<T>(
    app: &AndroidApp,
    operation: impl FnOnce(&mut jni::Env<'_>, &JObject) -> jni::errors::Result<T>,
) -> jni::errors::Result<T> {
    with_activity(app, |env, activity| {
        let gemma_bridge = env
            .get_field(
                activity,
                jni::jni_str!("gemmaBridge"),
                jni::jni_sig!(de.duecki.calibraw.GemmaBridge),
            )?
            .l()?;
        operation(env, &gemma_bridge)
    })
}
