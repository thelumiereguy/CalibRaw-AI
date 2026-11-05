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
