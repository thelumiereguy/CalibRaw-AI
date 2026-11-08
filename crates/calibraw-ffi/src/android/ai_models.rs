//! The AI editor's Java side through `AiModels`, its JNI facade, which hands
//! each call to the class that owns it: the selected model (`AiModelRouter`),
//! on-device Gemma (`GemmaBridge`, LiteRT-LM) or Claude Code in Termux
//! (`ClaudeBridge`), and Claude Code's setup (`ClaudeSetup`).

use super::*;
use jni::objects::JByteArray;

/// Makes `name` ("gemma" or "claude") the model the later calls go to, with
/// Claude Code running `claude_model` (a `--model` value), releasing the one
/// selected before. Returns at once.
pub fn select_ai_model(app: &AndroidApp, name: &str, claude_model: &str) -> Result<(), String> {
    with_ai_models(app, |env, ai_models| {
        let name = env.new_string(name)?;
        let claude_model = env.new_string(claude_model)?;
        env.call_method(
            ai_models,
            jni::jni_str!("select"),
            jni::jni_sig!((JString, JString) -> void),
            &[JValue::Object(&name), JValue::Object(&claude_model)],
        )?;
        Ok(())
    })
    .map_err(|error| format!("Could not select the AI model: {error:#}"))
}

/// How far Claude Code in Termux is set up, as a `ClaudeSetup.Status` code.
/// Blocks while Termux starts, up to about a minute, so run it on a worker.
pub fn claude_code_status(app: &AndroidApp) -> Result<String, String> {
    with_ai_models(app, |env, ai_models| {
        let value = env.call_method(
            ai_models,
            jni::jni_str!("claudeStatus"),
            jni::jni_sig!(() -> JString),
            &[],
        )?;
        java_string(env, value)
    })
    .map_err(|error| format!("Could not check Claude Code: {error:#}"))
}

/// Runs the Claude Code setup `step` ("permission", "install" or "sign_in").
/// Returns once Termux has the command.
pub fn claude_code_setup(app: &AndroidApp, step: &str) -> Result<(), String> {
    with_ai_models(app, |env, ai_models| {
        let step = env.new_string(step)?;
        env.call_method(
            ai_models,
            jni::jni_str!("claudeSetup"),
            jni::jni_sig!((JString) -> void),
            &[JValue::Object(&step)],
        )?;
        Ok(())
    })
    .map_err(|error| format!("Could not set up Claude Code: {error:#}"))
}

/// Whether the default network reaches the internet, as Android validated
/// it. Returns at once.
pub fn internet_available(app: &AndroidApp) -> Result<bool, String> {
    with_ai_models(app, |env, ai_models| {
        env.call_method(
            ai_models,
            jni::jni_str!("online"),
            jni::jni_sig!(() -> bool),
            &[],
        )?
        .z()
    })
    .map_err(|error| format!("Could not check the network: {error:#}"))
}

/// A triage's genre and emoji, reported while the model still writes the
/// rest of its card.
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
pub fn ai_triage(
    app: &AndroidApp,
    image_path: &Path,
    hint: &str,
    request_id: u64,
    badge_px: u32,
) -> Result<String, String> {
    let image_path = image_path.to_string_lossy().into_owned();
    let badge_px = i32::try_from(badge_px).map_err(|_| "badge size is too large".to_owned())?;
    with_ai_models(app, |env, ai_models| {
        let image_path = env.new_string(&image_path)?;
        let hint = env.new_string(hint)?;
        let value = env.call_method(
            ai_models,
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
    .map_err(|error| {
        format!(
            "The AI model could not triage the photo: {}",
            chat_error(error)
        )
    })
}

/// The newest triage headline reported since the last call.
pub fn take_triage_headline() -> Option<TriageHeadline> {
    TRIAGE_HEADLINE.lock().ok()?.take()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_de_duecki_calibraw_AiModels_nativeOnTriageHeadline<'local>(
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
            "AiModels.nativeOnTriageHeadline".to_owned()
        });
}

/// The text of the editor reply the model is writing, so far.
#[derive(Debug)]
pub struct EditorText {
    /// The `request_id` of the call the reply answers.
    pub request_id: u64,
    pub text: String,
}

/// The pieces of the newest reply since the last take, joined.
static EDITOR_TEXT: Mutex<Option<EditorText>> = Mutex::new(None);

/// Has the model ready a conversation with the AI editor's `system_prompt` in
/// the background, so the open photo's first request does not wait for it.
/// Returns at once.
pub fn ai_editor_prepare(app: &AndroidApp, system_prompt: &str) -> Result<(), String> {
    with_ai_models(app, |env, ai_models| {
        let system_prompt = env.new_string(system_prompt)?;
        env.call_method(
            ai_models,
            jni::jni_str!("prepareEditor"),
            jni::jni_sig!((JString) -> void),
            &[JValue::Object(&system_prompt)],
        )?;
        Ok(())
    })
    .map_err(|error| format!("The AI model could not prepare the editor: {error:#}"))
}

/// Starts an AI editor request on a fresh conversation that holds only
/// `system_prompt`, replacing the earlier request's. Blocks while the model
/// reads a prompt it has not read yet or Claude Code starts, so run it on a
/// worker.
pub fn ai_editor_start_request(app: &AndroidApp, system_prompt: &str) -> Result<(), String> {
    with_ai_models(app, |env, ai_models| {
        let system_prompt = env.new_string(system_prompt)?;
        env.call_method(
            ai_models,
            jni::jni_str!("startEditorRequest"),
            jni::jni_sig!((JString) -> void),
            &[JValue::Object(&system_prompt)],
        )?;
        Ok(())
    })
    .map_err(|error| {
        format!(
            "The AI model could not start the request: {}",
            chat_error(error)
        )
    })
}

/// Sends `text` to the running request, after the JPEGs at `images`, and
/// returns the model's reply as JSON (`AiModel.editorSend`, which takes the
/// paths one per line). Blocks for the whole reply, whose text arrives
/// earlier through [`take_editor_text`], tagged with `request_id`.
pub fn ai_editor_send(
    app: &AndroidApp,
    request_id: u64,
    text: &str,
    images: &[&Path],
) -> Result<String, String> {
    let image_paths = images
        .iter()
        .map(|path| path.to_string_lossy())
        .collect::<Vec<_>>()
        .join("\n");
    with_ai_models(app, |env, ai_models| {
        let text = env.new_string(text)?;
        let image_paths = env.new_string(&image_paths)?;
        let value = env.call_method(
            ai_models,
            jni::jni_str!("editorSend"),
            jni::jni_sig!((i64, JString, JString) -> JString),
            &[
                // Java has no unsigned long; the id comes back bit for bit.
                JValue::Long(request_id as i64),
                JValue::Object(&text),
                JValue::Object(&image_paths),
            ],
        )?;
        java_string(env, value)
    })
    .map_err(|error| format!("The AI model could not reply: {}", chat_error(error)))
}

/// Closes the AI editor request's conversation and cancels its running reply.
/// Returns at once.
pub fn ai_editor_close(app: &AndroidApp) -> Result<(), String> {
    with_ai_models(app, |env, ai_models| {
        env.call_method(
            ai_models,
            jni::jni_str!("closeEditor"),
            jni::jni_sig!(() -> void),
            &[],
        )?;
        Ok(())
    })
    .map_err(|error| format!("The AI model could not close the editor chat: {error:#}"))
}

/// Tells the selected model, and any selected later, whether Develop shows:
/// Claude Code keeps processes in Termux only while it does. Returns at once.
pub fn ai_editor_shown(app: &AndroidApp, shown: bool) -> Result<(), String> {
    with_ai_models(app, |env, ai_models| {
        env.call_method(
            ai_models,
            jni::jni_str!("editorShown"),
            jni::jni_sig!((bool) -> void),
            &[JValue::Bool(shown)],
        )?;
        Ok(())
    })
    .map_err(|error| format!("Could not tell the AI model whether Develop shows: {error:#}"))
}

/// Has Termux end the Claude Code processes that earlier runs of the app left
/// behind when Android killed it, where the app may run commands in Termux.
/// Returns at once.
pub fn end_stray_claude_processes(app: &AndroidApp) -> Result<(), String> {
    with_ai_models(app, |env, ai_models| {
        env.call_method(
            ai_models,
            jni::jni_str!("endStrayClaudeProcesses"),
            jni::jni_sig!(() -> void),
            &[],
        )?;
        Ok(())
    })
    .map_err(|error| format!("Could not end stray Claude Code processes: {error:#}"))
}

/// The newest editor reply's text written since the last call.
pub fn take_editor_text() -> Option<EditorText> {
    EDITOR_TEXT.lock().ok()?.take()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_de_duecki_calibraw_AiModels_nativeOnEditorText<'local>(
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
        .resolve_with::<LogContextErrorAndDefault, _>(|| "AiModels.nativeOnEditorText".to_owned());
}

/// `emoji` drawn by Android's colour emoji font in a `size_px` square, as
/// premultiplied RGBA rows, or `None` when the font has no glyph for it.
pub fn render_emoji(
    app: &AndroidApp,
    emoji: &str,
    size_px: u32,
) -> Result<Option<Vec<u8>>, String> {
    let size = i32::try_from(size_px).map_err(|_| "emoji size is too large".to_owned())?;
    let rgba = with_ai_models(app, |env, ai_models| {
        let emoji = env.new_string(emoji)?;
        let value = env
            .call_method(
                ai_models,
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

/// `error` as the AI editor's chat shows it: a Java exception by its message
/// alone, since the photographer reads it, with its stack trace logged. One
/// without a message (jni writes a null one as `<NULL>`) goes by its class.
fn chat_error(error: jni::errors::Error) -> String {
    match &error {
        jni::errors::Error::CaughtJavaException { name, msg, .. } => {
            log::warn!("{error:#}");
            if msg.is_empty() || msg == "<NULL>" {
                name.clone()
            } else {
                msg.clone()
            }
        }
        _ => format!("{error:#}"),
    }
}

fn with_ai_models<T>(
    app: &AndroidApp,
    operation: impl FnOnce(&mut jni::Env<'_>, &JObject) -> jni::errors::Result<T>,
) -> jni::errors::Result<T> {
    with_activity(app, |env, activity| {
        let ai_models = env
            .get_field(
                activity,
                jni::jni_str!("aiModels"),
                jni::jni_sig!(de.duecki.calibraw.AiModels),
            )?
            .l()?;
        operation(env, &ai_models)
    })
}
