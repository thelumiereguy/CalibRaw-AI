//! The soft keyboard: which view takes it (`CalibRawActivity.setSoftKeyboard`)
//! and what it typed into a text field (`TextInputView`), which the UI thread
//! reads with [`take_typed_input`].

use super::*;

/// The field the soft keyboard types into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftKeyboard {
    /// No field: NativeActivity's own view keeps the keyboard.
    None,
    /// A text field, with swipe and voice typing.
    Text,
    /// A number field, with the number layout.
    Number,
}

/// One edit the soft keyboard made in a text field, at the UI's cursor.
#[derive(Debug, PartialEq, Eq)]
pub enum TypedInput {
    Text(String),
    /// Deletes the character before the cursor.
    Backspace,
    /// Deletes the character after the cursor.
    Delete,
    Enter,
}

/// Edits in the order the keyboard made them; Java appends, the UI drains.
static TYPED_INPUT: Mutex<Vec<TypedInput>> = Mutex::new(Vec::new());

/// Moves the soft keyboard to `field`'s view. winit still decides whether
/// the keyboard is shown.
pub fn set_soft_keyboard(app: &AndroidApp, field: SoftKeyboard) -> Result<(), String> {
    // The values CalibRawActivity.setSoftKeyboard knows.
    let field = match field {
        SoftKeyboard::None => 0,
        SoftKeyboard::Text => 1,
        SoftKeyboard::Number => 2,
    };
    with_activity(app, |env, activity| {
        env.call_method(
            activity,
            jni::jni_str!("setSoftKeyboard"),
            jni::jni_sig!((i32) -> void),
            &[JValue::Int(field)],
        )?;
        Ok(())
    })
    .map_err(|error| format!("could not move the Android soft keyboard: {error:#}"))
}

/// The edits typed since the last call, oldest first.
pub fn take_typed_input() -> Vec<TypedInput> {
    TYPED_INPUT
        .lock()
        .map(|mut queue| std::mem::take(&mut *queue))
        .unwrap_or_default()
}

fn queue_typed_input(edits: impl IntoIterator<Item = TypedInput>) {
    if let Ok(mut queue) = TYPED_INPUT.lock() {
        queue.extend(edits);
    }
    request_repaint();
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_de_duecki_calibraw_TextInputView_nativeOnTextEdited<'local>(
    mut unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
    deleted_code_points: jni::sys::jint,
    inserted: JString<'local>,
) {
    unowned_env
        .with_env(|_env| -> jni::errors::Result<()> {
            let inserted = inserted.to_string();
            let deletions = (0..deleted_code_points.max(0)).map(|_| TypedInput::Backspace);
            let text = (!inserted.is_empty()).then_some(TypedInput::Text(inserted));
            queue_typed_input(deletions.chain(text));
            Ok(())
        })
        .resolve_with::<LogContextErrorAndDefault, _>(|| {
            "TextInputView.nativeOnTextEdited".to_owned()
        });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_de_duecki_calibraw_TextInputView_nativeOnKeyTyped<'local>(
    _unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
    key_code: jni::sys::jint,
) {
    // Android's KeyEvent codes, the only ones TextInputView sends.
    const KEYCODE_ENTER: jni::sys::jint = 66;
    const KEYCODE_DEL: jni::sys::jint = 67;
    const KEYCODE_FORWARD_DEL: jni::sys::jint = 112;
    let edit = match key_code {
        KEYCODE_DEL => TypedInput::Backspace,
        KEYCODE_FORWARD_DEL => TypedInput::Delete,
        KEYCODE_ENTER => TypedInput::Enter,
        _ => return,
    };
    queue_typed_input([edit]);
}
