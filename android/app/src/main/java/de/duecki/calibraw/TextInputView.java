package de.duecki.calibraw;

import android.content.Context;
import android.text.Editable;
import android.text.InputType;
import android.text.Selection;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.BaseInputConnection;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;

/**
 * Takes the soft keyboard's focus while the UI edits text, so the keyboard offers what it offers a
 * text field: swipe and voice typing, suggestions and corrections. NativeActivity's own view has
 * no input connection, so there the keyboard falls back to single key presses. The view is never
 * drawn: NativeActivity owns the window surface.
 *
 * <p>Typed text reaches the UI through {@link #nativeOnTextEdited} and {@link #nativeOnKeyTyped},
 * in order, not as key events: key events carry only characters with a key, and could overtake
 * the text.
 */
final class TextInputView extends View {
    TextInputView(Context context) {
        super(context);
        setFocusable(true);
        setFocusableInTouchMode(true);
    }

    @Override
    public boolean onCheckIsTextEditor() {
        return true;
    }

    @Override
    public InputConnection onCreateInputConnection(EditorInfo outAttrs) {
        outAttrs.inputType = InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_AUTO_CORRECT;
        outAttrs.imeOptions = EditorInfo.IME_ACTION_DONE
                | EditorInfo.IME_FLAG_NO_EXTRACT_UI
                | EditorInfo.IME_FLAG_NO_FULLSCREEN;
        outAttrs.initialSelStart = 0;
        outAttrs.initialSelEnd = 0;
        return new TypedTextConnection(this);
    }

    /**
     * The keyboard edits {@link #typed}, the text typed since the field took the keyboard, as it
     * would a text field's; the keyboard reads it to place spaces and suggestions. After each edit
     * the UI gets the difference to what it was sent before: deletions back to the first changed
     * character, then the new text after it. Text the keyboard is still composing, such as a swiped
     * word or a dictated phrase, is sent too and replaced when the keyboard changes it.
     *
     * <p>This assumes the UI's cursor stays at the end of the sent text, as it does while the
     * photographer types; text the field held before it took the keyboard is not part of
     * {@link #typed}, and a backspace past its start goes straight to the UI. An edit through a
     * method not overridden here, such as {@code replaceText}, reaches the UI with the next one.
     */
    private static final class TypedTextConnection extends BaseInputConnection {
        private final Editable typed = Editable.Factory.getInstance().newEditable("");
        // What the UI has of `typed`, as of the last edit sent.
        private String sent = "";

        TypedTextConnection(View view) {
            super(view, true);
        }

        @Override
        public Editable getEditable() {
            return typed;
        }

        @Override
        public boolean commitText(CharSequence text, int newCursorPosition) {
            boolean handled = super.commitText(text, newCursorPosition);
            sendEdit();
            return handled;
        }

        @Override
        public boolean setComposingText(CharSequence text, int newCursorPosition) {
            boolean handled = super.setComposingText(text, newCursorPosition);
            sendEdit();
            return handled;
        }

        @Override
        public boolean setComposingRegion(int start, int end) {
            boolean handled = super.setComposingRegion(start, end);
            sendEdit();
            return handled;
        }

        @Override
        public boolean finishComposingText() {
            boolean handled = super.finishComposingText();
            sendEdit();
            return handled;
        }

        @Override
        public boolean deleteSurroundingText(int beforeLength, int afterLength) {
            boolean handled = super.deleteSurroundingText(beforeLength, afterLength);
            sendEdit();
            return handled;
        }

        @Override
        public boolean deleteSurroundingTextInCodePoints(int beforeLength, int afterLength) {
            boolean handled = super.deleteSurroundingTextInCodePoints(beforeLength, afterLength);
            sendEdit();
            return handled;
        }

        /**
         * Keys the keyboard sends instead of editing, such as backspace with nothing composed and
         * Enter (also the Done action, which {@link BaseInputConnection#performEditorAction} sends
         * as Enter). Other keys reach NativeActivity's input queue as before.
         */
        @Override
        public boolean sendKeyEvent(KeyEvent event) {
            int keyCode = event.getKeyCode();
            boolean editKey = keyCode == KeyEvent.KEYCODE_DEL
                    || keyCode == KeyEvent.KEYCODE_FORWARD_DEL
                    || keyCode == KeyEvent.KEYCODE_ENTER;
            if (!editKey) {
                return super.sendKeyEvent(event);
            }
            if (event.getAction() != KeyEvent.ACTION_DOWN) {
                return true;
            }
            int cursor = Selection.getSelectionEnd(typed);
            if (keyCode == KeyEvent.KEYCODE_DEL && cursor > 0) {
                int start = Character.offsetByCodePoints(typed, cursor, -1);
                typed.delete(start, cursor);
                sendEdit();
            } else {
                sendEdit();
                nativeOnKeyTyped(keyCode);
            }
            return true;
        }

        private void sendEdit() {
            String now = typed.toString();
            int common = 0;
            int limit = Math.min(now.length(), sent.length());
            while (common < limit && now.charAt(common) == sent.charAt(common)) {
                common++;
            }
            // Never split a surrogate pair: the UI deletes whole code points.
            if (common > 0 && Character.isHighSurrogate(now.charAt(common - 1))) {
                common--;
            }
            int deleted = sent.codePointCount(common, sent.length());
            String inserted = now.substring(common);
            if (deleted > 0 || !inserted.isEmpty()) {
                nativeOnTextEdited(deleted, inserted);
            }
            sent = now;
        }
    }

    /** Deletes `deletedCodePoints` before the UI's cursor, then types `inserted` there. */
    private static native void nativeOnTextEdited(int deletedCodePoints, String inserted);

    /** Presses `keyCode` (`KEYCODE_DEL`, `KEYCODE_FORWARD_DEL` or `KEYCODE_ENTER`) in the UI. */
    private static native void nativeOnKeyTyped(int keyCode);
}
