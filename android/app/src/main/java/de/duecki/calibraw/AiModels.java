package de.duecki.calibraw;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The AI editor's model, as Rust reaches it: forwards each call to the model the photographer
 * selected in Settings ({@link #select}), and reports what a model writes while it writes it.
 * Rust calls the public methods synchronously from its worker threads. The selection is kept for
 * the process, as the models keep what they load.
 */
final class AiModels {
    private static final String LOG_TAG = "CalibRaw";
    // The names Rust selects a model by (AiEditorModel in platform/service.rs).
    private static final String GEMMA = "gemma";
    private static final String CLAUDE = "claude";
    // A string field of the triage card once its closing quote is written.
    static final Pattern GENRE_FIELD =
            Pattern.compile("\"genre\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");
    static final Pattern EMOJI_FIELD =
            Pattern.compile("\"emoji\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");

    // Guarded by AiModels.class.
    private static String selectedName;
    // Written under AiModels.class; volatile so a call reads it without waiting for a selection.
    private static volatile AiModel selected;

    private final Activity activity;

    AiModels(Activity activity) {
        this.activity = activity;
    }

    /**
     * Makes {@code name}, "gemma" or "claude", the model every later call goes to, releasing the
     * one selected before and activating the new one in the background. Returns at once.
     */
    public void select(String name) {
        synchronized (AiModels.class) {
            if (name.equals(selectedName)) {
                return;
            }
            AiModel model;
            if (GEMMA.equals(name)) {
                model = new GemmaBridge(activity);
            } else if (CLAUDE.equals(name)) {
                model = new ClaudeBridge(activity);
            } else {
                Log.w(LOG_TAG, "Unknown AI model " + name);
                return;
            }
            AiModel previous = selected;
            selected = model;
            selectedName = name;
            if (previous != null) {
                previous.release();
            }
            model.activate();
            Log.i(LOG_TAG, "AI model selected: " + name);
        }
    }

    public String triage(String imagePath, String hint, long requestId, int badgePx)
            throws Exception {
        return model().triage(imagePath, hint, requestId, badgePx);
    }

    public void prepareEditor(String systemPrompt) throws IOException {
        model().prepareEditor(systemPrompt);
    }

    public void startEditorRequest(String systemPrompt) throws Exception {
        model().startEditorRequest(systemPrompt);
    }

    public String editorSend(long requestId, String text, String imagePath) throws Exception {
        return model().editorSend(requestId, text, imagePath);
    }

    public void closeEditor() {
        AiModel model = selected;
        if (model != null) {
            model.closeEditor();
        }
    }

    /**
     * How far Claude Code in Termux is set up, as one of {@link ClaudeBridge}'s status codes.
     * Blocks for up to half a minute while Termux starts, so Rust runs it on a worker.
     */
    public String claudeStatus() {
        return ClaudeBridge.status(activity);
    }

    /** Runs the setup {@code step} Settings offers for Claude Code (ClaudeBridge.setup). */
    public void claudeSetup(String step) throws Exception {
        ClaudeBridge.setup(activity, step);
    }

    /**
     * Draws {@code emoji} centred in a {@code sizePx} square with the system's colour emoji font,
     * as premultiplied RGBA rows. Returns an empty array when the font has no glyph for it.
     */
    public byte[] renderEmoji(String emoji, int sizePx) {
        return drawEmoji(emoji, sizePx);
    }

    private static AiModel model() throws IOException {
        AiModel model = selected;
        if (model == null) {
            throw new IOException("No AI model is selected");
        }
        return model;
    }

    /**
     * Reports a triage's genre and emoji, with the emoji drawn in a {@code badgePx} square, before
     * its card is done.
     */
    static void reportTriageHeadline(long requestId, String genre, String emoji, int badgePx) {
        nativeOnTriageHeadline(requestId, genre, emoji, drawEmoji(emoji, badgePx));
    }

    /** Reports the next piece of an editor reply's text while the model writes it. */
    static void reportEditorText(long requestId, String text) {
        nativeOnEditorText(requestId, text);
    }

    private static native void nativeOnTriageHeadline(
            long requestId, String genre, String emoji, byte[] badge);

    private static native void nativeOnEditorText(long requestId, String text);

    /** The value of a string field the partial triage card {@code text} has finished, or null. */
    static String completeField(String text, Pattern field) {
        Matcher match = field.matcher(text);
        if (!match.find()) {
            return null;
        }
        try {
            // The match keeps its JSON escapes; decoding them as a JSON string undoes them.
            return new JSONObject("{\"v\":\"" + match.group(1) + "\"}").getString("v");
        } catch (JSONException error) {
            return null;
        }
    }

    private static byte[] drawEmoji(String emoji, int sizePx) {
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        if (emoji == null || emoji.isEmpty() || sizePx <= 0 || !paint.hasGlyph(emoji)) {
            return new byte[0];
        }
        paint.setTextSize(sizePx * 0.8f);
        paint.setTextAlign(Paint.Align.CENTER);
        Paint.FontMetrics metrics = paint.getFontMetrics();
        Bitmap bitmap = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888);
        new Canvas(bitmap).drawText(
                emoji, sizePx / 2f, sizePx / 2f - (metrics.ascent + metrics.descent) / 2f, paint);
        // ARGB_8888 stores premultiplied bytes in R, G, B, A order.
        ByteBuffer pixels = ByteBuffer.allocate(sizePx * sizePx * 4);
        bitmap.copyPixelsToBuffer(pixels);
        bitmap.recycle();
        return pixels.array();
    }

    static String asset(Context context, String path) throws IOException {
        try (InputStream input = context.getAssets().open(path)) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            for (int read; (read = input.read(buffer)) != -1; ) {
                output.write(buffer, 0, read);
            }
            return output.toString(StandardCharsets.UTF_8.name());
        }
    }
}
