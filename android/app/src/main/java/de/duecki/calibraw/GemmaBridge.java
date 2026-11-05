package de.duecki.calibraw;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.os.SystemClock;
import android.util.Log;

import com.google.ai.edge.litertlm.Backend;
import com.google.ai.edge.litertlm.Content;
import com.google.ai.edge.litertlm.Contents;
import com.google.ai.edge.litertlm.Conversation;
import com.google.ai.edge.litertlm.ConversationConfig;
import com.google.ai.edge.litertlm.Engine;
import com.google.ai.edge.litertlm.EngineConfig;
import com.google.ai.edge.litertlm.ExperimentalFlags;
import com.google.ai.edge.litertlm.Message;
import com.google.ai.edge.litertlm.MessageCallback;
import com.google.ai.edge.litertlm.ResponseFormat;
import com.google.ai.edge.litertlm.SamplerConfig;
import com.google.ai.edge.litertlm.ThinkingConfig;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Runs Gemma 4 E4B on the device through LiteRT-LM. Rust calls the public methods synchronously
 * from its worker threads. The engine is kept for the process and shared by every activity, and
 * one call uses it at a time.
 */
final class GemmaBridge {
    private static final String LOG_TAG = "CalibRaw";
    private static final String MODEL_FILE = "models/gemma-4-E4B-it.litertlm";
    private static final String TRIAGE_REQUEST = "Triage this photo.";
    // Triage wants a stable card, so sampling stays close to greedy.
    private static final SamplerConfig TRIAGE_SAMPLER = new SamplerConfig(1, 1.0, 0.0, 0);
    // A triage normally takes about ten seconds.
    private static final long TRIAGE_TIMEOUT_SECONDS = 120;
    // A string field of the card once its closing quote is written.
    private static final Pattern GENRE_FIELD =
            Pattern.compile("\"genre\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");
    private static final Pattern EMOJI_FIELD =
            Pattern.compile("\"emoji\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");

    // Guarded by GemmaBridge.class.
    private static Engine engine;
    // The next triage's conversation, made ahead of time since making one takes seconds. A
    // conversation keeps its history, so each triage takes this one and a new one is made after.
    // Guarded by GemmaBridge.class.
    private static Conversation nextTriage;

    private final Context context;

    GemmaBridge(Context context) {
        this.context = context.getApplicationContext();
    }

    /**
     * Loads the engine and the first triage's conversation on a background thread, so the first
     * photo's triage does not wait for them. A call that arrives meanwhile waits for the load.
     */
    void preload() {
        prepareTriageInBackground("Gemma preload");
    }

    /**
     * Returns the triage card for the JPEG at {@code imagePath} as JSON, shaped by
     * assets/gemma/triage_schema.json. {@code hint} is the photographer's own genre, or empty.
     *
     * <p>The card's genre and emoji come first and are reported through
     * {@link #nativeOnTriageHeadline} as soon as Gemma has written them, tagged with
     * {@code requestId} and with the emoji drawn in a {@code badgePx} square; the rest of the
     * card takes several seconds more.
     */
    public String triage(String imagePath, String hint, long requestId, int badgePx)
            throws Exception {
        try {
            synchronized (GemmaBridge.class) {
                return triageLocked(imagePath, hint, requestId, badgePx);
            }
        } finally {
            prepareTriageInBackground("Gemma triage preparation");
        }
    }

    private String triageLocked(String imagePath, String hint, long requestId, int badgePx)
            throws Exception {
        String request = hint == null || hint.isEmpty()
                ? TRIAGE_REQUEST
                : TRIAGE_REQUEST + " The photographer says the genre is: " + hint + ".";
        long started = SystemClock.elapsedRealtime();
        Conversation conversation = nextTriage != null ? nextTriage : createTriage(context);
        nextTriage = null;
        try (conversation) {
            Log.i(LOG_TAG, "Gemma triage conversation ready in "
                    + (SystemClock.elapsedRealtime() - started) + " ms");
            StringBuilder card = new StringBuilder();
            AtomicBoolean headlineReported = new AtomicBoolean();
            AtomicReference<Throwable> failure = new AtomicReference<>();
            CountDownLatch finished = new CountDownLatch(1);
            conversation.sendMessageAsync(
                    Contents.Companion.of(
                            new Content.ImageFile(imagePath), new Content.Text(request)),
                    new MessageCallback() {
                        @Override
                        public void onMessage(Message message) {
                            String text;
                            // Each message is the next piece of the card.
                            synchronized (card) {
                                text = card.append(message.toString()).toString();
                            }
                            if (headlineReported.get()) {
                                return;
                            }
                            String genre = completeField(text, GENRE_FIELD);
                            String emoji = completeField(text, EMOJI_FIELD);
                            if (genre != null && emoji != null
                                    && headlineReported.compareAndSet(false, true)) {
                                Log.i(LOG_TAG, "Gemma triage headline after "
                                        + (SystemClock.elapsedRealtime() - started) + " ms");
                                nativeOnTriageHeadline(
                                        requestId, genre, emoji, renderEmoji(emoji, badgePx));
                            }
                        }

                        @Override
                        public void onDone() {
                            finished.countDown();
                        }

                        @Override
                        public void onError(Throwable error) {
                            failure.set(error);
                            finished.countDown();
                        }
                    },
                    Collections.emptyMap(),
                    null,
                    null,
                    null,
                    null,
                    null,
                    ResponseFormat.json(asset(context, "gemma/triage_schema.json")));
            if (!finished.await(TRIAGE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                conversation.cancelProcess();
                throw new IOException(
                        "Gemma triage did not finish in " + TRIAGE_TIMEOUT_SECONDS + " s");
            }
            if (failure.get() != null) {
                throw new IOException("Gemma triage failed", failure.get());
            }
            String text;
            synchronized (card) {
                text = card.toString();
            }
            Log.i(LOG_TAG, "Gemma triage took " + (SystemClock.elapsedRealtime() - started)
                    + " ms: " + text);
            Log.i(LOG_TAG, "Gemma triage benchmark: " + conversation.getBenchmarkInfo());
            return text;
        }
    }

    /** The value of a string field the partial card {@code text} has finished, or null. */
    private static String completeField(String text, Pattern field) {
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

    /** Reports a triage's genre and emoji, with the emoji drawn, before its card is done. */
    private static native void nativeOnTriageHeadline(
            long requestId, String genre, String emoji, byte[] badge);

    private void prepareTriageInBackground(String threadName) {
        Thread preparer = new Thread(() -> {
            try {
                synchronized (GemmaBridge.class) {
                    if (nextTriage == null) {
                        nextTriage = createTriage(context);
                    }
                }
            } catch (Exception error) {
                Log.w(LOG_TAG, "Could not prepare Gemma triage", error);
            }
        }, threadName);
        preparer.setDaemon(true);
        preparer.start();
    }

    // Caller holds GemmaBridge.class.
    private static Conversation createTriage(Context context) throws IOException {
        Engine loaded = engine(context);
        long started = SystemClock.elapsedRealtime();
        Conversation conversation = loaded.createConversation(new ConversationConfig(
                Contents.Companion.of(asset(context, "gemma/triage_system.md")),
                Collections.emptyList(),
                Collections.emptyList(),
                TRIAGE_SAMPLER,
                true,
                null,
                Collections.emptyMap(),
                null,
                false,
                null,
                new ThinkingConfig(false, 0),
                true));
        Log.i(LOG_TAG, "Gemma triage conversation created in "
                + (SystemClock.elapsedRealtime() - started) + " ms");
        return conversation;
    }

    /**
     * Draws {@code emoji} centred in a {@code sizePx} square with the system's colour emoji font,
     * as premultiplied RGBA rows. Returns an empty array when the font has no glyph for it.
     */
    public byte[] renderEmoji(String emoji, int sizePx) {
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

    private static synchronized Engine engine(Context context) throws IOException {
        if (engine != null) {
            return engine;
        }
        File model = new File(mediaRoot(context), MODEL_FILE);
        if (!model.isFile()) {
            throw new IOException("Gemma model not found at " + model);
        }
        // The GPU program cache is slow to rebuild, so it lives in files, not cache.
        File cacheDir = new File(context.getFilesDir(), "litertlm-cache");
        if (!cacheDir.isDirectory() && !cacheDir.mkdirs()) {
            throw new IOException("Could not create " + cacheDir);
        }
        long started = SystemClock.elapsedRealtime();
        // Reports prefill and decode speed for the triage log.
        ExperimentalFlags.INSTANCE.setEnableBenchmark(true);
        Engine created = new Engine(new EngineConfig(
                model.getAbsolutePath(),
                new Backend.GPU(),
                new Backend.GPU(),
                null,
                null,
                null,
                cacheDir.getAbsolutePath()));
        created.initialize();
        Log.i(LOG_TAG, "Gemma engine loaded in " + (SystemClock.elapsedRealtime() - started)
                + " ms");
        engine = created;
        return created;
    }

    private static File mediaRoot(Context context) throws IOException {
        File[] mediaDirectories = context.getExternalMediaDirs();
        if (mediaDirectories != null) {
            for (File directory : mediaDirectories) {
                if (directory != null) {
                    return directory;
                }
            }
        }
        throw new IOException("Android shared media storage is unavailable");
    }

    private static String asset(Context context, String path) throws IOException {
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
