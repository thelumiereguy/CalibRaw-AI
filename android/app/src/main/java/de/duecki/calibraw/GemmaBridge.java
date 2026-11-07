package de.duecki.calibraw;

import android.content.Context;
import android.content.pm.ApplicationInfo;
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
import com.google.ai.edge.litertlm.JsonConvertersKt;
import com.google.ai.edge.litertlm.LiteRtLmJniException;
import com.google.ai.edge.litertlm.Message;
import com.google.ai.edge.litertlm.MessageCallback;
import com.google.ai.edge.litertlm.ResponseFormat;
import com.google.ai.edge.litertlm.SamplerConfig;
import com.google.ai.edge.litertlm.ThinkingConfig;
import com.google.ai.edge.litertlm.ToolCall;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Runs Gemma 4 E4B on the device through LiteRT-LM: the triage of each opened photo and the AI
 * editor's chat about it. Rust calls the public methods synchronously from its worker threads,
 * except {@link #closeEditor}, which returns at once. The engine is kept for the process and
 * shared by every activity, and one call uses it at a time.
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
    // The reply LiteRT-LM quotes in its message when it cannot parse a tool call.
    private static final Pattern FULL_RESPONSE = Pattern.compile("full response: (.*)");
    // The editor's replies are short, so greedy decoding keeps them predictable.
    private static final SamplerConfig EDITOR_SAMPLER = new SamplerConfig(1, 1.0, 0.0, 0);
    // An editor turn normally takes a few seconds.
    private static final long EDITOR_TURN_TIMEOUT_SECONDS = 120;
    // Tokens a conversation holds. A phase of an editor request took up to 2,447 (its system
    // prompt, a preview per message). LiteRT-LM's GPU backend reserves the whole context in every
    // layer, sliding-window ones included (LiteRT-LM #4096), so the reservation grows with this:
    // 16384 ran the phone out of memory after a triage, and 8192 left it swapping.
    private static final int MAX_NUM_TOKENS = 4096;

    // Guarded by GemmaBridge.class.
    private static Engine engine;
    // The next triage's conversation, made ahead of time since making one takes seconds. A
    // conversation keeps its history, so each triage takes this one and a new one is made after.
    // Guarded by GemmaBridge.class.
    private static Conversation nextTriage;
    // The editor's system prompt, already read by Gemma, by prompt. It names no photo, so it is
    // kept for the process, and opening a photo costs only its triage.
    // Each request runs on a clone of its prompt's: a clone took milliseconds, while a new
    // conversation's first prefill took 16-23 s after the process's first one. Guarded by
    // GemmaBridge.class.
    private static final Map<String, Conversation> editorBases = new LinkedHashMap<>();
    // Base conversations kept; each reserves MAX_NUM_TOKENS. A request runs on one prompt.
    private static final int MAX_EDITOR_BASES = 1;
    // The running request's conversation, or null. Written under GemmaBridge.class; volatile so
    // closeEditor can cancel its running turn without waiting for that lock.
    private static volatile Conversation editor;

    // For the evals (research/gemma-evals) only: the phone's log quota drops most of the app's
    // lines while LiteRT logs, so each triage card and the editor's messages, replies and request
    // ends are appended to files/gemma-trace.txt, in a debuggable build and only while
    // files/gemma-trace.on exists, which the evals create for their run. A release build never
    // traces. The app adds what each request set (app/ai/gemma/editor.rs) while the same switch
    // exists, which only run-as, and so only a debuggable build, can make. Null when the build is
    // not debuggable.
    private static volatile File traceFile;
    private static final String TRACE_SWITCH = "gemma-trace.on";

    private final Context context;

    GemmaBridge(Context context) {
        this.context = context.getApplicationContext();
        boolean debuggable =
                (this.context.getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
        traceFile = debuggable ? new File(this.context.getFilesDir(), "gemma-trace.txt") : null;
    }

    private static void trace(String line) {
        File file = traceFile;
        if (file == null || !new File(file.getParentFile(), TRACE_SWITCH).isFile()) {
            return;
        }
        synchronized (TRACE_LOCK) {
            try (PrintWriter out = new PrintWriter(new FileWriter(file, true))) {
                out.println(SystemClock.elapsedRealtime() + " " + line.replace('\n', '|'));
            } catch (IOException error) {
                Log.w(LOG_TAG, "Could not write the Gemma trace", error);
            }
        }
    }

    private static final Object TRACE_LOCK = new Object();

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
            trace("TRIAGE " + (SystemClock.elapsedRealtime() - started) + " ms " + text);
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
     * Has Gemma read the editor's {@code systemPrompt} on a background thread, so the open
     * photo's first request does not wait for it. Returns at once; a request that arrives
     * meanwhile waits for it.
     */
    public void prepareEditor(String systemPrompt) {
        Thread preparer = new Thread(() -> {
            try {
                synchronized (GemmaBridge.class) {
                    editorBase(context, systemPrompt);
                }
            } catch (Exception error) {
                Log.w(LOG_TAG, "Could not prepare the Gemma editor", error);
            }
        }, "Gemma editor preparation");
        preparer.setDaemon(true);
        preparer.start();
    }

    /**
     * Starts a request on a fresh conversation that holds only {@code systemPrompt}, replacing
     * the earlier request's. Blocks while Gemma reads a prompt it has not read yet.
     */
    public void startEditorRequest(String systemPrompt) throws Exception {
        synchronized (GemmaBridge.class) {
            Conversation previous = editor;
            editor = null;
            if (previous != null) {
                previous.close();
            }
            Conversation base = editorBase(context, systemPrompt);
            long started = SystemClock.elapsedRealtime();
            editor = base.clone();
            Log.i(LOG_TAG, "Gemma editor request conversation cloned in "
                    + (SystemClock.elapsedRealtime() - started) + " ms");
            trace("PHASE " + systemPrompt.substring(0, Math.min(60, systemPrompt.length())));
        }
    }

    // Caller holds GemmaBridge.class.
    private static Conversation editorBase(Context context, String systemPrompt)
            throws IOException {
        Conversation known = editorBases.get(systemPrompt);
        if (known != null) {
            return known;
        }
        // A prompt the app no longer sends, such as one from before an update.
        while (editorBases.size() >= MAX_EDITOR_BASES) {
            String oldest = editorBases.keySet().iterator().next();
            editorBases.remove(oldest).close();
        }
        Engine loaded = engine(context);
        long started = SystemClock.elapsedRealtime();
        // No tools are declared: the prompt lists them and Gemma writes its calls as text lines,
        // which the app parses (app/ai/calls.rs). The preface is read now, not at the first
        // request.
        Conversation base = loaded.createConversation(new ConversationConfig(
                Contents.Companion.of(systemPrompt),
                Collections.emptyList(),
                Collections.emptyList(),
                EDITOR_SAMPLER,
                false,
                null,
                Collections.emptyMap(),
                null,
                true,
                null,
                new ThinkingConfig(false, 0),
                false));
        Log.i(LOG_TAG, "Gemma editor prompt read in " + (SystemClock.elapsedRealtime() - started)
                + " ms, " + base.getTokenCount() + " tokens");
        editorBases.put(systemPrompt, base);
        return base;
    }

    /**
     * Sends the photographer's turn, {@code text} with the JPEG at {@code imagePath} before it
     * when that is not empty, and returns Gemma's reply as
     * {@code {"text": ..., "tool_calls": [{"name": ..., "arguments": {...}}]}}, the calls being
     * those Gemma wrote in its own call syntax that LiteRT-LM parsed. The reply's text also
     * streams through {@link #nativeOnEditorText}, tagged with {@code requestId}.
     */
    public String editorSend(long requestId, String text, String imagePath) throws Exception {
        Contents contents = imagePath == null || imagePath.isEmpty()
                ? Contents.Companion.of(text)
                : Contents.Companion.of(new Content.ImageFile(imagePath), new Content.Text(text));
        trace("SEND " + text);
        return editorTurn(requestId, Message.Companion.user(contents));
    }

    /**
     * Ends the editor's request, cancelling a running turn, which then fails: when a request
     * finishes, so its conversation does not hold memory until the next, and when another photo
     * opens. Returns at once; the request's conversation is closed on a background thread once no
     * call uses it, and the prompts' are kept for the next request.
     */
    public void closeEditor() {
        trace("CLOSE");
        Conversation closing = editor;
        if (closing != null) {
            try {
                closing.cancelProcess();
            } catch (IllegalStateException alreadyClosed) {
                // startEditorRequest replaced and closed it meanwhile.
            }
        }
        Thread closer = new Thread(() -> {
            synchronized (GemmaBridge.class) {
                // A request started meanwhile is not this one and stays open.
                if (closing != null && editor == closing) {
                    editor = null;
                    closing.close();
                }
            }
        }, "Gemma editor close");
        closer.setDaemon(true);
        closer.start();
    }

    private String editorTurn(long requestId, Message message) throws Exception {
        synchronized (GemmaBridge.class) {
            Conversation conversation = editor;
            if (conversation == null) {
                throw new IOException("The AI editor chat is closed");
            }
            long started = SystemClock.elapsedRealtime();
            StringBuilder reply = new StringBuilder();
            List<ToolCall> toolCalls = Collections.synchronizedList(new ArrayList<>());
            AtomicReference<Throwable> failure = new AtomicReference<>();
            CountDownLatch finished = new CountDownLatch(1);
            conversation.sendMessageAsync(
                    message,
                    new MessageCallback() {
                        @Override
                        public void onMessage(Message message) {
                            // Text arrives in pieces; tool calls arrive whole.
                            toolCalls.addAll(message.getToolCalls());
                            String text = message.toString();
                            if (text.isEmpty()) {
                                return;
                            }
                            synchronized (reply) {
                                reply.append(text);
                            }
                            nativeOnEditorText(requestId, text);
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
                    null);
            if (!finished.await(EDITOR_TURN_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                conversation.cancelProcess();
                throw new IOException(
                        "Gemma did not reply in " + EDITOR_TURN_TIMEOUT_SECONDS + " s");
            }
            if (failure.get() != null) {
                Log.w(LOG_TAG, "Gemma editor turn failed after "
                        + conversation.getTokenCount() + " tokens", failure.get());
                trace("FAILED " + failure.get());
                String unreadable = unreadableToolCall(failure.get());
                if (unreadable == null) {
                    throw new IOException(
                            "Gemma could not reply: " + failure.get(), failure.get());
                }
                // Gemma wrote its own call syntax and LiteRT-LM could not parse it, such as a
                // call with a brace too many; the caller reads it from the text as it reads
                // the calls it asked for (calls.rs).
                synchronized (reply) {
                    reply.append('\n').append(unreadable);
                }
            }
            // Gson, not org.json: argument numbers arrive as Gson's LazilyParsedNumber, which
            // Android's JSONObject writes as null.
            JsonArray calls = new JsonArray();
            synchronized (toolCalls) {
                for (ToolCall call : toolCalls) {
                    JsonObject json = new JsonObject();
                    json.addProperty("name", call.getName());
                    json.add("arguments", JsonConvertersKt.toJsonObject(call.getArguments()));
                    calls.add(json);
                }
            }
            JsonObject json = new JsonObject();
            synchronized (reply) {
                json.addProperty("text", reply.toString());
            }
            json.add("tool_calls", calls);
            String result = json.toString();
            Log.i(LOG_TAG, "Gemma editor turn took " + (SystemClock.elapsedRealtime() - started)
                    + " ms, " + conversation.getTokenCount() + " tokens: " + result);
            trace("REPLY " + (SystemClock.elapsedRealtime() - started) + " ms "
                    + conversation.getTokenCount() + " tok " + result);
            Log.i(LOG_TAG, "Gemma editor benchmark: " + conversation.getBenchmarkInfo());
            return result;
        }
    }

    /**
     * What Gemma wrote when {@code failure} is LiteRT-LM rejecting a tool call it could not
     * parse, such as one with an extra closing brace, or null for any other failure. LiteRT-LM
     * reports it only in the message: "Failed to parse tool calls ... full response: ...".
     */
    private static String unreadableToolCall(Throwable failure) {
        String message = failure.getMessage();
        if (!(failure instanceof LiteRtLmJniException) || message == null
                || !message.contains("Failed to parse tool calls")) {
            return null;
        }
        Matcher response = FULL_RESPONSE.matcher(message);
        return response.find() ? response.group(1).trim() : "";
    }

    /** Reports the next piece of an editor reply's text while Gemma writes it. */
    private static native void nativeOnEditorText(long requestId, String text);

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
                MAX_NUM_TOKENS,
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
