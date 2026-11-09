package de.duecki.calibraw.ai.gemma;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;

import de.duecki.calibraw.ai.AiModel;
import de.duecki.calibraw.ai.Assets;
import de.duecki.calibraw.ai.TriageCardStream;

import com.google.ai.edge.litertlm.Content;
import com.google.ai.edge.litertlm.Contents;
import com.google.ai.edge.litertlm.Conversation;
import com.google.ai.edge.litertlm.ConversationConfig;
import com.google.ai.edge.litertlm.Engine;
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Gemma 4 E4B on the device, through LiteRT-LM, as the AI editor's model: the triage of each
 * opened photo and the AI editor's chat about it. There is one for the process
 * ({@link #shared}): its engine holds gigabytes of GPU memory, so a second could not load, and
 * selecting Gemma again while it is still being released keeps the engine it has. Its calls
 * block, except {@link #closeEditor}, which returns at once, and one call uses the engine at a
 * time.
 */
public final class GemmaBridge implements AiModel {
    private static final String LOG_TAG = "CalibRaw";
    // Triage wants a stable card, so sampling stays close to greedy.
    private static final SamplerConfig TRIAGE_SAMPLER = new SamplerConfig(1, 1.0, 0.0, 0);
    // A triage normally takes about ten seconds.
    private static final long TRIAGE_TIMEOUT_SECONDS = 120;
    // The reply LiteRT-LM quotes in its message when it cannot parse a tool call.
    private static final Pattern FULL_RESPONSE = Pattern.compile("full response: (.*)");
    // The editor's replies are short, so greedy decoding keeps them predictable.
    private static final SamplerConfig EDITOR_SAMPLER = new SamplerConfig(1, 1.0, 0.0, 0);
    // An editor turn normally takes a few seconds.
    private static final long EDITOR_TURN_TIMEOUT_SECONDS = 120;
    // Base conversations kept; each reserves the engine's whole context. A request runs on one
    // prompt.
    private static final int MAX_EDITOR_BASES = 1;

    // Guarded by GemmaBridge.class.
    private static GemmaBridge shared;

    private final Context context;
    private final GemmaTrace trace;
    // Guarded by this.
    private Engine engine;
    // The next triage's conversation, made ahead of time since making one takes seconds. A
    // conversation keeps its history, so each triage takes this one and a new one is made after.
    // Guarded by this.
    private Conversation nextTriage;
    // The editor's system prompt, already read by Gemma, by prompt. It names no photo, so it is
    // kept while Gemma is selected, and opening a photo costs only its triage.
    // Each request runs on a clone of its prompt's: a clone took milliseconds, while a new
    // conversation's first prefill took 16-23 s after the process's first one. Guarded by this.
    private final Map<String, Conversation> editorBases = new LinkedHashMap<>();
    // The running request's conversation, or null. Written under this; volatile so closeEditor
    // can cancel its running turn without waiting for that lock.
    private volatile Conversation editor;
    // Whether Gemma is the selected model, so background preparations do not load the engine
    // again after release.
    private volatile boolean active;

    private GemmaBridge(Context context) {
        this.context = context;
        this.trace = new GemmaTrace(context);
    }

    /** The process's Gemma. */
    public static synchronized GemmaBridge shared(Context context) {
        if (shared == null) {
            shared = new GemmaBridge(context.getApplicationContext());
        }
        return shared;
    }

    /**
     * Loads the engine and the first triage's conversation on a background thread, so the first
     * photo's triage does not wait for them. A call that arrives meanwhile waits for the load.
     */
    @Override
    public void activate() {
        active = true;
        prepareTriageInBackground("Gemma preload");
    }

    /**
     * Cancels the editor's running turn and closes the engine and every conversation on a
     * background thread, once a running triage has finished, so the engine's memory is free for
     * the other model.
     */
    @Override
    public void release() {
        active = false;
        closeEditor();
        Thread releaser = new Thread(() -> {
            synchronized (this) {
                // Selected again meanwhile.
                if (active) {
                    return;
                }
                if (nextTriage != null) {
                    nextTriage.close();
                    nextTriage = null;
                }
                Conversation running = editor;
                editor = null;
                if (running != null) {
                    running.close();
                }
                for (Conversation base : editorBases.values()) {
                    base.close();
                }
                editorBases.clear();
                if (engine != null) {
                    try {
                        engine.close();
                    } catch (Exception error) {
                        Log.w(LOG_TAG, "Could not close the Gemma engine", error);
                    }
                    engine = null;
                }
                Log.i(LOG_TAG, "Gemma released");
            }
        }, "Gemma release");
        releaser.setDaemon(true);
        releaser.start();
    }

    /**
     * Returns the triage card for the JPEG at {@code imagePath} as JSON, shaped by
     * assets/gemma/triage_schema.json. The card's genre and emoji come first and go to
     * {@code headline} as soon as Gemma has written them; the rest of the card takes several
     * seconds more.
     */
    @Override
    public String triage(String imagePath, String hint, HeadlineListener headline)
            throws Exception {
        try {
            synchronized (this) {
                return triageLocked(imagePath, hint, headline);
            }
        } finally {
            prepareTriageInBackground("Gemma triage preparation");
        }
    }

    // Caller holds this.
    private String triageLocked(String imagePath, String hint, HeadlineListener headline)
            throws Exception {
        checkActive();
        String request = AiModel.triageRequest(hint);
        String schema = Assets.text(context, "gemma/triage_schema.json");
        long started = SystemClock.elapsedRealtime();
        Conversation conversation = nextTriage != null ? nextTriage : createTriage();
        nextTriage = null;
        try (conversation) {
            Log.i(LOG_TAG, "Gemma triage conversation ready in "
                    + (SystemClock.elapsedRealtime() - started) + " ms");
            TriageCardStream card = new TriageCardStream("Gemma", started, headline);
            Throwable failure = awaitReply(
                    conversation,
                    callback -> conversation.sendMessageAsync(
                            Contents.Companion.of(
                                    new Content.ImageFile(imagePath), new Content.Text(request)),
                            callback,
                            Collections.emptyMap(),
                            null,
                            null,
                            null,
                            null,
                            null,
                            ResponseFormat.json(schema)),
                    // Each message is the next piece of the card.
                    message -> card.append(message.toString()),
                    TRIAGE_TIMEOUT_SECONDS,
                    "Gemma triage did not finish in " + TRIAGE_TIMEOUT_SECONDS + " s");
            if (failure != null) {
                throw new IOException("Gemma triage failed", failure);
            }
            String text = card.text();
            Log.i(LOG_TAG, "Gemma triage took " + (SystemClock.elapsedRealtime() - started)
                    + " ms: " + text);
            trace.write("TRIAGE " + (SystemClock.elapsedRealtime() - started) + " ms " + text);
            Log.i(LOG_TAG, "Gemma triage benchmark: " + conversation.getBenchmarkInfo());
            return text;
        }
    }

    /**
     * Fails a call that waited for this while another model was selected, so it does not load
     * the engine again after release. Caller holds this.
     */
    private void checkActive() throws IOException {
        if (!active) {
            throw new IOException("Gemma is no longer the selected AI model");
        }
    }

    private void prepareTriageInBackground(String threadName) {
        Thread preparer = new Thread(() -> {
            try {
                synchronized (this) {
                    if (active && nextTriage == null) {
                        nextTriage = createTriage();
                    }
                }
            } catch (Exception error) {
                Log.w(LOG_TAG, "Could not prepare Gemma triage", error);
            }
        }, threadName);
        preparer.setDaemon(true);
        preparer.start();
    }

    // Caller holds this.
    private Conversation createTriage() throws IOException {
        Engine loaded = engine();
        long started = SystemClock.elapsedRealtime();
        Conversation conversation = loaded.createConversation(new ConversationConfig(
                Contents.Companion.of(Assets.text(context, "gemma/triage_system.md")),
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
    @Override
    public void prepareEditor(String systemPrompt) {
        Thread preparer = new Thread(() -> {
            try {
                synchronized (this) {
                    if (active) {
                        editorBase(systemPrompt);
                    }
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
    @Override
    public synchronized void startEditorRequest(String systemPrompt) throws Exception {
        Conversation previous = editor;
        editor = null;
        if (previous != null) {
            previous.close();
        }
        checkActive();
        Conversation base = editorBase(systemPrompt);
        long started = SystemClock.elapsedRealtime();
        editor = base.clone();
        Log.i(LOG_TAG, "Gemma editor request conversation cloned in "
                + (SystemClock.elapsedRealtime() - started) + " ms");
        trace.write("PHASE " + systemPrompt.substring(0, Math.min(60, systemPrompt.length())));
    }

    // Caller holds this.
    private Conversation editorBase(String systemPrompt) throws IOException {
        Conversation known = editorBases.get(systemPrompt);
        if (known != null) {
            return known;
        }
        // A prompt the app no longer sends, such as one from before an update.
        while (editorBases.size() >= MAX_EDITOR_BASES) {
            String oldest = editorBases.keySet().iterator().next();
            editorBases.remove(oldest).close();
        }
        Engine loaded = engine();
        long started = SystemClock.elapsedRealtime();
        // No tools are declared: Gemma writes a brief, which the app reads into tool calls
        // (app/ai/editor/chat/brief.rs). The preface is read now, not at the first request.
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
     * Sends the photographer's turn, {@code text} after the JPEG in {@code imagePaths}, and
     * returns Gemma's reply as
     * {@code {"text": ..., "tool_calls": [{"name": ..., "arguments": {...}}]}}, the calls being
     * those Gemma wrote in its own call syntax that LiteRT-LM parsed. Gemma takes one JPEG at
     * most. The reply's text also goes to {@code replyText} while Gemma writes it.
     */
    @Override
    public String editorSend(String text, List<String> imagePaths, TextListener replyText)
            throws Exception {
        if (imagePaths.size() > 1) {
            throw new IOException("Gemma takes one image per message, not " + imagePaths.size());
        }
        Contents contents = imagePaths.isEmpty()
                ? Contents.Companion.of(text)
                : Contents.Companion.of(
                        new Content.ImageFile(imagePaths.get(0)), new Content.Text(text));
        trace.write("SEND " + text);
        return editorTurn(Message.Companion.user(contents), replyText);
    }

    /**
     * Ends the editor's request, cancelling a running turn, which then fails: when a request
     * finishes, so its conversation does not hold memory until the next, and when another photo
     * opens. Returns at once; the request's conversation is closed on a background thread once no
     * call uses it, and the prompts' are kept for the next request.
     */
    @Override
    public void closeEditor() {
        trace.write("CLOSE");
        Conversation closing = editor;
        if (closing != null) {
            try {
                closing.cancelProcess();
            } catch (IllegalStateException alreadyClosed) {
                // startEditorRequest replaced and closed it meanwhile.
            }
        }
        Thread closer = new Thread(() -> {
            synchronized (this) {
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

    private synchronized String editorTurn(Message message, TextListener replyText)
            throws Exception {
        checkActive();
        Conversation conversation = editor;
        if (conversation == null) {
            throw new IOException("The AI editor chat is closed");
        }
        long started = SystemClock.elapsedRealtime();
        StringBuilder reply = new StringBuilder();
        List<ToolCall> toolCalls = Collections.synchronizedList(new ArrayList<>());
        Throwable failure = awaitReply(
                conversation,
                callback -> conversation.sendMessageAsync(
                        message,
                        callback,
                        Collections.emptyMap(),
                        null,
                        null,
                        null,
                        null,
                        null,
                        null),
                piece -> {
                    // Text arrives in pieces; tool calls arrive whole.
                    toolCalls.addAll(piece.getToolCalls());
                    String text = piece.toString();
                    if (text.isEmpty()) {
                        return;
                    }
                    synchronized (reply) {
                        reply.append(text);
                    }
                    replyText.onText(text);
                },
                EDITOR_TURN_TIMEOUT_SECONDS,
                "Gemma did not reply in " + EDITOR_TURN_TIMEOUT_SECONDS + " s");
        if (failure != null) {
            Log.w(LOG_TAG, "Gemma editor turn failed after "
                    + conversation.getTokenCount() + " tokens", failure);
            trace.write("FAILED " + failure);
            String unreadable = unreadableToolCall(failure);
            if (unreadable == null) {
                throw new IOException("Gemma could not reply: " + failure, failure);
            }
            // Gemma wrote its own call syntax and LiteRT-LM could not parse it, such as a call
            // with a brace too many; the caller reads it from the text as it reads the calls it
            // asked for (calls.rs).
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
        trace.write("REPLY " + (SystemClock.elapsedRealtime() - started) + " ms "
                + conversation.getTokenCount() + " tok " + result);
        Log.i(LOG_TAG, "Gemma editor benchmark: " + conversation.getBenchmarkInfo());
        return result;
    }

    /** Sends a message to a conversation, given the callback that hears its reply. */
    private interface Sender {
        void send(MessageCallback callback);
    }

    /**
     * Sends a message with {@code send} and waits for the reply, passing each message it writes
     * to {@code onMessage}. Returns the reply's failure, or null; when the reply does not finish
     * in {@code timeoutSeconds}, cancels it and throws {@code timeoutMessage}.
     */
    private static Throwable awaitReply(
            Conversation conversation,
            Sender send,
            Consumer<Message> onMessage,
            long timeoutSeconds,
            String timeoutMessage) throws IOException, InterruptedException {
        AtomicReference<Throwable> failure = new AtomicReference<>();
        CountDownLatch finished = new CountDownLatch(1);
        send.send(new MessageCallback() {
            @Override
            public void onMessage(Message message) {
                onMessage.accept(message);
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
        });
        if (!finished.await(timeoutSeconds, TimeUnit.SECONDS)) {
            conversation.cancelProcess();
            throw new IOException(timeoutMessage);
        }
        return failure.get();
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

    // Caller holds this.
    private Engine engine() throws IOException {
        if (engine == null) {
            engine = GemmaEngine.load(context);
        }
        return engine;
    }
}
