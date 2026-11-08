package de.duecki.calibraw;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Claude Code in Termux as the AI editor's model: the triage of each opened photo and the AI
 * editor's chat about it. Each conversation is one {@link ClaudeSession}.
 *
 * <p>A conversation's process starts in a few seconds, so the next triage's and the editor's are
 * started ahead of time, but only while Develop shows: leaving it ends every process. The editor's
 * conversation lasts while one photo is open in Develop, so Claude keeps the whole chat about it.
 * Triage and the editor use separate processes and may run at once.
 */
final class ClaudeBridge implements AiModel {
    private static final String LOG_TAG = "CalibRaw";
    private static final String RELEASED = "Claude Code is no longer the selected AI model";
    private static final String NOT_SHOWN = "Develop no longer shows";

    private final Context context;
    // Claude Code's --model value for every conversation: an alias such as "sonnet" or a full
    // model name, chosen in Settings (ClaudeModel in platform/service.rs).
    private final String model;
    // Whether another model was selected; then nothing is started or sent any more, so photos go
    // to Claude only while it is the selected model. Written under this, and checked under this
    // wherever a conversation is taken or installed, so release ends every conversation it misses.
    private volatile boolean released;
    // Whether Develop shows: conversations are started only then, and leaving it ends every one,
    // so no claude process waits in Termux while the photographer is elsewhere. Guarded by this,
    // and checked wherever a conversation is taken, started or installed, as released is.
    private boolean editorShown;
    // The next triage's conversation, started ahead of time; each triage takes it and starts the
    // next. Guarded by this.
    private ClaudeSession nextTriage;
    // The running triage's conversation, so release can end it. Guarded by this.
    private ClaudeSession runningTriage;
    // The open photo's editor conversation, started ahead of time with nextEditorPrompt.
    // Guarded by this.
    private ClaudeSession nextEditor;
    private String nextEditorPrompt;
    // The editor conversation, or null. Written under editorLock and this; volatile so
    // closeEditor can end its running turn without waiting for either.
    private volatile ClaudeSession editor;
    // How many times closeEditor was called, so a close that arrives while startEditorRequest
    // waits for its conversation to connect ends that conversation too.
    private final AtomicInteger editorCloses = new AtomicInteger();
    // Held for a turn of the editor's conversation, so its turns run one at a time.
    private final Object editorLock = new Object();
    private final Object triageLock = new Object();

    ClaudeBridge(Context context, String model) {
        this.context = context.getApplicationContext();
        this.model = model;
    }

    /**
     * Ends every conversation and its process, failing a running triage or editor turn; a call
     * that has not taken its conversation yet fails without sending anything.
     */
    @Override
    public void release() {
        synchronized (this) {
            released = true;
            endConversations();
        }
        Log.i(LOG_TAG, "Claude released");
    }

    /**
     * When Develop shows, ends the claude processes that earlier runs of the app left behind and
     * starts the next triage's conversation, so the first triage does not wait for it; when it no
     * longer shows, ends every conversation and its process as {@link #release} does. Returns at
     * once: starting reads assets and has Termux start processes, on a background thread.
     */
    @Override
    public void editorShown(boolean shown) {
        synchronized (this) {
            if (released || editorShown == shown) {
                return;
            }
            editorShown = shown;
            if (!shown) {
                endConversations();
                Log.i(LOG_TAG, "Claude conversations ended: Develop no longer shows");
                return;
            }
        }
        Thread preparer = new Thread(() -> {
            ClaudeSession.endEarlierRuns(context);
            prepareTriage();
        }, "Claude preload");
        preparer.setDaemon(true);
        preparer.start();
    }

    /** Ends every conversation and its process. Called under this. */
    private void endConversations() {
        if (runningTriage != null) {
            runningTriage.end();
        }
        if (nextTriage != null) {
            nextTriage.end();
            nextTriage = null;
        }
        if (nextEditor != null) {
            nextEditor.end();
            nextEditor = null;
            nextEditorPrompt = null;
        }
        // Also ends a request that is still connecting (startEditorRequest).
        closeEditor();
    }

    /**
     * Returns the triage card for the JPEG at {@code imagePath} as JSON, shaped by
     * assets/gemma/triage_schema.json, which Claude Code enforces.
     */
    @Override
    public String triage(String imagePath, String hint, HeadlineListener headline)
            throws Exception {
        synchronized (triageLock) {
            ClaudeSession session;
            synchronized (this) {
                // A triage that waited for triageLock while another model was selected or the
                // photographer left Develop.
                if (released || !editorShown) {
                    throw new IOException(released ? RELEASED : NOT_SHOWN);
                }
                session = nextTriage;
                nextTriage = null;
                if (session == null || session.hasEnded()) {
                    if (session != null) {
                        session.end();
                    }
                    session = ClaudeSession.start(context, "Claude triage", triageArguments());
                }
                runningTriage = session;
            }
            try {
                long started = SystemClock.elapsedRealtime();
                TriageCardStream card = new TriageCardStream("Claude", started, headline);
                JSONObject result = session.send(
                        ClaudeProtocol.userMessage(
                                AiModel.triageRequest(hint), Collections.singletonList(imagePath)),
                        new ClaudeSession.Listener() {
                            @Override
                            void onJson(String piece) {
                                // Claude writes the card as its structured-output tool's input.
                                card.append(piece);
                            }
                        });
                JSONObject structured = result.optJSONObject("structured_output");
                String text = structured != null ? structured.toString() : result.optString("result");
                Log.i(LOG_TAG, "Claude triage took " + (SystemClock.elapsedRealtime() - started)
                        + " ms, " + ClaudeProtocol.cost(result) + ": " + text);
                return text;
            } finally {
                synchronized (this) {
                    runningTriage = null;
                }
                session.end();
                prepareTriage();
            }
        }
    }

    private void prepareTriage() {
        synchronized (this) {
            if (released || !editorShown || (nextTriage != null && !nextTriage.hasEnded())) {
                return;
            }
            try {
                nextTriage = ClaudeSession.start(context, "Claude triage", triageArguments());
            } catch (IOException error) {
                nextTriage = null;
                Log.w(LOG_TAG, "Could not prepare Claude triage", error);
            }
        }
    }

    /**
     * Starts the editor's next conversation with {@code systemPrompt}, so the open photo's first
     * request does not wait for its process. Returns at once.
     */
    @Override
    public void prepareEditor(String systemPrompt) {
        synchronized (this) {
            if (released || !editorShown) {
                return;
            }
            if (nextEditor != null) {
                if (systemPrompt.equals(nextEditorPrompt) && !nextEditor.hasEnded()) {
                    return;
                }
                nextEditor.end();
                nextEditor = null;
                nextEditorPrompt = null;
            }
            try {
                nextEditor = ClaudeSession.start(context, "Claude editor", editorArguments(systemPrompt));
                nextEditorPrompt = systemPrompt;
            } catch (IOException error) {
                Log.w(LOG_TAG, "Could not prepare the Claude editor", error);
            }
        }
    }

    /**
     * Starts a fresh editor conversation with {@code systemPrompt}, ending the earlier one. Blocks
     * until its process has connected; fails when the editor is closed or another model is
     * selected meanwhile.
     */
    @Override
    public void startEditorRequest(String systemPrompt) throws Exception {
        synchronized (editorLock) {
            ClaudeSession session = null;
            int closes;
            synchronized (this) {
                ClaudeSession previous = editor;
                editor = null;
                if (previous != null) {
                    previous.end();
                }
                // A close from here on is for this request.
                closes = editorCloses.get();
                if (released || !editorShown) {
                    throw new IOException(released ? RELEASED : NOT_SHOWN);
                }
                if (nextEditor != null && systemPrompt.equals(nextEditorPrompt)
                        && !nextEditor.hasEnded()) {
                    session = nextEditor;
                    nextEditor = null;
                    nextEditorPrompt = null;
                }
                if (session == null) {
                    session = ClaudeSession.start(
                            context, "Claude editor", editorArguments(systemPrompt));
                }
            }
            long started = SystemClock.elapsedRealtime();
            try {
                session.awaitConnected();
            } catch (IOException | InterruptedException error) {
                session.end();
                throw error;
            }
            // Neither release nor closeEditor can see the session while it connects, so it is
            // installed only if neither came meanwhile; leaving Develop closes the editor.
            synchronized (this) {
                if (released || editorCloses.get() != closes) {
                    session.end();
                    throw new IOException(released ? RELEASED : "The AI editor chat was closed");
                }
                editor = session;
            }
            Log.i(LOG_TAG, "Claude editor conversation ready in "
                    + (SystemClock.elapsedRealtime() - started) + " ms");
        }
    }

    /**
     * Sends {@code text}, after the JPEGs at {@code imagePaths}, and returns Claude's reply as
     * {@code {"text": ..., "tool_calls": []}}: Claude writes its tool lines in the text, which the
     * app parses (app/ai/editor/calls.rs).
     */
    @Override
    public String editorSend(String text, List<String> imagePaths, TextListener replyText)
            throws Exception {
        synchronized (editorLock) {
            ClaudeSession session = editor;
            if (session == null) {
                throw new IOException("The AI editor chat is closed");
            }
            long started = SystemClock.elapsedRealtime();
            JSONObject result = session.send(
                    ClaudeProtocol.userMessage(text, imagePaths),
                    new ClaudeSession.Listener() {
                        @Override
                        void onText(String piece) {
                            replyText.onText(piece);
                        }
                    });
            JSONObject reply = new JSONObject();
            reply.put("text", result.optString("result"));
            reply.put("tool_calls", new JSONArray());
            String json = reply.toString();
            Log.i(LOG_TAG, "Claude editor turn took " + (SystemClock.elapsedRealtime() - started)
                    + " ms, " + ClaudeProtocol.cost(result) + ": " + json);
            return json;
        }
    }

    /**
     * Ends the editor conversation and its process, failing a running turn or a request that is
     * still connecting. Returns at once.
     */
    @Override
    public void closeEditor() {
        editorCloses.incrementAndGet();
        ClaudeSession closing = editor;
        if (closing != null) {
            closing.end();
        }
    }

    /**
     * Claude's triage prompt is its own: with Gemma's, the card's role named edits, which steered
     * the edits towards them rather than the photographer's words. The card's shape is Gemma's.
     */
    private List<String> triageArguments() throws IOException {
        return ClaudeProtocol.conversationArguments(
                model,
                Assets.text(context, "claude/triage_system.md"),
                Assets.text(context, "gemma/triage_schema.json"));
    }

    private List<String> editorArguments(String systemPrompt) {
        return ClaudeProtocol.conversationArguments(model, systemPrompt, null);
    }
}
