package de.duecki.calibraw;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.SystemClock;
import android.util.Base64;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Runs Claude Code in Termux for the triage of each opened photo and the AI editor's chat about
 * it, while it is the selected {@link AiModel}. Each conversation is one {@code claude -p}
 * process that Termux starts through its RUN_COMMAND intent (assets/claude/connect.sh); it
 * connects back to a loopback port of this app and speaks stream-json over that connection.
 *
 * <p>A conversation's process starts in a few seconds, so the next triage's and the editor's are
 * started ahead of time. The editor's conversation lasts while one photo is open, so Claude keeps
 * the whole chat about it. Triage and the editor use separate processes and may run at once.
 */
final class ClaudeBridge implements AiModel {
    private static final String LOG_TAG = "CalibRaw";
    private static final String TERMUX_PACKAGE = "com.termux";
    private static final String RUN_COMMAND_PERMISSION = "com.termux.permission.RUN_COMMAND";
    private static final String RUN_COMMAND_SERVICE = "com.termux.app.RunCommandService";
    private static final String RUN_COMMAND_ACTION = "com.termux.RUN_COMMAND";
    private static final String EXTRA_PATH = "com.termux.RUN_COMMAND_PATH";
    private static final String EXTRA_ARGUMENTS = "com.termux.RUN_COMMAND_ARGUMENTS";
    private static final String EXTRA_WORKDIR = "com.termux.RUN_COMMAND_WORKDIR";
    private static final String EXTRA_BACKGROUND = "com.termux.RUN_COMMAND_BACKGROUND";
    private static final String EXTRA_SESSION_ACTION = "com.termux.RUN_COMMAND_SESSION_ACTION";
    // Termux's action that opens a new session in front and switches to it.
    private static final String SESSION_ACTION_SWITCH_AND_OPEN = "0";
    private static final String BASH = "/data/data/com.termux/files/usr/bin/bash";
    private static final String TERMUX_HOME = "/data/data/com.termux/files/home";
    // Each request is set by the CalibRawActivity or one of its helpers; see their constants.
    private static final int RUN_COMMAND_PERMISSION_REQUEST = 1006;
    private static final String MODEL = "sonnet";
    private static final String TRIAGE_REQUEST = "Triage this photo.";
    private static final String RELEASED = "Claude Code is no longer the selected AI model";
    // Termux starting a process took about 3.4 s on the spike phone; a cold Termux takes longer.
    private static final int CONNECT_TIMEOUT_MS = 30_000;
    // The connection's first line, the token, arrives at once after it connects.
    private static final int TOKEN_TIMEOUT_MS = 5_000;
    // A turn took about 5 s on the spike phone.
    private static final long TURN_TIMEOUT_SECONDS = 120;
    // claude auth status answers within a few seconds of starting.
    private static final long STATUS_TIMEOUT_SECONDS = 60;

    // Setup status codes, read by Rust (ClaudeCodeStatus in platform/service.rs).
    static final String NO_TERMUX = "no_termux";
    static final String NO_RUN_COMMAND = "no_run_command";
    static final String NO_PERMISSION = "no_permission";
    static final String BLOCKED = "blocked";
    static final String NOT_INSTALLED = "not_installed";
    static final String SIGNED_OUT = "signed_out";
    static final String READY = "ready";
    static final String FAILED = "failed";

    private final Context context;
    // Whether another model was selected; then nothing is started or sent any more, so photos go
    // to Claude only while it is the selected model. Written under this, and checked under this
    // wherever a conversation is taken or installed, so release ends every conversation it misses.
    private volatile boolean released;
    // The next triage's conversation, started ahead of time; each triage takes it and starts the
    // next. Guarded by this.
    private Session nextTriage;
    // The running triage's conversation, so release can end it. Guarded by this.
    private Session runningTriage;
    // The open photo's editor conversation, started ahead of time with nextEditorPrompt.
    // Guarded by this.
    private Session nextEditor;
    private String nextEditorPrompt;
    // The editor conversation, or null. Written under editorLock and this; volatile so
    // closeEditor can end its running turn without waiting for either.
    private volatile Session editor;
    // How many times closeEditor was called, so a close that arrives while startEditorRequest
    // waits for its conversation to connect ends that conversation too.
    private final AtomicInteger editorCloses = new AtomicInteger();
    // Held for a turn of the editor's conversation, so its turns run one at a time.
    private final Object editorLock = new Object();
    private final Object triageLock = new Object();

    ClaudeBridge(Context context) {
        this.context = context.getApplicationContext();
    }

    /**
     * Starts the first triage's conversation on a background thread, so the first photo's triage
     * does not wait for it. Returns at once: starting it reads assets and has Termux start a
     * process.
     */
    @Override
    public void activate() {
        Thread preparer = new Thread(this::prepareTriage, "Claude preload");
        preparer.setDaemon(true);
        preparer.start();
    }

    /**
     * Ends every conversation and its process, failing a running triage or editor turn; a call
     * that has not taken its conversation yet fails without sending anything.
     */
    @Override
    public void release() {
        synchronized (this) {
            released = true;
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
            Session closing = editor;
            if (closing != null) {
                closing.end();
            }
        }
        Log.i(LOG_TAG, "Claude released");
    }

    /**
     * Returns the triage card for the JPEG at {@code imagePath} as JSON, shaped by
     * assets/gemma/triage_schema.json, which Claude Code enforces. The card's genre and emoji are
     * reported through {@link AiModels#reportTriageHeadline} once Claude has written them.
     */
    @Override
    public String triage(String imagePath, String hint, long requestId, int badgePx)
            throws Exception {
        synchronized (triageLock) {
            Session session;
            synchronized (this) {
                // A triage that waited for triageLock while another model was selected.
                if (released) {
                    throw new IOException(RELEASED);
                }
                session = nextTriage;
                nextTriage = null;
                if (session == null || session.hasEnded()) {
                    if (session != null) {
                        session.end();
                    }
                    session = Session.start(context, "Claude triage", triageArguments(context));
                }
                runningTriage = session;
            }
            try {
                String request = hint == null || hint.isEmpty()
                        ? TRIAGE_REQUEST
                        : TRIAGE_REQUEST + " The photographer says the genre is: " + hint + ".";
                long started = SystemClock.elapsedRealtime();
                StringBuilder card = new StringBuilder();
                AtomicBoolean headlineReported = new AtomicBoolean();
                JSONObject result = session.send(userMessage(request, imagePath), new Listener() {
                    @Override
                    public void onJson(String piece) {
                        // Each piece is the next part of the card, which Claude writes as the
                        // input of its structured-output tool.
                        String text = card.append(piece).toString();
                        if (headlineReported.get()) {
                            return;
                        }
                        String genre = AiModels.completeField(text, AiModels.GENRE_FIELD);
                        String emoji = AiModels.completeField(text, AiModels.EMOJI_FIELD);
                        if (genre != null && emoji != null
                                && headlineReported.compareAndSet(false, true)) {
                            Log.i(LOG_TAG, "Claude triage headline after "
                                    + (SystemClock.elapsedRealtime() - started) + " ms");
                            AiModels.reportTriageHeadline(requestId, genre, emoji, badgePx);
                        }
                    }
                });
                JSONObject structured = result.optJSONObject("structured_output");
                String text = structured != null ? structured.toString() : result.optString("result");
                Log.i(LOG_TAG, "Claude triage took " + (SystemClock.elapsedRealtime() - started)
                        + " ms, " + costOf(result) + ": " + text);
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
            if (released || (nextTriage != null && !nextTriage.hasEnded())) {
                return;
            }
            try {
                nextTriage = Session.start(context, "Claude triage", triageArguments(context));
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
            if (released) {
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
                nextEditor = Session.start(
                        context, "Claude editor", conversationArguments(systemPrompt, null));
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
            Session session = null;
            int closes;
            synchronized (this) {
                Session previous = editor;
                editor = null;
                if (previous != null) {
                    previous.end();
                }
                // A close from here on is for this request.
                closes = editorCloses.get();
                if (released) {
                    throw new IOException(RELEASED);
                }
                if (nextEditor != null && systemPrompt.equals(nextEditorPrompt)
                        && !nextEditor.hasEnded()) {
                    session = nextEditor;
                    nextEditor = null;
                    nextEditorPrompt = null;
                }
                if (session == null) {
                    session = Session.start(
                            context, "Claude editor", conversationArguments(systemPrompt, null));
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
            // installed only if neither came meanwhile.
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
     * Sends {@code text}, after the JPEG at {@code imagePath} when that is not empty, and returns
     * Claude's reply as {@code {"text": ..., "tool_calls": []}}: Claude writes its tool lines in
     * the text, which the app parses (app/ai/gemma/calls.rs). The text also streams through
     * {@link AiModels#reportEditorText}, tagged with {@code requestId}.
     */
    @Override
    public String editorSend(long requestId, String text, String imagePath) throws Exception {
        synchronized (editorLock) {
            Session session = editor;
            if (session == null) {
                throw new IOException("The AI editor chat is closed");
            }
            long started = SystemClock.elapsedRealtime();
            JSONObject result = session.send(userMessage(text, imagePath), new Listener() {
                @Override
                public void onText(String piece) {
                    AiModels.reportEditorText(requestId, piece);
                }
            });
            JSONObject reply = new JSONObject();
            reply.put("text", result.optString("result"));
            reply.put("tool_calls", new JSONArray());
            String json = reply.toString();
            Log.i(LOG_TAG, "Claude editor turn took " + (SystemClock.elapsedRealtime() - started)
                    + " ms, " + costOf(result) + ": " + json);
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
        Session closing = editor;
        if (closing != null) {
            closing.end();
        }
    }

    /**
     * How far Claude Code in Termux is set up, as the first missing step's status code, or
     * {@link #READY}. Blocks while Termux runs {@code claude auth status}.
     */
    static String status(Context context) {
        PackageManager packages = context.getPackageManager();
        try {
            packages.getPackageInfo(TERMUX_PACKAGE, 0);
        } catch (PackageManager.NameNotFoundException missing) {
            return NO_TERMUX;
        }
        // Termux's Google Play build has no RUN_COMMAND.
        try {
            packages.getPermissionInfo(RUN_COMMAND_PERMISSION, 0);
        } catch (PackageManager.NameNotFoundException missing) {
            return NO_RUN_COMMAND;
        }
        if (context.checkSelfPermission(RUN_COMMAND_PERMISSION)
                != PackageManager.PERMISSION_GRANTED) {
            return NO_PERMISSION;
        }
        Session session;
        try {
            session = Session.start(context, "Claude status", Arrays.asList("auth", "status"));
        } catch (IOException error) {
            Log.w(LOG_TAG, "Termux did not run the Claude Code status check", error);
            return BLOCKED;
        }
        try {
            if (!session.awaitEnd(STATUS_TIMEOUT_SECONDS) || !session.connected()) {
                // Termux refuses commands from other apps until allow-external-apps is set.
                return BLOCKED;
            }
            switch (session.exitCode()) {
                case 0:
                    return READY;
                case 1:
                    return SIGNED_OUT;
                case 127:
                    return NOT_INSTALLED;
                default:
                    Log.w(LOG_TAG, "Claude Code status check failed: " + session.ending());
                    return FAILED;
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            return FAILED;
        } finally {
            session.end();
        }
    }

    /**
     * Runs a setup step that Settings offers for Claude Code: "permission" asks for Termux's
     * RUN_COMMAND permission; "install" installs Claude Code (assets/claude/install.sh) and
     * "sign_in" runs its sign-in, each in a Termux session that is brought to the front.
     */
    static void setup(Activity activity, String step) throws IOException {
        switch (step) {
            case "permission":
                activity.runOnUiThread(() -> activity.requestPermissions(
                        new String[] {RUN_COMMAND_PERMISSION}, RUN_COMMAND_PERMISSION_REQUEST));
                return;
            case "install":
                runCommand(activity, new String[] {
                        "-c", AiModels.asset(activity, "claude/install.sh"), "calibraw-install"},
                        false);
                openTermux(activity);
                return;
            case "sign_in":
                runCommand(activity, new String[] {
                        "-c",
                        "claude auth login && printf '\\nSigned in. Go back to CalibRaw and tap"
                                + " Check again.\\n'",
                        "calibraw-sign-in"},
                        false);
                openTermux(activity);
                return;
            default:
                throw new IOException("Unknown Claude Code setup step " + step);
        }
    }

    private static void openTermux(Activity activity) {
        Intent launch = activity.getPackageManager().getLaunchIntentForPackage(TERMUX_PACKAGE);
        if (launch != null) {
            activity.runOnUiThread(() -> activity.startActivity(launch));
        }
    }

    /**
     * Has Termux run bash with {@code arguments}, in the background or in a session brought to
     * the front.
     */
    private static void runCommand(Context context, String[] arguments, boolean background)
            throws IOException {
        Intent intent = new Intent(RUN_COMMAND_ACTION);
        intent.setClassName(TERMUX_PACKAGE, RUN_COMMAND_SERVICE);
        intent.putExtra(EXTRA_PATH, BASH);
        intent.putExtra(EXTRA_ARGUMENTS, arguments);
        intent.putExtra(EXTRA_WORKDIR, TERMUX_HOME);
        intent.putExtra(EXTRA_BACKGROUND, background);
        if (!background) {
            intent.putExtra(EXTRA_SESSION_ACTION, SESSION_ACTION_SWITCH_AND_OPEN);
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent);
            } else {
                context.startService(intent);
            }
        } catch (SecurityException | IllegalStateException refused) {
            throw new IOException("Termux did not accept the command: " + refused, refused);
        }
    }

    private static List<String> triageArguments(Context context) throws IOException {
        return conversationArguments(
                AiModels.asset(context, "gemma/triage_system.md"),
                AiModels.asset(context, "gemma/triage_schema.json"));
    }

    /**
     * The arguments of a {@code claude -p} conversation over stream-json with
     * {@code systemPrompt} and nothing else: no tools, settings, MCP servers or saved session,
     * shaped by {@code jsonSchema} when that is not null. Flags without a value are written as
     * {@code --flag=}, so no argument is empty.
     */
    private static List<String> conversationArguments(String systemPrompt, String jsonSchema) {
        List<String> arguments = new ArrayList<>(Arrays.asList(
                "-p",
                "--input-format", "stream-json",
                "--output-format", "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--model", MODEL,
                "--system-prompt", systemPrompt,
                "--tools=",
                "--strict-mcp-config",
                "--mcp-config={\"mcpServers\":{}}",
                "--setting-sources=",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--permission-prompts", "none"));
        if (jsonSchema != null) {
            arguments.add("--json-schema");
            arguments.add(jsonSchema);
        }
        return arguments;
    }

    /** A stream-json user message: the JPEG at {@code imagePath}, when not empty, then text. */
    private static JSONObject userMessage(String text, String imagePath)
            throws IOException, JSONException {
        JSONArray content = new JSONArray();
        if (imagePath != null && !imagePath.isEmpty()) {
            JSONObject source = new JSONObject();
            source.put("type", "base64");
            source.put("media_type", "image/jpeg");
            source.put("data", Base64.encodeToString(readFile(imagePath), Base64.NO_WRAP));
            JSONObject image = new JSONObject();
            image.put("type", "image");
            image.put("source", source);
            content.put(image);
        }
        JSONObject textBlock = new JSONObject();
        textBlock.put("type", "text");
        textBlock.put("text", text);
        content.put(textBlock);
        JSONObject message = new JSONObject();
        message.put("role", "user");
        message.put("content", content);
        JSONObject line = new JSONObject();
        line.put("type", "user");
        line.put("message", message);
        return line;
    }

    private static byte[] readFile(String path) throws IOException {
        File file = new File(path);
        byte[] bytes = new byte[(int) file.length()];
        try (InputStream input = new FileInputStream(file)) {
            int offset = 0;
            while (offset < bytes.length) {
                int read = input.read(bytes, offset, bytes.length - offset);
                if (read < 0) {
                    throw new IOException("Could not read all of " + path);
                }
                offset += read;
            }
        }
        return bytes;
    }

    private static String costOf(JSONObject result) {
        return "$" + result.optDouble("total_cost_usd", 0.0);
    }

    /** What a conversation writes while Claude writes its reply. */
    private abstract static class Listener {
        /** The next piece of the reply's text. */
        void onText(String piece) {
        }

        /** The next piece of the structured output's JSON. */
        void onJson(String piece) {
        }
    }

    /**
     * One {@code claude} process that Termux runs and that connects back to this app. Its
     * connection is accepted and read on its own thread; {@link #send} waits for the reply's
     * result line.
     */
    private static final class Session {
        // Queued after the last result, when the process has ended or the connection closed.
        private static final JSONObject END = new JSONObject();
        private static final SecureRandom RANDOM = new SecureRandom();
        private static final InetAddress IPV4_LOOPBACK = ipv4Loopback();

        private final ServerSocket server;
        private final String token;
        private final String name;
        private final CountDownLatch connection = new CountDownLatch(1);
        private final LinkedBlockingQueue<JSONObject> results = new LinkedBlockingQueue<>();
        private final AtomicBoolean closed = new AtomicBoolean();
        private volatile Socket socket;
        private volatile Listener listener;
        private volatile boolean ended;
        private volatile int exitCode = -1;
        private volatile String stderr = "";

        private Session(ServerSocket server, String token, String name) {
            this.server = server;
            this.token = token;
            this.name = name;
        }

        /** Has Termux start {@code claude} with {@code claudeArguments}. Returns at once. */
        static Session start(Context context, String name, List<String> claudeArguments)
                throws IOException {
            // IPv4 loopback, where connect.sh connects: Android's getLoopbackAddress() is ::1.
            ServerSocket server = new ServerSocket(0, 1, IPV4_LOOPBACK);
            byte[] secret = new byte[16];
            RANDOM.nextBytes(secret);
            StringBuilder token = new StringBuilder();
            for (byte value : secret) {
                token.append(String.format("%02x", value & 0xff));
            }
            Session session = new Session(server, token.toString(), name);
            List<String> arguments = new ArrayList<>(Arrays.asList(
                    "-c",
                    AiModels.asset(context, "claude/connect.sh"),
                    "calibraw-claude",
                    Integer.toString(server.getLocalPort()),
                    session.token));
            arguments.addAll(claudeArguments);
            Thread reader = new Thread(session::run, name);
            reader.setDaemon(true);
            reader.start();
            try {
                runCommand(context, arguments.toArray(new String[0]), true);
            } catch (IOException error) {
                session.end();
                throw error;
            }
            return session;
        }

        /** Blocks until the process has connected, failing when Termux did not start it. */
        void awaitConnected() throws IOException, InterruptedException {
            connection.await(CONNECT_TIMEOUT_MS + TOKEN_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            if (socket == null) {
                throw new IOException("Termux did not start Claude Code; " + ending());
            }
        }

        boolean connected() {
            return socket != null;
        }

        /** Whether the process has ended or the connection is closed. */
        boolean hasEnded() {
            return ended || closed.get();
        }

        int exitCode() {
            return exitCode;
        }

        /**
         * Sends one user message and returns Claude's result line once the reply is done,
         * telling {@code listener} what it writes meanwhile.
         */
        JSONObject send(JSONObject message, Listener listener)
                throws IOException, InterruptedException {
            awaitConnected();
            this.listener = listener;
            try {
                OutputStream output = socket.getOutputStream();
                output.write((message.toString() + "\n").getBytes(StandardCharsets.UTF_8));
                output.flush();
                JSONObject result = results.poll(TURN_TIMEOUT_SECONDS, TimeUnit.SECONDS);
                if (result == null) {
                    end();
                    throw new IOException(
                            "Claude did not reply in " + TURN_TIMEOUT_SECONDS + " s");
                }
                if (result == END) {
                    throw new IOException("Claude Code stopped; " + ending());
                }
                if (result.optBoolean("is_error")) {
                    throw new IOException("Claude could not reply: " + result.optString("result"));
                }
                return result;
            } catch (InterruptedException interrupted) {
                // Ended, so a later send cannot read this turn's late reply as its own.
                end();
                Thread.currentThread().interrupt();
                throw interrupted;
            } catch (IOException error) {
                if (hasEnded()) {
                    throw new IOException("Claude Code stopped; " + ending(), error);
                }
                throw error;
            } finally {
                this.listener = null;
            }
        }

        /** Waits for the process to end; false when it has not after {@code seconds}. */
        boolean awaitEnd(long seconds) throws InterruptedException {
            long deadline = SystemClock.elapsedRealtime() + seconds * 1000;
            while (true) {
                long left = deadline - SystemClock.elapsedRealtime();
                if (left <= 0) {
                    return false;
                }
                if (results.poll(left, TimeUnit.MILLISECONDS) == END) {
                    return true;
                }
            }
        }

        /**
         * Ends the connection, which ends the process, and fails a waiting {@link #send}. Not
         * named close: the JNI contract check matches methods by name across all classes, and
         * Rust calls ReplayVideoEncoder.close.
         */
        void end() {
            if (!closed.compareAndSet(false, true)) {
                return;
            }
            closeQuietly(server);
            Socket connected = socket;
            if (connected != null) {
                closeQuietly(connected);
            }
            results.add(END);
        }

        /** How the process ended, for an error message. */
        String ending() {
            if (socket == null) {
                return "it did not connect back in " + CONNECT_TIMEOUT_MS / 1000 + " s";
            }
            if (exitCode < 0) {
                return "the connection closed";
            }
            return "exit code " + exitCode + (stderr.isEmpty() ? "" : ": " + stderr.trim());
        }

        private void run() {
            try {
                // Any app can connect to the loopback port, so connections without the token are
                // closed and the process's is waited for until the deadline.
                long deadline = SystemClock.elapsedRealtime() + CONNECT_TIMEOUT_MS;
                Socket accepted;
                BufferedReader reader;
                while (true) {
                    accepted = accept(deadline);
                    if (accepted == null) {
                        return;
                    }
                    try {
                        reader = new BufferedReader(new InputStreamReader(
                                accepted.getInputStream(), StandardCharsets.UTF_8));
                    } catch (IOException error) {
                        closeQuietly(accepted);
                        throw error;
                    }
                    if (token.equals(readToken(reader))) {
                        break;
                    }
                    closeQuietly(accepted);
                    Log.w(LOG_TAG, name + ": a connection without the token was refused");
                }
                closeQuietly(server);
                socket = accepted;
                // end() closes the socket once it is set; one that ran before is seen here, so
                // a send never writes to a session that has ended.
                if (closed.get()) {
                    closeQuietly(accepted);
                    return;
                }
                accepted.setSoTimeout(0);
                connection.countDown();
                for (String line; (line = reader.readLine()) != null; ) {
                    handle(line);
                }
            } catch (IOException error) {
                if (!closed.get()) {
                    Log.w(LOG_TAG, name + ": the connection failed", error);
                }
            } finally {
                closeQuietly(server);
                ended = true;
                connection.countDown();
                results.add(END);
            }
        }

        /**
         * Accepts the next connection, or returns null when none comes before {@code deadline}
         * (elapsedRealtime) or the session is closed.
         */
        private Socket accept(long deadline) throws IOException {
            long left = deadline - SystemClock.elapsedRealtime();
            try {
                // A timeout of 0 would wait forever.
                if (left <= 0) {
                    throw new SocketTimeoutException();
                }
                server.setSoTimeout((int) left);
                Socket accepted = server.accept();
                try {
                    accepted.setSoTimeout(TOKEN_TIMEOUT_MS);
                } catch (IOException error) {
                    closeQuietly(accepted);
                    throw error;
                }
                return accepted;
            } catch (SocketTimeoutException timeout) {
                Log.w(LOG_TAG, name + ": Termux did not start Claude Code in "
                        + CONNECT_TIMEOUT_MS / 1000 + " s");
                return null;
            } catch (IOException error) {
                if (closed.get()) {
                    return null;
                }
                throw error;
            }
        }

        /**
         * The connection's first line, read only as far as one character past the token's length
         * so a connection cannot make it buffer without end, or null when it is longer, does not
         * come in time or the connection fails.
         */
        private String readToken(BufferedReader reader) {
            StringBuilder line = new StringBuilder();
            try {
                while (line.length() <= token.length()) {
                    int next = reader.read();
                    if (next < 0 || next == '\n') {
                        return line.toString();
                    }
                    line.append((char) next);
                }
            } catch (IOException error) {
                if (!closed.get()) {
                    Log.w(LOG_TAG, name + ": a connection did not send the token", error);
                }
            }
            return null;
        }

        private void handle(String line) {
            JSONObject event;
            try {
                event = new JSONObject(line);
            } catch (JSONException notJson) {
                // claude auth status writes its JSON over several lines.
                return;
            }
            switch (event.optString("type")) {
                case "stream_event":
                    handleStreamEvent(event.optJSONObject("event"));
                    break;
                case "result":
                    results.add(event);
                    break;
                case "calibraw_exit":
                    String written = new String(
                            Base64.decode(event.optString("stderr"), Base64.DEFAULT),
                            StandardCharsets.UTF_8);
                    stderr = written;
                    exitCode = event.optInt("code", -1);
                    if (exitCode != 0) {
                        Log.w(LOG_TAG, name + ": Claude Code exited with " + exitCode + ": "
                                + written);
                    }
                    break;
                default:
                    break;
            }
        }

        private void handleStreamEvent(JSONObject event) {
            Listener current = listener;
            if (event == null || current == null
                    || !"content_block_delta".equals(event.optString("type"))) {
                return;
            }
            JSONObject delta = event.optJSONObject("delta");
            if (delta == null) {
                return;
            }
            switch (delta.optString("type")) {
                case "text_delta":
                    current.onText(delta.optString("text"));
                    break;
                case "input_json_delta":
                    current.onJson(delta.optString("partial_json"));
                    break;
                default:
                    break;
            }
        }

        private static InetAddress ipv4Loopback() {
            try {
                return InetAddress.getByAddress(new byte[] {127, 0, 0, 1});
            } catch (UnknownHostException impossible) {
                // getByAddress throws only for an address of the wrong length.
                throw new AssertionError(impossible);
            }
        }

        private static void closeQuietly(Closeable closeable) {
            try {
                closeable.close();
            } catch (IOException ignored) {
                // Already closed.
            }
        }
    }
}
