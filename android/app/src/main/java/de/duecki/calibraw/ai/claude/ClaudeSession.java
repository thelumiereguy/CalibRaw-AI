package de.duecki.calibraw.ai.claude;

import android.content.Context;
import android.os.SystemClock;
import android.util.Base64;
import android.util.Log;

import de.duecki.calibraw.ai.Assets;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.IOException;
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

/**
 * One {@code claude} process that Termux runs through assets/claude/connect.sh, which connects
 * back to this app on an IPv4 loopback port with a one-time token and then speaks stream-json over
 * that connection. The connection is accepted and read on the session's own thread; {@link #send}
 * waits for the reply's result line.
 */
public final class ClaudeSession {
    private static final String LOG_TAG = "CalibRaw";
    // Termux starting a process took about 3.4 s on the spike phone; a cold Termux takes longer.
    private static final int CONNECT_TIMEOUT_MS = 30_000;
    // The connection's first line, the token, arrives at once after it connects.
    private static final int TOKEN_TIMEOUT_MS = 5_000;
    // A turn took about 5 s on the spike phone.
    private static final long TURN_TIMEOUT_SECONDS = 120;
    // Before GENERATION, which is drawn from it as the class loads.
    private static final SecureRandom RANDOM = new SecureRandom();
    // This process's mark on every claude process it starts (connect.sh), so a later process can
    // end the ones this process leaves behind when Android kills it (cleanup.sh).
    private static final String GENERATION = randomHex();
    // Queued after the last result, when the process has ended or the connection closed.
    private static final JSONObject END = new JSONObject();
    private static final InetAddress IPV4_LOOPBACK = ipv4Loopback();

    /** What a session is told while Claude writes its reply. */
    abstract static class Listener {
        /** The next piece of the reply's text. */
        void onText(String piece) {
        }

        /** The next piece of the structured output's JSON. */
        void onJson(String piece) {
        }
    }

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

    private ClaudeSession(ServerSocket server, String token, String name) {
        this.server = server;
        this.token = token;
        this.name = name;
    }

    /**
     * Has Termux start {@code claude} with {@code claudeArguments}, as the thread named
     * {@code name} waits for it to connect. Returns at once.
     */
    static ClaudeSession start(Context context, String name, List<String> claudeArguments)
            throws IOException {
        String connect = Assets.text(context, "claude/connect.sh");
        // IPv4 loopback, where connect.sh connects: Android's getLoopbackAddress() is ::1.
        ServerSocket server = new ServerSocket(0, 1, IPV4_LOOPBACK);
        ClaudeSession session = new ClaudeSession(server, randomHex(), name);
        List<String> arguments = new ArrayList<>(Arrays.asList(
                "-c",
                connect,
                "calibraw-claude",
                Integer.toString(server.getLocalPort()),
                session.token,
                GENERATION));
        arguments.addAll(claudeArguments);
        Thread reader = new Thread(session::run, name);
        reader.setDaemon(true);
        reader.start();
        try {
            Termux.runInBackground(context, arguments.toArray(new String[0]));
        } catch (IOException error) {
            session.end();
            throw error;
        }
        return session;
    }

    /**
     * Has Termux end the claude processes that earlier runs of this app started, when CalibRaw may
     * run commands in Termux. When Android kills the app, their connection is not closed, so they
     * would wait on it without end. This process's own are spared, so it does not matter whether
     * this runs before or after the sessions it starts. Returns at once.
     */
    public static void endEarlierRuns(Context context) {
        if (!Termux.mayRunCommands(context)) {
            return;
        }
        try {
            Termux.runInBackground(context, new String[] {
                    "-c", Assets.text(context, "claude/cleanup.sh"), "calibraw-cleanup",
                    GENERATION});
            Log.i(LOG_TAG, "Asked Termux to end Claude Code processes of earlier runs");
        } catch (IOException error) {
            Log.w(LOG_TAG, "Could not end Claude Code processes of earlier runs", error);
        }
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
     * Sends one user message and returns Claude's result line once the reply is done, telling
     * {@code listener} what it writes meanwhile.
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
                throw new IOException("Claude did not reply in " + TURN_TIMEOUT_SECONDS + " s");
            }
            if (result == END) {
                throw new IOException("Claude Code stopped; " + ending());
            }
            if (result.optBoolean("is_error")) {
                throw new IOException(
                        "Claude could not reply: " + withAppHints(result.optString("result")));
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
     * Ends the connection, which ends the process, and fails a waiting {@link #send}. Not named
     * close: the JNI contract check matches methods by name across all classes, and Rust calls
     * ReplayVideoEncoder.close.
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

    /** 16 random bytes as 32 hex digits: a session's token, or the process's generation. */
    private static String randomHex() {
        byte[] secret = new byte[16];
        RANDOM.nextBytes(secret);
        StringBuilder hex = new StringBuilder();
        for (byte value : secret) {
            hex.append(String.format("%02x", value & 0xff));
        }
        return hex.toString();
    }

    /**
     * Claude Code's error text as the chat shows it. For a model it cannot run, Claude Code
     * suggests its /model command ("Run /model to pick a different model."), which these
     * conversations turn off, so that sentence points at Settings instead.
     */
    private static String withAppHints(String error) {
        return error.replaceAll("Run /model\\b[^.]*\\.?",
                "Pick another Claude model in Settings, under AI editor.");
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
            // end() closes the socket once it is set; one that ran before is seen here, so a send
            // never writes to a session that has ended.
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
     * The connection's first line, read only as far as one character past the token's length so a
     * connection cannot make it buffer without end, or null when it is longer, does not come in
     * time or the connection fails.
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
