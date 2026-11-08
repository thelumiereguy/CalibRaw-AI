package de.duecki.calibraw;

import android.app.Activity;
import android.content.Context;
import android.util.Log;

import java.io.IOException;
import java.util.Arrays;

/**
 * Claude Code's setup in Termux, which Settings checks and walks through one missing step at a
 * time: Termux's permission, allow-external-apps, the install and the sign-in.
 */
final class ClaudeSetup {
    private static final String LOG_TAG = "CalibRaw";
    // Each request is set by the CalibRawActivity or one of its helpers; see their constants.
    private static final int RUN_COMMAND_PERMISSION_REQUEST = 1006;
    // claude auth status answers within a few seconds of starting.
    private static final long STATUS_TIMEOUT_SECONDS = 60;

    /** How far the setup is, by the codes Rust reads (ClaudeCodeStatus in platform/service.rs). */
    enum Status {
        NO_TERMUX("no_termux"),
        NO_RUN_COMMAND("no_run_command"),
        NO_PERMISSION("no_permission"),
        BLOCKED("blocked"),
        NOT_INSTALLED("not_installed"),
        SIGNED_OUT("signed_out"),
        READY("ready"),
        FAILED("failed");

        private final String code;

        Status(String code) {
            this.code = code;
        }

        String code() {
            return code;
        }
    }

    /** A step Settings offers, by the names Rust uses (ClaudeCodeSetup in platform/service.rs). */
    enum Step {
        PERMISSION("permission"),
        INSTALL("install"),
        SIGN_IN("sign_in");

        private final String name;

        Step(String name) {
            this.name = name;
        }

        /** The step Rust names {@code name}, or null for a name it does not use. */
        static Step named(String name) {
            for (Step step : values()) {
                if (step.name.equals(name)) {
                    return step;
                }
            }
            return null;
        }
    }

    private ClaudeSetup() {}

    /**
     * The first step still missing, or {@link Status#READY}. Blocks while Termux runs
     * {@code claude auth status}.
     */
    static Status status(Context context) {
        if (!Termux.installed(context)) {
            return Status.NO_TERMUX;
        }
        if (!Termux.offersRunCommand(context)) {
            return Status.NO_RUN_COMMAND;
        }
        if (!Termux.mayRunCommands(context)) {
            return Status.NO_PERMISSION;
        }
        ClaudeSession session;
        try {
            session = ClaudeSession.start(
                    context, "Claude status", Arrays.asList("auth", "status"));
        } catch (IOException error) {
            Log.w(LOG_TAG, "Termux did not run the Claude Code status check", error);
            return Status.BLOCKED;
        }
        try {
            if (!session.awaitEnd(STATUS_TIMEOUT_SECONDS) || !session.connected()) {
                // Termux refuses commands from other apps until allow-external-apps is set.
                return Status.BLOCKED;
            }
            switch (session.exitCode()) {
                case 0:
                    return Status.READY;
                case 1:
                    return Status.SIGNED_OUT;
                case 127:
                    return Status.NOT_INSTALLED;
                default:
                    Log.w(LOG_TAG, "Claude Code status check failed: " + session.ending());
                    return Status.FAILED;
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            return Status.FAILED;
        } finally {
            session.end();
        }
    }

    /**
     * Runs {@code step}: asks for Termux's RUN_COMMAND permission, or installs Claude Code
     * (assets/claude/install.sh) or runs its sign-in, each in a Termux session brought to the
     * front.
     */
    static void run(Activity activity, Step step) throws IOException {
        switch (step) {
            case PERMISSION:
                activity.runOnUiThread(() -> activity.requestPermissions(
                        new String[] {Termux.RUN_COMMAND_PERMISSION},
                        RUN_COMMAND_PERMISSION_REQUEST));
                return;
            case INSTALL:
                Termux.runInFront(activity, new String[] {
                        "-c", Assets.text(activity, "claude/install.sh"), "calibraw-install"});
                return;
            case SIGN_IN:
                Termux.runInFront(activity, new String[] {
                        "-c",
                        "claude auth login && printf '\\nSigned in. Go back to CalibRaw and tap"
                                + " Check again.\\n'",
                        "calibraw-sign-in"});
                return;
        }
    }
}
