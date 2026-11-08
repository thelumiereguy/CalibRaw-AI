package de.duecki.calibraw;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * For the evals (research/gemma-evals) only: the phone's log quota drops most of the app's lines
 * while LiteRT logs, so each triage card and the editor's messages, replies and request ends are
 * appended to files/gemma-trace.txt, in a debuggable build and only while files/gemma-trace.on
 * exists, which the evals create for their run. A release build never traces. The app adds what
 * each request set (app/ai/editor/chat/gemma_brief.rs) while the same switch exists, which only
 * run-as, and so only a debuggable build, can make.
 */
final class GemmaTrace {
    private static final String LOG_TAG = "CalibRaw";
    private static final String SWITCH = "gemma-trace.on";

    // Null when the build is not debuggable.
    private final File file;

    GemmaTrace(Context context) {
        boolean debuggable =
                (context.getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
        file = debuggable ? new File(context.getFilesDir(), "gemma-trace.txt") : null;
    }

    /** Appends {@code line}, on one line, while the evals' switch exists. */
    synchronized void write(String line) {
        if (file == null || !new File(file.getParentFile(), SWITCH).isFile()) {
            return;
        }
        try (PrintWriter out = new PrintWriter(new FileWriter(file, true))) {
            out.println(SystemClock.elapsedRealtime() + " " + line.replace('\n', '|'));
        } catch (IOException error) {
            Log.w(LOG_TAG, "Could not write the Gemma trace", error);
        }
    }
}
