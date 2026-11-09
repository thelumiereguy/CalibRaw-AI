package de.duecki.calibraw.ai.gemma;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;

import com.google.ai.edge.litertlm.Backend;
import com.google.ai.edge.litertlm.Engine;
import com.google.ai.edge.litertlm.EngineConfig;
import com.google.ai.edge.litertlm.ExperimentalFlags;

import java.io.File;
import java.io.IOException;

/** Loads Gemma 4 E4B into a LiteRT-LM engine on the GPU, from the app's shared media storage. */
final class GemmaEngine {
    private static final String LOG_TAG = "CalibRaw";
    private static final String MODEL_FILE = "models/gemma-4-E4B-it.litertlm";
    // Tokens a conversation holds. A phase of an editor request took up to 2,447 (its system
    // prompt, a preview per message). LiteRT-LM's GPU backend reserves the whole context in every
    // layer, sliding-window ones included (LiteRT-LM #4096), so the reservation grows with this:
    // 16384 ran the phone out of memory after a triage, and 8192 left it swapping.
    private static final int MAX_NUM_TOKENS = 4096;

    private GemmaEngine() {}

    /** A newly loaded engine; takes seconds. */
    static Engine load(Context context) throws IOException {
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
        Engine engine = new Engine(new EngineConfig(
                model.getAbsolutePath(),
                new Backend.GPU(),
                new Backend.GPU(),
                null,
                MAX_NUM_TOKENS,
                null,
                cacheDir.getAbsolutePath()));
        engine.initialize();
        Log.i(LOG_TAG, "Gemma engine loaded in " + (SystemClock.elapsedRealtime() - started)
                + " ms");
        return engine;
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
}
