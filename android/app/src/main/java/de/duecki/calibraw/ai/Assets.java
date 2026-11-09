package de.duecki.calibraw.ai;

import android.content.Context;

import de.duecki.calibraw.BoundedStreams;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/** Text files the APK bundles in assets/, such as the AI models' prompts and Termux scripts. */
public final class Assets {
    private Assets() {}

    /** The UTF-8 text of the asset at {@code path}. */
    public static String text(Context context, String path) throws IOException {
        try (InputStream input = context.getAssets().open(path)) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            BoundedStreams.copy(input, output, Long.MAX_VALUE, "Asset " + path + " is too large");
            return output.toString(StandardCharsets.UTF_8.name());
        }
    }
}
