package de.duecki.calibraw;

import android.os.SystemClock;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A triage card that a model writes piece by piece, as JSON shaped by
 * assets/gemma/triage_schema.json. The card's genre and emoji come first; they are reported once
 * both are written, while the rest of the card takes seconds more. Pieces may arrive on any thread.
 */
final class TriageCardStream {
    private static final String LOG_TAG = "CalibRaw";
    // A string field of the card once its closing quote is written.
    private static final Pattern GENRE_FIELD =
            Pattern.compile("\"genre\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");
    private static final Pattern EMOJI_FIELD =
            Pattern.compile("\"emoji\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");

    private final String modelName;
    // SystemClock.elapsedRealtime() when the triage started, for the log.
    private final long started;
    private final AiModel.HeadlineListener headline;
    // Guarded by this.
    private final StringBuilder card = new StringBuilder();
    private boolean headlineReported;

    /** A card that {@code modelName} started writing at {@code started} (elapsedRealtime). */
    TriageCardStream(String modelName, long started, AiModel.HeadlineListener headline) {
        this.modelName = modelName;
        this.started = started;
        this.headline = headline;
    }

    /** Adds the card's next piece, reporting the genre and emoji once both are written. */
    void append(String piece) {
        String genre;
        String emoji;
        synchronized (this) {
            card.append(piece);
            if (headlineReported) {
                return;
            }
            String text = card.toString();
            genre = completeField(text, GENRE_FIELD);
            emoji = completeField(text, EMOJI_FIELD);
            if (genre == null || emoji == null) {
                return;
            }
            headlineReported = true;
        }
        Log.i(LOG_TAG, modelName + " triage headline after "
                + (SystemClock.elapsedRealtime() - started) + " ms");
        headline.onHeadline(genre, emoji);
    }

    /** The card as written so far. */
    synchronized String text() {
        return card.toString();
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
}
