package de.duecki.calibraw.ai;

import java.util.List;

/**
 * A model that triages each opened photo and runs the AI editor's chat about it: Gemma on the
 * device ({@link de.duecki.calibraw.ai.gemma.GemmaBridge}) or Claude Code in Termux
 * ({@link de.duecki.calibraw.ai.claude.ClaudeBridge}). The selected one is reached through
 * {@link AiModelRouter}. A model reports what it writes while it writes it through the listeners
 * each call is given, so it knows nothing of who listens.
 */
public interface AiModel {
    /** Told a triage card's genre and emoji once the model has written both. */
    interface HeadlineListener {
        void onHeadline(String genre, String emoji);
    }

    /** Told each next piece of an editor reply's text while the model writes it. */
    interface TextListener {
        void onText(String piece);
    }

    /** The message that asks for a triage, with the photographer's own genre when there is one. */
    static String triageRequest(String hint) {
        String request = "Triage this photo.";
        return hint == null || hint.isEmpty()
                ? request
                : request + " The photographer says the genre is: " + hint + ".";
    }

    /**
     * Starts what the model needs before its first call, in the background, when it is selected.
     * Returns at once. Nothing by default.
     */
    default void activate() {
    }

    /**
     * Frees what the model holds when another model is selected, cancelling the editor's running
     * turn. Returns at once.
     */
    void release();

    /**
     * Tells the model whether Develop shows, where the triage and the AI editor run. Returns at
     * once. Nothing by default.
     */
    default void editorShown(boolean shown) {
    }

    /**
     * Returns the triage card for the JPEG at {@code imagePath} as JSON, shaped by
     * assets/gemma/triage_schema.json. {@code hint} is the photographer's own genre, or empty. The
     * card's genre and emoji go to {@code headline} as soon as they are written.
     */
    String triage(String imagePath, String hint, HeadlineListener headline) throws Exception;

    /**
     * Has the model ready a conversation with the editor's {@code systemPrompt} in the background,
     * so the open photo's first request does not wait for it. Returns at once.
     */
    void prepareEditor(String systemPrompt);

    /**
     * Starts a fresh editor conversation that holds only {@code systemPrompt}, replacing the
     * earlier one.
     */
    void startEditorRequest(String systemPrompt) throws Exception;

    /**
     * Sends {@code text}, after the JPEGs at {@code imagePaths}, to the editor conversation and
     * returns the reply as {@code {"text": ..., "tool_calls": [{"name": ..., "arguments": {...}}]}},
     * the JSON Rust reads. Gemma is sent one JPEG at most. The reply's text also goes to
     * {@code replyText} while the model writes it.
     */
    String editorSend(String text, List<String> imagePaths, TextListener replyText)
            throws Exception;

    /**
     * Ends the editor conversation, cancelling a running turn, which then fails. Returns at once.
     */
    void closeEditor();
}
