package de.duecki.calibraw;

/**
 * A model that triages each opened photo and runs the AI editor's chat about it: Gemma on the
 * device ({@link GemmaBridge}) or Claude Code in Termux ({@link ClaudeBridge}). Rust reaches the
 * selected one through {@link AiModels}, which forwards these calls to it.
 */
interface AiModel {
    /**
     * Starts what the model needs before its first call, in the background, when the photographer
     * selects it or the app starts with it selected. Returns at once.
     */
    void activate();

    /**
     * Frees what the model holds when another model is selected, cancelling the editor's running
     * turn. Returns at once.
     */
    void release();

    /**
     * Tells the model whether Develop shows, where the triage and the AI editor run: Claude Code
     * keeps processes only while it does. Returns at once.
     */
    default void editorShown(boolean shown) {
    }

    /**
     * Returns the triage card for the JPEG at {@code imagePath} as JSON, shaped by
     * assets/gemma/triage_schema.json. {@code hint} is the photographer's own genre, or empty. The
     * card's genre and emoji are reported earlier through {@link AiModels#reportTriageHeadline}.
     */
    String triage(String imagePath, String hint, long requestId, int badgePx) throws Exception;

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
     * Sends {@code text}, after the JPEGs at {@code imagePaths}, one path per line or empty for
     * none, to the editor conversation and returns the reply as
     * {@code {"text": ..., "tool_calls": [{"name": ..., "arguments": {...}}]}}. Gemma is sent one
     * JPEG at most. The reply's text also streams through {@link AiModels#reportEditorText},
     * tagged with {@code requestId}.
     */
    String editorSend(long requestId, String text, String imagePaths) throws Exception;

    /**
     * Ends the editor conversation, cancelling a running turn, which then fails. Returns at once.
     */
    void closeEditor();
}
