package de.duecki.calibraw.ai;

import android.app.Activity;
import android.util.Log;

import de.duecki.calibraw.ai.claude.ClaudeSession;
import de.duecki.calibraw.ai.claude.ClaudeSetup;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * The AI editor's Java side as Rust calls it, through CalibRawActivity's aiModels field
 * (crates/calibraw-ffi/src/android/ai_models.rs): each method hands the call to the class that
 * owns it, and the native callbacks report what a model writes while it writes it. Rust calls
 * these synchronously from its worker threads, so they keep its JNI names and signatures.
 */
public final class AiModels {
    private static final String LOG_TAG = "CalibRaw";

    private final Activity activity;
    private final AiModelRouter router;

    public AiModels(Activity activity) {
        this.activity = activity;
        this.router = AiModelRouter.shared(activity);
    }

    /** Selects the model Rust names "gemma" or "claude" ({@link AiModelRouter#select}). */
    public void select(String name, String claudeModel) {
        AiModelRouter.Kind kind = AiModelRouter.Kind.named(name);
        if (kind == null) {
            Log.w(LOG_TAG, "Unknown AI model " + name);
            return;
        }
        router.select(kind, claudeModel);
    }

    public void editorShown(boolean shown) {
        router.editorShown(shown);
    }

    /** Has Termux end the Claude Code processes earlier runs of the app left behind. */
    public void endStrayClaudeProcesses() {
        ClaudeSession.endEarlierRuns(activity);
    }

    /**
     * Releases the selected model once the activity has finished: winit cannot start again in
     * this process (CalibRawActivity.handleBack), so nothing would use it. Returns at once.
     */
    public void releaseOnFinish() {
        router.releaseAll();
    }

    /**
     * The triage card for the JPEG at {@code imagePath}; its genre and emoji reach Rust earlier,
     * tagged with {@code requestId} and with the emoji drawn in a {@code badgePx} square.
     */
    public String triage(String imagePath, String hint, long requestId, int badgePx)
            throws Exception {
        return router.model().triage(imagePath, hint, (genre, emoji) -> nativeOnTriageHeadline(
                requestId, genre, emoji, EmojiRenderer.draw(emoji, badgePx)));
    }

    public void prepareEditor(String systemPrompt) throws IOException {
        router.model().prepareEditor(systemPrompt);
    }

    public void startEditorRequest(String systemPrompt) throws Exception {
        router.model().startEditorRequest(systemPrompt);
    }

    /**
     * Sends {@code text} after the JPEGs at {@code imagePaths}, one path per line or empty for
     * none; the reply's text reaches Rust while it is written, tagged with {@code requestId}.
     */
    public String editorSend(long requestId, String text, String imagePaths) throws Exception {
        return router.model().editorSend(
                text, lines(imagePaths), piece -> nativeOnEditorText(requestId, piece));
    }

    public void closeEditor() {
        router.closeEditor();
    }

    /**
     * How far Claude Code in Termux is set up, as a {@link ClaudeSetup.Status} code. Blocks for up
     * to a minute while Termux starts, so Rust runs it on a worker.
     */
    public String claudeStatus() {
        return ClaudeSetup.status(activity).code();
    }

    /** Runs the setup step Rust names "permission", "install" or "sign_in". */
    public void claudeSetup(String step) throws Exception {
        ClaudeSetup.Step named = ClaudeSetup.Step.named(step);
        if (named == null) {
            throw new IOException("Unknown Claude Code setup step " + step);
        }
        ClaudeSetup.run(activity, named);
    }

    /** Whether the default network reaches the internet. Claude Code needs it; Gemma does not. */
    public boolean online() {
        return Connectivity.online(activity);
    }

    public byte[] renderEmoji(String emoji, int sizePx) {
        return EmojiRenderer.draw(emoji, sizePx);
    }

    /** The non-empty lines of {@code text}, or none when it is null. */
    private static List<String> lines(String text) {
        List<String> lines = new ArrayList<>();
        for (String line : text == null ? new String[0] : text.split("\n")) {
            if (!line.isEmpty()) {
                lines.add(line);
            }
        }
        return lines;
    }

    private static native void nativeOnTriageHeadline(
            long requestId, String genre, String emoji, byte[] badge);

    private static native void nativeOnEditorText(long requestId, String text);
}
