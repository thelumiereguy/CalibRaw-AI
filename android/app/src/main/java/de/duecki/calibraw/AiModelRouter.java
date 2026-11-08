package de.duecki.calibraw;

import android.content.Context;
import android.util.Log;

import java.io.IOException;

/**
 * The AI model the photographer selected in Settings, one for the process: a model keeps what it
 * loads, Gemma's engine and Claude Code's conversations, while an activity is recreated, so the
 * selection outlives activities too. Selecting another model releases the one before.
 */
final class AiModelRouter {
    private static final String LOG_TAG = "CalibRaw";

    /** The models Rust selects, by the names it uses (AiEditorModel in platform/service.rs). */
    enum Kind {
        GEMMA("gemma"),
        CLAUDE("claude");

        private final String name;

        Kind(String name) {
            this.name = name;
        }

        /** The model Rust names {@code name}, or null for a name it does not use. */
        static Kind named(String name) {
            for (Kind kind : values()) {
                if (kind.name.equals(name)) {
                    return kind;
                }
            }
            return null;
        }
    }

    // Guarded by AiModelRouter.class.
    private static AiModelRouter shared;

    private final Context context;
    // Guarded by this.
    private Kind selectedKind;
    // The Claude model the selected ClaudeBridge runs, or null with Gemma. Guarded by this.
    private String selectedClaudeModel;
    // Written under this; volatile so a call reads it without waiting for a selection.
    private volatile AiModel selected;
    // Whether Develop shows, so a model selected meanwhile is told. Guarded by this.
    private boolean editorShown;

    private AiModelRouter(Context context) {
        this.context = context;
    }

    /** The process's router. */
    static synchronized AiModelRouter shared(Context context) {
        if (shared == null) {
            shared = new AiModelRouter(context.getApplicationContext());
        }
        return shared;
    }

    /**
     * Makes {@code kind} the model every later call goes to, Claude Code running
     * {@code claudeModel} (its --model value), releasing the one selected before and activating
     * the new one in the background. Gemma ignores {@code claudeModel}; another Claude model
     * replaces Claude Code's bridge, so no conversation started ahead of time keeps the model
     * before. Returns at once.
     */
    synchronized void select(Kind kind, String claudeModel) {
        boolean claude = kind == Kind.CLAUDE;
        if (kind == selectedKind && (!claude || claudeModel.equals(selectedClaudeModel))) {
            return;
        }
        AiModel model = claude ? new ClaudeBridge(context, claudeModel) : GemmaBridge.shared(context);
        AiModel previous = selected;
        selected = model;
        selectedKind = kind;
        selectedClaudeModel = claude ? claudeModel : null;
        if (previous != null) {
            previous.release();
        }
        model.activate();
        if (editorShown) {
            model.editorShown(true);
        }
        Log.i(LOG_TAG, "AI model selected: " + kind.name + (claude ? " " + claudeModel : ""));
    }

    /**
     * Tells the selected model, and any selected later, whether Develop shows: Claude Code starts
     * its conversations when it does and ends every one when it no longer does. Returns at once.
     */
    synchronized void editorShown(boolean shown) {
        editorShown = shown;
        AiModel model = selected;
        if (model != null) {
            model.editorShown(shown);
        }
    }

    /**
     * Releases the selected model and forgets the selection, ending Claude Code's conversations
     * and their processes. Returns at once.
     */
    synchronized void releaseAll() {
        AiModel previous = selected;
        selected = null;
        selectedKind = null;
        selectedClaudeModel = null;
        editorShown = false;
        if (previous != null) {
            previous.release();
        }
    }

    /** The selected model. */
    AiModel model() throws IOException {
        AiModel model = selected;
        if (model == null) {
            throw new IOException("No AI model is selected");
        }
        return model;
    }

    /** Ends the selected model's editor conversation, if a model is selected. Returns at once. */
    void closeEditor() {
        AiModel model = selected;
        if (model != null) {
            model.closeEditor();
        }
    }
}
