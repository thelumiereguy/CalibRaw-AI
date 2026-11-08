package de.duecki.calibraw;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;

import java.io.IOException;

/**
 * The Termux app, which runs commands for other apps through its RUN_COMMAND service once they
 * hold its permission and allow-external-apps is set in Termux.
 */
final class Termux {
    static final String RUN_COMMAND_PERMISSION = "com.termux.permission.RUN_COMMAND";
    private static final String PACKAGE = "com.termux";
    private static final String RUN_COMMAND_SERVICE = "com.termux.app.RunCommandService";
    private static final String RUN_COMMAND_ACTION = "com.termux.RUN_COMMAND";
    private static final String EXTRA_PATH = "com.termux.RUN_COMMAND_PATH";
    private static final String EXTRA_ARGUMENTS = "com.termux.RUN_COMMAND_ARGUMENTS";
    private static final String EXTRA_WORKDIR = "com.termux.RUN_COMMAND_WORKDIR";
    private static final String EXTRA_BACKGROUND = "com.termux.RUN_COMMAND_BACKGROUND";
    private static final String EXTRA_SESSION_ACTION = "com.termux.RUN_COMMAND_SESSION_ACTION";
    // Termux's action that opens a new session in front and switches to it.
    private static final String SESSION_ACTION_SWITCH_AND_OPEN = "0";
    private static final String BASH = "/data/data/com.termux/files/usr/bin/bash";
    private static final String HOME = "/data/data/com.termux/files/home";

    private Termux() {}

    static boolean installed(Context context) {
        try {
            context.getPackageManager().getPackageInfo(PACKAGE, 0);
            return true;
        } catch (PackageManager.NameNotFoundException missing) {
            return false;
        }
    }

    /** Whether this Termux offers RUN_COMMAND; its Google Play build does not. */
    static boolean offersRunCommand(Context context) {
        try {
            context.getPackageManager().getPermissionInfo(RUN_COMMAND_PERMISSION, 0);
            return true;
        } catch (PackageManager.NameNotFoundException missing) {
            return false;
        }
    }

    /** Whether CalibRaw holds the permission to run commands in Termux. */
    static boolean mayRunCommands(Context context) {
        return context.checkSelfPermission(RUN_COMMAND_PERMISSION)
                == PackageManager.PERMISSION_GRANTED;
    }

    /** Has Termux run bash with {@code arguments} in the background. Returns at once. */
    static void runInBackground(Context context, String[] arguments) throws IOException {
        runCommand(context, arguments, true);
    }

    /**
     * Has Termux run bash with {@code arguments} in a new session and brings Termux to the front,
     * so the photographer sees it run. Returns at once.
     */
    static void runInFront(Activity activity, String[] arguments) throws IOException {
        runCommand(activity, arguments, false);
        Intent launch = activity.getPackageManager().getLaunchIntentForPackage(PACKAGE);
        if (launch != null) {
            activity.runOnUiThread(() -> activity.startActivity(launch));
        }
    }

    private static void runCommand(Context context, String[] arguments, boolean background)
            throws IOException {
        Intent intent = new Intent(RUN_COMMAND_ACTION);
        intent.setClassName(PACKAGE, RUN_COMMAND_SERVICE);
        intent.putExtra(EXTRA_PATH, BASH);
        intent.putExtra(EXTRA_ARGUMENTS, arguments);
        intent.putExtra(EXTRA_WORKDIR, HOME);
        intent.putExtra(EXTRA_BACKGROUND, background);
        if (!background) {
            intent.putExtra(EXTRA_SESSION_ACTION, SESSION_ACTION_SWITCH_AND_OPEN);
        }
        try {
            context.startForegroundService(intent);
        } catch (SecurityException | IllegalStateException refused) {
            throw new IOException("Termux did not accept the command: " + refused, refused);
        }
    }
}
