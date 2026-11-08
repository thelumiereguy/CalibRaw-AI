package de.duecki.calibraw;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkCapabilities;

/** The device's network, as Android sees it. */
final class Connectivity {
    private Connectivity() {}

    /** Whether the default network reaches the internet, as Android validated it. */
    static boolean online(Context context) {
        ConnectivityManager connectivity = context.getSystemService(ConnectivityManager.class);
        if (connectivity == null) {
            return false;
        }
        NetworkCapabilities capabilities =
                connectivity.getNetworkCapabilities(connectivity.getActiveNetwork());
        return capabilities != null
                && capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                && capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED);
    }
}
