// app/src/main/java/io/scoregenius/app/LauncherActivity.java
package io.scoregenius.app;

import android.app.Activity;
import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.Resources;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.view.animation.AnimationUtils;
import android.view.animation.Interpolator;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.splashscreen.SplashScreen;

import com.google.androidbrowserhelper.trusted.TwaLauncher;
import androidx.browser.customtabs.CustomTabsIntent;

public class LauncherActivity extends Activity {

    private static final String TAG = "LauncherActivity";
    private static final String ORIGIN = "https://scoregenius.io";
    private static final String PREFS = "sg_launch_prefs";
    private static final String KEY_LAST_LAUNCH = "last_launch_ms";

    // Heuristic: treat as "cold" if user hasn’t launched in this many hours
    private static final long COLD_THRESHOLD_MS = 6L * 60L * 60L * 1000L;

    private static final int MIN_VISIBLE_FADE_MS = 200; // ensure some of the outro is visible

    private @Nullable TwaLauncher twaLauncher;
    private boolean launched = false;
    private volatile boolean splashHeld = true;

    // We will only finish() after onStop when this is true
    private boolean attemptedExternalLaunch = false;

    // Timings from resources (with safe defaults)
    private int SPLASH_MIN_MS;
    private int SPLASH_EXIT_FADE_MS;
    private int SPLASH_CONTENT_FADE_IN_MS;
    private int SPLASH_LAYOUT_MIN_MS;
    private int SPLASH_OUTRO_FADE_MS;
    private int SPLASH_OVERLAP_LAUNCH_MS;

    private @Nullable View contentRoot; // R.id.splash_root

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        // Android 12+ system splash (no-op on older)
        final SplashScreen splash = SplashScreen.installSplashScreen(this);
        splash.setKeepOnScreenCondition(() -> splashHeld);

        super.onCreate(savedInstanceState);

        // Safety: black window background to avoid launcher flash between activities
        try { getWindow().setBackgroundDrawable(new ColorDrawable(Color.BLACK)); } catch (Throwable ignored) {}

        // --- Load base timings ---
        SPLASH_MIN_MS             = getIntSafe(R.integer.splash_min_ms, 90);
        SPLASH_EXIT_FADE_MS       = getIntSafe(R.integer.splash_exit_fade_ms, 250);
        SPLASH_CONTENT_FADE_IN_MS = getIntSafe(R.integer.splash_content_fade_in_ms, 180);
        SPLASH_LAYOUT_MIN_MS      = getIntSafe(R.integer.splash_layout_min_ms, 600);
        SPLASH_OUTRO_FADE_MS      = getIntSafe(R.integer.splash_outro_fade_ms, 250);
        SPLASH_OVERLAP_LAUNCH_MS  = getIntSafe(R.integer.splash_overlap_launch_ms, 200);

        if (savedInstanceState != null) {
            launched = savedInstanceState.getBoolean("sg_launched", false);
        }

        // --- Cold-start heuristic tweak (helps mask DAL verify/renderer warmup on Pixel 8) ---
        final boolean isColdStart = isColdStart();
        final int extraHold   = isColdStart ? 200 : 0;  // keep system splash a touch longer
        final int extraOutro  = isColdStart ? 100 : 0;  // start/extend outro slightly later
        final int minMs       = SPLASH_MIN_MS + extraHold;
        final int outroMs     = SPLASH_OUTRO_FADE_MS + extraOutro;

        // Inflate optional overlay layout (safe if missing)
        try {
            setContentView(R.layout.activity_splash);
            contentRoot = findViewById(R.id.splash_root);
            if (contentRoot != null) contentRoot.setAlpha(0f);
        } catch (Resources.NotFoundException e) {
            Log.w(TAG, "No activity_splash layout; proceeding without overlay");
            contentRoot = null;
        }

        final Interpolator ios = AnimationUtils.loadInterpolator(this, android.R.interpolator.fast_out_slow_in);

        // Cross-fade: system splash → your overlay
        splash.setOnExitAnimationListener(provider -> {
            final View sysSplash = provider.getView();

            if (contentRoot != null) {
                contentRoot.setAlpha(0f);
                contentRoot.setScaleX(0.96f);
                contentRoot.setScaleY(0.96f);
                contentRoot.animate()
                        .alpha(1f)
                        .scaleX(1f)
                        .scaleY(1f)
                        .setDuration(SPLASH_CONTENT_FADE_IN_MS)
                        .setInterpolator(ios)
                        .start();
            }

            sysSplash.animate()
                    .alpha(0f)
                    .setDuration(SPLASH_EXIT_FADE_MS)
                    .setInterpolator(ios)
                    .setListener(new AnimatorListenerAdapter() {
                        @Override public void onAnimationEnd(Animator animation) {
                            provider.remove();
                        }
                    })
                    .start();
        });

        if (launched) {
            // Warm task relaunch: we expect Chrome/TWA to take over immediately.
            splashHeld = false;
            // Do NOT finish immediately here; defer finish to onStop so we don't flash the launcher.
            // We still trigger the launch path below.
        }
        launched = true;

        final Handler h = new Handler(Looper.getMainLooper());
        final Uri startUri = makeLaunchUri(getString(R.string.launchUrl));
        twaLauncher = new TwaLauncher(this);

        // 1) Release system splash → show overlay
        h.postDelayed(() -> splashHeld = false, minMs);

        // 2) Compute fade + launch windows (explicit control)
        final int fadeStartBase = Math.max(0, SPLASH_LAYOUT_MIN_MS - outroMs);
        int launchTarget = Math.max(
                fadeStartBase + MIN_VISIBLE_FADE_MS,
                SPLASH_LAYOUT_MIN_MS - SPLASH_OVERLAP_LAUNCH_MS
        );
        final int fadeStart = fadeStartBase;
        final int launchAt  = clamp(launchTarget, fadeStart, fadeStart + outroMs);

        Log.d(TAG, (isColdStart ? "[COLD] " : "[WARM] ")
                + "fadeStart=" + fadeStart + "ms, launchAt=" + launchAt
                + "ms, layoutMin=" + SPLASH_LAYOUT_MIN_MS + "ms");

        // 3) Start overlay OUTRO (visible fade) — BUT DO NOT finish here.
        h.postDelayed(() -> {
            if (contentRoot != null) {
                contentRoot.animate()
                        .alpha(0f)
                        .scaleX(0.98f)
                        .scaleY(0.98f)
                        .setDuration(outroMs)
                        .setInterpolator(AnimationUtils.loadInterpolator(this, android.R.interpolator.fast_out_linear_in))
                        .start();
            }
            // Keep our Activity alive (black background) until Chrome/TWA is actually shown.
        }, fadeStart);

        // 4) Launch TWA during the tail of the fade (handoff)
        h.postDelayed(() -> {
            try {
                attemptedExternalLaunch = true;
                twaLauncher.launch(startUri);
                // No transition animation — our black background + Chrome handoff is smoother
                try { overridePendingTransition(0, 0); } catch (Throwable ignored) {}
            } catch (Throwable twaError) {
                Log.e(TAG, "TWA launch failed; falling back to Custom Tabs", twaError);
                try {
                    attemptedExternalLaunch = true;
                    CustomTabsIntent cti = new CustomTabsIntent.Builder().build();
                    cti.intent.setData(startUri);
                    startActivity(cti.intent);
                    try { overridePendingTransition(0, 0); } catch (Throwable ignored) {}
                } catch (Throwable ctiError) {
                    Log.e(TAG, "Custom Tabs failed; falling back to ACTION_VIEW", ctiError);
                    try {
                        attemptedExternalLaunch = true;
                        Intent view = new Intent(Intent.ACTION_VIEW, startUri);
                        view.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                        startActivity(view);
                        try { overridePendingTransition(0, 0); } catch (Throwable ignored) {}
                    } catch (Throwable finalError) {
                        // Absolute last resort: we couldn't launch anything. Finish ourselves.
                        Log.e(TAG, "All browser handoffs failed.", finalError);
                        finish();
                    }
                }
            }
        }, launchAt);
    }

    @Override
    protected void onStop() {
        super.onStop();
        // Only now do we finish the trampoline, after the external activity is visible.
        if (attemptedExternalLaunch && !isFinishing()) {
            try { overridePendingTransition(0, 0); } catch (Throwable ignored) {}
            finish();
        }
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
    }

    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putBoolean("sg_launched", launched);
    }

    @Override
    protected void onDestroy() {
        // record last launch time for heuristic
        try {
            SharedPreferences sp = getSharedPreferences(PREFS, MODE_PRIVATE);
            sp.edit().putLong(KEY_LAST_LAUNCH, System.currentTimeMillis()).apply();
        } catch (Throwable ignored) {}

        if (twaLauncher != null) {
            try { twaLauncher.destroy(); } catch (Throwable ignored) {}
            twaLauncher = null;
        }
        super.onDestroy();
    }

    // ---------- helpers ----------

    private boolean isColdStart() {
        try {
            SharedPreferences sp = getSharedPreferences(PREFS, MODE_PRIVATE);
            long last = sp.getLong(KEY_LAST_LAUNCH, 0L);
            long now  = System.currentTimeMillis();
            return last == 0L || (now - last) >= COLD_THRESHOLD_MS;
        } catch (Throwable t) {
            return true; // be conservative
        }
    }

    private int getIntSafe(int resId, int fallback) {
        try { return getResources().getInteger(resId); } catch (Throwable t) { return fallback; }
    }

    private int clamp(int v, int min, int max) {
        return Math.max(min, Math.min(max, v));
    }

    /** Ensure absolute https URL; normalize "/app" → "/app/". */
    private Uri makeLaunchUri(String raw) {
        final String def = ORIGIN + "/app/";
        if (raw == null || raw.trim().isEmpty()) return Uri.parse(def);

        String t = raw.trim();
        if (t.startsWith("http://") || t.startsWith("https://")) {
            if (t.endsWith("/app")) t = t + "/";
            return Uri.parse(t);
        }
        if (t.startsWith("/")) {
            if ("/app".equals(t)) t = "/app/";
            return Uri.parse(ORIGIN + t);
        }
        if ("app".equals(t)) t = "app/";
        return Uri.parse(ORIGIN + "/" + t);
    }
}
