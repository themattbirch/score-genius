package io.scoregenius.app;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.view.animation.Interpolator;
import android.view.animation.AnimationUtils;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.splashscreen.SplashScreen;

import com.google.androidbrowserhelper.trusted.TwaLauncher;

public class LauncherActivity extends AppCompatActivity {

    private static final String TAG = "LauncherActivity";
    private static final int MIN_VISIBLE_FADE_MS = 200; // ensure you see some of the outro before launch

    private @Nullable TwaLauncher twaLauncher;
    private boolean launched = false;
    private volatile boolean splashHeld = true;

    // Timings from resources
    private int SPLASH_MIN_MS;
    private int SPLASH_EXIT_FADE_MS;
    private int SPLASH_CONTENT_FADE_IN_MS;
    private int SPLASH_LAYOUT_MIN_MS;
    private int SPLASH_OUTRO_FADE_MS;
    private int SPLASH_OVERLAP_LAUNCH_MS;

    private View contentRoot; // R.id.splash_root

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        final SplashScreen splash = SplashScreen.installSplashScreen(this);
        splash.setKeepOnScreenCondition(() -> splashHeld);

        super.onCreate(savedInstanceState);

        // Load timings
        SPLASH_MIN_MS             = getResources().getInteger(R.integer.splash_min_ms);
        SPLASH_EXIT_FADE_MS       = getResources().getInteger(R.integer.splash_exit_fade_ms);
        SPLASH_CONTENT_FADE_IN_MS = getResources().getInteger(R.integer.splash_content_fade_in_ms);
        SPLASH_LAYOUT_MIN_MS      = getResources().getInteger(R.integer.splash_layout_min_ms);
        SPLASH_OUTRO_FADE_MS      = getResources().getInteger(R.integer.splash_outro_fade_ms);
        SPLASH_OVERLAP_LAUNCH_MS  = getResources().getInteger(R.integer.splash_overlap_launch_ms);

        if (savedInstanceState != null) {
            launched = savedInstanceState.getBoolean("sg_launched", false);
        }

        setContentView(R.layout.activity_splash);
        contentRoot = findViewById(R.id.splash_root);
        if (contentRoot != null) contentRoot.setAlpha(0f);

        final Interpolator ios = AnimationUtils.loadInterpolator(this, android.R.interpolator.fast_out_slow_in);

        // Cross-fade: system splash → our layout
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
            Log.d(TAG, "Warm start; releasing splash and finishing trampoline.");
            splashHeld = false;
            new Handler(Looper.getMainLooper()).post(this::finish);
            return;
        }
        launched = true;

        final Handler h = new Handler(Looper.getMainLooper());
        final Uri startUri = Uri.parse(getString(R.string.launchUrl));
        twaLauncher = new TwaLauncher(this);

        // Release the Android 12 system splash quickly so our layout is visible
        h.postDelayed(() -> {
            Log.d(TAG, "Release system splash @ " + SPLASH_MIN_MS + "ms");
            splashHeld = false;
        }, SPLASH_MIN_MS);

        // Compute timing windows
        final int fadeStart = Math.max(0, SPLASH_LAYOUT_MIN_MS - SPLASH_OUTRO_FADE_MS);
        int launchTarget = Math.max(
            fadeStart + MIN_VISIBLE_FADE_MS,
            SPLASH_LAYOUT_MIN_MS - SPLASH_OVERLAP_LAUNCH_MS
        );
        final int launchAt = Math.min(Math.max(launchTarget, fadeStart), fadeStart + SPLASH_OUTRO_FADE_MS);

        Log.d(TAG, "fadeStart=" + fadeStart + "ms, launchAt=" + launchAt + "ms, layoutMin=" + SPLASH_LAYOUT_MIN_MS + "ms");

        // Start the OUTRO fade first (so it’s actually visible)
        h.postDelayed(() -> {
            Log.d(TAG, "Begin splash OUTRO fade (" + SPLASH_OUTRO_FADE_MS + "ms)");
            if (contentRoot != null) {
                contentRoot.animate()
                        .alpha(0f)
                        .scaleX(0.98f)
                        .scaleY(0.98f)
                        .setDuration(SPLASH_OUTRO_FADE_MS)
                        .setInterpolator(AnimationUtils.loadInterpolator(this, android.R.interpolator.fast_out_linear_in))
                        .withEndAction(() -> {
                            Log.d(TAG, "Splash OUTRO complete; finishing LauncherActivity.");
                            finish();
                            // No second override here; we already set transition on launch
                        })
                        .start();
            } else {
                // If no content root, just finish at layoutMin
                finish();
            }
        }, fadeStart);

        // Launch the TWA during the tail of the fade (or at end, per clamp above)
        h.postDelayed(() -> {
            Log.d(TAG, "Launching TWA at " + launchAt + "ms (during/after splash OUTRO).");
            twaLauncher.launch(startUri);
            // Activity transition for the handoff
            overridePendingTransition(R.anim.sg_fade_in, R.anim.sg_fade_out);
        }, launchAt);
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
        if (twaLauncher != null) {
            twaLauncher.destroy();
            twaLauncher = null;
        }
        super.onDestroy();
    }
}
