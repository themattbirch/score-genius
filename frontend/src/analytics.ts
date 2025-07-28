// src/analytics.ts

import { initializeApp } from "firebase/app";
// We assume `firebaseConfig` and `app` have already been initialized in main.tsx
import type { FirebaseApp } from "firebase/app";

let appInstance: FirebaseApp;

/**
 * Registers the Firebase App instance so Analytics can use it.
 * Call this once in main.tsx immediately after initializeApp(firebaseConfig).
 */
export function registerFirebaseApp(app: FirebaseApp) {
  appInstance = app;
}

/**
 * Lazy-loads Firebase Analytics on first user interaction or idle.
 */
export function setupAnalytics() {
  function init() {
    // Remove listeners so this only fires once
    ["pointerdown", "click", "touchstart"].forEach((evt) =>
      window.removeEventListener(evt, init, true)
    );

    import("firebase/analytics")
      .then(({ getAnalytics, logEvent }) => {
        const analytics = getAnalytics(appInstance);
        logEvent(analytics, "app_open");
        console.log("📊 Firebase Analytics initialized");
      })
      .catch((err) => console.error("Analytics load failed:", err));
  }

  // Schedule on idle or after timeout
  if ("requestIdleCallback" in window) {
    // @ts-ignore
    window.requestIdleCallback(init, { timeout: 10000 });
  } else {
    setTimeout(init, 5000);
  }

  // Kick off on first interaction
  ["pointerdown", "click", "touchstart"].forEach((evt) =>
    window.addEventListener(evt, init, { once: true, capture: true })
  );
}
