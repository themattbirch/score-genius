// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";
// ⛔️ REMOVED: No longer using the pwa-register helper
// import { registerSW } from "virtual:pwa-register";

// ─── React‑Query Client ──────────────────────────────────────────────────────
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 120_000,
      retry: 1,
    },
  },
});

// ─── Service Worker Registration ─────────────────────────────────────────────
const container = document.getElementById("root");
if (!container) throw new Error("Root element not found");

// ✅ REPLACED with native browser API for stability and explicit control
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/app/app-sw.js", { scope: "/app" })
      .then((reg) => {
        console.log(
          "✅ Service worker registered successfully with scope:",
          reg.scope
        );

        // This logic handles the "new version available" prompt
        reg.onupdatefound = () => {
          const installingWorker = reg.installing;
          if (installingWorker) {
            installingWorker.onstatechange = () => {
              if (installingWorker.state === "installed") {
                if (navigator.serviceWorker.controller) {
                  // Prompt the user to update to the new version
                  if (confirm("A new version is available. Reload?")) {
                    // The 'registerType: "autoUpdate"' in vite.config ensures
                    // the new service worker will listen for this message.
                    reg.waiting?.postMessage({ type: "SKIP_WAITING" });
                  }
                }
              }
            };
          }
        };
      })
      .catch((error) => {
        console.error("❌ SW registration failed:", error);
      });
  });

  // This logic reloads the page once the new service worker takes control
  let refreshing = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (refreshing) return;
    window.location.reload();
    refreshing = true;
  });
}

// ─── Render App ─────────────────────────────────────────────────────────────
ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/app">
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);

// ─── Lazy‑load Firebase Analytics on First Interaction ───────────────────────

// 1) Core init that dynamically pulls in the SDK
function initAnalytics() {
  // Remove all listeners so we only fire once
  ["pointerdown", "click", "touchstart"].forEach((evt) =>
    window.removeEventListener(evt, initAnalytics, true)
  );

  import("firebase/app")
    .then(({ initializeApp }) =>
      Promise.all([
        initializeApp({
          apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
          authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
          projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
          storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
          messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
          appId: import.meta.env.VITE_FIREBASE_APP_ID,
          measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID,
        }),
        import("firebase/analytics"),
      ])
    )
    .then(([app, { getAnalytics, logEvent }]) => {
      const analytics = getAnalytics(app);
      logEvent(analytics, "app_open");
      console.log("📊 Firebase Analytics initialized");
    })
    .catch((err) => console.error("Analytics load failed:", err));
}

// 2) Schedule it via idle callback or fallback timeout
function scheduleAnalyticsInit() {
  if ("requestIdleCallback" in window) {
    // @ts-ignore
    window.requestIdleCallback(initAnalytics, { timeout: 10_000 });
  } else {
    setTimeout(initAnalytics, 5_000);
  }
}

// 3) Kick off on first user interaction
["pointerdown", "click", "touchstart"].forEach((evt) =>
  window.addEventListener(evt, () => scheduleAnalyticsInit(), {
    once: true,
    capture: true,
  })
);

// 4) Also fire on idle if they never interact
scheduleAnalyticsInit();
