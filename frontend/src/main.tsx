// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";
// ✅ CHANGED: Using the official helper for registration
import { registerSW } from "virtual:pwa-register";

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

// ✅ CHANGED: Capture the return value of registerSW
const updateSW = registerSW({
  onNeedRefresh() {
    if (confirm("A new version is available. Reload?")) {
      updateSW?.();
    }
  },
  onOfflineReady() {
    console.log("App is ready to work offline.");
  },
});

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
