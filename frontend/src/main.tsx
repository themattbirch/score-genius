// src/main.tsx – gate Firebase Analytics behind user interaction

import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

// ─── React‑Query Client ──────────────────────────────────────────────────────
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 120_000,
      retry: 1,
    },
  },
});

// ─── Root Element & SW Registration ─────────────────────────────────────────
const container = document.getElementById("root");
if (!container) throw new Error("Root element not found");

const swUrl = import.meta.env.DEV ? "/dev-sw.js?dev-sw" : "/app-sw.js";
if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register(swUrl, { scope: "/app/" })
    .then((reg) => {
      if (reg.waiting) reg.waiting.postMessage({ type: "SKIP_WAITING" });
      reg.addEventListener("updatefound", () => {
        const w = reg.installing;
        if (!w) return;
        w.addEventListener("statechange", () => {
          if (w.state === "installed" && navigator.serviceWorker.controller) {
            w.postMessage({ type: "SKIP_WAITING" });
          }
        });
      });
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        location.reload();
      });
      reg.update().catch(() => {});
    })
    .catch((err) => console.error("❌ SW registration failed:", err));
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

// ─── Analytics on First Interaction ─────────────────────────────────────────
// 1) Define initAnalytics first
function initAnalytics() {
  // Remove listeners to avoid duplicates
  ["pointerdown", "click", "touchstart"].forEach((evt) =>
    window.removeEventListener(evt, initAnalytics, true)
  );

  import("firebase/app")
    .then(({ initializeApp }) =>
      import("firebase/analytics").then(({ getAnalytics, logEvent }) => {
        const firebaseConfig = {
          apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
          authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
          projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
          storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
          messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
          appId: import.meta.env.VITE_FIREBASE_APP_ID,
          measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID,
        } as const;

        const app = initializeApp(firebaseConfig);
        const analytics = getAnalytics(app);
        logEvent(analytics, "app_open");
        console.log("📊 Firebase Analytics initialized");
      })
    )
    .catch((err) => console.error("Analytics load failed:", err));
}

// 2) Schedule it on idle or after timeout
function scheduleAnalyticsInit() {
  if ("requestIdleCallback" in window) {
    (window as any).requestIdleCallback(initAnalytics, { timeout: 10_000 });
  } else {
    setTimeout(initAnalytics, 5_000);
  }
}

// 3) Wire up first‐interaction trigger
["pointerdown", "click", "touchstart"].forEach((evt) =>
  window.addEventListener(
    evt,
    () => {
      scheduleAnalyticsInit();
    },
    { once: true, capture: true }
  )
);

// 4) Also kick off on idle even if no interaction
scheduleAnalyticsInit();
