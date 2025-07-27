// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { registerSW } from "virtual:pwa-register";
import App from "./App";
import "./index.css";
import { initializeApp } from "firebase/app";
import { firebaseConfig } from "./firebaseConfig";

// ─── Firebase App Initialization ────────────────────────────────────────────
// Eagerly initialize your Firebase app with full config to avoid dynamic fetch fallback
const app = initializeApp(firebaseConfig);

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

if ("serviceWorker" in navigator && !import.meta.env.DEV) {
  window.addEventListener("load", async () => {
    try {
      const registration = await navigator.serviceWorker.register(
        "/app/app-sw.js",
        { scope: "/app/" }
      );

      registration.addEventListener("updatefound", () => {
        const newSW = registration.installing;
        if (newSW) {
          newSW.addEventListener("statechange", () => {
            if (
              newSW.state === "installed" &&
              navigator.serviceWorker.controller
            ) {
              if (confirm("🔄 New version available—reload now?")) {
                window.location.reload();
              }
            }
          });
        }
      });

      let refreshing = false;
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (!refreshing) {
          window.location.reload();
          refreshing = true;
        }
      });

      console.log(
        "✅ Service worker registered with scope:",
        registration.scope
      );
    } catch (err) {
      console.error("⚠️ SW registration failed:", err);
    }
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
function initAnalytics() {
  // Remove listeners so this only fires once
  ["pointerdown", "click", "touchstart"].forEach((evt) =>
    window.removeEventListener(evt, initAnalytics, true)
  );

  import("firebase/analytics")
    .then(({ getAnalytics, logEvent }) => {
      const analytics = getAnalytics(app);
      logEvent(analytics, "app_open");
      console.log("📊 Firebase Analytics initialized");
    })
    .catch((err) => console.error("Analytics load failed:", err));
}

function scheduleAnalyticsInit() {
  if ("requestIdleCallback" in window) {
    // @ts-ignore
    window.requestIdleCallback(initAnalytics, { timeout: 10_000 });
  } else {
    setTimeout(initAnalytics, 5_000);
  }
}

// Kick off analytics on first user interaction
["pointerdown", "click", "touchstart"].forEach((evt) =>
  window.addEventListener(evt, () => scheduleAnalyticsInit(), {
    once: true,
    capture: true,
  })
);

// Also fire if the user never interacts
scheduleAnalyticsInit();
