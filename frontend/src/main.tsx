// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

// 1. Core app dependencies are imported statically as before.
// Note: Firebase and Analytics imports are removed from the top.

// ─── React‑Query Client ──────────────────────────────────────────────────────
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 120_000, retry: 1 },
  },
});

// ─── Service Worker Registration (Already Optimized) ─────────────────────────
if ("serviceWorker" in navigator && !import.meta.env.DEV) {
  window.addEventListener("load", async () => {
    try {
      const registration = await navigator.serviceWorker.register(
        "/app/app-sw.js",
        { scope: "/app/" }
      );
      console.log(
        "✅ Service worker registered with scope:",
        registration.scope
      );

      // If there's already a waiting SW, tell it to activate immediately
      if (registration.waiting) {
        registration.waiting.postMessage({ type: "SKIP_WAITING" });
      }

      // Listen for new SW installations and skip waiting as soon as they're installed
      registration.addEventListener("updatefound", () => {
        const newSW = registration.installing;
        if (newSW) {
          newSW.addEventListener("statechange", () => {
            if (newSW.state === "installed") {
              newSW.postMessage({ type: "SKIP_WAITING" });
            }
          });
        }
      });

      // Auto-reload the page when the new SW takes control
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        window.location.reload();
      });
    } catch (err) {
      console.error("⚠️ SW registration failed:", err);
    }
  });
}

// ─── Render App (Happens Immediately) ───────────────────────────────────────
const container = document.getElementById("root");
if (!container) throw new Error("Root element not found");

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/app">
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);

// ─── Deferred Services Initialization ───────────────────────────────────────

// 2. All non-essential services are moved into a single async function.
async function initializeDeferredServices() {
  // Use dynamic imports to code-split Firebase and Analytics.
  const { initializeApp } = await import("firebase/app");
  const { firebaseConfig } = await import("./firebaseConfig");
  const { registerFirebaseApp, setupAnalytics } = await import("./analytics");

  // Initialize Firebase
  const app = initializeApp(firebaseConfig);
  registerFirebaseApp(app);

  // Setup Analytics
  setupAnalytics();
  console.log("🔥 Deferred services (Firebase, Analytics) initialized.");
}

// 3. A one-time listener triggers the initialization on the first user interaction.
const events = ["scroll", "mousemove", "touchstart", "keydown"];
const triggerInit = () => {
  initializeDeferredServices();
  events.forEach((event) => window.removeEventListener(event, triggerInit));
};

events.forEach((event) => {
  window.addEventListener(event, triggerInit, { once: true });
});
