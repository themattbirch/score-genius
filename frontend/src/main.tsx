// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { registerSW } from "virtual:pwa-register";
import App from "./App";
import "./index.css";

import { initializeApp, FirebaseApp } from "firebase/app";
import { firebaseConfig } from "./firebaseConfig";
import { registerFirebaseApp, setupAnalytics } from "./analytics";

// ─── Firebase App Initialization ────────────────────────────────────────────
const app: FirebaseApp = initializeApp(firebaseConfig);
// Make the app instance available for analytics
registerFirebaseApp(app);

// ─── React‑Query Client ──────────────────────────────────────────────────────
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 120_000, retry: 1 },
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
// This will code-split analytics into its own chunk
setupAnalytics();

// ─── Register Service Worker for PWA (optional immediate) ──────────────────
registerSW();
