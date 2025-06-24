// src/main.tsx
// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

// ─── Firebase Analytics ───────────────────────────────────────────────────────
import { initializeApp } from "firebase/app";
import { getAnalytics, logEvent } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyBwKUxO1kumvltOXrCvXISwNjWSw2D7S90",
  authDomain: "scoregenius-ddbbe.firebaseapp.com",
  projectId: "scoregenius-ddbbe",
  storageBucket: "scoregenius-ddbbe.firebasestorage.app",
  messagingSenderId: "462146250938",
  appId: "1:462146250938:web:e1fa4e5e8144ebbec0fa83",
  measurementId: "G-MGBC7QHT01",
};

const firebaseApp = initializeApp(firebaseConfig);
const analytics = getAnalytics(firebaseApp);
// Log an "app_open" each time the PWA bundle initializes:
logEvent(analytics, "app_open");
// ──────────────────────────────────────────────────────────────────────────────

// Create *one* client with your defaults
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 120_000,
      retry: 1,
    },
  },
});

const container = document.getElementById("root");
if (!container) throw new Error("Root element not found");

// 1) Attempt SW registration immediately
const swUrl = import.meta.env.DEV ? "/dev-sw.js?dev-sw" : "/app-sw.js";
console.log("📦 attempting SW registration at", swUrl);

if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register(swUrl, { scope: "/app/" })
    .then((reg) => {
      console.log("✅ SW registered, scope:", reg.scope);
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

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/app">
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
