// src/main.tsx
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

// ─── Mount Point Guard ───────────────────────────────────────────────────────
const container = document.getElementById("root");
if (!container) throw new Error("Root element not found");

// ─── Service‑Worker Registration (eager) ─────────────────────────────────────
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

// ─── Lazy Firebase Analytics (post‑LCP / idle) ───────────────────────────────
function initAnalytics() {
  // Skip entirely if measurement ID absent (e.g. local dev)
  if (!import.meta.env.VITE_FIREBASE_MEASUREMENT_ID) return;

  const idleCb: typeof requestIdleCallback =
    "requestIdleCallback" in window
      ? (window as any).requestIdleCallback
      : (cb) => setTimeout(cb, 0);

  idleCb(async () => {
    try {
      const [{ initializeApp }, { getAnalytics, logEvent }] = await Promise.all(
        [import("firebase/app"), import("firebase/analytics")]
      );

      const firebaseConfig = {
        apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
        authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
        projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
        storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
        messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
        appId: import.meta.env.VITE_FIREBASE_APP_ID,
        measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID,
      } as const;

      const firebaseApp = initializeApp(firebaseConfig);
      const analytics = getAnalytics(firebaseApp);
      logEvent(analytics, "app_open");
      console.log("📊 Firebase Analytics initialized lazily");
    } catch (err) {
      console.error("⚠️  Failed to init Firebase Analytics", err);
    }
  });
}

initAnalytics();

// ─── Render App ──────────────────────────────────────────────────────────────
ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/app">
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
