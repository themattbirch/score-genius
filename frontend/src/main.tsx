// src/main.tsx
import React, { useEffect, lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./index.css";

// lazy‑load the whole app for route‑based splitting
const App = lazy(() => import("./App"));

// ─── React‑Query setup ────────────────────────────────────────────────────────
const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 120_000, retry: 1 } },
});

function Bootstrap() {
  // 1) SW registration on idle
  useEffect(() => {
    if ("serviceWorker" in navigator && !import.meta.env.DEV) {
      window.requestIdleCallback?.(
        async () => {
          try {
            // dynamic import so pwa‑register doesn’t bloat your initial bundle
            const { registerSW } = await import("virtual:pwa-register");
            registerSW({ immediate: true });
          } catch (e) {
            console.warn("SW registration failed:", e);
          }
        },
        { timeout: 5_000 }
      );
    }
  }, []);

  // 2) Firebase Analytics on idle/interaction
  useEffect(() => {
    const initAnalytics = async () => {
      try {
        const [{ initializeApp }, { getAnalytics, logEvent }] =
          await Promise.all([
            import("firebase/app"),
            import("firebase/analytics"),
          ]);
        const app = initializeApp({
          apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
          authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
          projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
          storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
          messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
          appId: import.meta.env.VITE_FIREBASE_APP_ID,
          measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID,
        });
        const analytics = getAnalytics(app);
        logEvent(analytics, "app_open");
      } catch (err) {
        console.error("Analytics load failed:", err);
      }
    };

    // only fire once, after first user interaction or idle
    const schedule = () => {
      window.requestIdleCallback?.(initAnalytics, { timeout: 10_000 }) ??
        setTimeout(initAnalytics, 5_000);
      ["pointerdown", "click", "touchstart"].forEach((evt) =>
        window.removeEventListener(evt, schedule, true)
      );
    };
    ["pointerdown", "click", "touchstart"].forEach((evt) =>
      window.addEventListener(evt, schedule, { once: true, capture: true })
    );
    schedule(); // fallback if they never interact
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/app">
        <Suspense fallback={null}>
          <App />
        </Suspense>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

const root = document.getElementById("root");
if (!root) throw new Error("Root element not found");
ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <Bootstrap />
  </React.StrictMode>
);
