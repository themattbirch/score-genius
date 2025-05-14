// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

const queryClient = new QueryClient();
const container = document.getElementById("root");
if (!container) throw new Error("Root element not found");

// 1) Attempt registration immediately
const swUrl = import.meta.env.DEV ? "/dev-sw.js?dev-sw" : "/app-sw.js";
console.log("📦 attempting SW registration at", swUrl);

if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register(swUrl, { scope: "/app/" })
    .then((reg) => {
      console.log("✅ SW registered, scope:", reg.scope);

      // 1) If there’s already a waiting SW, tell it to skip waiting.
      if (reg.waiting) {
        reg.waiting.postMessage({ type: "SKIP_WAITING" });
      }

      // 2) When a new SW is found (on update), hook its state changes.
      reg.addEventListener("updatefound", () => {
        const w = reg.installing;
        if (!w) return;
        w.addEventListener("statechange", () => {
          if (w.state === "installed" && navigator.serviceWorker.controller) {
            w.postMessage({ type: "SKIP_WAITING" });
          }
        });
      });

      // 3) Reload the page when the new SW takes control.
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        location.reload();
      });

      // 4) Optional: check for updates on every load.
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
