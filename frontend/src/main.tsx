// frontend/src/main.tsx

import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

const queryClient = new QueryClient();
const container = document.getElementById("root");
if (!container) throw new Error("Root element not found");

const swUrl = "/app-sw.js";

if (
  (import.meta.env.PROD || import.meta.env.DEV) &&
  "serviceWorker" in navigator
) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register(swUrl, { scope: "/app/" })
      .then((reg) => {
        // 1) If there’s already a waiting SW, tell it to skip waiting.
        if (reg.waiting) {
          reg.waiting.postMessage({ type: "SKIP_WAITING" });
        }

        // 2) When a new SW is found (on update), hook its state changes.
        reg.addEventListener("updatefound", () => {
          const newWorker = reg.installing;
          if (!newWorker) return;
          newWorker.addEventListener("statechange", () => {
            if (
              newWorker.state === "installed" &&
              navigator.serviceWorker.controller
            ) {
              newWorker.postMessage({ type: "SKIP_WAITING" });
            }
          });
        });

        // 3) Reload the page when the new SW takes control.
        navigator.serviceWorker.addEventListener("controllerchange", () => {
          window.location.reload();
        });

        // 4) Optional: check for updates on every load.
        reg.update().catch(() => {});
      })
      .catch((err) =>
        console.error("Service-worker registration failed:", err)
      );
  });
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
