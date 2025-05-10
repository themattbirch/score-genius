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

/* ------- new SW registration, scoped to /app/ -------- */
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/app-sw.js", { scope: "/app/" })
      .catch((err) =>
        console.error("Service‑worker registration failed:", err)
      );
  });
}

/* ------- SW registration (production only) -------- */
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/app-sw.js", { scope: "/app/" })
      .then((reg) => {
        // When a new worker activates, reload the page so fresh HTML/CSS/JS load.
        navigator.serviceWorker.addEventListener("controllerchange", () => {
          window.location.reload();
        });
        // Optional: ask for an update check on every load
        reg.update().catch(() => {});
      })
      .catch((err) => console.error("SW registration failed:", err));
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
