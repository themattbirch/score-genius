// frontend/src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css"; // Assuming your CSS import

const queryClient = new QueryClient();

const container = document.getElementById("root");
if (!container) {
  throw new Error("Root element not found; check app.html");
}

ReactDOM.createRoot(container!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      {/* FIX: Remove the basename prop completely */}
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
