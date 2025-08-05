// frontend/src/contexts/online_context.tsx
import React, { createContext, useContext, useEffect, useState } from "react";

/**  How often to ping in ms (very cheap HEAD request) */
const HEARTBEAT_MS = 5_000;
const TEST_URL = "/app/offline.html"; // any small, precached asset

const OnlineCtx = createContext(true);
export const useOnline = () => useContext(OnlineCtx);

export const OnlineProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [online, setOnline] = useState(navigator.onLine);

  // Native browser events (these work on desktop)
  useEffect(() => {
    const handler = () => setOnline(navigator.onLine);
    window.addEventListener("online", handler);
    window.addEventListener("offline", handler);
    return () => {
      window.removeEventListener("online", handler);
      window.removeEventListener("offline", handler);
    };
  }, []);

  // Heartbeat – covers TWA / WebView where events don’t always fire
  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      try {
        // HEAD → no body download; `cache: 'no-store'` bypasses SW cache
        await fetch(TEST_URL, { method: "HEAD", cache: "no-store" });
        mounted && setOnline(true);
      } catch {
        mounted && setOnline(false);
      }
    };
    const id = window.setInterval(tick, HEARTBEAT_MS);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  return <OnlineCtx.Provider value={online}>{children}</OnlineCtx.Provider>;
};
