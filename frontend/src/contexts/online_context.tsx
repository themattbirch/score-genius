// frontend/src/contexts/online_context.tsx
import React, { createContext, useContext, useEffect, useState } from "react";

/** How often to ping in ms (cheap HEAD request) */
const HEARTBEAT_MS = 5000;

/**
 * Generate a unique URL each time to bypass SW and browser caches
 * We'll hit a non-existent resource so we rely on network errors
 */
const buildHeartbeatURL = () => `/favicon.ico?ts=${Date.now()}`;

const OnlineCtx = createContext<boolean>(true);
export const useOnline = () => useContext(OnlineCtx);

export const OnlineProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [online, setOnline] = useState<boolean>(navigator.onLine);

  // Browser online/offline events (reliable in desktop)
  useEffect(() => {
    const handleOnline = () => setOnline(true);
    const handleOffline = () => setOnline(false);

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // Heartbeat ping to catch TWA/WebView connectivity changes
  useEffect(() => {
    let isMounted = true;

    const heartbeat = async () => {
      try {
        await fetch(buildHeartbeatURL(), {
          method: "HEAD",
          cache: "reload",
          mode: "no-cors",
        });
        if (isMounted) setOnline(true);
      } catch {
        if (isMounted) setOnline(false);
      }
    };

    // Initial check & recurring interval
    heartbeat();
    const id = window.setInterval(heartbeat, HEARTBEAT_MS);
    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, []);

  return <OnlineCtx.Provider value={online}>{children}</OnlineCtx.Provider>;
};
