// frontend/src/contexts/online_context.tsx
import React, { createContext, useContext, useEffect, useState } from "react";

/** How often to ping in ms (cheap HEAD request) */
const HEARTBEAT_MS = 5000;
const TEST_URL = "/heartbeat.txt"; // precached asset for heartbeat

const OnlineCtx = createContext<boolean>(true);
export const useOnline = () => useContext(OnlineCtx);

export const OnlineProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [online, setOnline] = useState<boolean>(navigator.onLine);

  // Native browser events (reliable in desktop browsers)
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

  // Heartbeat ping to catch connectivity changes in WebView/TWA
  useEffect(() => {
    let isMounted = true;

    const heartbeat = async () => {
      try {
        await fetch(TEST_URL, { method: "HEAD", cache: "no-store" });
        if (isMounted) setOnline(true);
      } catch {
        if (isMounted) setOnline(false);
      }
    };

    // Initial check
    heartbeat();
    const intervalId = window.setInterval(heartbeat, HEARTBEAT_MS);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, []);

  return <OnlineCtx.Provider value={online}>{children}</OnlineCtx.Provider>;
};
