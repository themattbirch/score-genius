// src/hooks/use_network_status.ts
import { useState, useEffect } from "react";

/** Returns `true` when the browser is online. */
export function useNetworkStatus(): boolean {
  // DevTools “Offline” sets navigator.onLine = false immediately
  const [online, setOnline] = useState(navigator.onLine);

  useEffect(() => {
    const goOnline = () => setOnline(true);
    const goOffline = () => setOnline(false);

    window.addEventListener("online", goOnline);
    window.addEventListener("offline", goOffline);
    return () => {
      window.removeEventListener("online", goOnline);
      window.removeEventListener("offline", goOffline);
    };
  }, []);

  return online;
}
