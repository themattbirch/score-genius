// frontend/src/contexts/online_context.tsx
import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

/** Paths & timings */
const HEALTH_PATH = "/api/backend/health";
const TIMEOUT_MS = 2500; // fetch timeout
const INITIAL_GRACE_MS = 2500; // allow SW activation/boot to settle
const OFFLINE_VERIFY_DELAY_MS = 350;
const FAILS_THRESHOLD = 2; // consecutive failures before "hard offline"
const HEARTBEAT_MS = 5000; // fast check when (soft/hard) offline
const ONLINE_HEARTBEAT_MS = 60000; // slow check to self-heal when online

const OnlineCtx = createContext<boolean>(true);
export const useOnline = () => useContext(OnlineCtx);

/**
 * OnlineProvider is optimistic: it returns `true` unless we've CONFIRMED offline.
 * Confirmation requires BOTH:
 *   - the browser indicates offline (soft signal), and
 *   - >= FAILS_THRESHOLD consecutive failed probes to HEALTH_PATH,
 * after INITIAL_GRACE_MS from startup.
 */
export const OnlineProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  // Soft signal from browser events
  const [softOnline, _setSoftOnline] = useState<boolean>(true);
  const softOnlineRef = useRef<boolean>(true);
  const setSoftOnline = (v: boolean) => {
    softOnlineRef.current = v;
    _setSoftOnline(v);
  };

  // Hard decision managed by our probe logic
  const [hardOffline, _setHardOffline] = useState<boolean>(false);
  const hardOfflineRef = useRef<boolean>(false);
  const setHardOffline = (v: boolean) => {
    hardOfflineRef.current = v;
    _setHardOffline(v);
  };

  const failsRef = useRef<number>(0);
  const inFlightRef = useRef<AbortController | null>(null);
  const startedAtRef = useRef<number>(Date.now());
  const offlineVerifyTidRef = useRef<number | null>(null);

  /** Decide when to flip to hard offline (reads latest refs) */
  const evaluateHardOffline = () => {
    const elapsed = Date.now() - startedAtRef.current;
    const enoughFails = failsRef.current >= FAILS_THRESHOLD;
    const softSaysOffline = softOnlineRef.current === false;
    const allowAfterGrace = elapsed >= INITIAL_GRACE_MS;

    if (
      (softSaysOffline || hardOfflineRef.current) &&
      enoughFails &&
      allowAfterGrace
    ) {
      setHardOffline(true);
    }
  };

  /** Probe connectivity against the same-origin data plane */
  const verifyConnectivity = async () => {
    if (inFlightRef.current) return;
    const ctrl = new AbortController();
    inFlightRef.current = ctrl;

    const timeoutId = window.setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
      const res = await fetch(`${HEALTH_PATH}?ts=${Date.now()}`, {
        cache: "no-store",
        credentials: "omit",
        mode: "same-origin",
        signal: ctrl.signal,
      });
      // Treat any 2xx (including 204) as success; opaque also counts as non-failure.
      const ok = res.ok || res.type === "opaque";
      if (ok) {
        failsRef.current = 0;
        if (hardOfflineRef.current) setHardOffline(false);
      } else {
        failsRef.current += 1;
      }
    } catch (err: any) {
      // Abort is inconclusive; do not increment on AbortError
      if (err?.name !== "AbortError") {
        failsRef.current += 1;
      }
    } finally {
      clearTimeout(timeoutId);
      inFlightRef.current = null;
      evaluateHardOffline();
    }
  };

  /** Browser online/offline events */
  useEffect(() => {
    const onOnline = () => {
      setSoftOnline(true);
      setHardOffline(false);
      failsRef.current = 0;
      inFlightRef.current?.abort();
      inFlightRef.current = null;
      if (offlineVerifyTidRef.current !== null) {
        clearTimeout(offlineVerifyTidRef.current);
        offlineVerifyTidRef.current = null;
      }
    };

    const onOffline = () => {
      setSoftOnline(false);
      if (offlineVerifyTidRef.current !== null) {
        clearTimeout(offlineVerifyTidRef.current);
      }
      offlineVerifyTidRef.current = window.setTimeout(() => {
        void verifyConnectivity();
        offlineVerifyTidRef.current = null;
      }, OFFLINE_VERIFY_DELAY_MS);
    };

    window.addEventListener("online", onOnline);
    window.addEventListener("offline", onOffline);

    // If the browser reports offline at boot, verify shortly after startup.
    if (typeof navigator !== "undefined" && navigator.onLine === false) {
      setSoftOnline(false);
      offlineVerifyTidRef.current = window.setTimeout(() => {
        void verifyConnectivity();
        offlineVerifyTidRef.current = null;
      }, OFFLINE_VERIFY_DELAY_MS);
    }

    return () => {
      window.removeEventListener("online", onOnline);
      window.removeEventListener("offline", onOffline);
    };
  }, []);

  /** Initial verify, heartbeats, and self-heal on visibility change */
  useEffect(() => {
    // Initial verify shortly after mount to avoid SW races
    const initId = window.setTimeout(() => void verifyConnectivity(), 1000);

    // Fast heartbeat when soft/hard offline
    const fastId = window.setInterval(() => {
      if (!softOnlineRef.current || hardOfflineRef.current) {
        void verifyConnectivity();
      }
    }, HEARTBEAT_MS);

    // Slow heartbeat even when online (self-heal)
    const slowId = window.setInterval(() => {
      if (softOnlineRef.current && !hardOfflineRef.current) {
        void verifyConnectivity();
      }
    }, ONLINE_HEARTBEAT_MS);

    // Self-heal on focus/visibility
    const onVisible = () => {
      if (document.visibilityState === "visible") {
        if (hardOfflineRef.current || !softOnlineRef.current) {
          void verifyConnectivity();
        }
      }
    };
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      clearTimeout(initId);
      clearInterval(fastId);
      clearInterval(slowId);
      document.removeEventListener("visibilitychange", onVisible);
      inFlightRef.current?.abort();
      if (offlineVerifyTidRef.current !== null) {
        clearTimeout(offlineVerifyTidRef.current);
        offlineVerifyTidRef.current = null;
      }
    };
  }, []);

  // Expose: true unless we've confirmed offline
  const value = hardOffline ? false : true;
  return <OnlineCtx.Provider value={value}>{children}</OnlineCtx.Provider>;
};
