// frontend/src/contexts/online_context.tsx
import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

/** Heartbeat cadence (ms) */
const HEARTBEAT_MS = 5000;
/** Request timeout (ms) */
const TIMEOUT_MS = 2500;
/** Grace period after startup before we allow offline (ms) */
const INITIAL_GRACE_MS = 2500;
/** Delay before first verify when we get an 'offline' signal (ms) */
const OFFLINE_VERIFY_DELAY_MS = 350;
/** How many consecutive failures before declaring offline */
const FAILS_THRESHOLD = 2;

const OnlineCtx = createContext<boolean>(true);
export const useOnline = () => useContext(OnlineCtx);

/**
 * OnlineProvider is optimistic: it returns true unless we've CONFIRMED offline.
 * Confirmation = two consecutive failed /health checks after a short grace period.
 * This prevents cold-start false negatives and SW fallback HTML from tripping redirects.
 */
export const OnlineProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  // Soft signal from the browser; start optimistic (assume true)
  const [softOnline, setSoftOnline] = useState<boolean>(true);
  // Hard decision: only true offline when confirmed
  const [hardOffline, setHardOffline] = useState<boolean>(false);
  // Consecutive verification failures
  const failsRef = useRef<number>(0);
  // Prevent overlapping verifications
  const inFlightRef = useRef<AbortController | null>(null);
  // Startup clock (for grace period)
  const startedAtRef = useRef<number>(Date.now());

  // Browser online/offline events
  useEffect(() => {
    const onOnline = () => {
      setSoftOnline(true);
      setHardOffline(false);
      failsRef.current = 0;
      // cancel any in-flight verify
      inFlightRef.current?.abort();
      inFlightRef.current = null;
    };

    const onOffline = () => {
      setSoftOnline(false);
      // verify shortly after to confirm
      const t = setTimeout(() => {
        void verifyConnectivity();
      }, OFFLINE_VERIFY_DELAY_MS);
      return () => clearTimeout(t);
    };

    window.addEventListener("online", onOnline);
    window.addEventListener("offline", onOffline);

    // If the browser says offline at boot, still start optimistic but verify soon.
    if (typeof navigator !== "undefined" && navigator.onLine === false) {
      setSoftOnline(false);
      setTimeout(() => {
        void verifyConnectivity();
      }, OFFLINE_VERIFY_DELAY_MS);
    }

    return () => {
      window.removeEventListener("online", onOnline);
      window.removeEventListener("offline", onOffline);
    };
  }, []);

  // Verify connectivity against /health (expects JSON {status:"OK"})
  const verifyConnectivity = async () => {
    if (inFlightRef.current) return;
    const ctrl = new AbortController();
    inFlightRef.current = ctrl;
    try {
      const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
      const res = await fetch(`/health?ts=${Date.now()}`, {
        cache: "no-store",
        signal: ctrl.signal,
      });
      clearTimeout(timer);

      let ok = false;
      if (res.ok) {
        const ct = (res.headers.get("content-type") || "").toLowerCase();
        if (ct.includes("application/json")) {
          try {
            const body = await res.json();
            ok = body?.status === "OK";
          } catch {
            ok = false;
          }
        }
      }

      if (ok) {
        failsRef.current = 0;
        // Once verified, clear any hard offline
        setHardOffline(false);
      } else {
        failsRef.current += 1;
      }
    } catch (err: any) {
      // Abort during SW activation or race: treat as inconclusive (do not increment)
      if (err?.name !== "AbortError") {
        failsRef.current += 1;
      }
    } finally {
      inFlightRef.current = null;
      // Decide hardOffline status after each attempt
      evaluateHardOffline();
    }
  };

  // Decide when to flip to hard offline
  const evaluateHardOffline = () => {
    const elapsed = Date.now() - startedAtRef.current;
    const enoughFails = failsRef.current >= FAILS_THRESHOLD;
    // Only declare offline if the browser also thinks we're offline OR we already were hard offline
    const softSaysOffline = softOnline === false;
    const allowAfterGrace = elapsed >= INITIAL_GRACE_MS;

    if ((softSaysOffline || hardOffline) && enoughFails && allowAfterGrace) {
      setHardOffline(true);
    }
  };

  // Initial scheduled verify + periodic heartbeat
  useEffect(() => {
    // Initial verify slightly after startup to avoid SW install/activate races
    const init = setTimeout(() => {
      void verifyConnectivity();
    }, 1000);

    const id = window.setInterval(() => {
      // Only verify periodically if soft offline OR we’re currently hard offline
      if (!softOnline || hardOffline) {
        void verifyConnectivity();
      }
    }, HEARTBEAT_MS);

    return () => {
      clearTimeout(init);
      clearInterval(id);
      inFlightRef.current?.abort();
    };
  }, [softOnline, hardOffline]);

  // Exposed value: true unless CONFIRMED offline
  const value = hardOffline ? false : true;

  return <OnlineCtx.Provider value={value}>{children}</OnlineCtx.Provider>;
};
