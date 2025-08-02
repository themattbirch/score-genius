// frontend/src/components/ui/joyride_tour.tsx
import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
  useEffect,
  SetStateAction,
  useMemo,
  useCallback,
} from "react";
import Joyride, { Step, STATUS, CallBackProps, EVENTS } from "react-joyride";
import { useSport } from "@/contexts/sport_context";
import { useLocation } from "react-router-dom";
import { TourContext } from "@/contexts/tour_context";
import { CustomJoyrideTooltip } from "./custom_joyride_tooltip";

/* ------------------------------------------------------------------ */
/* 1. Steps Definition                                                */
/* ------------------------------------------------------------------ */
export const baseSteps: Step[] = [
  {
    target: '[data-tour="sport-switch"]',
    content: "Tap here to switch between NBA and MLB.",
    disableBeacon: true,
    placement: "bottom",
  },
  {
    target: '[data-tour="date-picker"]',
    content: "Use the calendar to select a game date...",
    disableBeacon: true,
    placement: "bottom",
  },
  {
    target: '[data-tour="game-card"]:first-of-type',
    content:
      "Each card shows teams and predictions. Tap the arrow to expand for more options.",
    disableBeacon: true,
    placement: "top",
  },
  {
    target: '[data-tour="snapshot-button"]:first-of-type',
    content:
      "Click 'H2H Stats' to see a detailed statistical snapshot for the matchup.",
    disableBeacon: true,
    placement: "right",
  },
  {
    target: '[data-tour="weather-badge"]',
    content:
      "For outdoor games (MLB/NFL), we provide real-time weather, including wind speed and direction, which can impact gameplay.",
    disableBeacon: true,
    placement: "left",
  },
  {
    target: '[data-tour="tab-stats"]',
    content:
      "TO CONTINUE: Click 'Stats' tab below. THEN: Click the green 'Next' button below this text.",
    disableBeacon: true,
    placement: "top",
  },
  {
    target: '[data-tour="stats-subtab-advanced"]',
    content:
      "Click sub-tabs like 'Advanced' or 'Players' (NBA only) to view different stat categories.",
    placement: "bottom",
    disableBeacon: true,
  },
  {
    target: '[data-tour="stats-column-winpct"]',
    content: "Click any column header like this one to sort teams and players.",
    placement: "bottom",
    disableBeacon: true,
  },
  {
    target: '[data-tour="tab-more"]',
    content:
      "Almost done! CLICK: 'More' tab for display options and info. THEN: Click the green 'Next' button below this text.",
    placement: "top",
    disableBeacon: true,
  },
  {
    target: '[data-tour="theme-toggle"]',
    content: "Light or dark? Flip the theme anytime using this toggle.",
    placement: "top",
    disableBeacon: true,
  },
];

/* ------------------------------------------------------------------ */
/* 2. Provider component                                              */
/* ------------------------------------------------------------------ */
export const TourProvider = ({ children }: { children: ReactNode }) => {
  const [run, setRun] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const { sport } = useSport();
  const location = useLocation();

  /* -------------------------------------------------- */
  /* Track live presence of at least one game card      */
  /* -------------------------------------------------- */
  const [hasGameCard, setHasGameCard] = useState<boolean>(() =>
    Boolean(document.querySelector('[data-tour="game-card"]'))
  );

  useEffect(() => {
    const updatePresence = () =>
      setHasGameCard(
        Boolean(document.querySelector('[data-tour="game-card"]'))
      );

    // Initial check
    updatePresence();

    // Observe DOM mutations for game‑card additions/removals
    const mo = new MutationObserver(updatePresence);
    mo.observe(document.body, { childList: true, subtree: true });
    return () => mo.disconnect();
  }, []);

  // unified scroll unlock helper
  const unlockScroll = useCallback(() => {
    ["body", "html", "main"].forEach((sel) => {
      const el = document.querySelector<HTMLElement>(sel);
      if (el) {
        el.style.overflow = "";
        el.style.position = "";
        el.style.paddingRight = "";
        el.style.top = "";
      }
    });
    document
      .querySelectorAll<HTMLElement>(".react-joyride__scroll-parent")
      .forEach((el) => {
        el.style.overflow = "";
        el.style.paddingRight = "";
      });
  }, []);

  const scrub = useCallback(() => {
    unlockScroll();
  }, [unlockScroll]);

  // teardown helper
  const cleanUpArtifacts = useCallback(() => {
    unlockScroll();
    document
      .querySelectorAll(
        ".react-joyride__overlay, .react-joyride__mask, .react-joyride__spotlight"
      )
      .forEach((n) => n.remove());
    document.documentElement.style.overflow = "auto";
    document.body.style.overflow = "auto";
  }, [unlockScroll]);

  // observer to keep DOM from locking scroll via mutation churn
  useEffect(() => {
    const observer = new MutationObserver(() => {
      scrub();
    });

    observer.observe(document.documentElement, {
      attributes: true,
      childList: true,
      subtree: true,
      attributeFilter: ["style", "class"],
    });

    return () => observer.disconnect();
  }, [scrub]);

  // reset scroll when location changes
  useEffect(() => {
    unlockScroll();
  }, [location.pathname, unlockScroll]);

  /* -------------------------------------------------- */
  /* Dynamically build steps, swapping in fallback      */
  /* -------------------------------------------------- */
  const tourSteps = useMemo(() => {
    // base list cloned so we never mutate source
    const steps: Step[] =
      sport === "NBA"
        ? baseSteps
            .filter((s) => s.target !== '[data-tour="weather-badge"]')
            .map((s) => ({ ...s }))
        : baseSteps.map((s) => ({ ...s }));

    // game‑card step is index 2
    if (!hasGameCard && steps[2]) {
      steps[2] = {
        target: '[data-tour="date-picker"]',
        content:
          "No games are scheduled for this date or sport. Pick another date or switch sport, then click **Next** to continue the tour.",
        disableBeacon: true,
        placement: "bottom",
      } as Step;
    }

    return steps;
  }, [sport, hasGameCard]);

  const handleJoyride = useCallback(
    (data: CallBackProps) => {
      const { status, type } = data;

      console.debug("Joyride callback:", data);

      // always ensure scroll is unlocked on every callback
      unlockScroll();

      const isTourFinished =
        status === STATUS.FINISHED ||
        status === STATUS.SKIPPED ||
        status === STATUS.ERROR;

      if (isTourFinished || type === EVENTS.TOUR_END) {
        if (run) {
          setRun(false);
          setStepIndex(0);
        }

        setTimeout(cleanUpArtifacts, 50);
      }
    },
    [run, cleanUpArtifacts, unlockScroll]
  );

  const start = () => {
    // force-remount Joyride by toggling run OFF → ON on the next tick
    setRun(false);
    setStepIndex(0);
    setTimeout(() => setRun(true), 0); // next macrotask
  };

  const contextValue = {
    start,
    setStepIndex,
    setRun,
    currentStepIndex: stepIndex,
    run,
  };

  return (
    <TourContext.Provider value={contextValue}>
      {children}
      {run && (
        <Joyride
          key={`tour-${stepIndex}`}
          steps={tourSteps}
          run={run}
          stepIndex={stepIndex}
          callback={handleJoyride}
          tooltipComponent={CustomJoyrideTooltip}
          disableScrolling={true}
          scrollToFirstStep
          styles={{ options: { zIndex: 9999 } }}
        />
      )}
    </TourContext.Provider>
  );
};
