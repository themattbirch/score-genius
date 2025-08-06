// frontend/src/components/ui/joyride_tour.tsx
import React, {
  useState,
  useMemo,
  useCallback,
  useEffect,
  ReactNode,
} from "react";
import Joyride, { Step, STATUS, CallBackProps, EVENTS } from "react-joyride";
import { useSport } from "@/contexts/sport_context";
import { TourContext } from "@/contexts/tour_context";
import { CustomJoyrideTooltip } from "./custom_joyride_tooltip";

// Base step definitions
export const baseSteps: Step[] = [
  {
    target: '[data-tour="sport-switch"]',
    content: "Tap here to switch between NFL, NBA and MLB.",
    disableBeacon: true,
    placement: "bottom",
  },
  {
    target: '[data-tour="game-card"]:first-of-type',
    content:
      "Each card shows teams and predictions. On mobile, tap to expand for more options.",
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

export const TourProvider = ({ children }: { children: ReactNode }) => {
  const [run, setRun] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const { sport } = useSport();

  // Re-add state tracking for game cards
  const [hasGameCard, setHasGameCard] = useState<boolean>(() =>
    Boolean(document.querySelector('[data-tour="game-card"]'))
  );

  // Re-add observer to detect when game cards appear
  useEffect(() => {
    const updatePresence = () => {
      const gameCardFound = Boolean(
        document.querySelector('[data-tour="game-card"]')
      );
      setHasGameCard(gameCardFound);
    };
    updatePresence();
    const observer = new MutationObserver(updatePresence);
    observer.observe(document.body, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, []);

  const cleanUpArtifacts = useCallback(() => {
    document.documentElement.style.overflow = "auto";
    document.body.style.overflow = "auto";
    document
      .querySelectorAll(".react-joyride__overlay, .react-joyride__spotlight")
      .forEach((n) => n.remove());
  }, []);

  const tourSteps = useMemo(() => {
    let steps = [...baseSteps];

    // RESTORED: Fallback for Game Card (now at index 1)
    const gameCardStep = steps[1];
    if (
      !hasGameCard &&
      gameCardStep &&
      typeof gameCardStep.target === "string" &&
      gameCardStep.target.includes("game-card")
    ) {
      gameCardStep.target = '[data-tour="sport-switch"]';
      gameCardStep.content =
        "No games scheduled. Switch sports or pick another date, then click Next.";
    }

    // Sport-specific filtering
    if (sport === "NBA") {
      steps = steps.filter(
        (step) => step.target !== '[data-tour="weather-badge"]'
      );
    }

    // UPDATED: NFL has a different stats screen layout
    if (sport === "NFL") {
      steps = steps.filter(
        (step) =>
          step.target !== '[data-tour="stats-subtab-advanced"]' &&
          step.target !== '[data-tour="stats-column-winpct"]'
      );
    }

    return steps;
  }, [sport, hasGameCard]);

  const handleJoyride = useCallback(
    (data: CallBackProps) => {
      const { status, type } = data;
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
    [run, cleanUpArtifacts]
  );

  const start = () => {
    setRun(false);
    setStepIndex(0);
    setTimeout(() => setRun(true), 0);
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
