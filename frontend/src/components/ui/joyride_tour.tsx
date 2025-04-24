// frontend/src/components/ui/joyride_tour.tsx
import React, { createContext, useContext, useState, ReactNode } from "react";
import Joyride, { Step, STATUS, CallBackProps } from "react-joyride";

/* ------------------------------------------------------------------ */
/* 1. Context so any component can trigger the tour                    */
/* ------------------------------------------------------------------ */
interface TourCtx {
  start: () => void;
}
const TourContext = createContext<TourCtx | undefined>(undefined);

export const useTour = () => {
  const ctx = useContext(TourContext);
  if (!ctx) throw new Error("useTour must be used inside <TourProvider>");
  return ctx;
};

/* ------------------------------------------------------------------ */
/* 2. Define the tour steps.                                           */
/*    Each `target` matches a `data-tour="..."` attr you’ll add.       */
/* ------------------------------------------------------------------ */
const steps: Step[] = [
  {
    target: '[data-tour="sport-switch"]',
    content: "Tap here to flip between NBA and MLB views.",
    placement: "bottom",
    disableBeacon: true,
  },
  {
    target: '[data-tour="date-picker"]',
    content: "Use the calendar to jump to any game date.",
    placement: "bottom",
    disableBeacon: true,
  },
  {
    target: '[data-tour="game-card"]:first-of-type',
    content:
      "Each card shows the matchup and our predicted score. Tap for full details.",
    placement: "bottom",
    disableBeacon: true,
  },
  {
    target: '[data-tour="tab-stats"]',
    content: "Ready for deeper numbers? Head to the Stats tab.",
    placement: "top",
    disableBeacon: true,
  },
  {
    target: '[data-tour="stats-column"]',
    content: "Click any column header to sort teams and players.",
    placement: "bottom",
    disableBeacon: true,
  },
  {
    target: '[data-tour="theme-toggle"]',
    content: "Light or dark? Flip the theme anytime in the More tab.",
    placement: "top",
    disableBeacon: true,
  },
];

/* ------------------------------------------------------------------ */
/* 3. Provider component                                               */
/* ------------------------------------------------------------------ */
export const TourProvider = ({ children }: { children: ReactNode }) => {
  const [run, setRun] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);

  const start = () => {
    setStepIndex(0);
    setRun(true);
  };

  const handleJoyride = (data: CallBackProps) => {
    const { status, index } = data;
    if (status === STATUS.FINISHED || status === STATUS.SKIPPED) {
      setRun(false);
    } else {
      setStepIndex(index);
    }
  };

  return (
    <TourContext.Provider value={{ start }}>
      {children}
      <Joyride
        steps={steps}
        run={run}
        stepIndex={stepIndex}
        continuous
        showSkipButton
        showProgress
        scrollToFirstStep
        callback={handleJoyride}
        styles={{
          options: {
            primaryColor: "#22c55e", // Tailwind green-500 for brand consistency
            zIndex: 9999,
          },
        }}
      />
    </TourContext.Provider>
  );
};
