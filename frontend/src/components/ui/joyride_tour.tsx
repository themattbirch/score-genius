// frontend/src/components/ui/joyride_tour.tsx
import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
  useEffect,
  SetStateAction,
} from "react";
import Joyride, {
  Step,
  STATUS,
  CallBackProps,
  ACTIONS,
  EVENTS,
  LIFECYCLE,
} from "react-joyride";
// --- Ensure this import is correct ---
import { CustomJoyrideTooltip } from "./custom_joyride_tooltip"; // Import from separate file

// --- CustomJoyrideTooltip definition should NOT be here ---
// It should be in its own file as imported above.

/* ------------------------------------------------------------------ */
/* 1. Context - ADD SETTERS                                           */
/* ------------------------------------------------------------------ */
interface TourCtx {
  start: () => void;
  // Add state setters and current index to context
  setStepIndex: React.Dispatch<SetStateAction<number>>;
  setRun: React.Dispatch<SetStateAction<boolean>>;
  currentStepIndex: number;
}
const TourContext = createContext<TourCtx | undefined>(undefined);

// Modify useTour to expose the full context value
export const useTour = (): TourCtx => {
  // Return TourCtx directly
  const ctx = useContext(TourContext);
  if (!ctx) throw new Error("useTour must be used inside <TourProvider>");
  return ctx;
};

/* ------------------------------------------------------------------ */
/* 2. Steps Definition                                                */
/* ------------------------------------------------------------------ */
const steps: Step[] = [
  // Step 1: Sport Switch (index 0)
  {
    target: '[data-tour="sport-switch"]',
    content: "Tap here to switch between NBA and MLB.",
    disableBeacon: true,
    placement: "bottom",
  },
  // Step 2: Games Tab (index 1)
  {
    target: '[data-tour="tab-games"]',
    content:
      "TO CONTINUE: Click the 'Games' tab below to switch screens. THEN: Click the green 'Next' button below this text.",
    disableBeacon: true,
    placement: "top",
  },
  // Step 3: Date Picker (index 2)
  {
    target: '[data-tour="date-picker"]',
    content: "Use the calendar to select a game date...",
    disableBeacon: true,
    placement: "bottom",
  },
  // Step 4: Game Card (index 3)
  {
    target: '[data-tour="game-card"]:first-of-type',
    content:
      "Each card shows teams, game information and Score Genius' score prediction.",
    disableBeacon: true,
    placement: "bottom",
  },
  // Step 5: Stats Tab (index 4)
  {
    target: '[data-tour="tab-stats"]',
    content:
      "TO CONTINUE: Click 'Stats' tab below. THEN: Click the green 'Next' button below this text.",
    disableBeacon: true,
    placement: "top",
  },

  // --- STEP 6: Sub-tabs (index 5) ---
  {
    target: '[data-tour="stats-subtab-advanced"]', // <<< UPDATE target attribute
    content:
      "Click sub-tabs like 'Advanced' or 'Players' (NBA only) to view different stat categories.", // Adjusted content
    placement: "bottom",
    disableBeacon: true,
  },
  // --- Step 7: Stats Column (index 6) ---
  {
    target: '[data-tour="stats-column-winpct"]',
    content: "Click any column header like this one to sort teams and players.",
    placement: "bottom",
    disableBeacon: true,
  },
  // --- STEP 8: More Tab (index 7) ---
  {
    target: '[data-tour="tab-more"]',
    content:
      "Almost done! CLICK: 'More' tab for display options and info. THEN: Click the green 'Next' button below this text.",
    placement: "top",
    disableBeacon: true,
  },
  // --- Step 9: Theme Toggle (index 8) ---
  {
    target: '[data-tour="theme-toggle"]',
    content: "Light or dark? Flip the theme anytime using this toggle.",
    placement: "top", // Or 'bottom'/'auto' depending on layout
    disableBeacon: true,
  },
];
/* ------------------------------------------------------------------ */
/* 3. Provider component                                              */
/* ------------------------------------------------------------------ */
export const TourProvider = ({ children }: { children: ReactNode }) => {
  const [run, setRun] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    console.log(`TourProvider state: run=${run}, stepIndex=${stepIndex}`);
  }, [run, stepIndex]);

  const start = () => {
    console.log("--- TourProvider: start function called ---");
    setStepIndex(0);
    setRun(true);
  };

  // --- MINIMAL handleJoyride - Only handles tour end/error events ---
  // It no longer needs to handle button actions like NEXT/PREV
  const handleJoyride = (data: CallBackProps) => {
    const { status, type, lifecycle, step, action, index } = data;

    console.log(
      `Joyride Minimal Callback: Action=${action}, Status=${status}, Type=${type}, Index=${index}, Lifecycle=${lifecycle}`,
      step?.target
    );

    // Only stop the tour on final states or errors reported by Joyride
    if (
      status === STATUS.FINISHED ||
      status === STATUS.SKIPPED ||
      status === STATUS.ERROR
    ) {
      console.log(`Tour Ended/Error via Joyride status: Status=${status}`);
      if (status === STATUS.ERROR) {
        console.error("Joyride Error:", data);
      }
      // Check run state before setting to prevent infinite loops if error occurs rapidly
      if (run) {
        setRun(false);
        setStepIndex(0); // Reset index too
      }
    } else if (type === EVENTS.TOUR_END) {
      console.log("Tour ended via TOUR_END event");
      if (run) {
        setRun(false);
        setStepIndex(0);
      }
    }
    // We ignore STEP_AFTER, TARGET_NOT_FOUND etc. for state changes now
  };

  // --- Provide setters and index in context value ---
  const contextValue = {
    start,
    setStepIndex,
    setRun,
    currentStepIndex: stepIndex,
  };

  return (
    <TourContext.Provider value={contextValue}>
      {" "}
      {/* Provide extended value */}
      {children}
      <Joyride
        steps={steps}
        run={run}
        stepIndex={stepIndex}
        callback={handleJoyride} // Minimal callback
        tooltipComponent={CustomJoyrideTooltip} // Custom tooltip handles buttons
        // --- REMOVE showSkipButton prop - button logic is now fully custom ---
        // showSkipButton
        scrollToFirstStep
        // debug
        styles={{ options: { zIndex: 9999 } }}
      />
    </TourContext.Provider>
  );
};

// export default TourProvider;
