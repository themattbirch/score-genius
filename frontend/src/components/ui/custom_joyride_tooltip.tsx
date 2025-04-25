// frontend/src/components/ui/custom_joyride_tooltip.tsx
import React from "react";
import type { TooltipRenderProps } from "react-joyride";
// --- Import the useTour hook ---
import { useTour } from "./joyride_tour"; // Adjust path if needed

export const CustomJoyrideTooltip: React.FC<TooltipRenderProps> = ({
  index,
  step,
  isLastStep,
  tooltipProps,
  // We no longer need backProps, primaryProps, skipProps from Joyride!
}) => {
  // --- Get state setters and current index from context ---
  const { setStepIndex, setRun, currentStepIndex } = useTour();

  // --- Custom Handler for NEXT ---
  const handleNext = () => {
    const nextIndex = currentStepIndex + 1;
    // Special delay logic after Step 2 (index 1)
    if (currentStepIndex === 1) {
      console.log("Custom Next on Step 2 (idx 1), delaying advancement...");
      setTimeout(() => {
        console.log(`Attempting to set stepIndex to ${nextIndex} after delay.`);
        setStepIndex(nextIndex);
      }, 300);
    } else {
      console.log(
        `Custom Next on Step ${currentStepIndex}, advancing state to stepIndex: ${nextIndex}`
      );
      setStepIndex(nextIndex);
    }
  };

  // --- Custom Handler for BACK ---
  const handleBack = () => {
    const prevIndex = currentStepIndex - 1;
    console.log(
      `Custom Back on Step ${currentStepIndex}, going back state to stepIndex: ${prevIndex}`
    );
    if (prevIndex >= 0) {
      setStepIndex(prevIndex);
    }
  };

  // --- Custom Handler for SKIP/CLOSE ---
  const handleSkip = () => {
    console.log("Custom Skip/Close clicked, stopping tour.");
    setRun(false); // Set run to false
    setStepIndex(0); // Reset index
  };

  return (
    <div
      {...tooltipProps}
      className="bg-white dark:bg-slate-800 text-slate-900 dark:text-white rounded-lg shadow-xl p-4 max-w-sm-full text-sm font-sans" // Example: Changed p-4 to p-5, max-w-xs to max-w-sm
    >
      <div className="mb-4 p-4">{step.content}</div>
      <div className="flex justify-between items-center pt-3 mt-2 border-t border-slate-200 dark:border-slate-600">
        {/* Skip Button */}
        <button
          onClick={handleSkip} // Use custom handler
          className="text-xs text-slate-500 dark:text-slate-400 hover:underline"
        >
          Skip {/* Or maybe Close? */}
        </button>

        {/* Back/Next Buttons Container */}
        <div className="flex gap-2">
          {/* Back Button */}
          {index > 0 && ( // Still use index prop for conditional rendering
            <button
              onClick={handleBack} // Use custom handler
              className="px-3 py-1 text-xs rounded bg-slate-200 dark:bg-slate-600 hover:bg-slate-300 dark:hover:bg-slate-500 text-slate-800 dark:text-slate-100"
            >
              Back
            </button>
          )}
          {/* Next/Finish Button */}
          <button
            onClick={isLastStep ? handleSkip : handleNext} // Use custom handler (Finish acts like Skip/Close)
            className="px-3 py-1 text-xs rounded bg-green-600 hover:bg-green-700 text-white font-medium"
            title={isLastStep ? "Finish" : "Next"}
            aria-label={isLastStep ? "Finish" : "Next"}
          >
            {isLastStep ? "Finish" : "Next"}
          </button>
        </div>
      </div>
    </div>
  );
};
