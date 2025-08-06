// frontend/src/components/ui/custom_joyride_tooltip.tsx
import React from "react";
import type { TooltipRenderProps } from "react-joyride";
import { useTour } from "@/contexts/tour_context";

const waitForElement = (
  selector: string,
  timeout = 3_000,
  poll = 100
): Promise<void> =>
  new Promise((resolve) => {
    const start = performance.now();
    const check = () => {
      const el = document.querySelector<HTMLElement>(selector);
      if (el && el.offsetWidth && el.offsetHeight) {
        return resolve();
      }
      if (performance.now() - start >= timeout) {
        console.warn(
          `[Joyride] waited ${timeout} ms – element not found`,
          selector
        );
        return resolve();
      }
      setTimeout(check, poll);
    };
    check();
  });

const STEP_READINESS: Record<number, string | null> = {
  1: '[data-tour="snapshot-button"]', // When on step 1 (Game Card), wait for H2H button
};

export const CustomJoyrideTooltip: React.FC<TooltipRenderProps> = ({
  index,
  step,
  isLastStep,
  tooltipProps,
}) => {
  const { setStepIndex, setRun, currentStepIndex } = useTour();

  const advance = async () => {
    const nextIndex = currentStepIndex + 1;
    const selector = STEP_READINESS[currentStepIndex];
    if (selector) {
      await waitForElement(selector);
    }
    setStepIndex(nextIndex);
  };

  const goBack = () => {
    const prev = currentStepIndex - 1;
    if (prev >= 0) setStepIndex(prev);
  };

  const stopTour = () => {
    setRun(false);
    setStepIndex(0);
  };

  const noGameCardPresent = !document.querySelector('[data-tour="game-card"]');
  // The Game Card step is now at index 1
  const isOnGameCardStep = currentStepIndex === 1;
  const targetSelector = typeof step.target === "string" ? step.target : "";
  const targetMissing =
    targetSelector && !document.querySelector(targetSelector);
  const showFallback = isOnGameCardStep && (noGameCardPresent || targetMissing);

  const fallbackContent = (
    <>
      <div className="font-medium mb-2">
        No games scheduled for this date/sport.
      </div>
      <div className="text-xs mb-3">
        To continue the tour, pick a different date or switch to a sport that
        has games.
      </div>
      <div className="text-[10px] text-slate-500">
        Once you have a game card visible, click Next to proceed.
      </div>
    </>
  );

  return (
    <div
      {...tooltipProps}
      className="bg-white dark:bg-slate-800 text-slate-900 dark:text-white rounded-lg shadow-xl p-5 max-w-sm text-sm font-sans"
    >
      <div className="mb-4">
        {showFallback ? fallbackContent : step.content}
      </div>

      <div className="flex justify-between items-center pt-3 mt-2 border-t border-slate-200 dark:border-slate-600">
        <button
          onClick={stopTour}
          className="text-xs text-slate-500 dark:text-slate-400 hover:underline"
        >
          {isLastStep ? "Finish" : "Skip"}
        </button>

        <div className="flex gap-2">
          {index > 0 && (
            <button
              onClick={goBack}
              className="px-3 py-1 text-xs rounded bg-slate-200 dark:bg-slate-600 hover:bg-slate-300 dark:hover:bg-slate-500 text-slate-800 dark:text-slate-100"
            >
              Back
            </button>
          )}
          <button
            onClick={showFallback ? stopTour : isLastStep ? stopTour : advance}
            className="px-3 py-1 text-xs rounded bg-green-600 hover:bg-green-700 text-white font-medium"
          >
            {showFallback ? "Got it" : isLastStep ? "Finish" : "Next"}
          </button>
        </div>
      </div>
    </div>
  );
};
