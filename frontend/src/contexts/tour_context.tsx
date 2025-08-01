// frontend/src/contexts/tour_context.tsx
import { createContext, useContext, SetStateAction } from "react";

export interface TourCtx {
  start: () => void;
  setStepIndex: React.Dispatch<SetStateAction<number>>;
  setRun: React.Dispatch<SetStateAction<boolean>>;
  currentStepIndex: number;
  run: boolean;
}

export const TourContext = createContext<TourCtx | undefined>(undefined);

export const useTour = (): TourCtx => {
  const ctx = useContext(TourContext);
  if (!ctx) {
    throw new Error("useTour must be used inside a TourProvider");
  }
  return ctx;
};
