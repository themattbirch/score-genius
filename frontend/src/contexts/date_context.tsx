// frontend/src/contexts/date_context.tsx

import React, { createContext, useContext, useState, ReactNode } from "react";
import { getLocalYYYYMMDD } from "@/utils/date";

// ── Context value interface ─────────────────────────────────
interface DateCtx {
  date: Date; // always a valid Date instance
  dateStringET: string; // “YYYY‑MM‑DD” in Eastern Time
  setDate: (d: Date) => void;
}

// ── Create context ───────────────────────────────────────────
const DateContext = createContext<DateCtx | undefined>(undefined);

// ── Provider component ───────────────────────────────────────
export function DateProvider({ children }: { children: ReactNode }) {
  // 1) Track the Date object
  const [date, setDate] = useState<Date>(new Date());

  // 2) Track the ET‑formatted string
  const [dateStringET, setDateStringET] = useState<string>(
    getLocalYYYYMMDD(date)
  );

  // Wrap setDate to also update the ET string
  function handleSetDate(d: Date) {
    setDate(d);
    setDateStringET(getLocalYYYYMMDD(d));
  }

  return (
    <DateContext.Provider
      value={{
        date,
        dateStringET,
        setDate: handleSetDate,
      }}
    >
      {children}
    </DateContext.Provider>
  );
}

// ── Hook for consuming context ───────────────────────────────
export function useDate(): DateCtx {
  const ctx = useContext(DateContext);
  if (!ctx) {
    throw new Error("useDate must be used within a DateProvider");
  }
  return ctx;
}
