import React, { createContext, useContext, useState, ReactNode } from 'react';

// ── Context value interface ─────────────────────────────────
interface DateCtx {
  date: Date;                    // always a valid Date instance
  setDate: (d: Date) => void;
}

// ── Create context ───────────────────────────────────────────
const DateContext = createContext<DateCtx | undefined>(undefined);

// ── Provider component ───────────────────────────────────────
export function DateProvider({ children }: { children: ReactNode }) {
  // Initialize with current date so 'date' is never null
  const [date, setDate] = useState<Date>(new Date());

  return (
    <DateContext.Provider value={{ date, setDate }}>
      {children}
    </DateContext.Provider>
  );
}

// ── Hook for consuming context ───────────────────────────────
export function useDate(): DateCtx {
  const ctx = useContext(DateContext);
  if (!ctx) {
    throw new Error('useDate must be used within a DateProvider');
  }
  return ctx;
}
