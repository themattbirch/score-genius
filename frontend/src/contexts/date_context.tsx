// frontend/src/contexts/date_context.tsx

import React, { createContext, useContext, useState, ReactNode, useCallback } from "react";
import { getLocalYYYYMMDD } from "@/utils/date";
import type { UnifiedGame } from "@/types";
import { useSport } from "./sport_context"; // if needed to filter by sport

// Placeholder: you need to supply this based on your data layer.
// For example, if you have a games cache or selector elsewhere, import and use it.
// Replace this with the real implementation.
async function fetchGamesForDate(dateString: string): Promise<UnifiedGame[]> {
  // TODO: replace with actual logic (e.g., read from in-memory cache, call API, etc.)
  return []; // empty fallback
}

// ── Context value interface ─────────────────────────────────
interface DateCtx {
  date: Date; // always a valid Date instance
  dateStringET: string; // “YYYY-MM-DD” in Eastern Time
  setDate: (d: Date) => void;

  // New additions:
  jumpToDate: (d: Date) => void;
  useGamesForSelectedDate: () => UnifiedGame[]; // synchronous; assumes you have cached data
  findNextDateWithGames: () => Promise<string | null>; // returns YYYY-MM-DD or null
}

// ── Create context ───────────────────────────────────────────
const DateContext = createContext<DateCtx | undefined>(undefined);

// ── Provider component ───────────────────────────────────────
export function DateProvider({ children }: { children: ReactNode }) {
  const [date, setDate] = useState<Date>(new Date());
  const [dateStringET, setDateStringET] = useState<string>(getLocalYYYYMMDD(date));

  const { sport } = useSport(); // if games are sport-scoped

  // Wrap setDate to also update the ET string
  function handleSetDate(d: Date) {
    setDate(d);
    setDateStringET(getLocalYYYYMMDD(d));
  }

  const jumpToDate = (d: Date) => {
    handleSetDate(d);
  };

  // Synchronous hook to get games for the current date. This assumes you have
  // cached/preloaded games elsewhere; if not, you’d need to make this async.
  const useGamesForSelectedDate = useCallback((): UnifiedGame[] => {
    // Replace with your real source. Example placeholder:
    // const allGames = useGameCacheForDate(dateStringET); // you might have this
    // return allGames.filter(g => g.sport === sport);
    // Temporary stub (empty)
    return [];
  }, [dateStringET, sport]);

  // Search forward up to N days for the next date with games
  const findNextDateWithGames = useCallback(async (): Promise<string | null> => {
    const MAX_LOOKAHEAD = 7; // e.g., search up to 7 days ahead
    const base = new Date(date);
    for (let offset = 1; offset <= MAX_LOOKAHEAD; offset++) {
      const candidate = new Date(base);
      candidate.setDate(base.getDate() + offset);
      const candidateStr = getLocalYYYYMMDD(candidate);
      // Replace with real fetch/lookup
      const games = await fetchGamesForDate(candidateStr);
      const filtered = sport
        ? games.filter((g) => g.sport === sport)
        : games;
      if (filtered.length > 0) {
        return candidateStr;
      }
    }
    return null;
  }, [date, sport]);

  return (
    <DateContext.Provider
      value={{
        date,
        dateStringET,
        setDate: handleSetDate,
        jumpToDate,
        useGamesForSelectedDate,
        findNextDateWithGames,
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
