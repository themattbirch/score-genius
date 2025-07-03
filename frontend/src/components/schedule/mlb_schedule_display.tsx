// frontend/src/components/schedule/mlb_schedule_display.tsx
import React, { useState, useEffect, useMemo, lazy, Suspense } from "react"; // Added lazy and Suspense
import { startOfDay, isBefore } from "date-fns";

import { useDate } from "@/contexts/date_context";
import { useMLBSchedule } from "@/api/use_mlb_schedule";
import { useNetworkStatus } from "@/hooks/use_network_status";

// GameCard is now lazy-loaded, so we don't need the direct import here.
// import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";
import type { UnifiedGame } from "@/types";

// NEW: Define the GameCard as a lazy-loaded component
const LazyGameCard = lazy(() => import("@/components/games/game_card"));

const formatLocalDate = (d: Date | null | undefined): string =>
  !d
    ? ""
    : `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
        d.getDate()
      ).padStart(2, "0")}`;

// Removed unused props for cleanup
const MLBScheduleDisplay: React.FC = () => {
  /* ── Hooks ───────────────────────────────────────────── */
  const { date } = useDate();
  const online = useNetworkStatus();

  const isoDate = formatLocalDate(date);
  const displayDate = date?.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
  });

  const today = startOfDay(new Date());
  const selectedDay = date ? startOfDay(date) : null;
  const isPastDate = selectedDay ? isBefore(selectedDay, today) : false;

  const [currentTime, setCurrentTime] = useState(Date.now);
  useEffect(() => {
    const id = setInterval(() => setCurrentTime(Date.now()), 60_000);
    return () => clearInterval(id);
  }, []);

  // This API hook is where the prediction data will come from
  const { data: games = [], isLoading, isError } = useMLBSchedule(isoDate);

  /* ── Hide finished games on same day ────────────────── */
  const filteredGames = useMemo(() => {
    if (isPastDate) return games;

    const buffer = 3.5 * 60 * 60 * 1000; // 3.5 h
    return games.filter((g: UnifiedGame) => {
      const ms = new Date(g.gameTimeUTC ?? "").getTime();
      return Number.isNaN(ms) ? true : currentTime < ms + buffer;
    });
  }, [games, currentTime, isPastDate]);

  const noGamesInitiallyScheduled = games.length === 0;
  const allGamesFilteredOut = games.length > 0 && filteredGames.length === 0;

  /* ── Render ──────────────────────────────────────────── */
  return (
    <div className="pt-4">
      {!online ? (
        /* ---- offline Message ---- */
        <p className="text-center text-slate-500 dark:text-slate-400">
          Live MLB schedule for {displayDate} requires internet. Please
          reconnect.
        </p>
      ) : isLoading ? (
        /* ---- loading ---- */
        <div className="p-4">
          <h2 className="text-lg font-semibold mb-3 italic animate-pulse text-gray-500 dark:text-text-primary">
            Loading MLB games for {displayDate}…
          </h2>
          <div className="space-y-4">
            {Array.from({ length: 4 }).map(
              (
                _,
                i // Using a fixed number for initial load
              ) => (
                <SkeletonBox key={i} className="h-24 w-full" />
              )
            )}
          </div>
        </div>
      ) : isError ? (
        /* ---- fetch error ---- */
        <p className="text-center text-slate-500 dark:text-slate-400">
          Error fetching MLB games for {displayDate}.
        </p>
      ) : (
        /* ---- success ---- */
        <div className="space-y-4">
          {filteredGames.length ? (
            // MODIFIED: Wrapped the list in Suspense for better UX
            <Suspense
              fallback={
                // This fallback shows a skeleton for each card while it loads
                <div className="space-y-4">
                  {Array.from({ length: filteredGames.length }).map((_, i) => (
                    <SkeletonBox key={i} className="h-24 w-full" />
                  ))}
                </div>
              }
            >
              <div className="space-y-4">
                {filteredGames.map((g) => (
                  // MODIFIED: Using the lazy-loaded component
                  <LazyGameCard key={g.id} game={g} />
                ))}
              </div>
            </Suspense>
          ) : noGamesInitiallyScheduled ? (
            <p className="mt-4 text-left text-text-secondary">
              No MLB games scheduled for {displayDate}.
            </p>
          ) : allGamesFilteredOut ? (
            <p className="mt-4 text-left text-text-secondary">
              All MLB games for {displayDate} have concluded.
            </p>
          ) : null}
        </div>
      )}
    </div>
  );
};

export default MLBScheduleDisplay;
