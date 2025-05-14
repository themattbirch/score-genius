// frontend/src/components/schedule/mlb_schedule_display.tsx
import React, { useState, useEffect, useMemo } from "react";
import { useDate } from "@/contexts/date_context";
import { useNBASchedule } from "@/api/use_nba_schedule";
import type { UnifiedGame } from "@/types";
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";
import { startOfDay, isBefore } from "date-fns";

interface ScheduleDisplayProps {
  showHeader?: boolean;
}

const formatLocalDate = (d: Date | null | undefined): string =>
  !d
    ? ""
    : `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
        d.getDate()
      ).padStart(2, "0")}`;

const NBAScheduleDisplay: React.FC<ScheduleDisplayProps> = ({
  showHeader = true,
}) => {
  // ── Hooks in fixed order ───────────────────────────────────────────────────
  const { date } = useDate();
  const isoDate = formatLocalDate(date);
  const displayDate = date?.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
  });

  const today = startOfDay(new Date());
  const selectedDay = date ? startOfDay(date) : null;
  const isPastDate = selectedDay ? isBefore(selectedDay, today) : false;

  const [currentTime, setCurrentTime] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setCurrentTime(Date.now()), 60_000);
    return () => clearInterval(id);
  }, []);

  const {
    data: games = [],
    isLoading: isLoadingGames,
    error: gamesError,
  } = useNBASchedule(isoDate);

  const filteredGames = useMemo(() => {
    if (isPastDate) return games;
    const buffer = 3.5 * 60 * 60 * 1000; // 3.5h
    return games.filter((g) => {
      const start = g.gameTimeUTC;
      if (!start) return true;
      const ms = new Date(start).getTime();
      return !Number.isNaN(ms) && currentTime < ms + buffer;
    });
  }, [games, currentTime, isPastDate]);

  const hasVisibleGames = filteredGames.length > 0;
  const noGamesInitiallyScheduled = games.length === 0;
  const allGamesFilteredOut = games.length > 0 && filteredGames.length === 0;

  // ── Single return with conditional branches ────────────────────────────────
  return (
    <div className="pt-4">
      {showHeader && (
        <h2 className="mb-3 text-left text-lg font-semibold text-slate-800 dark:text-text-primary">
          NBA Games for {displayDate}
        </h2>
      )}

      {isLoadingGames ? (
        <div className="p-4">
          <h2 className="text-lg text-left font-semibold mb-3 italic animate-pulse text-gray-500 dark:text-text-primary">
            Loading NBA Games for {isoDate}…
          </h2>
          <div className="space-y-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <SkeletonBox key={i} className="h-24 w-full" />
            ))}
          </div>
        </div>
      ) : gamesError ? (
        <div className="pt-4">
          <h2 className="mb-2 text-lg font-semibold text-red-600 dark:text-red-500">
            Error Loading NBA Games.
          </h2>
          <p className="text-red-500">Could not load game data.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {hasVisibleGames ? (
            filteredGames.map((g) => <GameCard key={g.id} game={g} />)
          ) : noGamesInitiallyScheduled ? (
            <p className="mt-4 text-left text-text-secondary">
              No NBA games scheduled for {displayDate}.
            </p>
          ) : allGamesFilteredOut ? (
            <p className="mt-4 text-left text-text-secondary">
              All NBA games for {displayDate} have concluded.
            </p>
          ) : null}
        </div>
      )}
    </div>
  );
};

export default NBAScheduleDisplay;
