// frontend/src/components/schedule/nba_schedule_display.tsx
import React, { useState, useEffect, useMemo } from "react";
import { startOfDay, isBefore } from "date-fns";

import { useDate } from "@/contexts/date_context";
import { useNBASchedule } from "@/api/use_nba_schedule";
import { useNetworkStatus } from "@/hooks/use_network_status";

import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";

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
  /* ── Hooks ───────────────────────────────────────────── */
  const { date } = useDate();
  const online = useNetworkStatus(); // ← NEW

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

  const { data: games = [], isLoading, isError } = useNBASchedule(isoDate);

  /* ── Filter out completed games on same day ──────────── */
  const filteredGames = useMemo(() => {
    if (isPastDate) return games;

    const buffer = 3.5 * 60 * 60 * 1000; // 3.5 h
    return games.filter((g) => {
      const ms = new Date(g.gameTimeUTC ?? "").getTime();
      return Number.isNaN(ms) ? true : currentTime < ms + buffer;
    });
  }, [games, currentTime, isPastDate]);

  const noGamesInitiallyScheduled = games.length === 0;
  const allGamesFilteredOut = games.length > 0 && filteredGames.length === 0;

  /* ── Render ──────────────────────────────────────────── */
  return (
    <div className="pt-4">
      {showHeader && (
        <h2 className="mb-3 text-left text-lg font-semibold text-slate-800 dark:text-text-primary">
          NBA Games for {displayDate}
        </h2>
      )}

      {/* ---- 1 ▸ browser offline ---- */}
      {!online ? (
        <p className="text-center text-slate-500 dark:text-slate-400">
          Error fetching NBA games for {displayDate}.
        </p>
      ) : isLoading ? (
        /* ---- 2 ▸ loading ---- */
        <div className="p-4">
          <h2 className="text-lg font-semibold mb-3 italic animate-pulse text-gray-500 dark:text-text-primary">
            Loading NBA games for {displayDate}…
          </h2>
          <div className="space-y-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <SkeletonBox key={i} className="h-24 w-full" />
            ))}
          </div>
        </div>
      ) : isError ? (
        /* ---- 3 ▸ fetch error ---- */
        <p className="text-center text-slate-500 dark:text-slate-400">
          Error fetching NBA games for {displayDate}.
        </p>
      ) : (
        /* ---- 4 ▸ success ---- */
        <div className="space-y-4">
          {filteredGames.length ? (
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
