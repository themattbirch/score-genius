// frontend/src/components/schedule/nba_schedule_display.tsx
import React, { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { startOfDay, isBefore } from "date-fns";
import { useDate } from "@/contexts/date_context";
import { useNBASchedule } from "@/api/use_nba_schedule";
import { useInjuries, type Injury } from "@/api/use_injuries";
import { useNetworkStatus } from "@/hooks/use_network_status";

import SkeletonBox from "@/components/ui/skeleton_box";

/* ────────────────────────────────────────────────────────── */
/* Helpers                                                   */
/* ────────────────────────────────────────────────────────── */

const formatLocalDate = (d: Date | null | undefined): string => {
  if (!d) return "";
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

/* ────────────────────────────────────────────────────────── */
/* Lazy-loaded sub-components                                */
/* ────────────────────────────────────────────────────────── */

const LazyGameCard = lazy(() => import("@/components/games/game_card"));
const LazyInjuryReport = lazy(
  () => import("@/components/schedule/nba_injury_report")
);

/* ────────────────────────────────────────────────────────── */
/* Main component                                            */
/* ────────────────────────────────────────────────────────── */

interface ScheduleDisplayProps {
  showHeader?: boolean;
}

const NBAScheduleDisplay: React.FC<ScheduleDisplayProps> = ({}) => {
  /* ── context & network status ─────────────────────────── */
  const dateCtx = useDate();
  if (!dateCtx) return null;

  const online = useNetworkStatus();
  const { date } = dateCtx;

  /* ── derived date info ───────────────────────────────── */
  const isoDate = formatLocalDate(date);
  const displayDate = date?.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
  });

  const today = startOfDay(new Date());
  const selectedDay = date ? startOfDay(date) : null;
  const isPastDate = selectedDay ? isBefore(selectedDay, today) : false;

  /* ── current time ticker (for live-game filtering) ─────── */
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(id);
  }, []);

  /* ── schedule query ───────────────────────────────────── */
  const {
    data: games = [],
    isLoading: isLoadingGames,
    isError: gamesError,
  } = useNBASchedule(isoDate);

  /* ── injuries query ───────────────────────────────────── */
  const {
    data: injuries = [],
    isLoading: isLoadingInjuries,
    error: injuriesError,
  } = useInjuries("NBA", isoDate);

  /* ── teams playing today (lower-case) ─────────────────── */
  const playingTeams = useMemo(() => {
    const set = new Set<string>();
    games.forEach(({ homeTeamName, awayTeamName }) => {
      [homeTeamName, awayTeamName].forEach((t) => {
        if (t) set.add(t.trim().toLowerCase());
      });
    });
    return set;
  }, [games]);

  /* ── injuries grouped by playing team ─────────────────── */
  const injuriesByTeam = useMemo(() => {
    const grouped: Record<string, Injury[]> = {};
    injuries.forEach((inj) => {
      const t = inj.team_display_name?.trim();
      if (!t) return;
      const key = t.toLowerCase();
      if (!playingTeams.has(key)) return;
      (grouped[t] ||= []).push(inj);
    });
    return grouped;
  }, [injuries, playingTeams]);

  const teamsWithInjuries = useMemo(
    () => Object.keys(injuriesByTeam).sort(),
    [injuriesByTeam]
  );

  /* ── filter out completed games (except past dates) ───── */
  const filteredGames = useMemo(() => {
    if (isPastDate) return games;

    const bufferMs = 3.5 * 60 * 60 * 1000;
    return games.filter(({ gameTimeUTC }) => {
      const ms = new Date(gameTimeUTC ?? "").getTime();
      return Number.isNaN(ms) ? true : now < ms + bufferMs;
    });
  }, [games, now, isPastDate]);

  const noGamesInitiallyScheduled = games.length === 0;
  const allGamesFilteredOut = games.length > 0 && filteredGames.length === 0;

  /* ─────────────────────────────────────────────────────── */
  /* Render                                                 */
  /* ─────────────────────────────────────────────────────── */

  if (!online)
    return (
      <p className="text-center text-slate-500 dark:text-slate-400">
        Live NBA schedule for {displayDate} requires internet. Please reconnect.
      </p>
    );

  if (isLoadingGames)
    return (
      <div className="space-y-4 px-4">
        <h2 className="animate-pulse text-center text-lg font-semibold italic text-gray-500 dark:text-text-secondary">
          Loading NBA games for {displayDate}…
        </h2>
        {Array.from({ length: 3 }).map((_, i) => (
          <SkeletonBox key={i} className="h-24 w-full" />
        ))}
      </div>
    );

  if (gamesError)
    return (
      <p className="text-center text-slate-500 dark:text-slate-400 px-4">
        Error fetching NBA games for {displayDate}.
      </p>
    );

  return (
    <div className="pt-4 space-y-8">
      {/* ── games list ── */}
      {filteredGames.length ? (
        <Suspense
          fallback={
            <div className="space-y-4">
              {Array.from({ length: 3 }).map((_, i) => (
                <SkeletonBox key={i} className="h-24 w-full" />
              ))}
            </div>
          }
        >
          <div className="space-y-4">
            {filteredGames.map((g) => (
              <LazyGameCard key={g.id} game={g} />
            ))}
          </div>
        </Suspense>
      ) : noGamesInitiallyScheduled ? (
        <p className="mt-4 text-left text-text-secondary">
          No NBA games scheduled for {displayDate}.
        </p>
      ) : allGamesFilteredOut ? (
        <p className="mt-4 text-left text-text-secondary">
          All NBA games for {displayDate} have concluded.
        </p>
      ) : null}

      {/* ── Injury Report ─────────────────────────────────────── */}
      {games.length > 0 && (
        <div className="mt-8 border-t border-border pt-6">
          {/* static header */}
          <h2 className="mb-3 text-left text-lg font-semibold text-slate-800 dark:text-text-primary">
            Daily Injury Report
          </h2>

          {/* body is lazy-loaded / skeleton-swapped */}
          <Suspense
            fallback={
              <div className="space-y-4">
                {teamsWithInjuries.map((team) => (
                  <SkeletonBox
                    key={team}
                    className="w-full rounded-md px-4 py-3 bg-slate-700/50 animate-pulse"
                  />
                ))}
              </div>
            }
          >
            <LazyInjuryReport
              displayDate={displayDate}
              isPastDate={isPastDate}
              allGamesFilteredOut={allGamesFilteredOut}
              isLoadingInjuries={isLoadingInjuries}
              injuriesError={injuriesError ?? undefined}
              teamsWithInjuries={teamsWithInjuries}
              injuriesByTeam={injuriesByTeam}
            />
          </Suspense>
        </div>
      )}
    </div>
  );
};

export default NBAScheduleDisplay;
