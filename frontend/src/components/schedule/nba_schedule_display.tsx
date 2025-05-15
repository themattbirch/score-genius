import React, { useState, useEffect, useMemo } from "react";
import { startOfDay, isBefore } from "date-fns";

import { useDate } from "@/contexts/date_context";
import { useNBASchedule } from "@/api/use_nba_schedule";
import { useInjuries, type Injury } from "@/api/use_injuries";
import { useNetworkStatus } from "@/hooks/use_network_status";

import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";
import { ChevronDown } from "lucide-react";

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
  const dateCtx = useDate();
  if (!dateCtx) {
    // Provider not ready yet – render nothing or a small loader
    return null; // <─ you can swap for <SkeletonBox …/> if preferred
  }
  const { date } = dateCtx;
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

  /* ── Queries ─────────────────────────────────────────── */
  const {
    data: games = [],
    isLoading: isLoadingGames,
    isError: gamesError,
  } = useNBASchedule(isoDate);

  const {
    data: injuries = [],
    isLoading: isLoadingInjuries,
    error: injuriesError,
  } = useInjuries("NBA", isoDate);

  /* 1 ▸ build today’s team-set (lower-case, trimmed) */
  const playingTeams = useMemo(() => {
    const set = new Set<string>();
    games.forEach((g) => {
      [g.homeTeamName, g.awayTeamName].forEach((t) => {
        if (t) set.add(t.trim().toLowerCase());
      });
    });
    return set;
  }, [games]);

  /* 2 ▸ group injuries only for those teams */
  const injuriesByTeam = useMemo(() => {
    const grouped: Record<string, Injury[]> = {};
    injuries.forEach((inj) => {
      if (!inj.team_display_name) return; // ← guard null / ''
      const teamKey = inj.team_display_name.trim().toLowerCase();
      if (!playingTeams.has(teamKey)) return; // ← skip non-playing teams
      (grouped[inj.team_display_name] ||= []).push(inj);
    });
    return grouped;
  }, [injuries, playingTeams]);

  const teamsWithInjuries = useMemo(
    () => Object.keys(injuriesByTeam).sort(),
    [injuriesByTeam]
  );

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

      {/* ── 1 ▸ browser offline / fetch errors / loading ── */}
      {!online ? (
        <p className="text-center text-slate-500 dark:text-slate-400">
          You’re offline. Can’t fetch NBA games for {displayDate}.
        </p>
      ) : isLoadingGames ? (
        <div className="p-4">
          <h2 className="mb-3 text-lg font-semibold italic animate-pulse text-gray-500 dark:text-text-primary">
            Loading NBA games for {displayDate}…
          </h2>
          <div className="space-y-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <SkeletonBox key={i} className="h-24 w-full" />
            ))}
          </div>
        </div>
      ) : gamesError ? (
        <p className="text-center text-slate-500 dark:text-slate-400">
          Error fetching NBA games for {displayDate}.
        </p>
      ) : (
        <>
          {/* ── 2 ▸ games list ─────────────────────────── */}
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

          {/* ── 3 ▸ daily injury report ────────────────── */}
          {games.length > 0 && (
            <div className="mt-8 border-t border-border pt-6">
              <h2 className="mb-3 text-left text-lg font-semibold text-slate-800 dark:text-text-primary">
                Daily Injury Report
              </h2>

              {isPastDate ? (
                <p className="text-left text-sm text-text-secondary">
                  Games have been completed. No injury statuses to report.
                </p>
              ) : allGamesFilteredOut ? (
                <p className="text-left text-sm text-text-secondary">
                  No remaining games today. No injury statuses to report.
                </p>
              ) : isLoadingInjuries ? (
                <p className="text-left text-sm italic text-text-secondary">
                  Loading injuries…
                </p>
              ) : injuriesError ? (
                <p className="text-left text-sm text-red-500">
                  Could not load injury report.
                </p>
              ) : teamsWithInjuries.length === 0 ? (
                <p className="text-left text-sm text-text-secondary">
                  No significant injuries reported for playing teams on{" "}
                  {displayDate}.
                </p>
              ) : (
                <div className="space-y-4">
                  {teamsWithInjuries.map((team) => (
                    <details
                      key={team}
                      className="app-card overflow-hidden group"
                    >
                      <summary className="flex cursor-pointer items-center justify-between gap-3 rounded-md px-4 py-3 text-slate-800 dark:text-text-primary hover:bg-gray-50 dark:hover:bg-gray-700/50 focus:outline-none focus:ring-2 focus:ring-green-400">
                        <span className="min-w-0 flex-1 font-medium">
                          {team}
                        </span>
                        <span className="flex-shrink-0 rounded-full border border-green-500 px-2.5 py-1 text-xs font-medium text-green-800 shadow-md dark:text-green-100">
                          {injuriesByTeam[team].length} available
                        </span>
                        <ChevronDown className="h-4 w-4 flex-shrink-0 transition-transform group-open:rotate-180" />
                      </summary>
                      <div className="mt-2 py-2">
                        <ul className="space-y-1">
                          {injuriesByTeam[team].map((inj) => (
                            <li
                              key={inj.id}
                              className="flex items-start justify-between rounded-md px-4 pt-3 hover:bg-gray-50 dark:hover:bg-gray-700/50"
                            >
                              <div className="flex-1 pr-4">
                                <p className="font-medium text-slate-800 dark:text-text-primary">
                                  {inj.player}
                                </p>
                                {inj.injury_type && (
                                  <p className="mt-1 text-xs text-gray-500 dark:text-text-secondary">
                                    {inj.injury_type}
                                  </p>
                                )}
                              </div>
                              <span className="ml-auto mr-10 flex-shrink-0 rounded-full border border-gray-300 bg-gray-100 px-2.5 py-1 text-xs font-medium text-slate-800 dark:border-border dark:bg-transparent dark:text-text-primary">
                                {inj.status}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </details>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default NBAScheduleDisplay;
