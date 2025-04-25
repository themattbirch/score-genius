// frontend/src/components/schedule/nba_schedule_display.tsx

import React, { useMemo } from "react";
import { useDate } from "@/contexts/date_context";
import { useNBASchedule } from "@/api/use_nba_schedule";
import { useInjuries, type Injury } from "@/api/use_injuries";
import type { UnifiedGame } from "@/types";
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";
import { ChevronDown } from "lucide-react";

const groupInjuriesByTeam = (inj: Injury[]) =>
  inj.reduce<Record<string, Injury[]>>((acc, i) => {
    const team = i.team_display_name;
    if (team) {
      (acc[team] ??= []).push(i);
    }
    return acc;
  }, {});

const formatDate = (d: Date | null) =>
  d ? d.toLocaleDateString("en-US", { month: "long", day: "numeric" }) : "";

const NBAScheduleDisplay: React.FC = () => {
  
  const { date } = useDate();
  const isoDate = date?.toISOString().slice(0, 10) ?? "";
  const displayDate = formatDate(date);

  console.log("[NBA schedule] isoDate =", isoDate);

  
  const {
    data: games,
    isLoading: loadingGames,
    error: gamesError,
  } = useNBASchedule(isoDate);

  const {
    data: injuries = [],
    isLoading: loadingInjuries,
    error: injuriesError,
  } = useInjuries("NBA", isoDate);

  const { teamsWithInjuries, injuriesByTeam } = useMemo(() => {
    if (!games?.length || !injuries.length) {
      return { teamsWithInjuries: [], injuriesByTeam: {} };
    }
    const playing = new Set(
      games.flatMap((g) => [g.homeTeamName, g.awayTeamName])
    );
    const grouped = groupInjuriesByTeam(injuries);
    const teams = [...playing].filter((t) => grouped[t]?.length).sort();
    return { teamsWithInjuries: teams, injuriesByTeam: grouped };
  }, [games, injuries]);

  if (gamesError) {
    return (
      <div className="p-4 text-center">
        <h2 className="mb-2 text-lg font-semibold text-red-600 dark:text-red-500">
          Error Loading NBA Games for {displayDate}
        </h2>
        <p className="text-red-500">Could not load game data.</p>
      </div>
    );
  }

  return (
    <div className="p-4">
      <h2
        className={`mb-3 text-center text-lg font-semibold ${
          loadingGames
            ? "animate-pulse italic text-gray-500 dark:text-text-secondary"
            : "text-gray-900 dark:text-text-primary"
        }`}
      >
        {loadingGames
          ? `Loading NBA Games for ${displayDate}…`
          : `NBA Games for ${displayDate}`}
      </h2>

      <div className="space-y-4">
        {loadingGames ? (
          <>
            {Array.from({ length: 3 }).map((_, i) => (
              <SkeletonBox key={i} className="h-24 w-full" />
            ))}
          </>
        ) : games && games.length > 0 ? (
          <>
            {games.map((game) => (
              <GameCard key={game.id} game={game} />
            ))}
          </>
        ) : (
          <p className="mt-4 text-center text-text-secondary">
            No NBA games scheduled for {displayDate}.
          </p>
        )}
      </div>

      {!loadingGames && games && games.length > 0 && (
        <div className="mt-8 border-t border-border pt-6">
          <h2 className="mb-3 text-center text-lg font-semibold text-gray-900 dark:text-text-primary">
            Daily Injury Report
          </h2>

          {loadingInjuries ? (
            <p className="text-center text-sm italic text-text-secondary">
              Loading injuries…
            </p>
          ) : injuriesError ? (
            <p className="text-center text-sm text-red-500">
              Could not load injury report.
            </p>
          ) : teamsWithInjuries.length === 0 ? (
            <p className="text-center text-sm text-text-secondary">
              No significant injuries reported for playing teams on{" "}
              {displayDate}.
            </p>
          ) : (
            <div className="space-y-2">
              {teamsWithInjuries.map((team) => (
                <details key={team} className="app-card overflow-hidden group">
                  <summary className="flex cursor-pointer items-start justify-between gap-2 bg-transparent p-3 text-gray-900 dark:text-text-primary">
                    <span className="min-w-0 flex-1 font-medium">{team}</span>
                    <span className="flex-shrink-0 rounded-full bg-orange-100 px-2.5 py-1 text-xs text-orange-800 dark:bg-orange-900/80 dark:text-orange-200">
                      {injuriesByTeam[team].length} available
                    </span>
                    <ChevronDown className="h-4 w-4 flex-shrink-0 transition-transform group-open:rotate-180" />
                  </summary>

                  <div className="border-t border-border bg-transparent p-3">
                    <ul className="space-y-1">
                      {injuriesByTeam[team].map((inj) => (
                        <li
                          key={inj.id}
                          className="flex items-center justify-between gap-2 py-1"
                        >
                          <span className="break-words text-gray-800 dark:text-text-primary">
                            {inj.player}
                            {inj.injury_type && (
                              <span className="ml-1 text-xs text-gray-500 dark:text-text-secondary">
                                ({inj.injury_type})
                              </span>
                            )}
                          </span>
                          <span className="whitespace-nowrap rounded border border-gray-300 bg-gray-100 px-1.5 py-0.5 text-xs font-medium text-gray-800 dark:border-border dark:bg-panel-hover dark:text-text-primary">
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
    </div>
  );
};

export default NBAScheduleDisplay;
