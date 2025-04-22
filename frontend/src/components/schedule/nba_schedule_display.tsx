// src/components/schedule/nba_schedule_display.tsx
import React, { useMemo } from "react"; // Import useMemo for data processing
import { useDate } from "@/contexts/date_context";
import { useNBASchedule } from "@/api/use_nba_schedule";
import { useInjuries, Injury } from "@/api/use_injuries";
import { UnifiedGame, Sport } from "@/types";
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";
import { ChevronDown } from "lucide-react";

// Helper function to group injuries by team name
const groupInjuriesByTeam = (injuries: Injury[]): Record<string, Injury[]> => {
  return injuries.reduce((acc, injury) => {
    const team = injury.team_display_name;
    if (team) {
      if (!acc[team]) {
        acc[team] = [];
      }
      acc[team].push(injury);
    }
    return acc;
  }, {} as Record<string, Injury[]>);
};

// Helper function to format date for display
const formatDisplayDate = (date: Date | null | undefined): string => {
  if (!date) {
    return "selected date"; // Or return an empty string "", or handle as needed
  }
  return date.toLocaleDateString("en-US", { month: "long", day: "numeric" });
};

const NBAScheduleDisplay: React.FC = () => {
  const { date } = useDate();
  // isoDate is kept for API calls and logic requiring YYYY-MM-DD
  const isoDate = date ? date.toISOString().slice(0, 10) : "";
  // NEW: displayDate is formatted for user-facing text
  const displayDate = formatDisplayDate(date);

  const {
    data: games,
    isLoading: loadingGames,
    error: gamesError,
  } = useNBASchedule("NBA" as Sport, isoDate);
  const {
    data: injuries = [],
    isLoading: loadingInjuries,
    error: injuriesError,
  } = useInjuries("NBA" as Sport, isoDate);

  const { teamsWithInjuries, injuriesByTeam } = useMemo(() => {
    if (
      !games ||
      !Array.isArray(games) ||
      !injuries ||
      !Array.isArray(injuries)
    ) {
      return { teamsWithInjuries: [], injuriesByTeam: {} };
    }
    const playingTeams = [
      ...new Set(games.flatMap((g) => [g.homeTeamName, g.awayTeamName])),
    ];
    const grouped = groupInjuriesByTeam(injuries);
    const teamsWithInjuries = playingTeams
      .filter((teamName) => grouped[teamName]?.length)
      .sort();
    return { teamsWithInjuries, injuriesByTeam: grouped };
  }, [games, injuries]);

  if (gamesError) {
    console.error("NBAScheduleDisplay: Error loading games.", gamesError);
    return (
      <div className="p-4">
        <h2 className="text-lg text-center font-semibold mb-3 text-red-600 dark:text-red-500">
          {/* CHANGED: Use displayDate */}
          Error Loading NBA Games for {displayDate}
        </h2>
        <p className="p-4 text-center text-red-500">
          Could not load game data.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4">
      <h2
        className={`text-lg text-center font-semibold mb-3 ${
          loadingGames
            ? "text-gray-500 dark:text-text-secondary italic animate-pulse"
            : "text-gray-900 dark:text-text-primary"
        }`}
      >
        {/* CHANGED: Use displayDate in both loading and loaded states */}
        {loadingGames
          ? `Loading NBA Games for ${displayDate}...`
          : `NBA Games for ${displayDate}`}
      </h2>
      <div className="space-y-4">
        {loadingGames ? (
          <>
            <SkeletonBox className="h-24 w-full" />
            <SkeletonBox className="h-24 w-full" />
            <SkeletonBox className="h-24 w-full" />
          </>
        ) : !Array.isArray(games) || games.length === 0 ? (
          <p className="text-text-secondary text-center mt-4">
            {/* UPDATED: Added displayDate for clarity */}
            No NBA games scheduled for {displayDate}.
          </p>
        ) : (
          games.map((game: UnifiedGame) => (
            <GameCard key={game.id} game={game} />
          ))
        )}
      </div>

      {!loadingGames && Array.isArray(games) && (
        <div className="mt-8 pt-6 border-t border-border">
          <h2 className="text-lg text-center font-semibold mb-3 text-gray-900 dark:text-text-primary">
            {/* CHANGED: Use displayDate */}
            Daily Injury Report
          </h2>
          {loadingInjuries ? (
            <p className="text-sm text-text-secondary italic text-center">
              Loading injuries...
            </p>
          ) : injuriesError ? (
            <p className="text-sm text-red-500 text-center">
              Could not load injury report.
            </p>
          ) : teamsWithInjuries.length === 0 ? (
            <p className="text-sm text-text-secondary text-center">
              {/* UPDATED: Added displayDate for clarity */}
              No significant injuries reported for playing teams on{" "}
              {displayDate}.
            </p>
          ) : (
            <div className="space-y-2">
              {teamsWithInjuries.map((teamName) => (
                <details
                  key={teamName}
                  className="app-card overflow-hidden group"
                >
                  {/* --- REVISED SUMMARY BELOW --- */}
                  {/* Keep items-start for top alignment */}
                  {/* Use gap-2 for spacing between elements */}
                  <summary className="flex items-start justify-between gap-2 p-3 bg-transparent text-gray-900 dark:text-text-primary cursor-pointer">
                    {/* Team Name: Allow grow/shrink/wrap */}
                    <span className="font-medium flex-1 min-w-0">
                      {teamName}
                    </span>

                    {/* Badge: Prevent shrinking - UPDATED STYLES */}
                    <span className="flex-shrink-0 px-2.5 py-1 text-xs rounded-full bg-orange-100 text-orange-800 dark:bg-orange-900/80 dark:text-orange-200">
                      {injuriesByTeam[teamName].length} available
                    </span>

                    {/* Chevron: Prevent shrinking */}
                    <ChevronDown className="flex-shrink-0 w-4 h-4 transition-transform group-open:rotate-180" />
                  </summary>

                  {/* Injury List Details (Unchanged) */}
                  <div className="border-t border-border bg-transparent p-3">
                    <ul className="space-y-1">
                      {injuriesByTeam[teamName].map((inj) => (
                        <li
                          key={inj.id}
                          className="flex items-center justify-between gap-2 py-1"
                        >
                          {/* Player + injury type */}
                          <span className="flex-1 break-words text-gray-800 dark:text-text-primary">
                            {inj.player}
                            {inj.injury_type && (
                              <span className="ml-1 text-xs text-gray-500 dark:text-text-secondary">
                                ({inj.injury_type})
                              </span>
                            )}
                          </span>

                          {/* Status badge */}
                          <span
                            className="
      whitespace-nowrap px-1.5 py-0.5 text-xs font-medium rounded border

      /* Light mode badge */
      border-gray-300 light:bg-gray-100 light:text-gray-800

      /* Dark mode badge */
      dark:border-border dark:bg-panel-hover dark:text-text-primary
    "
                          >
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
