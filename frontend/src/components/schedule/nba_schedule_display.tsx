// src/components/schedule/nba_schedule_display.tsx
import React, { useMemo } from "react"; // Import useMemo for data processing
import { useDate } from "@/contexts/date_context";
import { useNBASchedule } from "@/api/use_nba_schedule";
// Import injury hook and types
import { useInjuries, Injury } from "@/api/use_injuries";
// Import unified type and Sport type
import { UnifiedGame, Sport } from "@/types";
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";
import { ChevronDown } from "lucide-react"; // Optional: for accordion icon

// Helper function to group injuries by team name
const groupInjuriesByTeam = (injuries: Injury[]): Record<string, Injury[]> => {
  return injuries.reduce((acc, injury) => {
    const team = injury.team_display_name; // Assumes injury object has 'team' property with the name
    if (team) {
      // Only group if team name exists
      if (!acc[team]) {
        acc[team] = [];
      }
      acc[team].push(injury);
    }
    return acc;
  }, {} as Record<string, Injury[]>);
};

const NBAScheduleDisplay: React.FC = () => {
  const { date } = useDate();
  const isoDate = date ? date.toISOString().slice(0, 10) : "";

  // --- Fetch Game Schedule ---
  const {
    data: games,
    isLoading: loadingGames,
    error: gamesError,
  } = useNBASchedule("NBA" as Sport, isoDate);

  // --- Fetch Injuries (Only runs for NBA because of 'enabled' inside hook) ---
  const {
    data: injuries = [], // Default to empty array for easier handling
    isLoading: loadingInjuries,
    error: injuriesError,
  } = useInjuries("NBA" as Sport, isoDate); // Pass NBA and date

  // --- Process Data for Injury Display (Memoized) ---
  // Avoid reprocessing on every render unless games or injuries change
  const { teamsWithInjuries, injuriesByTeam } = useMemo(() => {
    // Ensure we have valid arrays before processing
    if (
      !games ||
      !Array.isArray(games) ||
      !injuries ||
      !Array.isArray(injuries)
    ) {
      return { teamsWithInjuries: [], injuriesByTeam: {} };
    }

    // 1. Get unique list of teams playing on the selected date
    const playingTeams = [
      ...new Set(games.flatMap((g) => [g.homeTeamName, g.awayTeamName])),
    ];

    // 2. Group all fetched injuries by their team name
    const grouped = groupInjuriesByTeam(injuries);

    // 3. Filter the playing teams list to only those that have reported injuries
    const teamsWithInjuries = playingTeams
      .filter((teamName) => grouped[teamName] && grouped[teamName].length > 0)
      .sort(); // Sort alphabetically

    return { teamsWithInjuries, injuriesByTeam: grouped };
  }, [games, injuries]); // Dependencies for useMemo
  // --- End Data Processing ---

  // --- Loading and Error States ---
  // Show main loading skeleton if games are loading
  if (loadingGames) {
    console.log(`%c[NBAScheduleDisplay] State: Loading Games`, "color: blue");
    return (
      <div className="p-4 space-y-4">
        {/* Skeletons for game cards */}
        <SkeletonBox className="h-24 w-full" />
        <SkeletonBox className="h-24 w-full" />
        <SkeletonBox className="h-24 w-full" />
        {/* Skeleton for injury section */}
        <SkeletonBox className="mt-8 pt-4 h-20 w-full" />
      </div>
    );
  }

  // Handle critical game fetching error first
  if (gamesError) {
    console.error("NBAScheduleDisplay: Error loading games.", gamesError);
    return <p className="p-4 text-red-500">Error loading NBA games.</p>;
  }

  // Handle case where games data isn't a valid array after loading
  if (!Array.isArray(games)) {
    console.log(
      `%c[NBAScheduleDisplay] State: Invalid Game data format`,
      "color: orange",
      games
    );
    return (
      <p className="p-4 text-text-secondary">No NBA games data available.</p>
    );
  }
  // --- End Loading/Error States ---

  // --- Render UI ---
  return (
    // Changed main div to only pad (space-y is now within sub-divs)
    <div className="p-4">
  {/* Games Report Section */}
  <h2 className="text-lg text-center font-semibold mb-3 text-gray-900 dark:text-text-primary">
    NBA Games for {isoDate}
  </h2>

  <div className="space-y-4">
    {games.length === 0 ? (
      <p className="text-text-secondary">
        No NBA games scheduled for this date.
      </p>
    ) : (
      games.map((game: UnifiedGame) => (
        <GameCard key={game.id} game={game} />
      ))
    )}
  </div>
      {/* Injury Report Section */}
      <div className="mt-8 pt-6 border-t border-border">
        <h2 className="text-lg text-center font-semibold mb-3 text-gray-900 dark:text-text-primary">
          {" "}
          NBA Injury Report ({isoDate})
        </h2>

        {loadingInjuries ? (
          <p className="text-sm text-text-secondary italic">
            Loading injuries...
          </p>
        ) : injuriesError ? (
          <p className="text-sm text-red-500">Could not load injury report.</p>
        ) : teamsWithInjuries.length === 0 ? (
          <p className="text-sm text-text-secondary">
            No significant injuries reported for playing teams on this date.
          </p>
        ) : (
          <div className="space-y-2">
            {teamsWithInjuries.map((teamName) => {
              const count = injuriesByTeam[teamName]?.length ?? 0;
              return (
                <details
                  key={teamName}
                  className="app-card overflow-hidden group"
                >
                  {/* === HEADER === */}
                  <summary
                    className={`
          flex items-center justify-between p-3
          
          /* LIGHT HEADER BG */
bg-transparent text-gray-900 dark:text-text-primary          /* DARK HEADER BG */
          
          cursor-pointer list-none select-none
          focus:outline-none focus:ring-0
        `}
                  >
                    <span className="flex-1 break-words font-medium">
                      {teamName}
                    </span>
                    <div className="flex items-center space-x-2 flex-shrink-0">
                      <span
                        className="
              whitespace-nowrap px-2 py-0.5 text-xs font-semibold rounded-full
              bg-red-100 text-red-800
              dark:bg-destructive/80 dark:text-destructive-foreground
            "
                      >
                        {count} available
                      </span>
                      <ChevronDown className="w-4 h-4 text-text-secondary transition-transform group-open:rotate-180" />
                    </div>
                  </summary>

                  {/* === PANEL === */}
                  <div className="border-t border-border bg-transparent p-3">
                    <ul className="space-y-1">
                      {injuriesByTeam[teamName].map((inj) => (
                        <li
                          key={inj.id}
                          className="flex items-center justify-between gap-2 py-1"
                        >
                          <span className="flex-1 break-words text-gray-800 dark:text-text-primary">
                            {inj.player}
                            {inj.injury_type && (
                              <span className="ml-1 text-xs text-gray-500 dark:text-text-secondary">
                                ({inj.injury_type})
                              </span>
                            )}
                          </span>
                          <span
                            className="whitespace-nowrap px-1.5 py-0.5 text-xs font-medium rounded border
                   border-gray-300 light:bg-gray-100 dark:bg-panel text-gray-800
                   dark:border-border dark:bg-panel-hover dark:text-text-primary"
                          >
                            {inj.status}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </details>
              );
            })}
          </div>
        )}
      </div>
    </div> /* <-- closes main wrapper <div className="p-4"> */
  ); /* <-- closes return( … ) */
}; /* <-- closes NBAScheduleDisplay component */

export default NBAScheduleDisplay;
