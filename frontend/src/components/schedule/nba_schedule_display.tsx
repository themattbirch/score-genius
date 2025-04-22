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
      {/* Game List Section */}
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
        {" "}
        {/* Added top padding & margin */}
        <h2 className="text-lg font-semibold mb-3 text-text-primary">
          NBA Injury Report ({isoDate}) {/* Show date for context */}
        </h2>
        {/* Handle Injury Loading/Error States Specifically */}
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
          // Render Accordion/List if injuries loaded successfully & exist
          <div className="space-y-2">
            {teamsWithInjuries.map((teamName) => (
              // Using HTML details/summary for simple accordion
              // Add group class for potential styling with group-open:
              <details
                key={teamName}
                className="bg-panel-muted rounded group border border-border shadow-sm"
              >
                <summary className="flex justify-between items-center p-2 cursor-pointer hover:bg-panel-hover list-none rounded-t">
                  <span className="font-medium text-text-primary">
                    {teamName}
                  </span>
                  {/* Injury Count Badge */}
                  <div className="flex items-center space-x-2">
                    <span className="text-xs bg-destructive/80 text-destructive-foreground rounded-full px-2 py-0.5">
                      {injuriesByTeam[teamName]?.length ?? 0}{" "}
                      {(injuriesByTeam[teamName]?.length ?? 0) === 1
                        ? "Player"
                        : "Players"}
                    </span>
                    {/* Optional: Chevron icon */}
                    <ChevronDown className="h-4 w-4 text-text-secondary group-open:rotate-180 transition-transform" />
                  </div>
                </summary>
                {/* Content of the accordion */}
                <div className="p-2 pl-4 border-t border-border bg-panel rounded-b">
                  <ul className="space-y-1">
                    {(injuriesByTeam[teamName] || []).map((inj) => (
                      <li
                        key={inj.id}
                        className="text-sm flex justify-between items-center text-text-secondary py-0.5"
                      >
                        {/* Left side: Player and Injury Type */}
                        <span
                          className="flex-grow truncate mr-2"
                          title={
                            inj.detail || `Type: ${inj.injury_type || "N/A"}`
                          }
                        >
                          {" "}
                          {/* Added type to title */}
                          {inj.player || "Unknown Player"}
                          {/* Display Injury Type if available */}
                          {inj.injury_type && (
                            <span className="ml-2 text-xs opacity-70">
                              ({inj.injury_type})
                            </span>
                          )}
                        </span>
                        {/* Right side: Status Badge */}
                        <span className="flex-shrink-0 font-medium text-text-primary text-xs py-0.5 px-1.5 rounded bg-panel-hover border border-border">
                          {inj.status || "N/A"}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </details>
            ))}
          </div>
        )}
      </div>{" "}
      {/* End Injury Section */}
    </div> // End Main Container
  );
};

export default NBAScheduleDisplay;
