// src/components/schedule/nba_schedule_display.tsx
import React from "react";
import { useDate } from "@/contexts/date_context";
// Assuming you renamed the hook and file, and updated the Game interface if needed
import { useNBASchedule, Game } from "@/api/use_nba_schedule";
import GameCard from "@/components/games/game_card"; // Needs to accept NBA Game type
import SkeletonBox from "@/components/ui/skeleton_box";

// Renamed old hook to useNBASchedule and updated its internals
// We assume useNBASchedule now only needs the date
// If it still takes sport, pass 'NBA' explicitly

const NBAScheduleDisplay: React.FC = () => {
  const { date } = useDate();
  // Make sure date processing is robust if date can be null/undefined initially
  const isoDate = date ? date.toISOString().slice(0, 10) : "";

  console.log(`%c[NBAScheduleDisplay] Preparing to call useNBASchedule`, 'color: blue', { isoDate });

  const {
    data: games, // useQuery should return Game[] or undefined
    isLoading: loadingGames,
    error: gamesError,
  } = useNBASchedule('NBA', isoDate); // Pass date, enable only if date is valid

  // Log hook results
  console.log(`%c[NBAScheduleDisplay] useNBASchedule results:`, 'color: blue', { loadingGames, gamesError, hasData: games !== undefined });


  // 1. Handle Loading State
  if (loadingGames) {
     console.log(`%c[NBAScheduleDisplay] State: Loading`, 'color: blue');
    // Use a fragment or adjust SkeletonBox props if needed
    return <div className="p-4"><SkeletonBox className="h-64 w-full" /></div>;
  }

  // 2. Handle Error State
  if (gamesError) {
    console.error("NBAScheduleDisplay: Error loading games.", gamesError);
     console.log(`%c[NBAScheduleDisplay] State: Error`, 'color: blue');
    return <p className="p-4 text-red-500">Error loading NBA games.</p>;
  }

  // 3. Validate that 'games' is an array (safer even if hook type says Game[])
  //    useQuery returns undefined initially before fetch/cache hit
  if (!Array.isArray(games)) {
     console.log(`%c[NBAScheduleDisplay] State: Data is not array (or undefined initially)`, 'color: blue', games);
     // It might be undefined before the first fetch, treat same as empty or loading briefly?
     // If it persists, it might be an API format issue returning non-array on success.
     // For now, treat as empty if not loading/error but also not array.
     return <p className="p-4 text-text-secondary">No NBA games scheduled for this date.</p>;
  }

  // 4. Render the list or empty message
   console.log(`%c[NBAScheduleDisplay] State: Valid Data. Rendering ${games.length} NBA games.`, 'color: blue');
  return (
    <div className="space-y-4 p-4">
      {games.length === 0 ? (
        <p className="text-text-secondary">No NBA games scheduled for this date.</p>
      ) : (
        // Ensure GameCard handles the NBA 'Game' type
        games.map((game: Game) => <GameCard key={game.id} game={game} />)
      )}
    </div>
  );
};

export default NBAScheduleDisplay;