// src/components/schedule/mlb_schedule_display.tsx
import React from "react";
import { useDate } from "@/contexts/date_context";
import { useMLBSchedule, MLBGame } from "@/api/use_mlb_schedule"; // Use the new hook and type
import GameCard from "@/components/games/game_card"; // TODO: Needs modification for MLBGame type!
import SkeletonBox from "@/components/ui/skeleton_box";

const MLBScheduleDisplay: React.FC = () => {
  const { date } = useDate();
  const isoDate = date ? date.toISOString().slice(0, 10) : "";

  console.log(`%c[MLBScheduleDisplay] Preparing to call useMLBSchedule`, 'color: green', { isoDate });

  const {
    data: games, // This should be MLBGame[] | undefined
    isLoading: loadingGames,
    error: gamesError,
  } = useMLBSchedule(isoDate); // Pass date, enable only if date is valid

  // Log hook results
  console.log(`%c[MLBScheduleDisplay] useMLBSchedule results:`, 'color: green', { loadingGames, gamesError, hasData: games !== undefined });


  // 1. Handle Loading State
  if (loadingGames) {
     console.log(`%c[MLBScheduleDisplay] State: Loading`, 'color: green');
    return <div className="p-4"><SkeletonBox className="h-64 w-full" /></div>;
  }

  // 2. Handle Error State
  if (gamesError) {
    console.error("MLBScheduleDisplay: Error loading games.", gamesError);
     console.log(`%c[MLBScheduleDisplay] State: Error`, 'color: green');
    // Ensure backend MLB endpoint exists and works, otherwise this error will show
    return <p className="p-4 text-red-500">Error loading MLB games.</p>;
  }

  // 3. Validate that 'games' is an array
  if (!Array.isArray(games)) {
     console.log(`%c[MLBScheduleDisplay] State: Data is not array (or undefined initially)`, 'color: green', games);
     // Treat as empty for now if invalid format comes from API
     return <p className="p-4 text-text-secondary">No MLB games scheduled for this date.</p>;
  }

  // 4. Render the list or empty message
  console.log(`%c[MLBScheduleDisplay] State: Valid Data. Rendering ${games.length} MLB games.`, 'color: green');
  return (
    <div className="space-y-4 p-4">
      {games.length === 0 ? (
        <p className="text-text-secondary">No MLB games scheduled for this date.</p>
      ) : (
        // *** TODO: Update GameCard to handle MLBGame type ***
        // It needs to know how to display MLB fields (pitchers, etc.)
        // For now, we pass the MLBGame typed data.
        games.map((game: MLBGame) => (
             // Make sure key is unique (game.game_id)
            <GameCard key={game.game_id} game={game as any} /> // Temp 'as any' or update GameCard prop type
        ))
      )}
    </div>
  );
};

export default MLBScheduleDisplay;