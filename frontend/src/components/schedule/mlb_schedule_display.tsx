// src/components/schedule/mlb_schedule_display.tsx
import React from "react";
import { useDate } from "@/contexts/date_context";
// Import the hook
import { useMLBSchedule } from "@/api/use_mlb_schedule";
// Import the UNIFIED game type
import { UnifiedGame } from "@/types"; // Adjust path if needed
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";

const MLBScheduleDisplay: React.FC = () => {
  const { date } = useDate();
  const isoDate = date ? date.toISOString().slice(0, 10) : "";

  console.log(
    `%c[MLBScheduleDisplay] Preparing to call useMLBSchedule`,
    "color: green",
    { isoDate }
  );

  const {
    data: games, // This should now be UnifiedGame[] | undefined
    isLoading, // Renamed for consistency
    error, // Renamed for consistency
  } = useMLBSchedule(isoDate);

  console.log(
    `%c[MLBScheduleDisplay] useMLBSchedule results:`,
    "color: green",
    { isLoading, error, hasData: games !== undefined }
  );

  if (isLoading) {
    console.log(`%c[MLBScheduleDisplay] State: Loading`, "color: green");
    return (
      <div className="p-4">
        <SkeletonBox className="h-64 w-full" />
      </div>
    );
  }

  if (error) {
    console.error("MLBScheduleDisplay: Error loading games.", error);
    console.log(`%c[MLBScheduleDisplay] State: Error`, "color: green");
    return <p className="p-4 text-red-500">Error loading MLB games.</p>;
  }

  // Validate that 'games' is an array (optional but safe)
  if (!Array.isArray(games)) {
    console.log(
      `%c[MLBScheduleDisplay] State: Data is not array`,
      "color: green",
      games
    );
    return (
      <p className="p-4 text-text-secondary">
        No MLB games scheduled for this date.
      </p>
    );
  }

  console.log(
    `%c[MLBScheduleDisplay] State: Valid Data. Rendering ${games.length} MLB games.`,
    "color: green"
  );
  return (
    <div className="space-y-4 p-4">
      {games.length === 0 ? (
        <p className="text-text-secondary">
          No MLB games scheduled for this date.
        </p>
      ) : (
        // Use UnifiedGame type and unified 'id'
        games.map((game: UnifiedGame) => <GameCard key={game.id} game={game} />)
      )}
    </div>
  );
};

export default MLBScheduleDisplay;
