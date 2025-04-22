// src/components/schedule/nba_schedule_display.tsx
import React from "react";
import { useDate } from "@/contexts/date_context";
// Import the hook
import { useNBASchedule } from "@/api/use_nba_schedule";
// Import the UNIFIED game type
import { UnifiedGame, Sport } from "@/types"; // Adjust path if needed
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";

const NBAScheduleDisplay: React.FC = () => {
  const { date } = useDate();
  const isoDate = date ? date.toISOString().slice(0, 10) : "";

  console.log(
    `%c[NBAScheduleDisplay] Preparing to call useNBASchedule`,
    "color: blue",
    { isoDate }
  );

  const {
    data: games, // This should now be UnifiedGame[] | undefined
    isLoading, // Renamed for consistency
    error, // Renamed for consistency
    // --- CORRECTED HOOK CALL ---
  } = useNBASchedule("NBA" as Sport, isoDate); // Pass 'NBA' and date

  console.log(`%c[NBAScheduleDisplay] useNBASchedule results:`, "color: blue", {
    isLoading,
    error,
    hasData: games !== undefined,
  });

  if (isLoading) {
    console.log(`%c[NBAScheduleDisplay] State: Loading`, "color: blue");
    return (
      <div className="p-4">
        <SkeletonBox className="h-64 w-full" />
      </div>
    );
  }

  if (error) {
    console.error("NBAScheduleDisplay: Error loading games.", error);
    console.log(`%c[NBAScheduleDisplay] State: Error`, "color: blue");
    return <p className="p-4 text-red-500">Error loading NBA games.</p>;
  }

  // Validate that 'games' is an array (optional but safe)
  if (!Array.isArray(games)) {
    console.log(
      `%c[NBAScheduleDisplay] State: Data is not array`,
      "color: blue",
      games
    );
    return (
      <p className="p-4 text-text-secondary">
        No NBA games scheduled for this date.
      </p>
    );
  }

  console.log(
    `%c[NBAScheduleDisplay] State: Valid Data. Rendering ${games.length} NBA games.`,
    "color: blue"
  );
  return (
    <div className="space-y-4 p-4">
      {games.length === 0 ? (
        <p className="text-text-secondary">
          No NBA games scheduled for this date.
        </p>
      ) : (
        // Use UnifiedGame type and unified 'id'
        games.map((game: UnifiedGame) => <GameCard key={game.id} game={game} />)
      )}
    </div>
  );
};

export default NBAScheduleDisplay;
