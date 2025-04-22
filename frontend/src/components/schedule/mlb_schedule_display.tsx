// src/components/schedule/mlb_schedule_display.tsx
import React from "react"; // Removed useMemo as it's not strictly needed here
import { useDate } from "@/contexts/date_context";
import { useMLBSchedule } from "@/api/use_mlb_schedule";
import { UnifiedGame } from "@/types"; // Adjust path if needed
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";

// Helper function to format date for display (consistent with NBA)
const formatDisplayDate = (date: Date | null | undefined): string => {
  if (!date) {
    return "selected date"; // Fallback text
  }
  // Using 'en-US' locale for "Month Day" format, e.g., "April 22"
  return date.toLocaleDateString('en-US', { month: 'long', day: 'numeric' });
};

const MLBScheduleDisplay: React.FC = () => {
  const { date } = useDate();
  // isoDate is kept for API calls (YYYY-MM-DD)
  const isoDate = date ? date.toISOString().slice(0, 10) : "";
  // displayDate is formatted for user-facing text ("Month Day")
  const displayDate = formatDisplayDate(date);

  const {
    data: games,
    isLoading, // Using the hook's loading state
    error,
  } = useMLBSchedule(isoDate); // Pass isoDate to the hook

  // --- Loading State ---
  if (isLoading) {
    return (
      <div className="p-4">
         {/* Consistent Loading Header */}
        <h2 className="text-lg text-center font-semibold mb-3 text-gray-500 dark:text-text-secondary italic animate-pulse">
          Loading MLB Games for {displayDate}...
        </h2>
         {/* Consistent Skeleton Structure */}
        <div className="space-y-4">
          <SkeletonBox className="h-24 w-full" />
          <SkeletonBox className="h-24 w-full" />
          <SkeletonBox className="h-24 w-full" />
        </div>
      </div>
    );
  }

  // --- Error State ---
  if (error) {
    console.error("MLBScheduleDisplay: Error loading games.", error);
    return (
       // Consistent Error Display Structure
      <div className="p-4">
        <h2 className="text-lg text-center font-semibold mb-3 text-red-600 dark:text-red-500">
          Error Loading MLB Games for {displayDate}
        </h2>
        <p className="p-4 text-center text-red-500">Could not load game data.</p>
      </div>
    );
  }

  // --- Data Display ---
  // Check if games is explicitly null or not an array (covers undefined from hook and potential API issues)
  const noGamesScheduled = !Array.isArray(games) || games.length === 0;

  return (
    <div className="p-4">
       {/* Consistent Data Header */}
      <h2 className="text-lg text-center font-semibold mb-3 text-gray-900 dark:text-text-primary">
        MLB Games for {displayDate}
      </h2>
      <div className="space-y-4">
        {noGamesScheduled ? (
          <p className="text-text-secondary text-center mt-4">
             {/* Updated message with displayDate */}
            No MLB games scheduled for {displayDate}.
          </p>
        ) : (
          // Render game cards if games array is valid and has items
          games.map((game: UnifiedGame) => <GameCard key={game.id} game={game} />)
        )}
      </div>
       {/* NOTE: Injury report section is not present in this MLB component, unlike the NBA one */}
    </div>
  );
};

export default MLBScheduleDisplay;