// src/screens/game_screen.tsx
import React from "react";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";
import { useSchedule, Game } from "@/api/use_schedule";
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";

const GamesScreen: React.FC = () => {
  console.log(`%c[GamesScreen] Rendering...`, "color: brown");
  const { sport } = useSport();
  const { date } = useDate();
  const isoDate = date ? date.toISOString().slice(0, 10) : ""; // Keep this check

  console.log(`%c[GamesScreen] Preparing to call useSchedule`, "color: brown", {
    sport,
    isoDate,
  });
  // --- FIX: Call useSchedule with only the expected 2 arguments ---
  const {
    data: rawData,
    isLoading: loadingGames,
    error: gamesError,
  } = useSchedule(sport, isoDate); // REMOVED the third argument
  // --- END FIX ---

  console.log(`%c[GamesScreen] useSchedule results:`, "color: brown", {
    loadingGames,
    gamesError,
    hasData: rawData !== undefined,
  });
  // 1. Handle Loading State
  if (loadingGames) {
    return <SkeletonBox className="h-64 w-full p-4" />;
  }

  // 2. Handle Error State
  if (gamesError) {
    console.error("GamesScreen: Error loading games.", gamesError);
    return <p className="p-4 text-red-500">Error loading games.</p>;
  }

  // 3. Log the actual data structure AFTER loading/error checks
  console.log("GamesScreen: Received rawData:", rawData);

  // 4. Extract the games array - *** ADJUST THIS LINE based on console log ***
  const games = rawData; // Example: Adjust as needed (e.g., rawData?.schedule)

  // 5. *** CRITICAL FIX ***: Validate that 'games' is actually an array
  if (!Array.isArray(games)) {
    console.error(
      "GamesScreen: Data received for 'games' is not an array!",
      games
    );
    // Return if data is invalid
    return (
      <p className="p-4 text-orange-500">Received invalid game data format.</p>
    );
  }

  // 6. Log and Render the list (Now we know 'games' is an array)
  // *** LOG MOVED HERE ***
  console.log(
    `%c[GamesScreen] State: Valid Data. Rendering ${games?.length ?? 0} games.`,
    "color: brown"
  );

  return (
    <div className="space-y-4 p-4">
      {games.length === 0 ? ( // Safe to use .length now
        <p className="text-text-secondary">No games scheduled for this date.</p>
      ) : (
        games.map((game: Game) => <GameCard key={game.id} game={game} />) // Safe to use .map now
      )}
    </div>
  );
};

export default GamesScreen;
