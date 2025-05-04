// src/components/schedule/mlb_schedule_display.tsx

import React, { useState, useEffect, useMemo } from "react"; // Added useState, useEffect, useMemo
import { useDate } from "@/contexts/date_context";
import { useMLBSchedule } from "@/api/use_mlb_schedule";
import { UnifiedGame } from "@/types"; // Adjust path if needed
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";
import { startOfDay, isBefore } from "date-fns";

const formatLocalDate = (d: Date | null | undefined): string => {
  if (!d) return "";
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(
    2,
    "0"
  )}-${String(d.getDate()).padStart(2, "0")}`;
};

const MLBScheduleDisplay: React.FC = () => {
  const { date } = useDate();
  const isoDate = formatLocalDate(date);

  // --- State for Current Time ---
  const [currentTime, setCurrentTime] = useState(() => Date.now());

  // --- Date Comparison Logic ---
  const today = startOfDay(new Date());
  const selectedDay = date ? startOfDay(date) : null;
  const isPastDate = selectedDay ? isBefore(selectedDay, today) : false;
  // --- End Date Comparison ---

  // --- Effect to Update Current Time ---
  useEffect(() => {
    const intervalId = setInterval(() => {
      setCurrentTime(Date.now());
      // console.log("Tick: Updating current time for MLB filtering"); // For debugging
    }, 60000); // Update every minute

    return () => clearInterval(intervalId); // Cleanup
  }, []); // Run only once on mount

  // Get MLB schedule data
  const { data: games, isLoading, error } = useMLBSchedule(isoDate);

  // --- Filtered Games Logic (Identical to NBA) ---
  const filteredGames = useMemo(() => {
    // --- If viewing a past date, show ALL fetched historical games ---
    if (isPastDate) {
      return games || []; // Return original games array (contains historical data)
    }

    // --- Otherwise (viewing TODAY), apply time filtering ---
    if (!games) return [];

    const nowMillis = currentTime; // Use state for current time
    const bufferMillis = 3.5 * 60 * 60 * 1000; // 3.5 hours

    return games.filter((game: UnifiedGame) => {
      const startTimeString = game.gameTimeUTC;
      if (!startTimeString) {
        console.warn(
          "MLB Game missing gameTimeUTC for filtering (today):",
          game.id
        );
        return true; // Keep missing times? Decide policy
      }
      try {
        const gameStartMillis = new Date(startTimeString).getTime();
        if (isNaN(gameStartMillis)) {
          console.warn("Invalid MLB game start time format (today):", game.id);
          return true; // Keep invalid times? Decide policy
        }
        const estimatedEndMillis = gameStartMillis + bufferMillis;
        return nowMillis < estimatedEndMillis; // Filter based on time
      } catch (e) {
        console.error("Error parsing MLB game date (today):", game.id, e);
        return true; // Keep on error? Decide policy
      }
    });
  }, [games, currentTime, isPastDate]); // Add isPastDate dependency

  // --- Loading State ---
  if (isLoading) {
    // Loading display remains the same
    return (
      <div className="p-4">
        <h2 className="text-lg text-center font-semibold mb-3 text-gray-500 dark:text-text-secondary italic animate-pulse">
          Loading MLB Games for {isoDate}...
        </h2>
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
    // Error display remains the same
    console.error("MLBScheduleDisplay: Error loading games.", error);
    return (
      <div className="p-4">
        <h2 className="text-lg text-center font-semibold mb-3 text-red-600 dark:text-red-500">
          Error Loading MLB Games for {isoDate}
        </h2>
        <p className="p-4 text-center text-red-500">
          Could not load game data.
        </p>
      </div>
    );
  }

  // --- Updated Data Display Logic ---
  // Determine display state based on filtering
  const hasVisibleGames = filteredGames.length > 0;
  const noGamesInitiallyScheduled = !Array.isArray(games) || games.length === 0;
  const allGamesFilteredOut = !noGamesInitiallyScheduled && !hasVisibleGames;

  return (
    <div className="p-4">
      {/* Consistent Data Header */}
      <h2 className="text-lg text-center font-semibold mb-3 text-gray-900 dark:text-text-primary">
        MLB Games for {isoDate}
      </h2>
      <div className="space-y-4">
        {/* Use filteredGames for rendering */}
        {
          hasVisibleGames ? (
            <>
              {filteredGames.map((game: UnifiedGame) => (
                <GameCard key={game.id} game={game} />
              ))}
            </>
          ) : // Handle cases with no games or all games finished
          noGamesInitiallyScheduled ? (
            <p className="text-text-secondary text-center mt-4">
              No MLB games scheduled for {isoDate}.
            </p>
          ) : allGamesFilteredOut ? (
            <p className="text-text-secondary text-center mt-4">
              All MLB games for {isoDate} have concluded.
            </p>
          ) : null // Should not happen
        }
      </div>
      {/* No Injury report section in MLB component */}
    </div>
  );
};

export default MLBScheduleDisplay;
