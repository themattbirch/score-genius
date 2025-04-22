// frontend/src/screens/game_detail_screen.tsx
import React from "react";
import { useParams } from "react-router-dom";
// Import Sport type if needed for casting or context hook
import { Sport } from "@/types"; // Assuming Sport type is here
// import { useSport } from "@/contexts/sport_context"; // Keep commented if not used directly
// import { useDate } from "@/contexts/date_context"; // Keep commented if not used directly
import { useInjuries, Injury } from "@/api/use_injuries"; // Keep if displaying injuries
// --- FIX 1: Corrected Import ---
// Import only the hook (named export). Remove 'Game'.
import { useNBASchedule } from "@/api/use_nba_schedule";
// Import the unified type definition
import { UnifiedGame } from "@/types";
// --- END FIX 1 ---
import SkeletonBox from "@/components/ui/skeleton_box";

const GameDetailScreen: React.FC = () => {
  const { gameId } = useParams<{ gameId: string }>();
  // const { sport } = useSport(); // Example: Get sport if needed elsewhere
  // const { date } = useDate();
  // const isoDate = date?.toISOString().slice(0, 10) ?? '';

  // TEMP VARS - Still recommend fetching details by gameId eventually
  const tempIsoDate = new Date().toISOString().slice(0, 10);
  const tempSport: Sport = "NBA"; // Use NBA for now, ensure type matches

  const {
    data: games = [],
    isLoading: loadingGames,
    error: gamesError,
  } = useNBASchedule(tempSport, tempIsoDate); // Correct arguments

  // Find the specific game AFTER the fetch completes
  const thisGame =
    !loadingGames && !gamesError
      ? // Use UnifiedGame type here
        games.find((g: UnifiedGame) => g.id === gameId)
      : undefined;

  // Fetch injuries
  const {
    data: injuries = [],
    isLoading: loadingInjuries,
    error: injuriesError,
    // --- FIX 2: Corrected Hook Call ---
    // Remove the third argument
  } = useInjuries(tempSport, tempIsoDate);
  // --- END FIX 2 ---

  // Combined Loading/Error States
  if (loadingGames || (loadingInjuries && !!thisGame)) {
    return <SkeletonBox className="h-screen w-full p-4" />;
  }
  if (gamesError) {
    console.error("GameDetailScreen: Error loading schedule", gamesError);
    return (
      <p className="p-4 text-red-500">Error loading game schedule data.</p>
    );
  }
  if (!thisGame) {
    return (
      <p className="p-4 text-orange-500">
        Game details not found for ID {gameId} on date {tempIsoDate}.
      </p>
    );
  }
  // Optional: Handle injury error separately
  // if (injuriesError) { console.error(...) }

  // --- Render details (assuming 'thisGame' is found) ---
  return (
    <div className="p-4 space-y-6">
      {/* --- Game Summary Card --- */}
      <div className="app-card p-4">
        <div className="flex items-start justify-between gap-4">
          {/* Left Side: Title and Time */}
          <div className="min-w-0 flex-1">
            <h2 className="text-xl font-semibold">
              {/* Use properties from UnifiedGame type */}
              {thisGame.awayTeamName} @ {thisGame.homeTeamName}
            </h2>
            <p className="text-sm text-text-secondary mt-1">
              {thisGame.game_date}
              {/* Check unified time properties */}
              {thisGame.tipoff ? (
                <>
                  {" / "}
                  {new Date(thisGame.tipoff).toLocaleTimeString([], {
                    hour: "numeric",
                    minute: "2-digit",
                  })}
                </>
              ) : thisGame.gameTimeUTC ? (
                <>
                  {" / "}
                  {new Date(thisGame.gameTimeUTC).toLocaleTimeString([], {
                    hour: "numeric",
                    minute: "2-digit",
                  })}
                </>
              ) : (
                "" /* No time */
              )}
            </p>
          </div>
          {/* Right Side: Score/Prediction and Odds */}
          <div className="flex-none text-right text-sm">
            {/* Conditional Score/Prediction using UnifiedGame properties */}
            {thisGame.home_final_score !== null &&
            thisGame.away_final_score !== null ? (
              <p className="font-semibold text-lg">
                {thisGame.away_final_score} – {thisGame.home_final_score}
                <span className="block text-xs font-normal text-text-secondary">
                  {" "}
                  (Final)
                </span>
              </p>
            ) : thisGame.predictionHome !== null &&
              thisGame.predictionAway !== null ? (
              <p className="font-medium">
                {thisGame.predictionAway} – {thisGame.predictionHome}
                <span className="block text-xs font-normal text-text-secondary">
                  {" "}
                  (Pred.)
                </span>
              </p>
            ) : (
              <p className="font-medium">---</p>
            )}

            {/* Odds using UnifiedGame properties */}
            {(thisGame.spread !== null || thisGame.total !== null) && ( // Assumes spread/total are unified names now
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                Spread {thisGame.spread ?? "N/A"}, Total{" "}
                {thisGame.total ?? "N/A"}
              </p>
            )}
            {/* TODO: Add logic to display historical odds / MLB odds */}
          </div>
        </div>
      </div>
      {/* --- End Game Summary Card --- */}

      {/* --- Injury Report Card --- */}
      <div className="app-card p-4">
        <h3 className="mb-2 font-semibold">Injury Report</h3>
        {injuries.length > 0 ? (
          <ul className="space-y-1 text-sm">
            {injuries.map((inj) => (
              // Ensure keys are unique and filtering works
              <li
                key={inj.id + "-" + inj.player}
                className="flex justify-between"
              >
                <span>
                  {inj.player} ({inj.team})
                </span>
                <span className="font-medium">{inj.status}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-text-secondary">
            No reported injuries for this date/sport.
          </p>
        )}
        {injuriesError && (
          <p className="text-red-500 text-xs mt-2">
            Could not load injury details.
          </p>
        )}
      </div>
      {/* --- End Injury Report Card --- */}
    </div>
  );
};

export default GameDetailScreen;
