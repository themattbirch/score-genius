// frontend/src/components/games/game_card.tsx
import React from "react";
import { UnifiedGame, Sport } from "@/types"; // Use UnifiedGame
import { useInjuries, Injury } from "@/api/use_injuries";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";

interface GameCardProps {
  game: UnifiedGame; // Expect the unified type
}

const GameCard: React.FC<GameCardProps> = ({ game }) => {
  const { sport } = useSport();
  const { date } = useDate();
  const isoDate = date ? date.toISOString().slice(0, 10) : "";

  // NOTE: Removed useInjuries hook call as 'teamInjuries' wasn't used in the return JSX.
  // If you need injury display later, you can uncomment this and the filter logic.
  // const { data: injuries = [] } = useInjuries(sport, isoDate);
  // const teamInjuries = injuries.filter(/* ... */);

  // Use unified fields directly
  const gameId = game.id;
  const homeTeamName = game.homeTeamName;
  const awayTeamName = game.awayTeamName;
  const displayTime = game.gameTimeUTC;
  const displayStatus = game.statusState;

  // Removed console.log statement

  return (
    <div className="app-card flex flex-col gap-2">
      {/* Top Row: Teams & Time */}
      <div className="flex items-start justify-between gap-4">
        {" "}
        {/* Changed items-center to items-start for top alignment */}
        {/* Left Side: Teams & Time */}
        <div className="min-w-0 flex-1">
          {/* --- MODIFIED SECTION START --- */}
          {/* Away Team */}
          <p className="font-semibold text-sm sm:text-base leading-tight">
            {" "}
            {/* Removed truncate */}
            {awayTeamName}
          </p>
          {/* Home Team - On a new line */}
          <p className="font-semibold text-sm sm:text-base leading-tight">
            @ {homeTeamName} {/* Kept the '@' prefix for context */}
          </p>
          {/* --- MODIFIED SECTION END --- */}

          {/* Time / Status - Added padding-top for spacing */}
          <p className="text-xs text-text-secondary pt-1">
            {/* Display time if available */}
            {
              displayTime
                ? new Date(displayTime).toLocaleTimeString([], {
                    hour: "numeric",
                    minute: "2-digit",
                  })
                : game.game_date // Fallback to date
            }
            {/* Display status if available and not obvious */}
            {displayStatus &&
              !displayStatus.toLowerCase().includes("final") &&
              !displayStatus.toLowerCase().includes("sched") &&
              ` (${displayStatus})`}
          </p>
        </div>
        {/* Right Side: Score/Prediction/Pitchers & Odds */}
        <div className="flex-none text-right text-sm">
          {/* === Main Display: Use dataType === */}
          {game.dataType === "historical" ? (
            // --- Historical Score ---
            <p className="font-semibold text-lg">
              {game.away_final_score ?? "-"} – {game.home_final_score ?? "-"}
              <span className="block text-xs font-normal text-text-secondary">
                {" "}
                (Final)
              </span>
            </p>
          ) : game.dataType === "schedule" ? (
            // --- Schedule Data ---
            <>
              {sport === "NBA" ? (
                // NBA Prediction
                <p className="font-medium">
                  {game.predictionAway ?? "-"} – {game.predictionHome ?? "-"}
                  <span className="block text-xs font-normal text-text-secondary">
                    {" "}
                    (Predicted Score)
                  </span>
                </p>
              ) : (
                // MLB Pitchers
                <>
                  <p className="text-xs font-normal text-[var(--color-text-secondary)] truncate max-w-[100px] sm:max-w-[150px]">
                    {game.awayPitcher ?? "TBD Pitcher"}
                    {game.awayPitcherHand && ` (${game.awayPitcherHand}HP)`}
                  </p>
                  <p className="text-xs font-normal text-[var(--color-text-secondary)] truncate max-w-[100px] sm:max-w-[150px]">
                    vs {game.homePitcher ?? "TBD Pitcher"}
                    {game.homePitcherHand && ` (${game.homePitcherHand}HP)`}
                  </p>
                </>
              )}
              {/* Schedule Odds Display */}
              {sport === "NBA"
                ? // Use standardized odds names if backend maps them
                  (game.spreadLine !== null || game.totalLine !== null) && (
                    <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                      Spread {game.spreadLine ?? "N/A"}, Total{" "}
                      {game.totalLine ?? "N/A"}
                    </p>
                  )
                : // MLB Odds
                  ((game.moneylineHome !== null &&
                    game.moneylineAway !== null) ||
                    game.totalLine !== null) && (
                    <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                      ML {game.moneylineAway ?? "N/A"} /{" "}
                      {game.moneylineHome ?? "N/A"}, O/U{" "}
                      {game.totalLine ?? "N/A"}
                    </p>
                  )}
            </>
          ) : (
            // Fallback
            <p className="font-medium">---</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default GameCard;
