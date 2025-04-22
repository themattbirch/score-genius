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

  const { data: injuries = [] } = useInjuries(sport, isoDate);

  // Use unified fields directly
  const gameId = game.id;
  const homeTeamName = game.homeTeamName;
  const awayTeamName = game.awayTeamName;
  // Use gameTimeUTC as the primary time source from unified type
  const displayTime = game.gameTimeUTC;
  const displayStatus = game.statusState;

  console.log(
    `📋 GameCard rendering for ${sport} ${game.dataType} game: ${gameId} ${isoDate}`
  );

  // Filter injuries using unified team names
  const teamInjuries = injuries.filter((inj: Injury) => {
    const injuryTeam = inj.team;
    // Ensure null/undefined checks if necessary
    return (
      injuryTeam && (injuryTeam === homeTeamName || injuryTeam === awayTeamName)
    );
  });

  return (
    <div className="app-card flex flex-col gap-2">
      {/* Top Row: Teams & Time */}
      <div className="flex items-center justify-between gap-4">
        {/* Left Side: Teams & Time */}
        <div className="min-w-0 flex-1">
          <p className="truncate font-semibold text-sm sm:text-base">
            {awayTeamName} @ {homeTeamName}
          </p>
          <p className="text-xs text-text-secondary">
            {/* Display time if available */}
            {displayTime
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
                    (Pred.)
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
          {/* TODO: Add logic to display historical closing odds if fetched */}
        </div>
      </div>

      {/* Bottom Row: Injury Chips */}
      {teamInjuries.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {teamInjuries.slice(0, 2).map((inj) => (
            <span
              key={`${game.id}-${inj.player}-${inj.status}`}
              className="pill bg-brand-orange text-xs"
              title={`${inj.player}: ${inj.detail}`}
            >
              {inj.player?.split(" ").pop()} {inj.status}
            </span>
          ))}
          {teamInjuries.length > 2 && (
            <span className="pill bg-brand-orange/60 text-xs">
              +{teamInjuries.length - 2} more
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default GameCard;
