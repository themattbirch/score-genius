// frontend/src/components/games/game_card.tsx
import React, { useState } from "react";
import { UnifiedGame, Sport } from "@/types";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";

interface GameCardProps {
  game: UnifiedGame;
}

const GameCard: React.FC<GameCardProps> = ({ game }) => {
  const { sport } = useSport();
  const { date } = useDate();
  const isoDate = date ? date.toISOString().slice(0, 10) : "";

  const gameId = game.id;
  const homeTeamName = game.homeTeamName;
  const awayTeamName = game.awayTeamName;
  const displayTime = game.gameTimeUTC;
  const displayStatus = game.statusState;

  const [showFeatures, setShowFeatures] = useState(false);

  return (
    <div className="app-card flex flex-col gap-4" data-tour="game-card">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1 max-w-md">
          <p className="font-semibold text-sm sm:text-base leading-tight">
            {awayTeamName}
          </p>
          <p className="font-semibold text-sm sm:text-base leading-tight">
            @ {homeTeamName}
          </p>
          <p className="text-xs text-text-secondary pt-1">
            {displayTime
              ? new Date(displayTime).toLocaleTimeString([], {
                  hour: "numeric",
                  minute: "2-digit",
                })
              : game.game_date}
            {displayStatus &&
              !["final", "sched", "pre"].some((s) =>
                displayStatus.toLowerCase().includes(s)
              ) &&
              ` (${displayStatus})`}
          </p>
        </div>

        <div className="w-36 md:w-auto text-right text-sm">
          {game.dataType === "historical" ? (
            <p className="font-semibold text-lg w-full">
              {game.away_final_score ?? "-"} – {game.home_final_score ?? "-"}
              <span className="block text-xs font-normal text-text-secondary">
                (Final)
              </span>
            </p>
          ) : game.dataType === "schedule" ? (
            sport === "NBA" ? (
              <p className="font-medium text-sky-500 dark:text-sky-400">
                {game.predictionAway?.toFixed(1) ?? "-"} –{" "}
                {game.predictionHome?.toFixed(1) ?? "-"}
                <span className="block text-xs font-normal text-text-secondary">
                  (Pred.)
                </span>
              </p>
            ) : (
              // ==========================================================
              // START: Corrected logic for MLB
              // ==========================================================
              <div>
                {/* Always show a "score" line. Show prediction if available, otherwise a placeholder. */}
                {game.predicted_home_runs != null &&
                game.predicted_away_runs != null ? (
                  <p className="font-medium text-sky-500 dark:text-sky-400">
                    {game.predicted_away_runs.toFixed(1)} –{" "}
                    {game.predicted_home_runs.toFixed(1)}
                    <span className="block text-xs font-normal text-text-secondary">
                      (Pred.)
                    </span>
                  </p>
                ) : (
                  <p className="font-medium text-text-secondary">-</p>
                )}

                {/* And always show the pitchers underneath for upcoming MLB games */}
                <div className="mt-1">
                  <p className="text-xs font-normal text-text-secondary">
                    {game.awayPitcher ?? "TBD"}{" "}
                    {game.awayPitcherHand && `(${game.awayPitcherHand})`}
                  </p>
                  <p className="text-xs font-normal text-text-secondary">
                    {game.homePitcher ?? "TBD"}{" "}
                    {game.homePitcherHand && `(${game.homePitcherHand})`}
                  </p>
                </div>
              </div>
              // ==========================================================
              // END: Corrected MLB logic
              // ==========================================================
            )
          ) : (
            <p className="font-medium w-full">—</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default GameCard;
