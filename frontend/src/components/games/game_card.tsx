// frontend/src/components/games/game_card.tsx

import React, { useState } from "react";
import { UnifiedGame, Sport } from "@/types";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";

// Import our new components
import SnapshotButton from './snapshot_button'; // Corrected path and snake_case
import WeatherBadge from './weather_badge';     // Corrected path and snake_case
import SnapshotModal from './snapshot_modal';   // Corrected path and snake_case

interface GameCardProps {
  game: UnifiedGame;
}

const GameCard: React.FC<GameCardProps> = ({ game }) => {
  const { sport } = useSport(); // This sport context can be used to determine MLB vs NBA for WeatherBadge
  const { date } = useDate();
  const isoDate = date ? date.toISOString().slice(0, 10) : "";

  const gameId = game.id;
  const homeTeamName = game.homeTeamName;
  const awayTeamName = game.awayTeamName;
  const displayTime = game.gameTimeUTC;
  const displayStatus = game.statusState;

  // State to control the visibility of the Snapshot Modal for this specific game card
  const [isSnapshotModalOpen, setIsSnapshotModalOpen] = useState(false);

  // Determine if it's an MLB game for WeatherBadge
  const isMLB = game.sport === 'MLB'; // Use game.sport prop, not context sport for specific game type

  // Handlers for the Snapshot Modal
  const handleOpenSnapshot = () => {
    setIsSnapshotModalOpen(true);
  };

  const handleCloseSnapshot = () => {
    setIsSnapshotModalOpen(false);
  };

  return (
    <div className="app-card flex flex-col gap-4" data-tour="game-card">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1 max-w-md">
          {/* NEW: Container for the Snapshot Button on the left */}
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
                    <div className="mb-1">
            <SnapshotButton onClick={handleOpenSnapshot} />
          </div>
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
            sport === "NBA" ? ( // Use the `sport` from context here for general display logic.
                                // For specific game data like predicted runs, use game.predicted_...
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

                {/* WeatherBadge for MLB games (FR-GC-2) */}
                {isMLB && (
                  <div className="mt-1">
                    <WeatherBadge />
                  </div>
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
      

      {/* The Snapshot Modal for this specific game card */}
      <SnapshotModal
        gameId={gameId}
        sport={game.sport as Sport} // Pass the specific game's sport (NBA or MLB)
        isOpen={isSnapshotModalOpen}
        onClose={handleCloseSnapshot}
      />
    </div>
  );
};

export default GameCard;