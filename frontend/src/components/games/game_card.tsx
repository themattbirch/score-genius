// frontend/src/components/games/game_card.tsx
import React, { useState } from "react";
import { UnifiedGame, Sport } from "@/types"; // Use UnifiedGame
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";
//import SnapshotCard from "../ui/snapshot_card";
//import { NBAGameFeaturesPanel } from "../ui/nba_game_features_panel";

interface GameCardProps {
  game: UnifiedGame; // Expect the unified type
}

const GameCard: React.FC<GameCardProps> = ({ game }) => {
  const { sport } = useSport();
  const { date } = useDate();
  const isoDate = date ? date.toISOString().slice(0, 10) : "";

  // Use unified fields directly
  const gameId = game.id;
  const homeTeamName = game.homeTeamName;
  const awayTeamName = game.awayTeamName;
  const displayTime = game.gameTimeUTC;
  const displayStatus = game.statusState;

  const [showFeatures, setShowFeatures] = useState(false);

  return (
    <div className="app-card flex flex-col gap-4" data-tour="game-card">
      {/* Top Row: Teams & Time */}
      <div className="flex items-start justify-between gap-4">
        {/* Teams & Time */}
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

        {/* Score / Prediction / Pitchers */}
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
              <p className="font-medium w-full">
                {game.predictionAway ?? "-"} – {game.predictionHome ?? "-"}
                <span className="block text-xs font-normal text-text-secondary">
                  (Predicted)
                </span>
              </p>
            ) : (
              <>
                <p className="text-xs font-normal text-text-secondary">
                  {game.awayPitcher ?? "TBD"}{" "}
                  {game.awayPitcherHand && `(${game.awayPitcherHand}HP)`}
                </p>
                <p className="text-xs font-normal text-text-secondary">
                  {game.homePitcher ?? "TBD"}{" "}
                  {game.homePitcherHand && `(${game.homePitcherHand}HP)`}
                </p>
              </>
            )
          ) : (
            <p className="font-medium w-full">—</p>
          )}
        </div>
      </div>

      {/* Snapshot + Drill-In 
       <div className="border-t pt-4">
       <SnapshotCard gameId={gameId} />

        <button
          className="mt-2 text-sm underline hover:text-green-600"
          onClick={() => setShowFeatures((v) => !v)}
        >
          {showFeatures ? "Hide full stats" : "View full stats"}
        </button>

        {showFeatures && (
          <div className="mt-4">
            <NBAGameFeaturesPanel gameId={gameId} />
          </div>
                  )}
      </div>
              */}
    </div>
  );
};

export default GameCard;
