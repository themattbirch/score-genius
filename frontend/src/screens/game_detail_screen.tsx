// frontend/src/screens/game_detail_screen.tsx

import React from "react";
import { useParams } from "react-router-dom";

import { useNBASchedule } from "@/api/use_nba_schedule";
import { useInjuries } from "@/api/use_injuries";
import type { UnifiedGame } from "@/types";
import SnapshotCard from "@/components/ui/snapshot_card";

import SkeletonBox from "@/components/ui/skeleton_box";

const GameDetailScreen: React.FC = () => {
  const { gameId = "" } = useParams<{ gameId?: string }>();
  console.log("🏷️ GameDetailScreen mounted with gameId:", gameId);
  const isoDate = new Date().toISOString().slice(0, 10);

  /* ── Fetch schedule row ─────────────────────────────────────── */
  const {
    data: games = [],
    isLoading: loadingGames,
    error: gamesError,
  } = useNBASchedule(isoDate);

  const thisGame: UnifiedGame | undefined = games.find((g) => g.id === gameId);

  /* ── Fetch injuries ─────────────────────────────────────────── */
  const {
    data: injuries = [],
    isLoading: loadingInjuries,
    error: injuriesError,
  } = useInjuries("NBA", isoDate);

  /* ── Early-return states ────────────────────────────────────── */
  if (loadingGames || (loadingInjuries && thisGame)) {
    return <SkeletonBox className="h-screen w-full p-4" />;
  }
  if (gamesError) {
    return (
      <p className="p-4 text-red-500">Error loading game schedule data.</p>
    );
  }
  if (!thisGame) {
    return (
      <p className="p-4 text-orange-500">
        Game {gameId} not found for {isoDate}.
      </p>
    );
  }

  /* ── Render ─────────────────────────────────────────────────── */
  return (
    <div className="p-4 space-y-6">
      if (true) return{" "}
      <p className="p-4 bg-yellow-100 text-black">
        [DEBUG] detail screen for gameId: {gameId}
      </p>
      ;{/* ── Game summary ── */}
      <div className="app-card p-4">
        <h3 className="mb-2 font-semibold">Game Snapshot</h3>
        <SnapshotCard gameId={gameId} />
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <h2 className="text-xl font-semibold">
              {thisGame.awayTeamName} @ {thisGame.homeTeamName}
            </h2>
            <p className="mt-1 text-sm text-text-secondary">
              {thisGame.game_date}
              {thisGame.gameTimeUTC && (
                <>
                  {" / "}
                  {new Date(thisGame.gameTimeUTC).toLocaleTimeString([], {
                    hour: "numeric",
                    minute: "2-digit",
                  })}
                </>
              )}
            </p>
          </div>

          <div className="flex-none text-right text-sm">
            {thisGame.home_final_score != null &&
            thisGame.away_final_score != null ? (
              <p className="text-lg font-semibold">
                {thisGame.away_final_score} – {thisGame.home_final_score}
                <span className="block text-xs font-normal text-text-secondary">
                  (Final)
                </span>
              </p>
            ) : thisGame.predictionHome != null &&
              thisGame.predictionAway != null ? (
              <p className="font-medium">
                {thisGame.predictionAway} – {thisGame.predictionHome}
                <span className="block text-xs font-normal text-text-secondary">
                  (Pred.)
                </span>
              </p>
            ) : (
              <p className="font-medium">—</p>
            )}

            {(thisGame.spreadLine != null || thisGame.totalLine != null) && (
              <p className="mt-1 text-xs text-text-secondary">
                Spread {thisGame.spreadLine ?? "N/A"}, Total{" "}
                {thisGame.totalLine ?? "N/A"}
              </p>
            )}
          </div>
        </div>
      </div>
      {/* ── Injury report ── */}
      <div className="app-card p-4">
        <h3 className="mb-2 font-semibold">Injury Report</h3>
        {injuries.length ? (
          <ul className="space-y-1 text-sm">
            {injuries.map((inj) => (
              <li key={inj.id} className="flex justify-between">
                <span>
                  {inj.player}
                  {inj.injury_type && ` (${inj.injury_type})`} —{" "}
                  <em>{inj.team_display_name}</em>
                </span>
                <span className="font-medium">{inj.status}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-text-secondary">No reported injuries.</p>
        )}
        {injuriesError && (
          <p className="mt-2 text-xs text-red-500">
            Could not load injury details.
          </p>
        )}
      </div>
    </div>
  );
};

export default GameDetailScreen;
