// frontend/src/screens/game_detail_screen.tsx
import React from "react";
import { useParams } from "react-router-dom";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";
import { useInjuries } from "@/api/use_injuries";
import { useSchedule, Game } from "@/api/use_schedule";
import SkeletonBox from "@/components/ui/skeleton_box";

const GameDetailScreen: React.FC = () => {
  const { gameId } = useParams<{ gameId: string }>();
  const { sport } = useSport();
  const { date } = useDate();
  const isoDate = date.toISOString().slice(0, 10);

  // 1) Fetch injuries
  const {
    data: injuries = [],
    isLoading: loadingInjuries,
    error: injuriesError,
  } = useInjuries(sport, isoDate);

  // 2) (Optional) Fetch full day’s schedule and pick this game
  const {
    data: games = [],
    isLoading: loadingGames,
    error: gamesError,
  } = useSchedule(sport, isoDate);
  const thisGame = games.find((g: Game) => g.id === gameId);

  if (loadingInjuries || loadingGames) {
    return <SkeletonBox className="h-64 w-full" />;
  }
  if (injuriesError || gamesError) {
    return <p className="text-red-500">Error loading game data.</p>;
  }

  return (
    <div className="p-4 space-y-6">
      {/* Game summary */}
      {thisGame ? (
        <div className="app-card p-4">
          <h2 className="text-xl font-bold">
            {thisGame.awayTeam} @ {thisGame.homeTeam}
          </h2>
          <p className="text-sm text-text-secondary mb-2">
            Tip‑off:{" "}
            {new Date(thisGame.tipoff).toLocaleTimeString([], {
              hour: "numeric",
              minute: "2-digit",
            })}
          </p>
          <p className="font-medium">
            Prediction: {thisGame.predictionAway} – {thisGame.predictionHome}
          </p>
          <p className="text-xs text-text-secondary">
            Spread {thisGame.spread}, Total {thisGame.total}
          </p>
        </div>
      ) : (
        <p className="text-text-secondary">Game details not found.</p>
      )}

      {/* Injury Report */}
      <div className="app-card p-4">
        <h3 className="mb-2 font-semibold">Injury Report</h3>
        {injuries.length ? (
          <ul className="space-y-1 text-sm">
            {injuries.map((inj) => (
              <li key={inj.id} className="flex justify-between">
                <span>
                  {inj.player} ({inj.team})
                </span>
                <span className="font-medium">{inj.status}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-text-secondary">No reported injuries.</p>
        )}
      </div>
    </div>
  );
};

export default GameDetailScreen;
