import React from "react";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";
import { useSchedule, Game } from "@/api/use_schedule";
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";

const GamesScreen: React.FC = () => {
  const { sport } = useSport();
  const { date } = useDate();
  const isoDate = date.toISOString().slice(0, 10);

  const {
    data: games = [],
    isLoading: loadingGames,
    error: gamesError,
  } = useSchedule(sport, isoDate);

  if (loadingGames) return <SkeletonBox className="h-64 w-full" />;
  if (gamesError) return <p className="text-red-500">Error loading games.</p>;

  return (
    <div className="space-y-4 p-4">
      {games.length === 0 ? (
        <p className="text-text-secondary">No games scheduled.</p>
      ) : (
        games.map((game: Game) => <GameCard key={game.id} game={game} />)
      )}
    </div>
  );
};

export default GamesScreen;
