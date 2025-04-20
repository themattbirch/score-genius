// frontend/src/components/games/game_card.tsx
import React from 'react';
import { Game } from '@/api/use_schedule';
import { useInjuries, Injury } from '@/api/use_injuries';
import { useSport } from '@/contexts/sport_context';
import { useDate } from '@/contexts/date_context';

const GameCard: React.FC<{ game: Game }> = ({ game }) => {
  const { sport } = useSport();
  const { date  } = useDate();
  const isoDate   = date.toISOString().slice(0, 10);

  // Fetch injuries here
  const { data: injuries = [] } = useInjuries(sport, isoDate);

  console.log("📋 GameCard mount for", game.id, isoDate);

  // Now Injuries is typed, so filter callbacks infer correctly
  const teamInjuries = injuries.filter(
    (inj: Injury) =>
      inj.team === game.homeTeam || inj.team === game.awayTeam
  );

  return (
    <div className="app-card flex flex-col gap-2">
      {/* teams & tip‑off */}
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <p className="truncate font-semibold">
            {game.awayTeam} @ {game.homeTeam}
          </p>
          <p className="text-xs text-text-secondary">
            {new Date(game.tipoff).toLocaleTimeString([], {
              hour: 'numeric',
              minute: '2-digit',
            })}
          </p>
        </div>

        <div className="text-right text-sm">
          <p className="font-medium">
            {game.predictionAway} – {game.predictionHome}
          </p>
          <p className="text-xs text-text-secondary">
            Spread {game.spread}, Total {game.total}
          </p>
        </div>
      </div>

      {/* injury chips */}
      {teamInjuries.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {teamInjuries.slice(0, 2).map((inj) => (
            <span
              key={inj.id}
              className="pill bg-brand-orange text-xs"
              title={inj.detail}
            >
              {inj.player.split(' ')[1]} {inj.status}
            </span>
          ))}
          {teamInjuries.length > 2 && (
            <span className="pill bg-brand-orange/60 text-xs">
              +{teamInjuries.length - 2} more
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default GameCard;
