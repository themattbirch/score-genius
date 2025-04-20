// frontend/src/screens/game_screen.tsx

import React from 'react';
import SkeletonBox from '@/components/ui/skeleton_box';
import { useParams } from 'react-router-dom';
import { useInjuries } from '@/api/use_injuries';
import { useSport } from '@/contexts/sport_context';
import { useDate } from '@/contexts/date_context';

const GameDetailScreen: React.FC = () => {
  const { gameId } = useParams();
  const { sport }  = useSport();
  const { date }   = useDate();

  const { data: injuries } = useInjuries(
    sport,
    date.toISOString().slice(0, 10)
  );

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-lg font-semibold">Game {gameId}</h2>

      <div className="app-card">
        <h3 className="mb-2 font-semibold">Injury Report</h3>

        {injuries?.length ? (
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