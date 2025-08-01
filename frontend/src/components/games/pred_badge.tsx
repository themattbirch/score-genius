// components/games/pred_badge.tsx
import React from "react";

interface PredBadgeProps {
  away: number;
  home: number;
}

const PredBadge: React.FC<PredBadgeProps> = ({ away, home }) => {
  const label = `${away.toFixed(1)} to ${home.toFixed(1)}, predicted score`;
  return (
    <span className="pred-badge flex flex-col items-center" aria-label={label}>
      <span className="score">
        {away.toFixed(1)} – {home.toFixed(1)}
      </span>
      <span className="label text-[10px] opacity-80 -mt-0.5">
        (predicted score)
      </span>
    </span>
  );
};

export default PredBadge;
