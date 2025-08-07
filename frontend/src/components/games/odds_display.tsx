// frontend/src/components/games/odds_display.tsx

import React from "react";

interface OddsDisplayProps {
  sport: "NBA" | "MLB" | "NFL";
  moneylineHome?: string | number | null;
  moneylineAway?: string | number | null;
  spreadLine?: number | null;
  totalLine?: number | null;
}

const formatOdd = (odd?: string | number | null): string => {
  if (odd === null || odd === undefined || odd === 0) return "N/A";
  const num = Number(odd);
  if (num > 0) return `+${num}`;
  return String(num);
};

const OddsDisplay: React.FC<OddsDisplayProps> = ({
  sport,
  moneylineHome,
  moneylineAway,
  spreadLine,
  totalLine,
}) => {
  return (
    <div className="mt-2 flex items-center flex-nowrap gap-x-1 text-xs text-text-secondary font-medium tracking-tight">
      <span className="whitespace-nowrap">
        ML: {formatOdd(moneylineHome)} / {formatOdd(moneylineAway)}
      </span>
      {sport !== "MLB" && (
        <>
          <span className="mx-1 select-none">|</span>
          <span className="whitespace-nowrap">{formatOdd(spreadLine)}</span>
        </>
      )}
      <span className="mx-1 select-none">|</span>
      <span className="whitespace-nowrap">
        O/U: {totalLine != null ? String(totalLine) : "N/A"}
      </span>
    </div>
  );
};

export default OddsDisplay;
