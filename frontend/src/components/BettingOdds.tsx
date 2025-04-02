// frontend/src/components/BettingOdds.tsx
import React from "react";

interface BettingOddsProps {
  odds: {
    team: string;
    oddsValue: number;
  }[];
}

const BettingOdds: React.FC<BettingOddsProps> = ({ odds }) => {
  return (
    <div className="betting-odds">
      <h3>Betting Odds</h3>
      <ul>
        {odds.map((bet, index) => (
          <li key={index}>
            {bet.team}: {bet.oddsValue}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default BettingOdds;
