// frontend/src/components/ScoreBoard.tsx
import React from "react";

interface ScoreBoardProps {
  homeTeam: string;
  awayTeam: string;
  homeScore: number;
  awayScore: number;
  status: string;
}

const ScoreBoard: React.FC<ScoreBoardProps> = ({
  homeTeam,
  awayTeam,
  homeScore,
  awayScore,
  status,
}) => {
  return (
    <div className="scoreboard">
      <div className="teams">
        <div className="team home">
          <h2>{homeTeam}</h2>
          <p>{homeScore}</p>
        </div>
        <div className="team away">
          <h2>{awayTeam}</h2>
          <p>{awayScore}</p>
        </div>
      </div>
      <div className="status">
        <p>{status}</p>
      </div>
    </div>
  );
};

export default ScoreBoard;
