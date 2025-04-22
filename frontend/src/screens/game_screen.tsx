// frontend/src/screens/game_screen.tsx
import React from "react";
import { useSport } from "@/contexts/sport_context";
import NBAScheduleDisplay from "@/components/schedule/nba_schedule_display";
import MLBScheduleDisplay from "@/components/schedule/mlb_schedule_display";

const GamesScreen: React.FC = () => {
  const { sport } = useSport();

  return (
    <div className="games-screen-container">
      {sport === "NBA" && <NBAScheduleDisplay key="nba-display" />}
      {sport === "MLB" && <MLBScheduleDisplay key="mlb-display" />}
    </div>
  );
};

export default GamesScreen;
