// src/screens/game_screen.tsx
import React from "react";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context"; // Keep if Date Selector UI is here

// Import the new display components (adjust paths if needed)
import NBAScheduleDisplay from "@/components/schedule/nba_schedule_display";
import MLBScheduleDisplay from "@/components/schedule/mlb_schedule_display";

// TODO: Import your Date Selector component if it belongs here
// import DateSelector from "@/components/ui/date_selector";

const GamesScreen: React.FC = () => {
  console.log(`%c[GamesScreen] Rendering Container...`, "color: brown");
  const { sport } = useSport();
  // const { date, setDate } = useDate(); // Keep/use if Date Selector is rendered here

  return (
    // Container div
    <div className="games-screen-container">

      {/* TODO: Place your Date Selector component here if desired */}
      {/* Example: <DateSelector selectedDate={date} onDateChange={setDate} /> */}
      {/* Add padding or margins as needed */}

      {/* Conditionally render the correct schedule display */}
      {/* Using a key ensures the component fully remounts when the sport changes */}
      {sport === 'NBA' && <NBAScheduleDisplay key="nba-display" />}
      {sport === 'MLB' && <MLBScheduleDisplay key="mlb-display" />}

    </div>
  );
};

export default GamesScreen;