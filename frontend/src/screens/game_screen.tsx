// frontend/src/screens/game_screen.tsx
import React from "react";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";
import { Calendar as CalendarIcon } from "lucide-react";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";

import NBAScheduleDisplay from "@/components/schedule/nba_schedule_display";
import MLBScheduleDisplay from "@/components/schedule/mlb_schedule_display";
import type { UnifiedGame } from "@/types";
// --- FIX: Ensure the 'sport' property is explicitly added to each game object ---
const mockGames: UnifiedGame[] = [ // Explicitly type the array to ensure compliance
  {
    id: "717904", // Example MLB game ID
    game_date: "2025-06-13", // Use current/relevant date for testing
    scheduled_time: "8:10 PM ET",
    homeTeamName: "Houston Astros",
    awayTeamName: "Chicago White Sox",
    predicted_home_runs: 4.8,
    predicted_away_runs: 4.3,
    // --- ADD THIS LINE ---
    sport: "MLB", // <--- CRITICAL FIX: Add the sport property
    // --- Ensure all other required properties for UnifiedGame are present ---
    dataType: "schedule",
    homePitcher: "Shane Smith",
    awayPitcher: "Lance McCullers Jr.",
    homePitcherHand: "R",
    awayPitcherHand: "R",
    gameTimeUTC: "2025-06-13T00:10:00Z", // Example UTC time
    statusState: "pre",
    // Add any other properties from UnifiedGame that are not nullable and need values
  },
  {
    id: "401766125", // This is the NBA game ID from your logs
    game_date: "2025-06-13",
    scheduled_time: "8:30 PM ET",
    homeTeamName: "Indiana Pacers", // Note: Your log showed Indiana Pacers as home, OKC as away. Adjust to match actual data
    awayTeamName: "Oklahoma City Thunder",
    predictionHome: 113.5, // NBA uses predictionHome/Away
    predictionAway: 118.1,
    // --- ADD THIS LINE ---
    sport: "NBA", // <--- CRITICAL FIX: Add the sport property
    // --- Ensure all other required properties for UnifiedGame are present ---
    dataType: "schedule",
    gameTimeUTC: "2025-06-13T00:30:00Z", // Example UTC time
    statusState: "pre",
    // Add any other properties from UnifiedGame that are not nullable and need values
  },
  // Add more games as needed for testing
];


const GamesScreen: React.FC = () => {
  const { sport } = useSport();
  const { date, setDate } = useDate();

  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  return (
    <main className="pt-6 px-6 md:px-8 lg:px-12">
      {/* ─── Toolbar: Title + Date picker ─── */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="mb-3 text-left text-lg font-semibold text-slate-800 dark:text-text-primary">
          {sport} Games for {formattedDate}
        </h1>

        <div className="flex items-center gap-3">
          <Popover>
            <PopoverTrigger asChild>
              <button
                data-tour="date-picker"
                className="inline-flex items-center gap-2 rounded-lg border px-2 md:px-4 py-2 text-sm
                     border-slate-300 bg-white text-slate-700 hover:bg-gray-50
                     dark:border-slate-600/60 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
              >
                <CalendarIcon size={16} strokeWidth={1.8} />
                {formattedDate}
              </button>
            </PopoverTrigger>
            <PopoverContent
              side="bottom"
              align="end"
              sideOffset={8}
              className="bg-[var(--color-panel)] rounded-lg shadow-lg p-4 w-[20rem]"
            >
              <Calendar
                selected={date}
                onSelect={(d) => d && setDate(d)}
                className="calendar-reset [--rdp-cell-size:2.5rem]"
              />
            </PopoverContent>
          </Popover>
        </div>
      </div>

      {/* ─── The schedule itself ─── */}
      {sport === "NBA" ? (
        <NBAScheduleDisplay key="nba-display" />
      ) : (
        <MLBScheduleDisplay key="mlb-display" />
      )}
    </main>
  );
};

export default GamesScreen;
