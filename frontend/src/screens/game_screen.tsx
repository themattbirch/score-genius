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

const GamesScreen: React.FC = () => {
  const { sport } = useSport();
  const { date, setDate } = useDate();

  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  return (
    <main className="pt-6 px-4 md:px-8 lg:px-1">
      {/* ─── Toolbar: Title + Date picker ─── */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold">
          {sport} Games for {formattedDate}
        </h1>

        <div className="flex items-center gap-3">
          <Popover>
            <PopoverTrigger asChild>
              <button
                className="inline-flex items-center gap-1 rounded-lg border px-3 py-1 text-sm
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
        <NBAScheduleDisplay key="nba-display" showHeader={false} />
      ) : (
        <MLBScheduleDisplay key="mlb-display" showHeader={false} />
      )}
    </main>
  );
};

export default GamesScreen;
