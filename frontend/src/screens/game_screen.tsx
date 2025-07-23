// frontend/src/screens/game_screen.tsx

import React from "react";
import { FixedSizeList as List, ListChildComponentProps } from "react-window";
// @ts-ignore react-virtualized-auto-sizer has no type declarations
import AutoSizer from "react-virtualized-auto-sizer";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";
import { Calendar as CalendarIcon } from "lucide-react";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import GameCard from "../components/games/game_card";
import { useMLBSchedule } from "@/api/use_mlb_schedule";
import { useNBASchedule } from "@/api/use_nba_schedule";
import NBAScheduleDisplay from "@/components/schedule/nba_schedule_display";
import MLBScheduleDisplay from "@/components/schedule/mlb_schedule_display";
import type { UnifiedGame } from "@/types";

// Height per card (match your Tailwind h‑class)
const CARD_HEIGHT = 128;

// Virtualized row renderer
const Row = ({
  index,
  style,
  data,
}: ListChildComponentProps<UnifiedGame[]>) => (
  <div style={style} className="px-4">
    <GameCard game={data[index]} />
  </div>
);

const GamesScreen: React.FC = () => {
  const { sport } = useSport();
  const { date, setDate } = useDate();

  // Display date as "Jul 23"
  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  // API needs "YYYY-MM-DD"
  const apiDate = date.toISOString().split("T")[0];

  // Fetch schedule for the right sport & date
  const { data: games = [], isLoading } =
    sport === "NBA" ? useNBASchedule(apiDate) : useMLBSchedule(apiDate);

  return (
    <main className="flex flex-col flex-1 pt-6 px-6 md:px-8 lg:px-12">
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-lg font-semibold dark:text-text-primary">
          {sport} Games for {formattedDate}
        </h1>
        <Popover>
          <PopoverTrigger asChild>
            <button
              data-tour="date-picker"
              className="inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm
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
            className="p-4 w-[20rem] bg-[var(--color-panel)] rounded-lg shadow-lg"
          >
            <Calendar
              selected={date}
              onSelect={(d) => d && setDate(d)}
              className="calendar-reset [--rdp-cell-size:2.5rem]"
            />
          </PopoverContent>
        </Popover>
      </div>

      {/* Content area */}
      <div className="flex-1">
        {isLoading ? (
          <p className="text-sm text-gray-400">Loading…</p>
        ) : games.length === 0 ? (
          sport === "NBA" ? (
            <NBAScheduleDisplay key="nba-fallback" />
          ) : (
            <MLBScheduleDisplay key="mlb-fallback" />
          )
        ) : (
          <AutoSizer>
            {({ height, width }: { height: number; width: number }) => (
              <List
                height={height}
                width={width}
                itemCount={games.length}
                itemSize={CARD_HEIGHT}
                itemData={games}
              >
                {Row}
              </List>
            )}
          </AutoSizer>
        )}
      </div>
    </main>
  );
};

export default React.memo(GamesScreen);
