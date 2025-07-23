// frontend/src/screens/game_screen.tsx

import React, { useLayoutEffect, useRef, useState, memo } from "react";
import { FixedSizeList as List, ListChildComponentProps } from "react-window";
// @ts-ignore – no types published
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
import GameCard from "@/components/games/game_card";
import { useMLBSchedule } from "@/api/use_mlb_schedule";
import { useNBASchedule } from "@/api/use_nba_schedule";
import NBAScheduleDisplay from "@/components/schedule/nba_schedule_display";
import MLBScheduleDisplay from "@/components/schedule/mlb_schedule_display";
import type { UnifiedGame } from "@/types";

/* -------------------------------------------------------------------------- */
/*                               List helpers                                 */
/* -------------------------------------------------------------------------- */

const Row = ({
  index,
  style,
  data,
}: ListChildComponentProps<UnifiedGame[]>) => (
  <div style={style} className="box-border">
    <GameCard game={data[index]} />
  </div>
);

/* -------------------------------------------------------------------------- */
/*                                 Screen                                      */
/* -------------------------------------------------------------------------- */

const GamesScreen: React.FC = () => {
  const { sport } = useSport();
  const { date, setDate } = useDate();

  /* -------------------------- fetch the schedule -------------------------- */
  const apiDate = date.toISOString().split("T")[0];
  const { data: games = [], isLoading } =
    sport === "NBA" ? useNBASchedule(apiDate) : useMLBSchedule(apiDate);

  /* ---------------------- measure card height once ------------------------ */
  const [rowHeight, setRowHeight] = useState<number | null>(null);
  const probeRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (rowHeight == null && probeRef.current) {
      const h = probeRef.current.getBoundingClientRect().height;
      if (h) setRowHeight(Math.ceil(h));
    }
  }, [rowHeight, games]);

  /* -------------------------- render helpers ----------------------------- */
  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  /* ---------------------------------------------------------------------- */
  /*                                JSX                                     */
  /* ---------------------------------------------------------------------- */
  return (
    <main className="flex flex-col flex-1 h-full overflow-hidden pt-6 px-6 md:px-8 lg:px-12">
      {/* Toolbar */}
      <div className="flex-none mb-6 flex items-center justify-between">
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

      {/* Content */}
      <div className="flex-1 min-h-0">
        {isLoading ? (
          <p className="text-sm text-gray-400">Loading…</p>
        ) : games.length === 0 ? (
          sport === "NBA" ? (
            <NBAScheduleDisplay key="nba-fallback" />
          ) : (
            <MLBScheduleDisplay key="mlb-fallback" />
          )
        ) : rowHeight == null ? (
          // Render one invisible card to measure actual height
          <div ref={probeRef} className="opacity-0 pointer-events-none px-4">
            <GameCard game={games[0]} />
          </div>
        ) : (
          <AutoSizer>
            {({ height, width }: { height: number; width: number }) => (
              <List
                height={height}
                width={width}
                itemCount={games.length}
                itemSize={rowHeight}
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

export default memo(GamesScreen);
