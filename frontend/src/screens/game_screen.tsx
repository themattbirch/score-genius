// frontend/src/screens/game_screen.tsx

import React from "react";
import { FixedSizeList as List, ListChildComponentProps } from "react-window";
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

// Fallback height for each card (px) – match your Tailwind h‑class
const CARD_HEIGHT = 128;

// Virtualized row renderer
const Row = ({
  index,
  style,
  data,
}: ListChildComponentProps<UnifiedGame[]>) => {
  const game = data[index];
  return (
    <div style={style} className="px-4">
      <GameCard game={game} />
    </div>
  );
};

const GamesScreen: React.FC = () => {
  const { sport } = useSport();
  const { date, setDate } = useDate();

  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  // Choose the correct schedule hook
  const { data: games = [], isLoading } =
    sport === "NBA"
      ? useNBASchedule(formattedDate)
      : useMLBSchedule(formattedDate);

  // Compute list height (subtract header + bottom bar)
  const listHeight =
    typeof window !== "undefined" ? window.innerHeight - 160 : 600;

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

      {/* ─── Virtualized schedule or fallback ─── */}
      {isLoading ? (
        <p className="text-sm text-gray-400">Loading…</p>
      ) : games.length === 0 ? (
        sport === "NBA" ? (
          <NBAScheduleDisplay key="nba-fallback" />
        ) : (
          <MLBScheduleDisplay key="mlb-fallback" />
        )
      ) : (
        <List
          height={listHeight}
          itemCount={games.length}
          itemSize={CARD_HEIGHT}
          width="100%"
          itemData={games}
        >
          {Row}
        </List>
      )}
    </main>
  );
};

export default React.memo(GamesScreen);
