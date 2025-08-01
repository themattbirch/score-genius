import React, { memo } from "react";
import { Calendar as CalendarIcon } from "lucide-react";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import GameCard from "@/components/games/game_card";
import { useMLBSchedule } from "@/api/use_mlb_schedule";
import { useNBASchedule } from "@/api/use_nba_schedule";
// 🏈 1. Import the new NFL hook and display component
import { useNFLSchedule } from "@/api/use_nfl_schedule";
import NBAScheduleDisplay from "@/components/schedule/nba_schedule_display";
import MLBScheduleDisplay from "@/components/schedule/mlb_schedule_display";
import NFLScheduleDisplay from "@/components/schedule/nfl_schedule_display";
import SkeletonBox from "@/components/ui/skeleton_box";
import { getLocalYYYYMMDD } from "@/utils/date";

const useGamesForSport = (sport: "NBA" | "MLB" | "NFL", apiDate: string) => {
  // Only enable the hook that matches the currently selected sport
  const nflQuery = useNFLSchedule(apiDate, { enabled: sport === "NFL" });
  const mlbQuery = useMLBSchedule(apiDate, { enabled: sport === "MLB" });
  const nbaQuery = useNBASchedule(apiDate, { enabled: sport === "NBA" });

  switch (sport) {
    case "NFL":
      return nflQuery;
    case "NBA":
      return nbaQuery;
    case "MLB":
      return mlbQuery;
    default:
      // Should not happen
      return { data: [], isLoading: false, isError: true };
  }
};

const GamesScreen: React.FC = () => {
  const { sport } = useSport();
  const { date, setDate } = useDate();

  const apiDate = getLocalYYYYMMDD(date);
  // 🏈 2. Use a dynamic hook selector for cleaner data fetching
  const {
    data: games = [],
    isLoading,
    isError,
  } = useGamesForSport(sport, apiDate);

  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  // 👇 ADD THIS CONSOLE LOG
  console.log("GAMES SCREEN STATE:", {
    sport,
    isLoading,
    isError,
    "games object": games,
    "games.length": games?.length,
  });

  /* -------------------------------------------------- render */
  return (
    <main className="flex flex-col flex-1 overflow-hidden">
      {/* Sticky filters-bar */}
      <div className="filters-bar contain-layout">
        <h1 className="text-base sm:text-lg font-semibold">
          {sport} Games for {formattedDate}
        </h1>
        <Popover>
          <PopoverTrigger asChild>
            <button
              data-tour="date-picker"
              className="pill border text-sm gap-1 bg-surface hover:bg-surface-hover border-border-subtle focus-ring"
            >
              <CalendarIcon size={16} strokeWidth={1.8} />
              {formattedDate}
            </button>
          </PopoverTrigger>
          <PopoverContent
            side="bottom"
            align="end"
            sideOffset={8}
            className="p-4 w-[20rem] bg-[var(--color-panel)] rounded-lg shadow-lg contain-layout"
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
      <section className="flex-1 overflow-y-auto px-6 py-6 space-y-6 contain-layout">
        {sport === "NFL" && (
          <p className="text-sm text-text-secondary">
            Note: No score predictions for preseason games.
          </p>
        )}

        {isLoading ? (
          /* skeleton list */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Array.from({ length: 6 }).map((_, i) => (
              <SkeletonBox key={i} className="app-card h-32 w-full" />
            ))}
          </div>
        ) : games.length === 0 ? (
          // 🏈 3. Add the NFL fallback component
          <>
            {sport === "NBA" && <NBAScheduleDisplay key="nba-fallback" />}
            {sport === "MLB" && <MLBScheduleDisplay key="mlb-fallback" />}
            {sport === "NFL" && <NFLScheduleDisplay key="nfl-fallback" />}
          </>
        ) : (
          /* responsive grid of cards */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {games.map((g) => (
              <GameCard key={g.id} game={g} />
            ))}
          </div>
        )}
      </section>
    </main>
  );
};

export default memo(GamesScreen);
