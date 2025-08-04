// frontend/src/screens/game_screen.tsx
import React, { memo, useMemo } from "react";
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
import { useNFLSchedule } from "@/api/use_nfl_schedule";
import NBAScheduleDisplay from "@/components/schedule/nba_schedule_display";
import MLBScheduleDisplay from "@/components/schedule/mlb_schedule_display";
import NFLScheduleDisplay from "@/components/schedule/nfl_schedule_display";
import SkeletonBox from "@/components/ui/skeleton_box";
import { getLocalYYYYMMDD } from "@/utils/date";
import { isGameStale } from "@/game";

/* ------------------------------------------------------------ */
/* Hook selector                                                */
/* ------------------------------------------------------------ */
const useGamesForSport = (sport: "NBA" | "MLB" | "NFL", apiDate: string) => {
  const nfl = useNFLSchedule(apiDate, { enabled: sport === "NFL" });
  const mlb = useMLBSchedule(apiDate, { enabled: sport === "MLB" });
  const nba = useNBASchedule(apiDate, { enabled: sport === "NBA" });

  switch (sport) {
    case "NFL":
      return nfl;
    case "MLB":
      return mlb;
    case "NBA":
      return nba;
    default:
      return { data: [], isLoading: false, isError: true };
  }
};

/* ------------------------------------------------------------ */
/* Screen component                                             */
/* ------------------------------------------------------------ */
const GamesScreen: React.FC = () => {
  const { sport } = useSport();
  const { date, setDate } = useDate();

  const apiDate = getLocalYYYYMMDD(date);
  const {
    data: games = [],
    isLoading,
    isError,
  } = useGamesForSport(sport, apiDate);

  const visibleGames = useMemo(
    () => games.filter((g) => !isGameStale(g)),
    [games]
  );

  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  /* dev inspect */
  if (import.meta.env.DEV) {
    console.log("GAMES SCREEN STATE:", {
      sport,
      isLoading,
      isError,
      rawCount: games.length,
      visibleCount: visibleGames.length,
    });
  }

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
            <button className="pill border text-sm gap-1 bg-surface hover:bg-surface-hover border-border-subtle focus-ring">
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
          /* skeleton list while fetching */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {Array.from({ length: 6 }).map((_, i) => (
              <SkeletonBox key={i} className="app-card h-32 w-full" />
            ))}
          </div>
        ) : games.length === 0 ? (
          /* API returned nothing → fall back to legacy schedule displays */
          <>
            {sport === "NBA" && <NBAScheduleDisplay />}
            {sport === "MLB" && <MLBScheduleDisplay />}
            {sport === "NFL" && <NFLScheduleDisplay />}
          </>
        ) : visibleGames.length === 0 ? (
          /* All games have concluded / stale-filtered out */
          <p className="text-left text-text-secondary">
            All {sport} games for {formattedDate} have concluded.
          </p>
        ) : (
          /* responsive grid of active cards */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {visibleGames.map((g, idx) => (
              <GameCard key={g.id} game={g} isFirst={idx === 0} />
            ))}
          </div>
        )}
      </section>
    </main>
  );
};

export default memo(GamesScreen);
