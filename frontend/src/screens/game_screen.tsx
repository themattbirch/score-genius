// frontend/src/screens/game_screen.tsx
import React, { memo, useMemo } from "react";
import { useOnline } from "@/contexts/online_context";
import OfflineBanner from "@/components/offline_banner";
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
const useGamesForSport = (
  sport: "NBA" | "MLB" | "NFL",
  apiDate: string,
  options: { enabled: boolean }
) => {
  const nfl = useNFLSchedule(apiDate, {
    enabled: sport === "NFL" && options.enabled,
  });
  const mlb = useMLBSchedule(apiDate, {
    enabled: sport === "MLB" && options.enabled,
  });
  const nba = useNBASchedule(apiDate, {
    enabled: sport === "NBA" && options.enabled,
  });

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
  const online = useOnline();
  const { sport } = useSport();
  const { date, setDate } = useDate();
  const apiDate = getLocalYYYYMMDD(date);

  // Early return if offline
  if (!online) {
    window.location.href = "/app/offline.html"; // << always show offline shell    );
    return null;
  }

  // Only fetch when online
  const {
    data: games = [],
    isLoading,
    isError,
  } = useGamesForSport(sport, apiDate, { enabled: online });

  const visibleGames = useMemo(
    () => games.filter((g) => !isGameStale(g)),
    [games]
  );

  // --- pick out the first game that hasn't started and isn't final ---
  const firstUpcomingGameId = useMemo(() => {
    const GAME_STALE_MS = 3.5 * 60 * 60 * 1000;
    const now = Date.now();
    return visibleGames.find((g) => {
      const src = g.gameTimeUTC ?? g.game_date;
      if (!src) return false;
      const start = new Date(src).getTime();
      const inProgress = now >= start && now < start + GAME_STALE_MS;
      const status = (g.statusState ?? "").toLowerCase();
      const isFinal =
        ["final", "ended", "ft", "post-game", "postgame", "completed"].some(
          (s) => status.includes(s)
        ) ||
        (g.away_final_score != null && g.home_final_score != null);
      return !inProgress && !isFinal;
    })?.id;
  }, [visibleGames]);

  // Put upcoming games first, then in-progress, then finals
  const sortedGames = useMemo(() => {
    const GAME_STALE_MS = 3.5 * 60 * 60 * 1000;
    const now = Date.now();

    const rank = (g: (typeof games)[number]) => {
      const src = g.gameTimeUTC ?? g.game_date;
      if (!src) return 2; // push unknowns to bottom

      const start = new Date(src).getTime();
      const inProgress = now >= start && now < start + GAME_STALE_MS;

      const status = (g.statusState ?? "").toLowerCase();
      const isFinal =
        ["final", "ended", "ft", "post-game", "postgame", "completed"].some(
          (s) => status.includes(s)
        ) ||
        (g.away_final_score != null && g.home_final_score != null);

      if (!inProgress && !isFinal) return 0; // upcoming
      if (inProgress) return 1; // live
      return 2; // final
    };

    return [...visibleGames].sort((a, b) => rank(a) - rank(b));
  }, [visibleGames]);

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
            {sortedGames.map((g) => (
              <GameCard
                key={g.id}
                game={g}
                isFirst={g.id === firstUpcomingGameId}
              />
            ))}
          </div>
        )}
      </section>
    </main>
  );
};

export default memo(GamesScreen);
