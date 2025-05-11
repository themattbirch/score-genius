// frontend/src/components/schedule/nba_schedule_display.tsx

import React, { useState, useEffect, useMemo } from "react"; // Added useState, useEffect
import { useDate } from "@/contexts/date_context";
import { useNBASchedule } from "@/api/use_nba_schedule";
import { useInjuries, type Injury } from "@/api/use_injuries";
import type { UnifiedGame } from "@/types";
import GameCard from "@/components/games/game_card";
import SkeletonBox from "@/components/ui/skeleton_box";
import { ChevronDown } from "lucide-react";
import { startOfDay, isBefore } from "date-fns";
// Consider adding a date library for robust parsing if needed:
// import { parseISO } from 'date-fns';

const groupInjuriesByTeam = (inj: Injury[]) =>
  inj.reduce<Record<string, Injury[]>>((acc, i) => {
    const team = i.team_display_name;
    if (team) {
      (acc[team] ??= []).push(i);
    }
    return acc;
  }, {});

const formatLocalDate = (d: Date | null | undefined): string => {
  if (!d) return "";
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(
    2,
    "0"
  )}-${String(d.getDate()).padStart(2, "0")}`;
};

const NBAScheduleDisplay: React.FC = () => {
  const { date } = useDate();
  const isoDate = formatLocalDate(date);
  const displayDate =
    date?.toLocaleDateString("en-US", {
      month: "long",
      day: "numeric",
    }) ?? "";

  const today = startOfDay(new Date()); // Get the very beginning of today
  const selectedDay = date ? startOfDay(date) : null; // Get the beginning of the selected day
  // Check if the selected day is strictly before today
  const isPastDate = selectedDay ? isBefore(selectedDay, today) : false;

  // Alternative using native Date (less robust, use with caution):
  /*
  const today = new Date();
  today.setHours(0, 0, 0, 0); // Set to beginning of today
  const selectedDay = date ? new Date(date) : null;
  if (selectedDay) {
      selectedDay.setHours(0, 0, 0, 0); // Set to beginning of selected day
  }
  const isPastDate = selectedDay ? selectedDay.getTime() < today.getTime() : false;
  */
  // --- End Date Comparison ---

  // --- State for Current Time ---
  const [currentTime, setCurrentTime] = useState(() => Date.now());

  // --- Effect to Update Current Time ---
  useEffect(() => {
    // Update time every 60 seconds (1 minute)
    const intervalId = setInterval(() => {
      setCurrentTime(Date.now());
      console.log("Tick: Updating current time for filtering"); // For debugging
    }, 60000);

    // Cleanup function to clear the interval when the component unmounts
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array ensures this runs only once on mount

  console.log("[NBA schedule] isoDate used for API =", isoDate);
  console.log("[NBA schedule] displayDate used for UI =", displayDate);

  const {
    data: games,
    isLoading: loadingGames,
    error: gamesError,
  } = useNBASchedule(isoDate);

  const {
    data: injuries = [],
    isLoading: loadingInjuries,
    error: injuriesError,
  } = useInjuries("NBA", isoDate);

  // --- Filtered Games Logic ---
  const filteredGames = useMemo(() => {
    if (!games) return [];

    const nowMillis = currentTime;
    const bufferMillis = 3 * 60 * 60 * 1000; // 3 hours in milliseconds

    console.log(
      `Filtering ${games.length} games against time: ${new Date(
        nowMillis
      ).toLocaleString()}`
    ); // For debugging

    return games.filter((game: UnifiedGame) => {
      // --- Use the CORRECT field name ---
      const startTimeString = game.gameTimeUTC; // <--- Corrected field name

      // This check now correctly handles if gameTimeUTC is null or undefined
      if (!startTimeString) {
        console.warn("Game missing gameTimeUTC for filtering:", game.id);
        return true; // Keep games with missing times? Or return false to hide?
      }

      try {
        // Use Date constructor for ISO 8601 strings. Use a library (date-fns, dayjs) for more complex formats.
        const gameStartMillis = new Date(startTimeString).getTime();

        if (isNaN(gameStartMillis)) {
          console.warn(
            "Invalid game start time format (gameTimeUTC):",
            startTimeString,
            "Game ID:",
            game.id
          );
          return true; // Keep games with invalid times? Or return false?
        }

        const estimatedEndMillis = gameStartMillis + bufferMillis;

        // Keep the game if the current time is BEFORE the estimated end time
        const shouldShow = nowMillis < estimatedEndMillis;

        // Debugging log per game
        // console.log(`Game ${game.id} (${game.awayTeamAbbr} @ ${game.homeTeamAbbr}): Start ${new Date(gameStartMillis).toLocaleString()}, Est End ${new Date(estimatedEndMillis).toLocaleString()}, Show: ${shouldShow}`);

        return shouldShow;
      } catch (e) {
        console.error(
          "Error parsing game date (gameTimeUTC):",
          startTimeString,
          "Game ID:",
          game.id,
          e
        );
        return true; // Keep game if there's a parsing error? Or return false?
      }
    });
  }, [games, currentTime]); // Re-run filter when games data or currentTime changes

  // Injury logic depends on the original 'games' list to know which teams are playing *today*
  // It should NOT use filteredGames, otherwise injury reports might disappear prematurely.
  const { teamsWithInjuries, injuriesByTeam } = useMemo(() => {
    // Use original 'games' list here
    if (!games?.length || !injuries.length) {
      return { teamsWithInjuries: [], injuriesByTeam: {} };
    }
    const playing = new Set(
      games.flatMap((g) => [g.homeTeamName, g.awayTeamName])
    );
    const grouped = groupInjuriesByTeam(injuries);
    const teams = [...playing].filter((t) => grouped[t]?.length).sort();
    return { teamsWithInjuries: teams, injuriesByTeam: grouped };
  }, [games, injuries]); // Keep dependencies on original games and injuries

  if (gamesError) {
    // Error display remains the same...
    return (
      <div className="p-4 text-center">
        <h2 className="mb-2 text-lg font-semibold text-red-600 dark:text-red-500">
          Error Loading NBA Games for {displayDate}
        </h2>
        <p className="text-red-500">Could not load game data.</p>
      </div>
    );
  }

  // Determine if there are games to show *after* filtering
  const hasVisibleGames = !loadingGames && filteredGames.length > 0;
  // Determine if the initial load finished and resulted in zero games *before* filtering
  const noGamesInitiallyScheduled =
    !loadingGames && (!games || games.length === 0);
  // Determine if games were scheduled but all have been filtered out
  const allGamesFilteredOut =
    !loadingGames && games && games.length > 0 && filteredGames.length === 0;

  return (
    <div className="p-4">
      {/* Header remains mostly the same */}
      <h1
        className={`mb-3 text-center text-lg font-semibold ${
          loadingGames
            ? "animate-pulse italic text-gray-500 dark:text-text-primary"
            : "text-slate-800 dark:text-text-primary"
        }`}
      >
        {loadingGames
          ? `Loading NBA Games for ${displayDate}…`
          : `NBA Games for ${displayDate}`}
      </h1>

      <div className="space-y-4">
        {loadingGames ? (
          <>
            {Array.from({ length: 3 }).map((_, i) => (
              <SkeletonBox key={i} className="h-24 w-full" />
            ))}
          </>
        ) : hasVisibleGames ? (
          <>
            {filteredGames.map((game: UnifiedGame) => (
              <GameCard key={game.id} game={game} />
            ))}
          </>
        ) : noGamesInitiallyScheduled ? (
          <p className="mt-4 text-center text-text-secondary">
            No NBA games scheduled for {displayDate}.
          </p>
        ) : allGamesFilteredOut ? (
          <p className="mt-4 text-center text-text-secondary">
            All NBA games for {displayDate} have concluded.
          </p>
        ) : null}
      </div>

      {/* Injury Report */}
      {!loadingGames && games && games.length > 0 && (
        <div className="mt-8 border-t border-border pt-6">
          <h2 className="mb-3 text-center text-lg font-semibold text-slate-800 dark:text-text-primary">
            Daily Injury Report
          </h2>

          {isPastDate ? (
            <p className="text-center text-sm text-text-secondary">
              Games have been completed. No injury statuses to report.
            </p>
          ) : (
            <>
              {allGamesFilteredOut ? (
                <p className="text-center text-sm text-text-secondary">
                  No remaining games for today. No injury statuses to report.
                </p>
              ) : loadingInjuries ? (
                <p className="text-center text-sm italic text-text-secondary">
                  Loading injuries…
                </p>
              ) : injuriesError ? (
                <p className="text-center text-sm text-red-500">
                  Could not load injury report.
                </p>
              ) : teamsWithInjuries.length === 0 ? (
                <p className="text-center text-sm text-text-secondary">
                  No significant injuries reported for playing teams on{" "}
                  {displayDate}.
                </p>
              ) : (
                <div className="space-y-4">
                  {teamsWithInjuries.map((team) => (
                    <details
                      key={team}
                      className="app-card overflow-hidden group"
                    >
                      <summary className="flex cursor-pointer items-center justify-between gap-3 px-4 py-3 rounded-md text-slate-800 dark:text-text-primary hover:bg-gray-50 dark:hover:bg-gray-700/50  focus:outline-none focus:ring-2 focus:ring-green-400">
                        <span className="min-w-0 flex-1 font-medium">
                          {team}
                        </span>
                        <span className="flex-shrink-0 rounded-full border border-green-500 shadow-md px-2.5 py-1 text-xs font-medium light:text-green-800 dark:text-green-100">
                          {injuriesByTeam[team].length} available
                        </span>
                        <ChevronDown className="h-4 w-4 flex-shrink-0 transition-transform group-open:rotate-180" />
                      </summary>
                      {/* ← single wrapper div around hr + ul */}
                      <div className="mt-2 py-2">
                        {/* optional, you can omit if you like */}
                        <ul className="space-y-1">
                          {injuriesByTeam[team].map((inj) => (
                            <li
                              key={inj.id}
                              className="flex items-start justify-between px-4 pt-3 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700/50"
                            >
                              <div className="flex-1 pr-4">
                                <p className="font-medium text-slate-800 dark:text-text-primary">
                                  {inj.player}
                                </p>
                                {inj.injury_type && (
                                  <p className="mt-1 text-xs text-gray-500 dark:text-text-secondary">
                                    {inj.injury_type}
                                  </p>
                                )}
                              </div>
                              <span
                                className="
                                ml-auto mr-10
                                flex-shrink-0
                                rounded-full
                                border border-gray-300
                                bg-gray-100 px-2.5 py-1
                                text-xs font-medium text-slate-800
                                dark:border-border dark:bg-transparent dark:text-text-primary
                              "
                              >
                                {inj.status}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </details>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default NBAScheduleDisplay;
