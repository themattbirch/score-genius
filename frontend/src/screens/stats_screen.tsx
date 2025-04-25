// frontend/src/screens/stats_screen.tsx
import React, { useMemo, useState, useEffect } from "react";
import { useSport } from "../contexts/sport_context";
import { useDate } from "../contexts/date_context";
import { useTeamStats } from "../api/use_team_stats";
import { usePlayerStats, UnifiedPlayerStats } from "../api/use_player_stats";
// Import BOTH NBA and MLB advanced stats hooks and types
import {
  useAdvancedStats as useNbaAdvancedStats,
  AdvancedTeamStats as NbaAdvancedTeamStats,
} from "../api/use_nba_advanced_stats";
import {
  useMlbAdvancedStats,
  MlbAdvancedTeamStats,
} from "../api/use_mlb_advanced_stats"; // <-- IMPORTED
import type { UnifiedTeamStats, Sport } from "../types";
import SkeletonBox from "@/components/ui/skeleton_box";
import { ChevronsUpDown } from "lucide-react";

// --- Type definitions ---
type SortDir = "asc" | "desc";
type TeamSortKey = keyof UnifiedTeamStats | string;
type PlayerSortKey = keyof UnifiedPlayerStats;
type NbaAdvancedSortKey = keyof NbaAdvancedTeamStats; // For potential NBA advanced sorting
type MlbAdvancedSortKey = keyof MlbAdvancedTeamStats; // For potential MLB advanced sorting

// Define keys needing percentage formatting
const pctKeys = new Set<string>([
  "three_pct",
  "ft_pct",
  "wins_all_percentage",
  "efg_pct", // NBA Advanced
  "tov_pct", // NBA Advanced
  "oreb_pct", // NBA Advanced
  "win_pct", // MLB Advanced
  "pythagorean_win_pct", // MLB Advanced
  "home_away_win_pct_split", // MLB Advanced
]);

// Define keys needing 1 decimal place (adjust as needed)
const oneDecimalKeys = new Set<string>([
  "points_for_avg_all", // NBA Teams
  "points_against_avg_all", // NBA Teams
  "runs_for_avg_all", // MLB Teams
  "runs_against_avg_all", // MLB Teams
  "pace", // NBA Advanced
  "off_rtg", // NBA Advanced
  "def_rtg", // NBA Advanced
  "run_differential_avg", // MLB Advanced
  "home_away_run_diff_avg_split", // MLB Advanced
]);

// Define keys needing 0 decimal places (integers)
const zeroDecimalKeys = new Set<string>([
  "games_played", // Both
  "points", // NBA Players
  "rebounds", // NBA Players
  "assists", // NBA Players
  "minutes", // NBA Players (can be float, but often displayed int)
  "run_differential", // MLB Advanced
  "expected_wins", // MLB Advanced
  "luck_factor", // MLB Advanced
  "wins", // MLB Advanced
  "runs_for", // MLB Advanced
  "runs_against", // MLB Advanced
]);

// --- Component Definition ---
const StatsScreen: React.FC = () => {
  const { sport } = useSport();
  const { date } = useDate();

  // --- Season Logic (unmodified) ---
  const defaultSeason = useMemo(() => {
    if (sport === "MLB") return date.getUTCFullYear();
    return date.getUTCMonth() >= 6
      ? date.getUTCFullYear()
      : date.getUTCFullYear() - 1;
  }, [sport, date]);

  const [season, setSeason] = useState<number>(defaultSeason);
  useEffect(() => setSeason(defaultSeason), [defaultSeason]);

  // --- Sub-tab state - NOW applies to both sports ---
  // Default to 'teams' for both
  const [subTab, setSubTab] = useState<"teams" | "players" | "advanced">(
    "teams"
  );

  // --- Player filters state (NBA only) ---
  const [playerSearch, setPlayerSearch] = useState("");

  // --- Season fetch check (unmodified) ---
  const canFetchSelectedSeason = useMemo(() => {
    if (sport === "MLB") return true;
    return season <= defaultSeason;
  }, [sport, season, defaultSeason]);

  // --- Effect to reset state when sport changes ---
  useEffect(() => {
    // Always reset to 'teams' tab when sport changes
    setSubTab("teams");
    // Clear NBA-specific state if switching away from NBA
    if (sport !== "NBA") {
      setPlayerSearch("");
      // Consider resetting NBA sort states if desired
      // setPlayerSort(null);
      // setNbaAdvancedSort(null);
    }
    // Consider resetting MLB sort state if switching away from MLB
    // if (sport !== 'MLB') {
    //   setMlbAdvancedSort(null);
    // }
  }, [sport]); // Re-run when sport changes

  // --- QUERIES with refined 'enabled' flags ---
  const {
    data: teamData,
    isLoading: teamLoading,
    error: teamError,
  } = useTeamStats({
    sport,
    season,
    // Enable if fetchable AND 'teams' tab is active (for EITHER sport)
    enabled: canFetchSelectedSeason && subTab === "teams",
  });

  const {
    data: playerData, // NBA player data only
    isLoading: playerLoading,
    error: playerError,
  } = usePlayerStats({
    sport: "NBA",
    season,
    search: playerSearch,
    // Enable ONLY if fetchable, NBA is selected AND players tab is active
    enabled: canFetchSelectedSeason && sport === "NBA" && subTab === "players",
  });

  // --- NBA Advanced Stats Query ---
  const {
    data: nbaAdvancedData,
    isLoading: nbaAdvancedLoading,
    error: nbaAdvancedError,
  } = useNbaAdvancedStats({
    // Renamed import alias
    sport: "NBA", // Explicitly NBA
    season,
    // Enable ONLY if fetchable, NBA is selected, AND advanced tab is active
    enabled: canFetchSelectedSeason && sport === "NBA" && subTab === "advanced",
  });

  // --- MLB Advanced Stats Query --- NEW ---
  const {
    data: mlbAdvancedData,
    isLoading: mlbAdvancedLoading,
    error: mlbAdvancedError,
  } = useMlbAdvancedStats({
    sport: "MLB", // Explicitly MLB
    season,
    // Enable ONLY if fetchable, MLB is selected, AND advanced tab is active
    enabled: canFetchSelectedSeason && sport === "MLB" && subTab === "advanced",
  });

  // --- Season dropdown options (unmodified) ---
  const seasonOptions = useMemo(
    () =>
      Array.from({ length: 5 }).map((_, i) => {
        const yr = defaultSeason - i;
        return {
          value: yr,
          label:
            sport === "NBA" ? `${yr}-${String(yr + 1).slice(-2)}` : String(yr),
        };
      }),
    [defaultSeason, sport]
  );

  // --- Sorting state ---
  const [teamSort, setTeamSort] = useState<{
    key: TeamSortKey;
    dir: SortDir;
  } | null>(
    {
      key: sport === "MLB" ? "wins_all_percentage" : "wins_all_percentage",
      dir: "desc",
    } // Default sort based on sport
  );
  const [playerSort, setPlayerSort] = useState<{
    key: PlayerSortKey;
    dir: SortDir;
  } | null>(null); // NBA only
  // TODO: Add sorting state for NBA Advanced and MLB Advanced if desired
  // const [nbaAdvancedSort, setNbaAdvancedSort] = useState<...>(null);
  // const [mlbAdvancedSort, setMlbAdvancedSort] = useState<...>(null);

  // Update default team sort when sport changes
  useEffect(() => {
    setTeamSort({
      key: sport === "MLB" ? "wins_all_percentage" : "wins_all_percentage",
      dir: "desc",
    });
  }, [sport]);

  // --- Toggle Sort Functions (keep existing logic) ---
  const toggleTeamSort = (k: TeamSortKey) =>
    setTeamSort(
      (prev) =>
        prev?.key === k
          ? { key: k, dir: prev.dir === "asc" ? "desc" : "asc" }
          : { key: k, dir: "asc" } // Default to ascending on first click
    );

  const togglePlayerSort = (k: PlayerSortKey) =>
    setPlayerSort((prev) =>
      prev?.key === k
        ? { key: k, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );
  // TODO: Add toggle functions for NBA/MLB Advanced if sorting is implemented

  // --- Dynamic Headers Definitions (keep existing logic) ---
  const teamHeaders: { label: string; key: TeamSortKey }[] = useMemo(() => {
    if (sport === "MLB") {
      return [
        { label: "Team", key: "team_name" },
        { label: "Win %", key: "wins_all_percentage" },
        { label: "Runs For", key: "runs_for_avg_all" },
        { label: "Runs Vs", key: "runs_against_avg_all" },
        { label: "Streak", key: "current_form" },
      ];
    } else {
      // NBA
      return [
        { label: "Team", key: "team_name" },
        { label: "Win %", key: "wins_all_percentage" },
        { label: "Off Pts", key: "points_for_avg_all" },
        { label: "Def Pts", key: "points_against_avg_all" },
        { label: "Streak", key: "current_form" },
      ];
    }
  }, [sport]);

  const playerHeaders: { label: string; key: PlayerSortKey }[] = [
    { label: "Player", key: "player_name" }, // NBA only
    { label: "Team", key: "team_name" },
    { label: "Pts", key: "points" },
    { label: "Reb", key: "rebounds" },
    { label: "Ast", key: "assists" },
    { label: "3P%", key: "three_pct" },
    { label: "FT%", key: "ft_pct" },
    { label: "Min", key: "minutes" },
    { label: "GP", key: "games_played" },
  ];

  // --- Sorted data logic (keep existing) ---
  const sortedTeams = useMemo(() => {
    if (!teamData) return [];
    const out = [...teamData];
    if (teamSort) {
      // ... sorting logic remains the same ...
      const { key, dir } = teamSort;
      out.sort((a, b) => {
        const av =
          a[key as keyof UnifiedTeamStats] ??
          (typeof a[key as keyof UnifiedTeamStats] === "string"
            ? ""
            : -Infinity); // Handle nulls better for numeric sort
        const bv =
          b[key as keyof UnifiedTeamStats] ??
          (typeof b[key as keyof UnifiedTeamStats] === "string"
            ? ""
            : -Infinity);
        const diff =
          typeof av === "number" && typeof bv === "number"
            ? av - bv
            : String(av).localeCompare(String(bv));
        return dir === "asc" ? diff : -diff;
      });
    }
    return out;
  }, [teamData, teamSort]);

  const sortedPlayers = useMemo(() => {
    // NBA only
    if (!playerData) return [];
    const out = [...playerData];
    if (playerSort) {
      const { key, dir } = playerSort;
      out.sort((a, b) => {
        const av = a[key as keyof UnifiedPlayerStats] ?? -Infinity; // Handle nulls
        const bv = b[key as keyof UnifiedPlayerStats] ?? -Infinity;
        const diff =
          typeof av === "number" && typeof bv === "number"
            ? av - bv
            : String(av).localeCompare(String(bv));
        return dir === "asc" ? diff : -diff;
      });
    }
    return out;
  }, [playerData, playerSort]);
  // TODO: Add sorted memoized data for NBA/MLB Advanced if sorting is implemented

  // --- Reusable header cell component (unmodified) ---
  const HeaderCell = ({
    /* ... props ... */ label,
    active,
    onClick,
    align = "right",
    className = "",
  }: {
    label: string;
    active: boolean;
    onClick: () => void;
    align?: "left" | "right";
    className?: string;
  }) => (
    <th
      onClick={onClick}
      title="Click to sort"
      className={` bg-gray-50 dark:bg-gray-800 cursor-pointer select-none py-2 px-3 font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700/30 ${
        align === "left" ? "text-left" : "text-center"
      } group ${className} `}
    >
      <span className="inline-flex items-center">
        {" "}
        {label}{" "}
        <ChevronsUpDown
          size={12}
          className="ml-1 opacity-30 dark:opacity-40 group-hover:opacity-60 dark:group-hover:opacity-70 transition-opacity"
        />{" "}
      </span>
    </th>
  );

  // --- Helper for cell formatting ---
  const formatCell = (value: any, key: string): React.ReactNode => {
    if (value == null || value === -Infinity) return "–"; // Handle null/undefined/-Infinity explicitly
    if (pctKeys.has(key)) {
      return typeof value === "number"
        ? (value * 100).toFixed(1) + "%"
        : String(value); // Adjust percentage formatting for MLB if needed
    } else if (oneDecimalKeys.has(key)) {
      return typeof value === "number" ? value.toFixed(1) : String(value);
    } else if (zeroDecimalKeys.has(key)) {
      return typeof value === "number" ? value.toFixed(0) : String(value);
    } else if (typeof value === "number") {
      return value.toFixed(1); // Default numeric formatting
    }
    return String(value); // Default string formatting
  };

  // --- Render Team Table function (use formatCell helper) ---
  const renderTeamTable = () => {
    if (teamLoading)
      return (
        <div className="p-4 space-y-3">
          {" "}
          {Array.from({ length: 8 }).map((_, i) => (
            <SkeletonBox key={i} className="h-10 w-full rounded-lg" />
          ))}{" "}
        </div>
      );
    if (teamError)
      return (
        <div className="p-4 text-center text-red-400">
          Failed to load team stats.
        </div>
      );
    if (!sortedTeams?.length)
      return (
        <div className="p-4 text-center text-gray-500">
          No team stats available for this season.
        </div>
      );

    return (
      <div className="overflow-x-auto rounded-xl border border-gray-300 dark:border-gray-700">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700">
            <tr>
              {teamHeaders.map(({ label, key }, index) => (
                <HeaderCell
                  key={String(key)}
                  label={label}
                  active={teamSort?.key === key}
                  onClick={() => toggleTeamSort(key)}
                  align={index === 0 ? "left" : "right"}
                  className={
                    index > 0
                      ? "border-l border-gray-300 dark:border-gray-700"
                      : ""
                  }
                />
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-300 dark:divide-gray-700 bg-white dark:bg-gray-900">
            {sortedTeams.map((team) => (
              <tr
                key={team.team_id}
                className="hover:bg-gray-50 dark:hover:bg-gray-800/60"
              >
                {teamHeaders.map(({ key }, index) => {
                  const raw = team[key as keyof UnifiedTeamStats];
                  const cell = formatCell(raw, String(key)); // Use helper
                  let cellSpecificClasses =
                    index === 0
                      ? `px-3 text-left font-medium whitespace-nowrap text-gray-900 dark:text-gray-100`
                      : `px-3 text-center border-l border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-400`;
                  return (
                    <td
                      key={String(key)}
                      className={`py-2 ${cellSpecificClasses}`}
                    >
                      {" "}
                      {cell}{" "}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // --- Render Player Table function (NBA only - use formatCell) ---
  const renderPlayerTable = () => {
    if (sport !== "NBA") return null; // Guard clause
    if (playerLoading)
      return (
        <div className="space-y-3">
          {" "}
          {Array.from({ length: 15 }).map((_, i) => (
            <SkeletonBox key={i} className="h-10 w-full rounded-lg" />
          ))}{" "}
        </div>
      );
    if (playerError)
      return (
        <div className="text-center text-red-400">
          Problem loading player stats.
        </div>
      );
    if (!sortedPlayers?.length)
      return (
        <div className="text-center opacity-60">
          No results {playerSearch ? `for “${playerSearch}”` : ""}
        </div>
      );

    return (
      <div className="overflow-x-auto rounded-xl border border-gray-300 dark:border-gray-700">
        <table className="w-full min-w-max text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700">
            <tr>
              {playerHeaders.map(({ label, key }, i) => (
                <HeaderCell
                  key={key}
                  label={label}
                  active={playerSort?.key === key}
                  onClick={() => togglePlayerSort(key)}
                  align={i < 2 ? "left" : "right"}
                  className={
                    i === 0
                      ? `sticky left-0 z-30 bg-gray-50 dark:bg-gray-800 before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-gray-700`
                      : "border-l border-gray-300 dark:border-gray-700"
                  }
                />
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-300 dark:divide-gray-700 bg-white dark:bg-gray-900">
            {sortedPlayers.map((p) => (
              <tr
                key={p.player_id}
                className="hover:bg-gray-50 dark:hover:bg-gray-800/60"
              >
                {playerHeaders.map(({ key }, i) => {
                  const raw = p[key as keyof UnifiedPlayerStats];
                  const display = formatCell(raw, String(key)); // Use helper
                  return i === 0 ? (
                    <td
                      key={key}
                      className={` sticky left-0 z-20 bg-white dark:bg-gray-900 px-3 text-cent font-medium whitespace-nowrap text-gray-900 dark:text-gray-100 before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-gray-700 `}
                    >
                      {" "}
                      {display}{" "}
                    </td>
                  ) : (
                    <td
                      key={key}
                      className="py-2 px-3 text-left whitespace-nowrap border-l border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-400"
                    >
                      {" "}
                      {display}{" "}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // --- Render NBA Advanced Stats Table function (use formatCell) ---
  const renderNbaAdvancedTable = () => {
    if (sport !== "NBA") return null;

    const advancedHeaders: {
      label: string;
      key: keyof NbaAdvancedTeamStats;
    }[] = [
      { label: "Team", key: "team_name" },
      { label: "Pace", key: "pace" },
      { label: "OffRtg", key: "off_rtg" },
      { label: "DefRtg", key: "def_rtg" },
      { label: "eFG%", key: "efg_pct" },
      { label: "TOV%", key: "tov_pct" },
      { label: "ORB%", key: "oreb_pct" },
      { label: "GP", key: "games_played" },
    ];

    if (nbaAdvancedLoading)
      return (
        <div className="p-4 space-y-3">
          {" "}
          {Array.from({ length: 15 }).map((_, i) => (
            <SkeletonBox key={i} className="h-10 w-full rounded-lg" />
          ))}{" "}
        </div>
      );
    if (nbaAdvancedError)
      return (
        <div className="p-4 text-center text-red-400">
          Problem loading advanced stats.
        </div>
      );
    if (!nbaAdvancedData?.length)
      return (
        <div className="p-4 text-center text-gray-500 dark:text-gray-400">
          No advanced stats available for this season.
        </div>
      );

    // TODO: Add sorting logic if needed, using nbaAdvancedData, nbaAdvancedSort, toggleNbaAdvancedSort

    return (
      <div className="overflow-x-auto rounded-xl border border-gray-300 dark:border-gray-700">
        <table className="w-full min-w-max text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700">
            <tr>
              {advancedHeaders.map(({ label, key }, i) => (
                <HeaderCell
                  key={key}
                  label={label}
                  active={false /* TODO: Add sort active state */}
                  onClick={() => {} /* TODO: Add toggle sort */}
                  align={i === 0 ? "left" : "right"}
                  className={
                    i === 0
                      ? `sticky left-0 z-30 bg-gray-50 dark:bg-gray-800 before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-gray-700`
                      : "border-l border-gray-300 dark:border-gray-700"
                  }
                />
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-300 dark:divide-gray-700 bg-white dark:bg-gray-900">
            {nbaAdvancedData.map(
              (
                team // Use unsorted data for now
              ) => (
                <tr
                  key={team.team_name}
                  className="hover:bg-gray-50 dark:hover:bg-gray-800/60"
                >
                  {advancedHeaders.map(({ key }, i) => {
                    const raw = team[key as keyof NbaAdvancedTeamStats];
                    const display = formatCell(raw, String(key)); // Use helper
                    return i === 0 ? (
                      <td
                        key={key}
                        className={` sticky left-0 z-20 bg-white dark:bg-gray-900 px-3 text-left font-medium whitespace-nowrap text-gray-900 dark:text-gray-100 before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-gray-700 `}
                      >
                        {" "}
                        {display}{" "}
                      </td>
                    ) : (
                      <td
                        key={key}
                        className="py-2 px-3 text-center whitespace-nowrap border-l border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-400"
                      >
                        {" "}
                        {display}{" "}
                      </td>
                    );
                  })}
                </tr>
              )
            )}
          </tbody>
        </table>
      </div>
    );
  };

  // --- Render MLB Advanced Stats Table function --- NEW ---
  const renderMlbAdvancedTable = () => {
    if (sport !== "MLB") return null; // Guard clause

    // Define Headers for the MLB Advanced Stats Table
    const mlbAdvancedHeaders: {
      label: string;
      key: keyof MlbAdvancedTeamStats;
    }[] = [
      { label: "Team", key: "team_name" },
      { label: "Win%", key: "win_pct" },
      { label: "Pyth W%", key: "pythagorean_win_pct" }, // Shortened label
      { label: "Run Diff", key: "run_differential" },
      { label: "Run Diff Avg", key: "run_differential_avg" },
      { label: "Luck", key: "luck_factor" },
      { label: "GP", key: "games_played" },
      // Add more if desired: expected_wins, home_away_win_pct_split etc.
    ];

    // Loading State
    if (mlbAdvancedLoading) {
      return (
        <div className="p-4 space-y-3">
          {" "}
          {Array.from({ length: 15 }).map((_, i) => (
            <SkeletonBox key={i} className="h-10 w-full rounded-lg" />
          ))}{" "}
        </div>
      );
    }
    // Error State
    if (mlbAdvancedError) {
      return (
        <div className="p-4 text-center text-red-400">
          Problem loading MLB advanced stats.
        </div>
      );
    }
    // Empty State
    if (!mlbAdvancedData?.length) {
      return (
        <div className="p-4 text-center text-gray-500 dark:text-gray-400">
          No MLB advanced stats available for this season.
        </div>
      );
    }

    // TODO: Add sorting logic if needed, using mlbAdvancedData, mlbAdvancedSort, toggleMlbAdvancedSort

    return (
      <div className="overflow-x-auto rounded-xl border border-gray-300 dark:border-gray-700">
        <table className="w-full min-w-max text-sm">
          {/* ─────── HEADERS ─────── */}
          <thead className="bg-gray-50 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700">
            <tr>
              {mlbAdvancedHeaders.map(({ label, key }, i) => (
                <HeaderCell
                  key={key}
                  label={label}
                  active={false /* TODO: Add sort active state */}
                  onClick={() => {} /* TODO: Add toggle sort */}
                  align={i === 0 ? "left" : "right"}
                  className={
                    i === 0
                      ? `sticky left-0 z-30 bg-gray-50 dark:bg-gray-800 before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-gray-700`
                      : "border-l border-gray-300 dark:border-gray-700"
                  }
                />
              ))}
            </tr>
          </thead>

          {/* ─────── BODY ─────── */}
          <tbody className="divide-y divide-gray-300 dark:divide-gray-700 bg-white dark:bg-gray-900">
            {mlbAdvancedData.map(
              (
                team // Use unsorted data for now
              ) => (
                <tr
                  key={team.team_id}
                  className="hover:bg-gray-50 dark:hover:bg-gray-800/60"
                >
                  {mlbAdvancedHeaders.map(({ key }, i) => {
                    const raw = team[key as keyof MlbAdvancedTeamStats];
                    const display = formatCell(raw, String(key)); // Use helper

                    return i === 0 ? (
                      /* ── STICKY TEAM NAME ── */
                      <td
                        key={key}
                        className={` sticky left-0 z-20 bg-white dark:bg-gray-900 px-3 text-left font-medium whitespace-nowrap text-gray-900 dark:text-gray-100 before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-gray-700 `}
                      >
                        {display}
                      </td>
                    ) : (
                      /* ── STAT COLUMNS ── */
                      <td
                        key={key}
                        className="py-2 px-3 text-center whitespace-nowrap border-l border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-400"
                      >
                        {display}
                      </td>
                    );
                  })}
                </tr>
              )
            )}
          </tbody>
        </table>
      </div>
    );
  };

  // --- FINAL RENDER ---
  return (
    <section className="p-4 space-y-4">
      {/* Row for Tabs/Title and Season Picker */}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
        {/* Sub-Tab Navigation Area */}
        <div className="flex-shrink-0">
          <div className="flex gap-1 rounded-lg bg-gray-200 dark:bg-gray-800 p-1 text-sm">
            {/* Define tabs based on sport */}
            {(sport === "NBA"
              ? (["teams", "players", "advanced"] as const)
              : (["teams", "advanced"] as const)
            ) // MLB tabs
              .map((tab) => (
                <button
                  key={tab}
                  className={`rounded-md px-3 py-1 transition-colors text-xs sm:text-sm ${
                    subTab === tab
                      ? "bg-green-600 text-white shadow-sm" // Active tab
                      : "text-gray-600 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-700" // Inactive tab
                  }`}
                  // Disable 'players' tab if sport is MLB (it won't be in the array, but belts & suspenders)
                  disabled={sport === "MLB" && tab === "players"}
                  onClick={() => setSubTab(tab)}
                >
                  {/* Capitalize tab name */}
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
          </div>
        </div>

        {/* Season picker - always shown */}
        <select
          value={season}
          onChange={(e) => setSeason(Number(e.target.value))}
          className="align-baseline rounded-lg bg-gray-200 text-gray-900 dark:bg-gray-800 dark:text-white py-1 text-sm outline-none focus:ring focus:ring-green-500/50"
        >
          {seasonOptions.map(({ value, label }) => (
            <option key={value} value={value}>
              {" "}
              {label}{" "}
            </option>
          ))}
        </select>
      </div>

      {/* Player Filters - Only show for NBA Players Tab */}
      {sport === "NBA" && subTab === "players" && (
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:gap-4">
          <input
            type="text"
            placeholder="Search player…"
            value={playerSearch}
            onChange={(e) => setPlayerSearch(e.target.value)}
            className="flex-grow rounded-lg border border-gray-400 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-1.5 text-sm text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500 outline-none focus:ring-1 focus:ring-green-600 md:max-w-xs"
          />
        </div>
      )}

      {/* Table Display Area - Conditional Rendering based on sport and subTab */}
      <div key={`${sport}-${subTab}`}>
        {" "}
        {/* Key forces remount on sport/tab change */}
        {sport === "NBA"
          ? // --- NBA Table Rendering ---
            subTab === "teams"
            ? renderTeamTable()
            : subTab === "players"
            ? renderPlayerTable()
            : subTab === "advanced"
            ? renderNbaAdvancedTable() // Use NBA specific render func
            : null // Should not happen
          : // --- MLB Table Rendering ---
          subTab === "teams"
          ? renderTeamTable()
          : subTab === "advanced"
          ? renderMlbAdvancedTable() // <-- Use NEW MLB render func
          : null // Should not happen
        }
      </div>
    </section>
  );
};

export default StatsScreen;
