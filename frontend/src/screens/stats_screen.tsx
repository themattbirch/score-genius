// frontend/src/screens/stats_screen.tsx
import React, { useMemo, useState, useEffect } from "react";
import { startOfDay, isBefore } from "date-fns";
import { useSport } from "../contexts/sport_context";
import { useDate } from "../contexts/date_context";
import { useNetworkStatus } from "@/hooks/use_network_status";
import { useTeamStats } from "../api/use_team_stats";
import { usePlayerStats } from "../api/use_player_stats";
import { useAdvancedStats as useNbaAdvancedStats } from "../api/use_nba_advanced_stats";
import { useMlbAdvancedStats } from "../api/use_mlb_advanced_stats";
import type {
  UnifiedPlayerStats,
  UnifiedTeamStats,
  NbaAdvancedTeamStats,
  MlbAdvancedTeamStats,
  Sport,
} from "@/types";
import SkeletonBox from "@/components/ui/skeleton_box";
import { ChevronsUpDown } from "lucide-react";

import { Calendar as CalendarIcon } from "lucide-react";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";

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
  const { date, setDate } = useDate();
  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  const online = useNetworkStatus();

  // --- Season Logic (updated for Oct 22 [2025-26 NBA season] rollover) ---
  const defaultSeason = useMemo(() => {
    if (sport === "MLB") return date.getUTCFullYear();

    const month = date.getUTCMonth() + 1; // JS months are 0–11 → +1 for 1–12
    const day = date.getUTCDate();

    // Only roll into the new NBA season on or after Oct 22 UTC
    return month > 10 || (month === 10 && day >= 22)
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

  // --- Calculate Dynamic Heading Text ---
  const headingText = useMemo(() => {
    switch (subTab) {
      case "teams":
        return `${sport} Team Rankings`; // e.g., "NBA Team Rankings" or "MLB Team Rankings"
      case "players":
        return "NBA Player Statistics"; // Only shown for NBA anyway
      case "advanced":
        return `${sport} Advanced Statistics`; // e.g., "NBA Advanced Statistics" or "MLB Advanced Statistics"
      default:
        return `${sport} Statistics`; // Fallback (shouldn't normally be reached)
    }
  }, [sport, subTab]); // Recalculate when sport or subTab changes

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
    enabled: online && canFetchSelectedSeason && subTab === "teams",
  });

  const {
    data: playerData,
    isLoading: playerLoading,
    error: playerError,
  } = usePlayerStats({
    sport: "NBA",
    season,
    search: playerSearch,
    // Enable ONLY if fetchable, NBA is selected AND players tab is active
    enabled:
      online &&
      canFetchSelectedSeason &&
      sport === "NBA" &&
      subTab === "players",
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
    enabled:
      online &&
      canFetchSelectedSeason &&
      sport === "NBA" &&
      subTab === "advanced",
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
    enabled:
      online &&
      canFetchSelectedSeason &&
      sport === "MLB" &&
      subTab === "advanced",
  });
  /* ---------- 🚧 EARLY-RETURN WHEN OFFLINE ------------------------------- */
  const offlineBanner = !online && (
    <section className="p-4 space-y-4">
      <h1 className="text-xl text-slate-800 dark:text-text-primary font-semibold">
        Offline mode
      </h1>
      <p className="text-gray-500 dark:text-text-secondary">
        You appear to be offline. Reconnect to load the latest stats.
      </p>
    </section>
  );

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
  } | null>(null);

  const [nbaAdvancedSort, setNbaAdvancedSort] = useState<{
    key: NbaAdvancedSortKey;
    dir: SortDir;
  } | null>({ key: "off_rtg", dir: "desc" });

  const [mlbAdvancedSort, setMlbAdvancedSort] = useState<{
    key: MlbAdvancedSortKey;
    dir: SortDir;
  } | null>({ key: "run_differential", dir: "desc" });

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

  const toggleNbaAdvancedSort = (k: NbaAdvancedSortKey) =>
    setNbaAdvancedSort((p) =>
      p?.key === k
        ? { key: k, dir: p.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  const toggleMlbAdvancedSort = (k: MlbAdvancedSortKey) =>
    setMlbAdvancedSort((p) =>
      p?.key === k
        ? { key: k, dir: p.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  // --- Dynamic Headers Definitions (keep existing logic) ---
  /* ---------------- TEAM HEADERS ---------------- */
  const teamHeaders = useMemo(
    () =>
      sport === "MLB"
        ? [
            { label: "Team", key: "team_name" },
            { label: "Win %", key: "wins_all_percentage" },
            { label: "Runs For", key: "runs_for_avg_all" },
            { label: "Runs Vs", key: "runs_against_avg_all" },
            { label: "Streak", key: "current_form" },
          ]
        : [
            { label: "Team", key: "team_name" },
            { label: "Win %", key: "wins_all_percentage" },
            { label: "Off Pts", key: "points_for_avg_all" },
            { label: "Def Pts", key: "points_against_avg_all" },
            { label: "Streak", key: "current_form" },
          ],
    [sport]
  );

  /* ---------------- PLAYER HEADERS (NBA) ---------------- */
  const playerHeaders = [
    { label: "Player", key: "player_name" },
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

  // --- Sorted NBA Advanced ---
  const sortedNbaAdvanced = useMemo(() => {
    if (!nbaAdvancedData) return [];
    const out = [...nbaAdvancedData];
    if (nbaAdvancedSort) {
      const { key, dir } = nbaAdvancedSort;
      out.sort((a, b) => {
        const av = a[key] ?? -Infinity;
        const bv = b[key] ?? -Infinity;
        const diff =
          typeof av === "number" && typeof bv === "number"
            ? av - bv
            : String(av).localeCompare(String(bv));
        return dir === "asc" ? diff : -diff;
      });
    }
    return out;
  }, [nbaAdvancedData, nbaAdvancedSort]);

  // --- Sorted MLB Advanced ---
  const sortedMlbAdvanced = useMemo(() => {
    if (!mlbAdvancedData) return [];
    const out = [...mlbAdvancedData];
    if (mlbAdvancedSort) {
      const { key, dir } = mlbAdvancedSort;
      out.sort((a, b) => {
        const av = a[key] ?? -Infinity;
        const bv = b[key] ?? -Infinity;
        const diff =
          typeof av === "number" && typeof bv === "number"
            ? av - bv
            : String(av).localeCompare(String(bv));
        return dir === "asc" ? diff : -diff;
      });
    }
    return out;
  }, [mlbAdvancedData, mlbAdvancedSort]);

  // --- Reusable header cell component (unmodified) ---
  const HeaderCell = ({
    label,
    active,
    onClick,
    align = "right",
    className = "",
    ...rest
  }: {
    label: string;
    active: boolean;
    onClick: () => void;
    align?: "left" | "right";
    className?: string;
    [key: string]: any;
  }) => (
    <th
      {...rest}
      onClick={onClick}
      title="Click to sort"
      className={` bg-gray-50 dark:bg-[var(--color-panel)] cursor-pointer select-none py-2 px-3 font-medium text-slate-600 dark:text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700/30 ${
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
        <div className="p-4 text-center text-text-secondary">
          No team stats available for this season.
        </div>
      );

    return (
      <div className="overflow-x-auto rounded-xl border border-gray-300 dark:border-slate-600/60 bg-white dark:bg-[var(--color-panel)]">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 dark:bg-[var(--color-panel)] border-b border-gray-300 dark:border-slate-600/60">
            {/* Ensure no extra spaces/newlines directly within <tr> */}
            <tr>
              {teamHeaders.map(({ label, key }, index) => {
                const isTargetColumn = key === "wins_all_percentage";
                const tourAttribute = isTargetColumn
                  ? "stats-column-winpct"
                  : undefined;
                // Assuming HeaderCell renders a <th> element
                return (
                  <HeaderCell
                    key={String(key)}
                    label={label}
                    active={teamSort?.key === key}
                    onClick={() => toggleTeamSort(key as TeamSortKey)}
                    align={index === 0 ? "left" : "right"}
                    // Reminder: className value "border-l ..." might still be incomplete
                    className={
                      index > 0
                        ? "border-l border-gray-300 dark:border-slate-600/60"
                        : ""
                    }
                    data-tour={tourAttribute}
                  />
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-300 dark:divide-slate-600/60 bg-white dark:bg-[var(--color-panel)]">
            {sortedTeams.map((team) => (
              <tr
                key={team.team_id}
                className="hover:bg-gray-50 dark:hover:bg-gray-800/60"
              >
                {teamHeaders.map(({ key }, index) => {
                  const raw = team[key as keyof UnifiedTeamStats];
                  const cell = formatCell(raw, String(key));
                  let cellSpecificClasses =
                    index === 0
                      ? `px-3 text-left font-medium whitespace-nowrap text-slate-800 dark:text-text-primary`
                      : `px-3 text-center border-l border-gray-300 dark:border-slate-600/60 text-slate-800 dark:text-text-primary`;
                  // Ensure no stray whitespace inside the <td> either
                  return (
                    <td
                      key={String(key)}
                      className={`py-2 ${cellSpecificClasses}`}
                    >
                      {/* Remove space here if present */}
                      {cell}
                      {/* Remove space here if present */}
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
      <div className="overflow-x-auto rounded-xl border border-gray-300 dark:border-slate-600/60 bg-white dark:bg-[var(--color-panel)]">
        <table className="w-full min-w-max text-sm">
          <thead className="bg-gray-50 dark:bg-[var(--color-panel)] border-b border-gray-300 dark:border-slate-600/60">
            <tr>
              {playerHeaders.map(({ label, key }, i) => (
                <HeaderCell
                  key={key}
                  label={label}
                  active={playerSort?.key === key}
                  onClick={() => togglePlayerSort(key as PlayerSortKey)}
                  align={i < 2 ? "left" : "right"}
                  className={
                    i === 0
                      ? `sticky left-0 z-30 bg-gray-50 dark:bg-[var(--color-panel)] before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-slate-600/60`
                      : "border-l border-gray-300 dark:border-slate-600/60"
                  }
                />
              ))}
            </tr>
          </thead>
          <tbody
            className="divide-y divide-gray-300 dark:divide-slate-600/60
            bg-white dark:bg-[var(--color-panel)]"
          >
            {sortedPlayers.map((p, rowIdx) => (
              <tr
                /* ① use compound key so duplicates never collide            */
                /* ② rowIdx is stable because list length doesn’t change     */
                key={`${p.player_id}-${rowIdx}`}
                className="hover:bg-gray-50 dark:hover:bg-gray-800/60"
              >
                {playerHeaders.map(({ key }, i) => {
                  const raw = p[key as keyof UnifiedPlayerStats];
                  const display = formatCell(raw, String(key));

                  // first column stays sticky + left-aligned
                  if (i === 0) {
                    return (
                      <td
                        key={key}
                        className="
                sticky left-0 z-20
                bg-white dark:bg-[var(--color-panel)]
                px-3
                text-left font-medium
                whitespace-nowrap
                text-slate-800 dark:text-text-primary
                before:content-[''] before:absolute before:inset-y-0
                before:right-0 before:w-px
                before:bg-gray-300 dark:before:bg-slate-600/60
              "
                      >
                        {display}
                      </td>
                    );
                  }

                  // team column (i===1) left-align, all others center
                  const alignClass = i === 1 ? "text-left" : "text-center";

                  return (
                    <td
                      key={key}
                      className={`
              py-2 px-3
              ${alignClass}
              whitespace-nowrap
              border-l border-gray-300 dark:border-slate-600/60
              text-gray-700 dark:text-text-primary
            `}
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

  const renderNbaAdvancedTable = () => {
    if (sport !== "NBA") return null;

    const nbaAdvancedHeaders = [
      { label: "Team", key: "team_name" },
      { label: "Pace", key: "pace" },
      { label: "OffRtg", key: "off_rtg" },
      { label: "DefRtg", key: "def_rtg" },
      { label: "eFG%", key: "efg_pct" },
      { label: "TOV%", key: "tov_pct" },
      { label: "ORB%", key: "oreb_pct" },
      { label: "GP", key: "games_played" },
    ] as const;

    if (nbaAdvancedLoading) {
      return (
        <div className="p-4 space-y-3">
          {Array.from({ length: 8 }).map((_, i) => (
            <SkeletonBox key={i} className="h-10 w-full rounded-lg" />
          ))}
        </div>
      );
    }
    if (nbaAdvancedError) {
      return (
        <div className="p-4 text-center text-red-400">
          Problem loading advanced stats.
        </div>
      );
    }
    if (!nbaAdvancedData?.length) {
      return (
        <div className="p-4 text-center text-gray-500 dark:text-gray-400">
          No advanced stats available for this season.
        </div>
      );
    }

    return (
      <div className="overflow-x-auto rounded-xl border border-gray-300 dark:border-slate-600/60 bg-white dark:bg-[var(--color-panel)]">
        <table className="w-full min-w-max text-sm">
          <thead className="bg-gray-50 dark:bg-[var(--color-panel)] border-b border-gray-300 dark:border-slate-600/60">
            <tr>
              {nbaAdvancedHeaders.map(({ label, key }, idx) => (
                <HeaderCell
                  key={key}
                  label={label}
                  active={false}
                  onClick={() =>
                    toggleNbaAdvancedSort(key as NbaAdvancedSortKey)
                  }
                  align={idx === 0 ? "left" : "right"}
                  className={
                    idx === 0
                      ? "sticky left-0 z-30 bg-gray-50 dark:bg-[var(--color-panel)] before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-slate-600/60"
                      : "border-l border-gray-300 dark:border-slate-600/60"
                  }
                />
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-300 dark:divide-slate-600/60 bg-white dark:bg-[var(--color-panel)]">
            {sortedNbaAdvanced.map((team, rowIdx) => (
              <tr
                key={`nba-adv-${team.team_id ?? team.team_name}-${rowIdx}`}
                className="hover:bg-gray-50 dark:hover:bg-gray-800/60"
              >
                {nbaAdvancedHeaders.map(({ key }, idx) => {
                  const raw = team[key as keyof NbaAdvancedTeamStats];
                  const display = formatCell(raw, key);
                  const cellKey = `nba-adv-${team.team_id}-${key}`;
                  return idx === 0 ? (
                    <td
                      key={cellKey}
                      className="sticky left-0 z-20 bg-white dark:bg-[var(--color-panel)] px-3 text-left font-medium whitespace-nowrap text-slate-800 dark:text-text-primary before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-slate-600/60"
                    >
                      {display}
                    </td>
                  ) : (
                    <td
                      key={cellKey}
                      className="py-2 px-3 text-center whitespace-nowrap border-l border-gray-300 dark:border-slate-600/60 text-slate-800 dark:text-text-primary"
                    >
                      {display}
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

  // --- Render MLB Advanced Stats Table function ---
  const renderMlbAdvancedTable = () => {
    if (sport !== "MLB") return null; // Guard clause

    // Define Headers for the MLB Advanced Stats Table
    const mlbAdvancedHeaders: {
      label: string;
      key: string;
    }[] = [
      { label: "Team", key: "team_name" },
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
          ))}
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
      <div className="overflow-x-auto rounded-xl border border-gray-300 dark:border-slate-600/60">
        <table className="w-full min-w-max text-sm">
          {/* ─────── HEADERS ─────── */}
          <thead className="bg-gray-50 dark:bg-[var(--color-panel)] border-b border-gray-300 dark:border-slate-600/60">
            <tr>
              {mlbAdvancedHeaders.map(({ label, key }, i) => (
                <HeaderCell
                  key={key}
                  label={label}
                  active={false}
                  onClick={() =>
                    toggleMlbAdvancedSort(key as MlbAdvancedSortKey)
                  }
                  align={i === 0 ? "left" : "right"}
                  className={
                    i === 0
                      ? "sticky left-0 z-30 bg-gray-50 dark:bg-[var(--color-panel)] before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-slate-600/60"
                      : "border-l border-gray-300 dark:border-slate-600/60"
                  }
                />
              ))}
            </tr>
          </thead>

          {/* ─────── BODY ─────── */}
          <tbody className="divide-y divide-gray-300 dark:divide-slate-600/60 bg-white dark:bg-[var(--color-panel)]">
            {sortedMlbAdvanced.map(
              (
                team // Use unsorted data for now
              ) => (
                <tr key={team.team_id}>
                  {/* use team_name for React key */}
                  {mlbAdvancedHeaders.map(({ key }, i) => {
                    const raw = team[key as keyof MlbAdvancedTeamStats];
                    const display = formatCell(raw, key);

                    return i === 0 ? (
                      /* ── STICKY TEAM NAME ── */
                      <td
                        key={key}
                        className={` sticky left-0 z-20 bg-white dark:bg-[var(--color-panel)] px-3 text-left font-medium whitespace-nowrap text-slate-800 dark:text-text-primary before:content-[''] before:absolute before:inset-y-0 before:right-0 before:w-px before:bg-gray-300 dark:before:bg-slate-600/60 `}
                      >
                        {display}
                      </td>
                    ) : (
                      /* ── STAT COLUMNS ── */
                      <td
                        key={key}
                        className="py-2 px-3 text-center whitespace-nowrap border-l border-gray-300 dark:border-slate-600/60 text-gray-700 dark:text-text-primary"
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
    // Main container with vertical spacing between direct children
    <section className="p-6 md:px-8 lg:px-12 space-y-4 text-slate-800 dark:text-text-primary">
      {offlineBanner}
      {/* --- Row 1: Controls (Sub-Tabs + Season Picker) --- */}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
        {/* Left: Sub-Tabs (unchanged) */}
        <div className="flex-shrink-0">
          <div className="flex gap-1 rounded-lg bg-gray-200 dark:bg-[var(--color-panel)] p-1 text-sm">
            {(sport === "NBA"
              ? (["teams", "players", "advanced"] as const)
              : (["teams", "advanced"] as const)
            ).map((tab) => {
              const tourAttribute =
                tab === "advanced" ? "stats-subtab-advanced" : undefined;
              return (
                <button
                  key={tab}
                  data-tour={tourAttribute}
                  className={`rounded-md py-2 px-4 transition-colors text-xs sm:text-sm ${
                    subTab === tab
                      ? "bg-green-600 text-white shadow-sm"
                      : "text-gray-600 dark:text-text-secondary hover:bg-gray-300 dark:hover:bg-gray-700"
                  }`}
                  onClick={() => setSubTab(tab)}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              );
            })}
          </div>
        </div>

        {/* Right: Season Picker as calendar-style button */}
        <div className="flex items-center gap-3">
          <Popover>
            <PopoverTrigger asChild>
              <button
                className="inline-flex items-center gap-2
      rounded-lg border px-2 md:px-4 py-2 text-sm
      border-slate-300 bg-white text-slate-700 hover:bg-gray-50
      dark:border-slate-600/60 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
              >
                <CalendarIcon size={16} strokeWidth={1.8} />
                {sport === "NBA"
                  ? `${season}-${String(season + 1).slice(-2)}`
                  : String(season)}
              </button>
            </PopoverTrigger>

            <PopoverContent
              side="bottom"
              align="end"
              sideOffset={8}
              className="
    bg-white dark:bg-[var(--color-panel)]
    rounded-lg shadow-lg p-2 w-[12rem]"
            >
              <div className="space-y-1">
                {seasonOptions.map(({ value, label }) => (
                  <button
                    key={value}
                    onClick={() => setSeason(value)}
                    className={`
          w-full text-left px-3 py-2 rounded text-sm transition-colors
          text-slate-800 dark:text-slate-200             /* text colour */
          hover:bg-gray-100 dark:hover:bg-slate-700      /* hover bg    */
          ${value === season ? "font-semibold text-green-600" : ""}
        `}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </PopoverContent>
          </Popover>
        </div>
      </div>
      {/* --- END OF ROW 1 --- */}

      {/* --- Row 2: Dynamic H1 Heading (Positioned correctly now) --- */}
      <h1 className="text-xl font-semibold text-slate-800 dark:text-text-primary py-2 md:py-4">
        {headingText}
      </h1>

      {/* --- END OF ROW 2 --- */}

      {/* Player Filters - Only show for NBA Players Tab */}
      {sport === "NBA" && subTab === "players" && (
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:gap-4">
          <input
            type="text"
            placeholder="Search player…"
            value={playerSearch}
            onChange={(e) => setPlayerSearch(e.target.value)}
            className="flex-grow rounded-lg border border-gray-400 dark:border-slate-600/60 bg-white dark:bg-[var(--color-panel)] px-3 py-1.5 text-sm text-gray-800 dark:text-text-primary placeholder:text-gray-400 dark:placeholder:text-text-secondary outline-none focus:ring-1 focus:ring-green-600 md:max-w-xs"
          />
        </div>
      )}

      {/* Table Display Area - Conditional Rendering based on sport and subTab */}
      <div key={`${sport}-${subTab}`}>
        {" "}
        {/* Key forces remount on sport/tab change */}
        {
          sport === "NBA"
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
