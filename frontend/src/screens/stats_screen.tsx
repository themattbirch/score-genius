// frontend/src/screens/stats_screen.tsx
import React, { useMemo, useState, useEffect } from "react";
import { useSport } from "../contexts/sport_context";
import { useDate } from "../contexts/date_context";
// Assuming useTeamStats can fetch MLB team stats too by just changing the 'sport' prop
import { useTeamStats } from "../api/use_team_stats";
import { usePlayerStats, UnifiedPlayerStats } from "../api/use_player_stats";
import type { UnifiedTeamStats } from "../types"; // Assuming UnifiedTeamStats works for MLB too
import SkeletonBox from "@/components/ui/skeleton_box";
import { ChevronsUpDown } from "lucide-react";

// --- Type definitions (keep these) ---
type SortDir = "asc" | "desc";
type TeamSortKey = keyof UnifiedTeamStats | string; // Allow string if keys differ wildly
type PlayerSortKey = keyof UnifiedPlayerStats;
const pctKeys = new Set(["three_pct", "ft_pct", "wins_all_percentage"]); // Added win %

const StatsScreen: React.FC = () => {
  const { sport } = useSport(); // Get current sport ('NBA' | 'MLB')
  const { date } = useDate();

  // --- Season logic (adjusts based on sport) ---
  const defaultSeason = useMemo(() => {
    if (sport === "MLB") return date.getUTCFullYear();
    // NBA season spans years
    return date.getUTCMonth() >= 6 // July or later means current year starts season
      ? date.getUTCFullYear()
      : date.getUTCFullYear() - 1; // Jan-June means previous year started season
  }, [sport, date]);

  const [season, setSeason] = useState<number>(defaultSeason);
  // Update season if defaultSeason changes (due to sport or date change)
  useEffect(() => setSeason(defaultSeason), [defaultSeason]);

  // --- Sub-tab state - only relevant for NBA ---
  const [subTab, setSubTab] = useState<"teams" | "players" | "advanced">(
    "teams"
  );

  // --- Player filters state - only relevant for NBA players ---
  const [playerSearch, setPlayerSearch] = useState("");

  const canFetchSelectedSeason = useMemo(() => {
    // Allow MLB fetching for now (can be refined later)
    if (sport === "MLB") {
      return true;
    }
    // For NBA, only allow if selected season is the current/past default season or earlier
    // (e.g., if defaultSeason is 2024, allow 2024, 2023, etc., but not 2025)
    return season <= defaultSeason;
  }, [sport, season, defaultSeason]);
  // --- End of added block ---

  // --- Effect to reset state when sport changes ---
  useEffect(() => {
    // When switching TO MLB, force view to 'teams' and clear NBA filters
    if (sport === "MLB") {
      setSubTab("teams");
      setPlayerSearch("");
    }
    // Optional: reset to 'teams' when switching back TO NBA?
    // else if (sport === 'NBA' && subTab === 'players') {
    // If you want to always default NBA to 'teams' uncomment below
    // setSubTab('teams');
    // }
  }, [sport]); // Run this effect when sport changes

  // --- QUERIES with conditional 'enabled' flags ---
  const {
    data: teamData,
    isLoading: teamLoading,
    error: teamError,
  } = useTeamStats({
    // Assumes this hook handles both sports
    sport,
    season,
    // Enable if MLB is selected, OR if NBA is selected AND teams tab is active
    enabled:
      canFetchSelectedSeason &&
      (sport === "MLB" || (sport === "NBA" && subTab === "teams")),
  });

  const {
    data: playerData, // NBA player data only
    isLoading: playerLoading,
    error: playerError,
  } = usePlayerStats({
    sport: "NBA", // This hook might be NBA specific
    season,
    search: playerSearch,
    // Enable ONLY if NBA is selected AND players tab is active
    enabled: canFetchSelectedSeason && sport === "NBA" && subTab === "players",
  });

  // --- Season dropdown options ---
  const seasonOptions = useMemo(
    () =>
      Array.from({ length: 5 }).map((_, i) => {
        const yr = defaultSeason - i;
        return {
          value: yr,
          // Format label based on sport
          label:
            sport === "NBA"
              ? `${yr}-${String(yr + 1).slice(-2)}` // e.g., 2023-24
              : String(yr), // e.g., 2024
        };
      }),
    [defaultSeason, sport]
  );

  // --- Sorting state ---
  const [teamSort, setTeamSort] = useState<{
    key: TeamSortKey;
    dir: SortDir;
  } | null>(null);
  const [playerSort, setPlayerSort] = useState<{
    key: PlayerSortKey;
    dir: SortDir;
  } | null>(null); // NBA only

  const toggleTeamSort = (k: TeamSortKey) =>
    setTeamSort((prev /* ... keep existing logic ... */) =>
      prev?.key === k
        ? { key: k, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  const togglePlayerSort = (k: PlayerSortKey) =>
    setPlayerSort((prev /* ... keep existing logic ... */) =>
      prev?.key === k
        ? { key: k, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  // --- Dynamic Headers Definitions based on Sport ---
  // Define headers dynamically based on the sport
  const teamHeaders: { label: string; key: TeamSortKey }[] = useMemo(() => {
    if (sport === "MLB") {
      // Define MLB Team Stat Headers
      // IMPORTANT: Ensure the 'key' values match the property names
      // returned by your useTeamStats hook for MLB data
      return [
        { label: "Team", key: "team_name" },
        { label: "Win %", key: "wins_all_percentage" }, // Example common stat
        // Add other relevant MLB stats keys like 'runs_for_avg_all', 'runs_against_avg_all' etc.
        { label: "Runs Score", key: "runs_for_avg_all" },
        { label: "Runs Allow", key: "runs_against_avg_all" },
        { label: "Streak", key: "current_form" }, // Example
      ];
    } else {
      // NBA Headers
      return [
        { label: "Team", key: "team_name" },
        { label: "Win %", key: "wins_all_percentage" },
        { label: "Off Pts", key: "points_for_avg_all" },
        { label: "Def Pts", key: "points_against_avg_all" },
        { label: "Streak", key: "current_form" },
      ];
    }
  }, [sport]);

  // Player headers only needed for NBA
  const playerHeaders: { label: string; key: PlayerSortKey }[] = [
    { label: "Player", key: "player_name" },
    { label: "Team", key: "team_name" },
    { label: "PTS", key: "points" },
    { label: "REB", key: "rebounds" },
    { label: "AST", key: "assists" },
    { label: "3P%", key: "three_pct" },
    { label: "FT%", key: "ft_pct" },
    { label: "MIN", key: "minutes" },
    { label: "GP", key: "games_played" }, // Added GP
  ];

  // --- Sorted data logic (should work as is) ---
  const sortedTeams = useMemo(() => {
    if (!teamData) return [];
    const out = [...teamData];
    if (teamSort) {
      const { key, dir } = teamSort;
      out.sort((a, b) => {
        const av =
          a[key as keyof UnifiedTeamStats] ??
          (typeof a[key as keyof UnifiedTeamStats] === "string" ? "" : 0);
        const bv =
          b[key as keyof UnifiedTeamStats] ??
          (typeof b[key as keyof UnifiedTeamStats] === "string" ? "" : 0);
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
    // Only used for NBA
    if (!playerData) return [];
    const out = [...playerData];
    if (playerSort) {
      const { key, dir } = playerSort;
      out.sort((a, b) => {
        const av = (a[key] ?? 0) as number; // Assuming player stats are numeric for sorting
        const bv = (b[key] ?? 0) as number;
        return dir === "asc" ? av - bv : bv - av;
      });
    }
    return out;
  }, [playerData, playerSort]);

  // --- Reusable header cell component (no changes needed) ---
  // --- Reusable header cell component ---
  const HeaderCell = ({
    label,
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
      // ADD light mode text color, ADJUST dark mode if needed
      className={`
        bg-gray-50 dark:bg-gray-800
        cursor-pointer select-none py-2 px-3 font-medium
        text-gray-600 dark:text-gray-400
        hover:bg-gray-200 dark:hover:bg-gray-700/30
        ${align === "left" ? "text-left" : "text-center"}
        group
        ${className} // Correctly append the passed className prop here
      `}
    >
      <span className="inline-flex items-center">
        {label}
        <ChevronsUpDown
          size={12}
          // Ensure icon color contrasts in both modes
          className="ml-1 opacity-30 dark:opacity-40 group-hover:opacity-60 dark:group-hover:opacity-70 transition-opacity"
        />
      </span>
    </th>
  );

  // --- Render Team Table function (now uses dynamic headers) ---
  const renderTeamTable = () => {
    // Loading/Error states
    if (teamLoading) {
      /* ... Skeleton ... */
      return (
        <div className="p-4 space-y-3">
          {" "}
          {Array.from({ length: 8 }).map((_, i) => (
            <SkeletonBox key={i} className="h-10 w-full rounded-lg" />
          ))}{" "}
        </div>
      );
    }
    if (teamError) {
      /* ... Error message ... */
      return (
        <div className="p-4 text-center text-red-400">
          Failed to load team stats.
        </div>
      );
    }
    if (!sortedTeams?.length) {
      // Check if sortedTeams is empty
      return (
        <div className="p-4 text-center text-gray-500">
          No team stats available for this season.
        </div>
      );
    }

    return (
      <div className="overflow-x-auto rounded-xl border border-gray-700">
        <table className="min-w-full divide-y divide-gray-700 text-sm">
          <thead className="bg-gray-800">
            {/* thead content using teamHeaders and HeaderCell... */}
            <tr>
              {teamHeaders.map(({ label, key }) => (
                <HeaderCell
                  key={String(key)}
                  label={label}
                  active={teamSort?.key === key}
                  onClick={() => toggleTeamSort(key)}
                  // Apply first:text-left directly to header cell too
                  align={key === "team_name" ? "left" : "right"} // Center stat headers
                />
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedTeams.map((team) => (
              <tr
                key={team.team_id}
                className="border-b border-gray-700 last:border-none hover:bg-gray-800/60"
              >
                {teamHeaders.map(({ key }, index) => {
                  const raw = team[key as keyof UnifiedTeamStats];
                  let cell: React.ReactNode = "–";
                  // Cell formatting logic (remains the same)
                  if (raw != null) {
                    if (pctKeys.has(String(key))) {
                      cell = ((raw as number) * 100).toFixed(1) + "%";
                    } else if (typeof raw === "number") {
                      cell = raw.toFixed(1);
                    } else {
                      cell = String(raw); // Ensure it's a string if not number/null
                    }
                  }

                  // --- ADD THIS LINE BACK ---
                  // --- NEW TD STYLING LOGIC using index ---
                  let cellSpecificClasses = "";
                  // Base vertical padding only
                  const baseVerticalPadding = "py-2";

                  if (key === "player_name") {
                    // Sticky Player cell - ADD border-r and ensure px-3
                    cellSpecificClasses = `
                      sticky left-0
                      bg-white dark:bg-gray-900
                      px-3 /* Horizontal Padding */
                      text-left font-medium whitespace-nowrap
                      text-gray-900 dark:text-gray-100
                      border-r border-gray-200 dark:border-gray-700 /* <<< ADDED RIGHT BORDER */
                    `;
                  } else if (key === "team_name") {
                    // Team cell - Needs px-3 and already has border-l
                    cellSpecificClasses = `
                      px-3 /* Horizontal Padding */
                      text-left font-medium whitespace-nowrap
                      border-l border-gray-200 dark:border-gray-700
                      text-gray-700 dark:text-gray-400
                    `;
                  } else {
                    // Default Stat cells - Add px-3 and border-l for consistency
                    cellSpecificClasses = `
                      px-3 /* Horizontal Padding */
                      text-center
                      border-l border-gray-200 dark:border-gray-700 /* Add border */
                      text-gray-700 dark:text-gray-400
                    `;
                  }

                  return (
                    // Combine base vertical padding with specific styles
                    <td
                      key={key}
                      className={`${baseVerticalPadding} ${cellSpecificClasses}`}
                    >
                      {cell ?? "–"} {/* Correctly uses cell */}
                    </td>
                  );
                  // --- END OF TD ELEMENT ---
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // --- Render Player Table function (only relevant for NBA) ---
  const renderPlayerTable = () => {
    // This should only be called when sport === 'NBA'
    if (playerLoading) {
      /* ... Skeleton ... */
      return (
        <div className="space-y-3">
          {" "}
          {Array.from({ length: 15 }).map((_, i) => (
            <SkeletonBox key={i} className="h-10 w-full rounded-lg" />
          ))}{" "}
        </div>
      );
    }
    if (playerError) {
      /* ... Error message ... */
      return (
        <div className="text-center text-red-400">
          Problem loading player stats.
        </div>
      );
    }
    if (!sortedPlayers?.length) {
      /* ... No results message ... */
      return (
        <div className="text-center opacity-60">
          No results {playerSearch ? `for “${playerSearch}”` : ""}
        </div>
      );
    }

    // Uses playerHeaders
    return (
      // Use consistent border colors
      <div className="overflow-x-auto rounded-xl border border-gray-300 dark:border-gray-700">
        <table className="min-w-full text-sm border-separate border-spacing-0">
          <thead className="bg-gray-50 dark:bg-gray-800">
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
                      ? // sticky “Player” header
                        "sticky left-0 z-20 bg-gray-50 dark:bg-gray-800 border-r border-gray-300 dark:border-gray-700"
                      : ""
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
                  const v = p[key];
                  const display =
                    typeof v === "number"
                      ? pctKeys.has(String(key))
                        ? v.toFixed(1) + "%"
                        : v.toFixed(key === "games_played" ? 0 : 1)
                      : v;
                  if (i === 0) {
                    // sticky player name cell
                    return (
                      <td
                        key={key}
                        className="
                    sticky left-0 z-20
                    bg-white dark:bg-gray-900
                    border-r border-gray-300 dark:border-gray-700
                    py-2 px-3
                    text-left font-medium whitespace-nowrap
                    text-gray-900 dark:text-gray-100
                  "
                      >
                        {display}
                      </td>
                    );
                  } else {
                    // team + stats
                    return (
                      <td
                        key={key}
                        className={`
                    py-2 px-3
                    ${
                      key === "team_name"
                        ? "text-left font-medium whitespace-nowrap"
                        : "text-center"
                    }
                    border-l border-gray-300 dark:border-gray-700
                    text-gray-700 dark:text-gray-400
                  `}
                      >
                        {display ?? "–"}
                      </td>
                    );
                  }
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }; // End renderPlayerTable

  const renderAdvancedTable = () => {
    if (sport !== "NBA") return null; // Only for NBA
    // TODO: Implement fetching and rendering for advanced stats
    return (
      <div className="p-4 text-center text-gray-500 dark:text-gray-400">
        Advanced Stats Table Placeholder
        {/* Add SkeletonBox here later */}
      </div>
    );
  };

  // --- FINAL RENDER ---
  return (
    <section className="p-4 space-y-4">
      {/* Row for Conditional Tabs/Title and Season Picker */}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
        {/* Element changes based on sport */}
        <div className="flex-shrink-0">
          {sport === "NBA" ? (
            // NBA: Show Sub-Tabs - MODIFY to include 'advanced'
            // Try slightly less padding/gap for three buttons
            <div className="flex gap-1 rounded-lg bg-gray-200 dark:bg-gray-800 p-1 text-sm">
              {/* MODIFY array to map */}
              {(["teams", "players", "advanced"] as const).map((tab) => (
                <button
                  key={tab}
                  // MODIFY button text and maybe padding
                  className={`rounded-md px-3 py-1 transition-colors text-xs sm:text-sm ${
                    // Reduced padding slightly, smaller text base
                    subTab === tab
                      ? "bg-green-600 text-white shadow-sm" // Active tab
                      : "text-gray-600 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-700" // Inactive tab
                  }`}
                  onClick={() => setSubTab(tab)}
                >
                  {/* MODIFY text generation */}
                  {tab === "advanced"
                    ? "Advanced"
                    : tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>
          ) : (
            // MLB: Show Title Heading
            <h1 className="text-lg font-semibold tracking-wide text-gray-900 dark:text-gray-100">
              {sport} Team Rankings
            </h1>
          )}
        </div>

        {/* Season picker - always shown */}
        <select
          value={season}
          onChange={(e) => setSeason(Number(e.target.value))}
          className="align-baseline rounded-lg bg-gray-200 text-gray-900 dark:bg-gray-800 dark:text-white py-1 text-sm outline-none focus:ring focus:ring-green-500/50"
        >
          {seasonOptions.map(({ value, label }) => (
            <option key={value} value={value}>
              {label}
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

      {/* Table Display Area */}
      {/* Use updated key to force remount/reset when view changes */}
      <div key={`${sport}-${sport === "NBA" ? subTab : "teams"}`}>
        {sport === "NBA"
          ? // NBA shows table based on subTab
            subTab === "teams"
            ? renderTeamTable()
            : subTab === "players"
            ? renderPlayerTable()
            : renderAdvancedTable() // Render placeholder
          : // MLB always shows Team Table
            renderTeamTable()}
      </div>
    </section>
  );
};

export default StatsScreen;
