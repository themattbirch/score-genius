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
const pctKeys = new Set(["fg_pct", "three_pct", "ft_pct", "wins_all_percentage"]); // Added win %

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
  const [subTab, setSubTab] = useState<"teams" | "players">("teams");

  // --- Player filters state - only relevant for NBA players ---
  const [playerSearch, setPlayerSearch] = useState("");
  const [minMp, setMinMp] = useState(0); // Default from RPC

  const canFetchSelectedSeason = useMemo(() => {
    // Allow MLB fetching for now (can be refined later)
    if (sport === 'MLB') {
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
    if (sport === 'MLB') {
      setSubTab('teams');
      setPlayerSearch('');
      setMinMp(0);
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
  } = useTeamStats({ // Assumes this hook handles both sports
    sport,
    season,
    // Enable if MLB is selected, OR if NBA is selected AND teams tab is active
    enabled: canFetchSelectedSeason && (sport === 'MLB' || (sport === 'NBA' && subTab === 'teams')),
  });

  const {
    data: playerData, // NBA player data only
    isLoading: playerLoading,
    error: playerError,
  } = usePlayerStats({
    sport: 'NBA', // This hook might be NBA specific
    season,
    minMp,
    search: playerSearch,
    // Enable ONLY if NBA is selected AND players tab is active
    enabled: canFetchSelectedSeason && sport === 'NBA' && subTab === 'players',
  });

  // --- Season dropdown options ---
  const seasonOptions = useMemo(() =>
      Array.from({ length: 5 }).map((_, i) => {
        const yr = defaultSeason - i;
        return {
          value: yr,
          // Format label based on sport
          label: sport === "NBA"
            ? `${yr}-${String(yr + 1).slice(-2)}` // e.g., 2023-24
            : String(yr), // e.g., 2024
        };
      }),
    [defaultSeason, sport]
  );

  // --- Sorting state ---
  const [teamSort, setTeamSort] = useState<{ key: TeamSortKey; dir: SortDir } | null>(null);
  const [playerSort, setPlayerSort] = useState<{ key: PlayerSortKey; dir: SortDir } | null>(null); // NBA only

  const toggleTeamSort = (k: TeamSortKey) =>
    setTeamSort((prev) => /* ... keep existing logic ... */
       prev?.key === k
        ? { key: k, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  const togglePlayerSort = (k: PlayerSortKey) =>
    setPlayerSort((prev) => /* ... keep existing logic ... */
       prev?.key === k
        ? { key: k, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  // --- Dynamic Headers Definitions based on Sport ---
  // Define headers dynamically based on the sport
  const teamHeaders: { label: string; key: TeamSortKey }[] = useMemo(() => {
    if (sport === 'MLB') {
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
    } else { // NBA Headers
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
    { label: "STL", key: "steals" },
    { label: "BLK", key: "blocks" },
    { label: "FG%", key: "fg_pct" },
    { label: "3P%", key: "three_pct" },
    { label: "FT%", key: "ft_pct" },
    { label: "MIN", key: "minutes" },
    { label: "GP", key: "games_played"}, // Added GP
  ];

  // --- Sorted data logic (should work as is) ---
  const sortedTeams = useMemo(() => {
    if (!teamData) return [];
    const out = [...teamData];
    if (teamSort) {
      const { key, dir } = teamSort;
      out.sort((a, b) => {
        const av = a[key as keyof UnifiedTeamStats] ?? (typeof a[key as keyof UnifiedTeamStats] === 'string' ? '' : 0);
        const bv = b[key as keyof UnifiedTeamStats] ?? (typeof b[key as keyof UnifiedTeamStats] === 'string' ? '' : 0);
        const diff = typeof av === "number" && typeof bv === "number"
            ? av - bv
            : String(av).localeCompare(String(bv));
        return dir === "asc" ? diff : -diff;
      });
    }
    return out;
  }, [teamData, teamSort]);

  const sortedPlayers = useMemo(() => { // Only used for NBA
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
  const HeaderCell = ({ label, active, onClick, align = "right" }: { label: string; active: boolean; onClick: () => void; align?: "left" | "right"; }) => (
    <th onClick={onClick} title="Click to sort" className={` cursor-pointer select-none py-2 px-3 font-medium hover:bg-gray-700/30 ${align === "left" ? "text-left" : "text-right"} group `} >
      <span className="inline-flex items-center"> {label} <ChevronsUpDown size={12} className="ml-1 opacity-10 group-hover:opacity-40 transition-opacity" /> </span>
    </th>
  );

  // --- Render Team Table function (now uses dynamic headers) ---
  const renderTeamTable = () => {
    // Loading/Error states
    if (teamLoading) { /* ... Skeleton ... */
        return ( <div className="p-4 space-y-3"> {Array.from({ length: 8 }).map((_, i) => ( <SkeletonBox key={i} className="h-10 w-full rounded-lg" /> ))} </div> );
    }
    if (teamError) { /* ... Error message ... */
        return <div className="p-4 text-center text-red-400">Failed to load team stats.</div>;
    }
    if (!sortedTeams?.length) { // Check if sortedTeams is empty
        return <div className="p-4 text-center text-gray-500">No team stats available for this season.</div>;
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
              <tr key={team.team_id} className="border-b border-gray-700 last:border-none hover:bg-gray-800/60" >
                {teamHeaders.map(({ key }) => {
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

                  // --- TD ELEMENT WITH 'first:' VARIANT STYLING ---
                  return (
                    <td
                      key={String(key)} // Use the column key for the cell's key within this row
                      className="
                        py-2 px-3                      /* Standard padding */
                        text-center                    /* Default: Center align stat columns */
                        first:text-left                /* Override: Left-align the first cell (Team Name) */
                        first:font-medium              /* Override: Make first cell font medium */
                        first:whitespace-nowrap        /* Override: Prevent wrapping on first cell */
                        first:text-gray-100            /* Override: First cell text color (dark mode adjusted) */
                        dark:first:text-gray-100       /* Ensure dark mode color for first cell */
                        text-gray-400                  /* Default text color for other cells (stats) */
                        dark:text-gray-400             /* Ensure dark mode color for other cells */
                      "
                    >
                      {cell}
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
    if (playerLoading) { /* ... Skeleton ... */
        return ( <div className="space-y-3"> {Array.from({ length: 15 }).map((_, i) => ( <SkeletonBox key={i} className="h-10 w-full rounded-lg" /> ))} </div> );
    }
    if (playerError) { /* ... Error message ... */
        return <div className="text-center text-red-400">Problem loading player stats.</div>;
    }
    if (!sortedPlayers?.length) { /* ... No results message ... */
        return <div className="text-center opacity-60">No results {playerSearch ? `for “${playerSearch}”` : ''}</div>;
    }

    // Uses playerHeaders
    return (
      <div className="overflow-x-auto rounded-xl border border-gray-700">
        <table className="min-w-max divide-y divide-gray-700 text-sm">
          <thead className="bg-gray-800">
            <tr>
              {playerHeaders.map(({ label, key }) => (
                <HeaderCell key={key} label={label} active={playerSort?.key === key} onClick={() => togglePlayerSort(key)} align={key === "player_name" ? "left" : "right"} />
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedPlayers.map((p) => (
              <tr key={p.player_id} className="border-b border-gray-700 last:border-none hover:bg-gray-800/60">
                {playerHeaders.map(({ key }) => {
                  const v = p[key];
                  const display = typeof v === "number"
                      ? pctKeys.has(String(key)) // Use pctKeys for formatting
                        ? v.toFixed(1) + "%" // Already 0-100 from RPC
                        : v.toFixed(key === 'minutes' || key === 'points' || key === 'rebounds' || key === 'assists' || key === 'steals' || key === 'blocks' ? 1 : 0) // Example: more decimals for averages
                      : v;
                  const alignClass = key === "player_name" ? "sticky left-0 bg-gray-900 dark:bg-gray-950 text-left font-medium whitespace-nowrap px-3" : "text-right px-3"; // Added dark bg for sticky
                  return ( <td key={key} className={`py-2 ${alignClass}`}> {display ?? "–"} </td> );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // --- FINAL RENDER ---
  return (
    <section className="p-4 space-y-4">
      {/* Row for Conditional Tabs/Title and Season Picker */}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3">

        {/* Element changes based on sport */}
        <div className="flex-shrink-0"> {/* Wrap conditional element */}
          {sport === 'NBA' ? (
            // NBA: Show Sub-Tabs
            <div className="flex gap-1 rounded-lg bg-gray-800 p-1 text-sm">
              {(["teams", "players"] as const).map((tab) => (
                <button
                  key={tab}
                  className={`rounded-md px-3 py-1 transition-colors ${
                    subTab === tab
                      ? "bg-green-600 text-white shadow-sm"
                      : "text-gray-300 hover:bg-gray-700"
                  }`}
                  onClick={() => setSubTab(tab)}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>
          ) : (
            // MLB: Show Title Heading
            <h1 className="text-lg font-semibold tracking-wide text-gray-100">
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
      {sport === 'NBA' && subTab === 'players' && (
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:gap-4">
          <input
            type="text"
            placeholder="Search player…"
            value={playerSearch}
            onChange={(e) => setPlayerSearch(e.target.value)}
            className="flex-grow rounded-lg bg-gray-800 px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-green-600 md:max-w-xs"
          />
          <label className="flex items-center gap-2 text-sm text-gray-300">
            Min MPG
            <input
              type="number"
              min={0} max={48} step={1} value={minMp}
              onChange={(e) => setMinMp(Number(e.target.value) || 0)}
              className="w-16 rounded bg-gray-800 px-2 py-1 text-right text-sm outline-none focus:ring-1 focus:ring-green-600"
            />
          </label>
        </div>
      )}

      {/* Table Display Area */}
      {/* Use updated key to force remount/reset when view changes */}
      <div key={`${sport}-${sport === 'NBA' ? subTab : 'teams'}`}>
        {sport === 'NBA' ? (
            // NBA shows table based on subTab
             subTab === 'teams' ? renderTeamTable() : renderPlayerTable()
        ) : (
            // MLB always shows Team Table
            renderTeamTable()
        )}
      </div>

    </section>
  );
};

export default StatsScreen;