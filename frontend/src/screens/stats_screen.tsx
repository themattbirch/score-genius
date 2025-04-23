import React, { useMemo, useState, useEffect } from "react";
import { useSport } from "../contexts/sport_context";
import { useDate } from "../contexts/date_context";
import { useTeamStats } from "../api/use_team_stats";
import { UnifiedTeamStats, Sport } from "../types";
import SkeletonBox from "@/components/ui/skeleton_box";
import { ChevronsUpDown } from "lucide-react";

type SortKey = keyof UnifiedTeamStats;

export const StatsScreen: React.FC = () => {
  const { sport } = useSport();
  const { date } = useDate();

  // 1️⃣ Compute defaultSeason based on sport
  const defaultSeason = useMemo(() => {
    if (sport === "MLB") {
      // MLB is calendar-year
      return date.getUTCFullYear();
    } else {
      // NBA spans two years; start year is July cutoff
      return date.getUTCMonth() >= 6
        ? date.getUTCFullYear()
        : date.getUTCFullYear() - 1;
    }
  }, [sport, date]);

  // 2️⃣ Manage season state (and reset it on sport change)
  const [season, setSeason] = useState<number>(defaultSeason);
  useEffect(() => {
    setSeason(defaultSeason);
  }, [defaultSeason]);

  const handleSeasonChange = (e: React.ChangeEvent<HTMLSelectElement>) =>
    setSeason(Number(e.target.value));

  // 3️⃣ Fetch data
  const canFetch = sport === "NBA" ? season <= defaultSeason : true;
  const { data, isLoading, error } = useTeamStats({
    sport,
    season,
    enabled: canFetch,
  });

  // 4️⃣ Sorting state
  const [sortConfig, setSortConfig] = useState<{
    key: SortKey;
    direction: "asc" | "desc";
  } | null>(null);
  const requestSort = (key: SortKey) =>
    setSortConfig((prev) =>
      prev?.key === key
        ? { key, direction: prev.direction === "asc" ? "desc" : "asc" }
        : { key, direction: "asc" }
    );

  // 5️⃣ Build season dropdown options
  const seasonOptions = useMemo(() => {
    return Array.from({ length: 5 }).map((_, i) => {
      const year = defaultSeason - i;
      const label =
        sport === "NBA"
          ? `${year}-${String(year + 1).slice(-2)}` // e.g. "2024-25"
          : String(year); // e.g. "2025"
      return { value: year, label };
    });
  }, [defaultSeason, sport]);

  // 6️⃣ Decide average columns
  const avgCols =
    sport === "MLB"
      ? [
          { label: "Runs For", key: "runs_for_avg_all" as SortKey },
          { label: "Runs Allow", key: "runs_against_avg_all" as SortKey },
        ]
      : [
          { label: "Off Pts", key: "points_for_avg_all" as SortKey },
          {
            label: "Def Pts",
            key: "points_against_avg_all" as SortKey,
          },
        ];

  // 7️⃣ Table headers
  const headers = useMemo(
    () => [
      { label: "Team", key: "team_name" as SortKey },
      { label: "Win %", key: "wins_all_percentage" as SortKey },
      ...avgCols,
      { label: "Streak", key: "current_form" as SortKey },
    ],
    [avgCols]
  );

  // 8️⃣ Sort the data
  const sortedData = useMemo(() => {
    if (!data) return [];
    const arr = [...data];
    if (sortConfig) {
      const { key, direction } = sortConfig;
      arr.sort((a, b) => {
        const aVal = a[key] ?? 0;
        const bVal = b[key] ?? 0;
        if (typeof aVal === "number" && typeof bVal === "number") {
          return aVal - bVal;
        }
        return String(aVal).localeCompare(String(bVal));
      });
      if (direction === "desc") arr.reverse();
    }
    return arr;
  }, [data, sortConfig]);

  // 9️⃣ Loading & error states
  if (isLoading) {
    return (
      <div className="p-4 space-y-3">
        {Array.from({ length: 10 }).map((_, i) => (
          <SkeletonBox key={i} className="h-10 w-full rounded-lg" />
        ))}
      </div>
    );
  }
  if (error) {
    return (
      <div className="p-4 text-center text-red-400">
        Something went wrong fetching stats.
      </div>
    );
  }

  // 🔟 Render
  return (
    <section className="p-4">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-lg font-semibold tracking-wide">
          {sport} Team Rankings
        </h1>
        <select
          value={season}
          onChange={handleSeasonChange}
          className="align-baseline rounded-lg bg-gray-200 text-gray-900 dark:bg-gray-800 dark:text-white py-1 text-sm outline-none focus:ring focus:ring-green-500/50"
        >
          {seasonOptions.map(({ value, label }) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-gray-700">
        <table className="min-w-full divide-y divide-gray-700 text-sm">
          <thead className="bg-gray-800">
            <tr>
              {headers.map(({ label, key }) => {
                const alignClass =
                  key === "team_name" ? "text-left" : "text-center";
                const active = sortConfig?.key === key;
                const arrow = active
                  ? sortConfig!.direction === "asc"
                    ? " ▲"
                    : " ▼"
                  : "";

                return (
                  <th
                    key={key}
                    title="Click to sort"
                    onClick={() => requestSort(key)}
                    className={`
    ${alignClass} group py-2 px-3 font-medium cursor-pointer select-none text-white hover:bg-gray-700/30
  `}
                  >
                    <span className="inline-flex items-center">
                      {label}
                      <ChevronsUpDown
                        size={12}
                        className="
        ml-1 
        opacity-10 
        group-hover:opacity-40 
        transition-opacity
      "
                      />
                    </span>
                    {arrow}
                  </th>
                );
              })}
            </tr>
          </thead>

          <tbody>
            {sortedData.map((team) => (
              <tr
                key={team.team_id}
                className="border-b border-gray-200 dark:border-gray-700 last:border-none hover:bg-gray-800/60"
              >
                {headers.map(({ key }) => {
                  // decide left vs center
                  const alignClass =
                    key === "team_name"
                      ? "text-left font-medium whitespace-nowrap text-gray-900 dark:text-gray-100"
                      : "text-center";

                  // pick the right content (%, number, or string)
                  let cell: React.ReactNode = "–";
                  const v = team[key];
                  if (v != null) {
                    if (key === "wins_all_percentage") {
                      cell = ((v as number) * 100).toFixed(1) + "%";
                    } else if (typeof v === "number") {
                      cell = (v as number).toFixed(1);
                    } else {
                      cell = v;
                    }
                  }

                  return (
                    <td
                      key={key}
                      className="
                    py-2 px-3
                    text-center                       /* center everything by default */
                    first:text-left                   /* first cell (team) is left-aligned */
                    first:font-medium                 /* and bold for emphasis */
                    first:whitespace-nowrap           /* no wrapping on the team name */
                    first:text-gray-900 first:dark:text-gray-100  /* team-name color */
                  "
                    >
                      {cell}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default StatsScreen;
