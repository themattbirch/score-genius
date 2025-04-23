// frontend/src/screens/stats_screen.tsx
import React, { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSport } from "../contexts/sport_context";
import { useDate } from "../contexts/date_context";
import { fetchTeamStats } from "../api/use_nba_team_stats";
import { UnifiedTeamStats, Sport } from "../types";
import SkeletonBox from "@/components/ui/skeleton_box";

type SortKey =
  | "team_name"
  | "wins_all_percentage"
  | "points_for_avg_all"
  | "points_against_avg_all"
  | "current_form";

export const StatsScreen: React.FC = () => {
  const { sport } = useSport();
  const { date } = useDate();

  // Default season logic
  const currentYear = useMemo(() => {
    const m = date.getUTCMonth();
    return m >= 6 ? date.getUTCFullYear() : date.getUTCFullYear() - 1;
  }, [date]);

  const [season, setSeason] = useState<number>(currentYear);

  // Sorting state
  const [sortConfig, setSortConfig] = useState<{
    key: SortKey;
    direction: "asc" | "desc";
  } | null>(null);

  // Data fetch
  const { data, isLoading, error } = useQuery<UnifiedTeamStats[]>({
    queryKey: ["teamStats", sport, season],
    queryFn: () => fetchTeamStats({ sport: sport as Sport, season }),
    staleTime: 1000 * 60 * 30,
  });

  // Handle header click
  const requestSort = (key: SortKey) => {
    setSortConfig((prev) => {
      if (prev?.key === key) {
        // toggle direction
        return {
          key,
          direction: prev.direction === "asc" ? "desc" : "asc",
        };
      }
      return { key, direction: "asc" };
    });
  };

  // Sorted data memo
  const sortedData = useMemo(() => {
    if (!data) return [];
    const sortable = [...data];
    if (sortConfig) {
      const { key, direction } = sortConfig;
      sortable.sort((a, b) => {
        const aVal = a[key] ?? 0;
        const bVal = b[key] ?? 0;

        // numeric sort if both are numbers
        if (typeof aVal === "number" && typeof bVal === "number") {
          return aVal - bVal;
        }
        // string fallback
        return String(aVal).localeCompare(String(bVal));
      });
      if (direction === "desc") sortable.reverse();
    }
    return sortable;
  }, [data, sortConfig]);

  const handleSeasonChange = (e: React.ChangeEvent<HTMLSelectElement>) =>
    setSeason(Number(e.target.value));

  // Loading state
  if (isLoading) {
    return (
      <div className="p-4 space-y-3">
        {Array.from({ length: 10 }).map((_, i) => (
          <SkeletonBox key={i} className="h-10 w-full rounded-lg" />
        ))}
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="p-4 text-center text-red-400">
        Something went wrong fetching stats.
      </div>
    );
  }

  // Render
  return (
    <section className="p-4">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between gap-4">
        <h1 className="text-lg font-semibold tracking-wide">
          {sport} {season} Team Rankings
        </h1>
        <select
          value={season}
          onChange={handleSeasonChange}
          className="rounded-lg bg-gray-200 text-gray-900 dark:bg-gray-800 dark:text-white px-3 py-1 text-sm outline-none focus:ring focus:ring-green-500/50"
        >
          {Array.from({ length: 5 }).map((_, i) => {
            const yr = currentYear - i;
            return (
              <option key={yr} value={yr}>
                {yr}
              </option>
            );
          })}
        </select>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-gray-700">
        <table className="min-w-full divide-y divide-gray-700 text-sm">
          <thead className="bg-gray-800">
            <tr>
              {[
                { label: "Team", key: "team_name" },
                { label: "Win %", key: "wins_all_percentage" },
                { label: "Avg Pts For", key: "points_for_avg_all" },
                { label: "Avg Pts Against", key: "points_against_avg_all" },
                { label: "Form", key: "current_form" },
              ].map(({ label, key }) => {
                const cfg = sortConfig;
                const isActive = cfg?.key === key;
                const dirArrow =
                  isActive && (cfg.direction === "asc" ? " ▲" : " ▼");
                const alignClass =
                  key === "team_name" ? "text-left" : "text-right";
                return (
                  <th
                    key={key}
                    onClick={() => requestSort(key as SortKey)}
                    className={`py-2 px-3 font-medium cursor-pointer select-none text-white ${alignClass}`}
                  >
                    {label}
                    {dirArrow}
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
                <td className="py-2 px-3 text-left font-medium whitespace-nowrap text-gray-900 dark:text-gray-100">
                  {team.team_name}
                </td>

                <td className="py-2 px-3 text-right">
                  {team.wins_all_percentage != null
                    ? (team.wins_all_percentage * 100).toFixed(1) + "%"
                    : "–"}
                </td>

                <td className="py-2 px-3 text-right">
                  {team.points_for_avg_all != null
                    ? team.points_for_avg_all.toFixed(1)
                    : "–"}
                </td>

                <td className="py-2 px-3 text-right">
                  {team.points_against_avg_all != null
                    ? team.points_against_avg_all.toFixed(1)
                    : "–"}
                </td>

                <td className="py-2 px-3 text-right">
                  {team.current_form ?? "–"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default StatsScreen;
