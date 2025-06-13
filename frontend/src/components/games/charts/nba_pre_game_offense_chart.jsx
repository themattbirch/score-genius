// frontend/src/components/games/charts/nba_pre_game_offense_chart.jsx

import React from "react";
import PropTypes from "prop-types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

/**
 * NBA Pre-Game Offensive Efficiency Chart.
 * Displays comparison of Offensive Rating, eFG%, and TOV% between Home and Away teams.
 *
 * @param {Array<Object>} data - An array of objects with 'metric', 'Home', and 'Away' properties.
 * Example: [{ metric: "OffRtg", Home: 1.15, Away: 1.125 }, ...] (OffRtg normalized)
 */
const NbaPreGameOffenseChart = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-text-secondary py-12">
        No pre-game offensive stats available.
      </div>
    );
  }

  const HomeColor = "#4ade80"; // Consistent with other charts
  const AwayColor = "#60a5fa"; // Consistent with other charts

  return (
    <div className="p-2 rounded-lg bg-[var(--color-panel)] shadow-md my-4 h-[240px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{
            top: 10,
            right: 0,
            left: -25,
            bottom: 0,
          }}
          barCategoryGap="20%"
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--color-panel-border)"
          />
          <XAxis
            dataKey="metric"
            stroke="var(--color-text-secondary)"
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke="var(--color-text-secondary)"
            tickLine={false}
            axisLine={false}
            // --- FIX: Custom tick formatter for YAxis to display original scale ---
            tickFormatter={(value) => {
              if (value >= 0.5) {
                // Heuristic: Values >= 0.5 are likely percentages like eFG% (0.53) or TOV% (0.135) * 100 for display
                // OffRtg (1.15) also falls here.
                // Let's format all as percentages for consistency (e.g. 115%, 53%, 13%)
                // Or, we can use the metric context if needed.
                // For simplicity and clarity on a single axis, let's display as numbers or percentages.
                // A value like 1.15 for OffRtg or 0.53 for eFG% should be converted.
                return `${(value * 100).toFixed(0)}`; // e.g., 1.15 -> 115, 0.53 -> 53
              }
              // Fallback for very small values if any, or general numbers
              return value.toFixed(1);
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--color-panel)",
              border: "none",
              borderRadius: "4px",
            }}
            labelStyle={{ color: "var(--color-text-primary)" }}
            itemStyle={{ color: "var(--color-text-secondary)" }}
            // --- FIX: Custom formatter for Tooltip to display original scale ---
            formatter={(value, name, props) => {
              const metric = props.payload.metric;
              if (metric === "OffRtg") {
                return `${(value * 100).toFixed(1)}`; // Convert back to original scale (e.g., 1.15 -> 115.0)
              }
              // eFG% and TOV% are already 0.xx, so multiply by 100 and add %
              return `${(value * 100).toFixed(1)}%`;
            }}
          />
          <Legend wrapperStyle={{ color: "var(--color-text-primary)" }} />
          <Bar dataKey="Home" fill={HomeColor} name="Home Team" />
          <Bar dataKey="Away" fill={AwayColor} name="Away Team" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

NbaPreGameOffenseChart.propTypes = {
  data: PropTypes.arrayOf(
    PropTypes.shape({
      metric: PropTypes.string.isRequired,
      Home: PropTypes.number.isRequired,
      Away: PropTypes.number.isRequired,
    })
  ),
};

NbaPreGameOffenseChart.defaultProps = {
  data: [],
};

export default NbaPreGameOffenseChart;
