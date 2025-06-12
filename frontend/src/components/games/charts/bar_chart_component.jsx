// frontend/src/components/games/charts/bar_chart_component.jsx

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
 * Bar Chart Component
 * Displays bar chart data for game snapshots.
 * Dynamically adapts for NBA (single value) and MLB (home/away splits).
 *
 * @param {Array<Object>} data - Bar chart data from snapshot JSON.
 * - NBA Post-Game: [{ name: 'Q1', value: X }, { name: 'Q2', value: Y }, ...]
 * - NBA Pre-Game: [{ name: 'Avg Pts For', value: X }]
 * - MLB Post-Game: [{ name: 'Inn 1', Home: X, Away: Y }, ...]
 * - MLB Pre-Game: [{ name: 'Avg Runs For', Home: X, Away: Y }, { name: 'Avg Runs Against', Home: X, Away: Y }]
 * @param {'NBA' | 'MLB'} sport - The sport type to determine chart rendering logic.
 */
const BarChartComponent = ({ data, sport }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-text-secondary py-12">
        No bar chart data available.
      </div>
    );
  }

  // Determine if it's a multi-series chart (MLB home/away) or single-series (NBA quarters/avg pts)
  const isMultiSeries =
    sport === "MLB" && data.some((d) => "Home" in d && "Away" in d);

  const HomeColor = "#4ade80"; // brand-green-light equivalent for charts
  const AwayColor = "#60a5fa"; // A light blue
  const NBAColor = "#fbbf24"; // brand-orange equivalent for charts

  return (
    <div className="p-2 rounded-lg bg-[var(--color-panel)] shadow-md my-4 h-[240px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{
            top: 10,
            right: 0,
            left: -25, // Adjust left margin to fit YAxis labels
            bottom: 0,
          }}
          barCategoryGap="20%" // Space between bar categories
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--color-panel-border)"
          />
          <XAxis
            dataKey="name"
            stroke="var(--color-text-secondary)"
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke="var(--color-text-secondary)"
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--color-panel)",
              border: "none",
              borderRadius: "4px",
            }}
            labelStyle={{ color: "var(--color-text-primary)" }}
            itemStyle={{ color: "var(--color-text-secondary)" }}
            formatter={(value) => value.toFixed(1)} // Format numbers
          />
          {isMultiSeries ? (
            <>
              <Legend wrapperStyle={{ color: "var(--color-text-primary)" }} />
              <Bar dataKey="Home" fill={HomeColor} name="Home" />
              <Bar dataKey="Away" fill={AwayColor} name="Away" />
            </>
          ) : (
            <Bar dataKey="value" fill={NBAColor} name="Points" />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

BarChartComponent.propTypes = {
  data: PropTypes.array,
  sport: PropTypes.oneOf(["NBA", "MLB"]).isRequired,
};

BarChartComponent.defaultProps = {
  data: [],
};

export default BarChartComponent;
