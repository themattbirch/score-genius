// frontend/src/components/games/charts/pie_chart_component.jsx

import React from "react";
import PropTypes from "prop-types";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";

/**
 * Pie Chart Component
 * Displays data in a pie chart format (e.g., scoring distribution).
 *
 * @param {Array<Object>} data - Pie chart data from snapshot JSON.
 * Example: [{ category: '2P FG Made', value: X, color: '#hex' }, ...]
 */
const PieChartComponent = ({ data }) => {
  // Define default colors if not provided in data, or use a palette
  const defaultColors = [
    "#4ade80",
    "#60a5fa",
    "#fbbf24",
    "#f87171",
    "#a78bfa",
    "#facc15",
  ]; // A palette of friendly colors

  if (!data || data.length === 0) {
    return (
      <div className="text-center text-text-secondary py-12">
        No pie chart data available.
      </div>
    );
  }

  // Assign colors from the data's 'color' property, or fall back to default palette
  const chartData = data.map((item, index) => ({
    ...item,
    color: item.color || defaultColors[index % defaultColors.length],
  }));

  // Check if data is just a placeholder for "Pre-Game Distribution N/A" from NBA
  const isPlaceholder =
    chartData.length === 1 &&
    chartData[0].category === "Pre-Game Distribution N/A";

  if (isPlaceholder) {
    return (
      <div className="text-center text-text-secondary py-12">
        Pre-Game Scoring Distribution N/A.
      </div>
    );
  }

  return (
    <div className="p-2 rounded-lg bg-[var(--color-panel)] shadow-md my-4 h-[240px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            dataKey="value"
            nameKey="category"
            cx="50%"
            cy="50%"
            outerRadius={80} // Adjust size as needed
            fill="#8884d8" // Default fill, overridden by Cell colors
            labelLine={false}
            // Optional: label={({ percent }) => `${(percent * 100).toFixed(0)}%`} // Show percentage on slices
            label={({ category, value }) => `${category} (${value})`} // Show category and value
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--color-panel)",
              border: "none",
              borderRadius: "4px",
            }}
            labelStyle={{ color: "var(--color-text-primary)" }}
            itemStyle={{ color: "var(--color-text-secondary)" }}
            formatter={(value) => value.toLocaleString()} // Format numbers
          />
          <Legend wrapperStyle={{ color: "var(--color-text-primary)" }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

PieChartComponent.propTypes = {
  data: PropTypes.array,
};

PieChartComponent.defaultProps = {
  data: [],
};

export default PieChartComponent;
