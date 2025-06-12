// frontend/src/components/games/charts/radar_chart_component.jsx

import React from "react";
import PropTypes from "prop-types";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";

/**
 * Radar Chart Component
 * Displays team advanced metrics in a radar chart format.
 *
 * @param {Array<Object>} data - Radar chart data from snapshot JSON.
 * Example: [{ metric: 'Pace', home_value: 100.0, away_value: 98.5 }, ...]
 */
const RadarChartComponent = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-text-secondary py-12">
        No radar chart data available.
      </div>
    );
  }

  const HomeColor = "#4ade80"; // brand-green-light equivalent
  const AwayColor = "#60a5fa"; // A light blue

  return (
    <div className="p-2 rounded-lg bg-[var(--color-panel)] shadow-md my-4 h-[240px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid stroke="var(--color-panel-border)" />
          <PolarAngleAxis
            dataKey="metric"
            stroke="var(--color-text-secondary)"
            tick={{ fill: "var(--color-text-primary)", fontSize: 10 }}
          />
          {/* PolarRadiusAxis can be added if needed for scaling reference, but often omitted for simplicity */}
          {/* <PolarRadiusAxis angle={90} domain={[0, 'auto']} /> */}
          <Radar
            name="Home Team"
            dataKey="home_value"
            stroke={HomeColor}
            fill={HomeColor}
            fillOpacity={0.6}
          />
          <Radar
            name="Away Team"
            dataKey="away_value"
            stroke={AwayColor}
            fill={AwayColor}
            fillOpacity={0.6}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--color-panel)",
              border: "none",
              borderRadius: "4px",
            }}
            labelStyle={{ color: "var(--color-text-primary)" }}
            itemStyle={{ color: "var(--color-text-secondary)" }}
            formatter={(value) => value.toFixed(2)} // Format numbers
          />
          <Legend wrapperStyle={{ color: "var(--color-text-primary)" }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

RadarChartComponent.propTypes = {
  data: PropTypes.array,
};

RadarChartComponent.defaultProps = {
  data: [],
};

export default RadarChartComponent;
