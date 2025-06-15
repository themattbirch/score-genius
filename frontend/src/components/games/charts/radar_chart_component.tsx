// frontend/src/components/games/charts/radar_chart_component.tsx

import React from "react";
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
import { useTheme } from "@/contexts/theme_context"; // Import useTheme context
import { RadarChartData } from "@/types"; // Import type

// 1. Define the interface for RadarChartComponent's props
interface RadarChartComponentProps {
  data?: RadarChartData[]; // The data prop can be optional and defaults to []
}

/**
 * Radar Chart Component
 * Displays team advanced metrics in a radar chart format.
 *
 * @param {Array<Object>} data - Radar chart data from snapshot JSON.
 * Example: [{ metric: 'Pace', home_value: 100.0, away_value: 98.5 }, ...]
 */
// 2. Apply the interface to the functional component
const RadarChartComponent: React.FC<RadarChartComponentProps> = ({
  data = [],
}) => {
  // Get the current theme
  const { theme } = useTheme();

  // Define color values based on the theme for Recharts properties
  const textColorPrimary = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const textColorSecondary = theme === "dark" ? "#9ca3af" : "#475569";
  const panelBgColor = theme === "dark" ? "#161b22" : "#f8fafc";
  const subtleBorderColor =
    theme === "dark" ? "rgba(51, 65, 85, 0.6)" : "#e2e8f0";

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
    <div className="p-2 rounded-lg bg-[var(--color-panel)] shadow-md h-[240px] w-full overflow-hidden">
      {" "}
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid stroke={subtleBorderColor} />
          <PolarAngleAxis
            dataKey="metric"
            stroke={textColorSecondary}
            tick={{ fill: textColorPrimary, fontSize: 10 }}
          />
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
              backgroundColor: panelBgColor,
              border: `1px solid ${subtleBorderColor}`,
              borderRadius: "4px",
            }}
            labelStyle={{ color: textColorPrimary }}
            itemStyle={{ color: textColorSecondary }}
            // 3. Type the 'value' in the formatter and ensure it's a number
            formatter={(value: number | string) => {
              if (typeof value === "number") {
                return value.toFixed(2);
              }
              return value; // Return as is if not a number
            }}
          />
          <Legend wrapperStyle={{ color: textColorPrimary }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

// PropTypes and defaultProps are removed as they are replaced by TypeScript interface
// RadarChartComponent.propTypes = {
//   data: PropTypes.array,
// };
// RadarChartComponent.defaultProps = {
//   data: [],
// };

export default RadarChartComponent;
