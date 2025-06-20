import React from "react";
import { PieChart, Pie, Cell, Tooltip } from "recharts";
import { useTheme } from "@/contexts/theme_context";
import { PieChartDataItem, Sport } from "@/types";
import { TooltipProps } from "recharts"; // Import TooltipProps for typing
import {
  ValueType,
  NameType,
} from "recharts/types/component/DefaultTooltipContent";

// --- ★ START OF FIX ★ ---
// 1. Define a new, fully custom tooltip component
const CustomPieTooltip: React.FC<
  TooltipProps<ValueType, NameType> & { sport?: Sport }
> = ({ active, payload, sport }) => {
  const { theme } = useTheme(); // We can use theme hook inside the custom component
  const textColorPrimary = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const panelBgColor = theme === "dark" ? "#161b22" : "#f8fafc";
  const subtleBorderColor =
    theme === "dark" ? "rgba(51, 65, 85, 0.6)" : "#e2e8f0";

  if (active && payload && payload.length) {
    const dataName = payload[0].name; // e.g., "Home Offense vs Opp. Hand..."
    const dataValue = payload[0].value; // e.g., 4.1

    return (
      <div
        style={{
          backgroundColor: panelBgColor,
          border: `1px solid ${subtleBorderColor}`,
          borderRadius: "8px",
          padding: "8px",
          maxWidth: "240px",
          whiteSpace: "normal",
          fontSize: "12px",
          color: textColorPrimary,
        }}
      >
        {/* For MLB, we only show the name because the value is already in it */}
        {sport === "MLB" && <p>{dataName}</p>}

        {/* For NBA, we show both the name (e.g., "Home") and the value */}
        {sport === "NBA" && (
          <p>
            {dataName}: {Number(dataValue).toLocaleString()}
          </p>
        )}
      </div>
    );
  }

  return null;
};
// --- ★ END OF FIX ★ ---

interface PieChartComponentProps {
  data?: PieChartDataItem[];
  sport?: Sport;
}

const PieChartComponent: React.FC<PieChartComponentProps> = ({
  data = [],
  sport,
}) => {
  const { theme } = useTheme();
  const textColorPrimary = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const defaultColors = [
    "#4ade80",
    "#60a5fa",
    "#fbbf24",
    "#f87171",
    "#a78bfa",
    "#facc15",
  ];

  if (!data || data.length === 0) {
    return (
      <div className="text-center text-text-secondary py-12">
        No pie chart data available.
      </div>
    );
  }

  const chartData = data.map((item, index) => ({
    ...item,
    color: item.color || defaultColors[index % defaultColors.length],
  }));

  const isPlaceholder =
    chartData.length === 1 &&
    chartData[0].category === "Pre-Game Distribution N/A";
  if (isPlaceholder) return null;

  return (
    <div className="flex flex-col items-center">
      <PieChart width={150} height={150}>
        {/* --- ★ START OF FIX ★ --- */}
        {/* 2. Tell the Tooltip to use our new custom component */}
        <Tooltip
          cursor={{ fill: "rgba(255, 255, 255, 0.1)" }}
          content={<CustomPieTooltip sport={sport} />}
        />
        {/* --- ★ END OF FIX ★ --- */}

        <Pie
          data={chartData}
          dataKey="value"
          nameKey="category"
          cx="50%"
          cy="50%"
          outerRadius={70}
          innerRadius={0}
          labelLine={false}
          label={false}
          stroke="none"
        >
          {chartData.map((entry) => (
            <Cell key={`cell-${entry.category}`} fill={entry.color} />
          ))}
        </Pie>
      </PieChart>

      <div className="w-full mt-2 text-xs space-y-1 flex flex-col items-center">
        {chartData.map((entry) => (
          <div
            key={entry.category}
            className="flex items-center justify-center gap-2"
          >
            <div
              className="w-3 h-3 rounded-sm flex-shrink-0"
              style={{ backgroundColor: entry.color }}
            />
            <span style={{ color: textColorPrimary }}>{entry.category}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PieChartComponent;
