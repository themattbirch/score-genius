// frontend/src/components/games/charts/pie_chart_component.tsx

import React from "react";
import { PieChart, Pie, Cell, Tooltip } from "recharts";
import { useTheme } from "@/contexts/theme_context";
import { PieChartDataItem } from "@/types";
import {
  ValueType,
  NameType,
} from "recharts/types/component/DefaultTooltipContent";

interface PieChartComponentProps {
  data?: PieChartDataItem[];
}

const PieChartComponent: React.FC<PieChartComponentProps> = ({ data = [] }) => {
  const { theme } = useTheme();

  const textColorPrimary = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const panelBgColor = theme === "dark" ? "#161b22" : "#f8fafc";
  const subtleBorderColor =
    theme === "dark" ? "rgba(51, 65, 85, 0.6)" : "#e2e8f0";
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

  if (isPlaceholder) {
    return null;
  }

  return (
    <div className="flex flex-col items-center">
      <PieChart width={150} height={150}>
        <Tooltip
          contentStyle={{
            backgroundColor: panelBgColor,
            border: `1px solid ${subtleBorderColor}`,
            borderRadius: "4px",
          }}
          formatter={(value: ValueType) =>
            typeof value === "number" ? value.toLocaleString() : String(value)
          }
        />
        <Pie
          data={chartData}
          dataKey="value"
          nameKey="category"
          cx="50%"
          cy="50%"
          outerRadius={70}
          innerRadius={0} /* FIX #1: Set to 0 for a solid pie chart */
          labelLine={false}
          label={false}
          stroke="none"
        >
          {chartData.map((entry) => (
            <Cell key={`cell-${entry.category}`} fill={entry.color} />
          ))}
        </Pie>
      </PieChart>

      {/* FIX #2: Make the legend container a centered flex column */}
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
