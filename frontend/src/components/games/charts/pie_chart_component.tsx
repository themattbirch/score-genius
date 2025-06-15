// frontend/src/components/games/charts/pie_chart_component.tsx

import React from "react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";
import { useTheme } from "@/contexts/theme_context";
import { PieChartDataItem } from "@/types"; // Import type
import {
  ValueType,
  NameType,
} from "recharts/types/component/DefaultTooltipContent"; // Import for Tooltip formatter

// --- START OF REQUIRED CHANGES ---
// 1. Explicitly define props interface for the component
interface PieChartComponentProps {
  data?: PieChartDataItem[]; // <--- CRITICAL: Explicitly type data here
}

// 2. Use React.FC with the defined props interface
const PieChartComponent: React.FC<PieChartComponentProps> = ({ data = [] }) => {
  // --- END OF REQUIRED CHANGES ---

  const { theme } = useTheme();

  const textColorPrimary = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const textColorSecondary = theme === "dark" ? "#9ca3af" : "#475569";
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

  // Now 'item' will be correctly inferred as PieChartDataItem due to PieChartComponentProps
  // This resolves "Spread types may only be created from object types."
  // And "Property 'color' does not exist on type 'never'."
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
    <div className="p-2 rounded-lg bg-[var(--color-panel)] shadow-md h-[240px] w-full overflow-hidden">
      {" "}
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            dataKey="value"
            nameKey="category"
            cx="50%"
            cy="50%"
            outerRadius={80}
            labelLine={false}
            label={false} // ← remove all inline slice labels
          >
            {chartData.map((entry, index) => (
              <Cell key={index} fill={entry.color} />
            ))}
          </Pie>

          <Tooltip
            contentStyle={{
              backgroundColor: panelBgColor,
              border: `1px solid ${subtleBorderColor}`,
              borderRadius: "4px",
            }}
            labelStyle={{ color: textColorPrimary }}
            itemStyle={{ color: textColorSecondary }}
            formatter={(value: ValueType) =>
              typeof value === "number" ? value.toLocaleString() : String(value)
            }
          />

          <Legend wrapperStyle={{ color: textColorPrimary }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PieChartComponent;
