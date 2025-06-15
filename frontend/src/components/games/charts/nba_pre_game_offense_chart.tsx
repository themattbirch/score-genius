// frontend/src/components/games/charts/nba_pre_game_offense_chart.tsx

import React from "react";
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
import { useTheme } from "@/contexts/theme_context";
import { NbaPreGameOffenseDataItem } from "@/types";
// Import specific types for tooltip formatter clarity from Recharts
import {
  ValueType,
  NameType,
  Payload as TooltipPayload, // <--- ALIAS Payload to TooltipPayload for clarity
} from "recharts/types/component/DefaultTooltipContent";
// Removed 'TooltipProps' import for formatter, as it's not needed there.
// import { TooltipProps } from 'recharts';

interface NbaPreGameOffenseChartProps {
  data?: NbaPreGameOffenseDataItem[];
}

const NbaPreGameOffenseChart: React.FC<NbaPreGameOffenseChartProps> = ({
  data = [],
}) => {
  const { theme } = useTheme();

  const textColorPrimary = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const textColorSecondary = theme === "dark" ? "#9ca3af" : "#475569";
  const panelBgColor = theme === "dark" ? "#161b22" : "#f8fafc";
  const subtleBorderColor =
    theme === "dark" ? "rgba(51, 65, 85, 0.6)" : "#e2e8f0";

  if (!data || data.length === 0) {
    return (
      <div className="text-center text-text-secondary py-12">
        No pre-game offensive stats available.
      </div>
    );
  }

  const HomeColor = "#4ade80";
  const AwayColor = "#60a5fa";

  return (
    <div className="p-2 rounded-lg bg-[var(--color-panel)] shadow-md h-[240px] w-full overflow-hidden">
      {" "}
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
          <CartesianGrid strokeDasharray="3 3" stroke={subtleBorderColor} />
          <XAxis
            dataKey="metric"
            stroke={textColorSecondary}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke={textColorSecondary}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value: ValueType) => {
              if (typeof value === "number") {
                if (value >= 0.5) {
                  return `${(value * 100).toFixed(0)}`;
                }
                return value.toFixed(1);
              }
              return String(value);
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: panelBgColor,
              border: `1px solid ${subtleBorderColor}`,
              borderRadius: "4px",
            }}
            labelStyle={{ color: textColorPrimary }}
            itemStyle={{ color: textColorSecondary }}
            // FIX: Correctly type the third argument as TooltipPayload
            // It represents the data of *one* item shown in the tooltip, not the Tooltip's props.
            formatter={(
              value: ValueType,
              name: NameType,
              item: TooltipPayload<ValueType, NameType>
            ) => {
              // Now 'item' contains the individual data entry for the tooltip.
              // We access the original data object through 'item.payload'.
              // Add checks to ensure item.payload exists and has the 'metric' property.
              const metric =
                item.payload &&
                typeof item.payload === "object" &&
                "metric" in item.payload
                  ? (item.payload as NbaPreGameOffenseDataItem).metric
                  : undefined;

              if (typeof value === "number") {
                if (metric === "OffRtg") {
                  return `${(value * 100).toFixed(1)}`;
                }
                return `${(value * 100).toFixed(1)}%`;
              }
              return `${value}`;
            }}
          />
          <Legend wrapperStyle={{ color: textColorPrimary }} />
          <Bar dataKey="Home" fill={HomeColor} name="Home Team" />
          <Bar dataKey="Away" fill={AwayColor} name="Away Team" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default NbaPreGameOffenseChart;
