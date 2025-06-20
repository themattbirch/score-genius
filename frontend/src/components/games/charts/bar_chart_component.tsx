// frontend/src/components/games/charts/bar_chart_component.tsx

// frontend/src/components/games/charts/bar_chart_component.tsx

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
import { Sport, BarChartData } from "@/types";
import {
  ValueType,
  NameType,
} from "recharts/types/component/DefaultTooltipContent";

interface BarChartComponentProps {
  data?: BarChartData[];
  sport?: Sport;
}

const BarChartComponent: React.FC<BarChartComponentProps> = ({
  data = [],
  sport = "NBA",
}) => {
  const { theme } = useTheme();

  // --- ★★★ START OF DEBUG CODE ★★★ ---
  // This will log the data this component receives every time it renders.
  console.log("BarChartComponent rendered with data:", data);
  // --- ★★★ END OF DEBUG CODE ★★★ ---

  const textColorPrimary = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const textColorSecondary = theme === "dark" ? "#9ca3af" : "#475569";
  const panelBgColor = theme === "dark" ? "#161b22" : "#f8fafc";
  const subtleBorderColor =
    theme === "dark" ? "rgba(51, 65, 85, 0.6)" : "#e2e8f0";

  if (!data || data.length === 0) {
    // This also helps us debug. If we see this message, the data array is empty.
    console.log("BarChartComponent determined data is empty or unavailable.");
    return (
      <div className="text-center text-text-secondary py-12">
        No bar chart data available.
      </div>
    );
  }

  const isMultiSeries = data.some(
    (d: BarChartData) => d.Home !== undefined && d.Away !== undefined
  );

  const HomeColor = "#4ade80";
  const AwayColor = "#60a5fa";
  const NBAColor = "#fbbf24";

  return (
    <div className="p-2 rounded-lg bg-[var(--color-panel)] shadow-md h-[240px] w-full overflow-hidden">
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
            dataKey="category"
            stroke={textColorSecondary}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke={textColorSecondary}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: panelBgColor,
              border: `1px solid ${subtleBorderColor}`,
              borderRadius: "4px",
            }}
            labelStyle={{ color: textColorPrimary }}
            itemStyle={{ color: textColorSecondary }}
            formatter={(value: ValueType, name: NameType) => {
              if (typeof value === "number") {
                return [value.toFixed(1), name];
              }
              return [value, name];
            }}
          />
          {isMultiSeries ? (
            <>
              <Legend wrapperStyle={{ color: textColorPrimary }} />
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

export default BarChartComponent;
