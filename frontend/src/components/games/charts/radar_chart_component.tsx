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
import { useTheme } from "@/contexts/theme_context";
import { RadarChartData } from "@/types";

interface RadarChartComponentProps {
  data?: RadarChartData[];
}

/**
 * Radar Chart Component
 * Displays Home vs Away and Away vs Home deltas in advanced metrics,
 * with symmetric grid labels.
 */
const RadarChartComponent: React.FC<RadarChartComponentProps> = ({
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
        No radar chart data available.
      </div>
    );
  }

  // Calculate Home - Away delta, invert lower-is-better metrics
  const deltaData = data.map((d) => {
    let home = d.home_value ?? 0;
    let away = d.away_value ?? 0;
    if (d.metric === "DefRtg" || d.metric === "TOV%") {
      const maxRef = d.metric === "DefRtg" ? 230 : 0.27;
      home = maxRef - home;
      away = maxRef - away;
    }
    const delta = home - away;
    return { metric: d.metric, delta, inverse: -delta };
  });

  // Determine symmetric domain for radius axis
  const maxVal = Math.max(
    ...deltaData.flatMap((d) => [Math.abs(d.delta), Math.abs(d.inverse)])
  );
  const domain: [number, number] = [-maxVal, maxVal];

  return (
    <div className="p-2 rounded-lg bg-[var(--color-panel)] shadow-md h-[240px] w-full overflow-hidden">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={deltaData}>
          <PolarGrid stroke={subtleBorderColor} />
          <PolarAngleAxis
            dataKey="metric"
            stroke={textColorSecondary}
            tick={{ fill: textColorPrimary, fontSize: 10 }}
          />
          <PolarRadiusAxis
            angle={30}
            domain={domain}
            tickCount={5}
            tick={{ fill: textColorSecondary, fontSize: 8 }}
          />
          <Radar
            name="Home vs Away"
            dataKey="delta"
            stroke="#4ade80"
            fill="#4ade80"
            fillOpacity={0.6}
          />
          <Radar
            name="Away vs Home"
            dataKey="inverse"
            stroke="#60a5fa"
            fill="#60a5fa"
            fillOpacity={0.4}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: panelBgColor,
              border: `1px solid ${subtleBorderColor}`,
              borderRadius: "4px",
            }}
            labelStyle={{ color: textColorPrimary }}
            itemStyle={{ color: textColorSecondary }}
            formatter={(value: number, name: string, props) => {
              const metric = (props.payload as any).metric as string;
              const sign = value >= 0 ? "+" : "";
              const decimals = metric === "eFG%" || metric === "TOV%" ? 3 : 1;
              return [`${sign}${value.toFixed(decimals)}`, name];
            }}
          />
          <Legend wrapperStyle={{ color: textColorPrimary }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RadarChartComponent;
