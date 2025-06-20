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

const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  const p0 = payload[0]?.payload;
  if (!p0) return null;
  const homeRaw = p0.home_raw;
  const awayRaw = p0.away_raw;
  const homeIdx = p0.home_idx.toFixed(1);
  const awayIdx = p0.away_idx.toFixed(1);
  return (
    <div className="rounded-md p-2 bg-[var(--color-panel)] border border-border-muted text-xs">
      <div className="font-medium mb-1 text-[var(--color-text-primary)]">
        {label}
      </div>
      <div className="flex flex-col gap-1 text-[var(--color-text-secondary)]">
        <div className="flex items-center gap-1">
          <span
            className="w-2 h-2 rounded-sm inline-block"
            style={{ background: "#4ade80" }}
          />
          Home&nbsp;•&nbsp;<span className="font-medium">{homeRaw}</span>
          <span className="opacity-70">&nbsp;(idx&nbsp;{homeIdx})</span>
        </div>
        <div className="flex items-center gap-1">
          <span
            className="w-2 h-2 rounded-sm inline-block"
            style={{ background: "#60a5fa" }}
          />
          Away&nbsp;•&nbsp;<span className="font-medium">{awayRaw}</span>
          <span className="opacity-70">&nbsp;(idx&nbsp;{awayIdx})</span>
        </div>
      </div>
    </div>
  );
};

const RadarChartComponent: React.FC<RadarChartComponentProps> = ({
  data = [],
}) => {
  const { theme } = useTheme();
  const textColorPrimary = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const textColorSecondary = theme === "dark" ? "#9ca3af" : "#475569";
  const subtleBorderColor = theme === "dark" ? "rgba(51,65,85,.6)" : "#e2e8f0";

  if (!data.length) {
    return (
      <div className="text-center text-text-secondary py-12">
        No radar chart data available.
      </div>
    );
  }

  return (
    <div className="p-2 rounded-lg bg-[var(--color-panel)] shadow-md h-[260px] w-full overflow-hidden">
      <ResponsiveContainer width="100%" height="100%">
        {/* --- ★★★ THE FIX ★★★ --- */}
        {/* We adjust cy and outerRadius to fix the layout permanently */}
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="65%">
          <PolarGrid stroke={subtleBorderColor} radialLines={false} />
          <PolarAngleAxis
            dataKey="metric"
            stroke={textColorSecondary}
            tick={{ fill: textColorPrimary, fontSize: 12 }}
          />
          <PolarRadiusAxis
            domain={[0, 100]}
            tickCount={5}
            tick={{ fill: textColorSecondary, fontSize: 8 }}
          />
          <Radar
            name="Home vs Away"
            dataKey="home_idx"
            stroke="#4ade80"
            strokeWidth={2}
            fill="#4ade80"
            fillOpacity={0.25}
            dot={false}
          />
          <Radar
            name="Away vs Home"
            dataKey="away_idx"
            stroke="#60a5fa"
            strokeWidth={2}
            fill="#60a5fa"
            fillOpacity={0.25}
            dot={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            verticalAlign="bottom"
            wrapperStyle={{ color: textColorPrimary, fontSize: 11 }}
            iconSize={10}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RadarChartComponent;
