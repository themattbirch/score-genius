// frontend/src/components/snapshot_card.tsx
import React, { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  PieChart,
  Pie,
  Cell,
} from "recharts";

interface HeadlineStat {
  label: string;
  value: number | string;
}

interface ChartEntry {
  [key: string]: any;
}

interface SnapshotData {
  headlineStats: { label: string; value: number | string }[];
  barData: { name: string; value: number }[];
  radarData: { metric: string; value: number }[];
  pieData: { category: string; value: number; color: string }[];
}

interface SnapshotCardProps {
  gameId: string;
}

export default function SnapshotCard({ gameId }: SnapshotCardProps) {
  const [data, setData] = useState<SnapshotData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();

  const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/snapshots/${gameId}`)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((json: SnapshotData) => setData(json))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [gameId, API_BASE]);

  if (loading) return <div>Loading snapshot…</div>;
  if (error) return <div className="text-red-500">Error: {error}</div>;
  if (!data) return <div>No snapshot available.</div>;

  return (
    <div className="grid grid-cols-1 gap-4 bg-gray-50 dark:bg-gray-800 p-4 rounded-md">
      {/* Headline stats */}
      <div className="grid grid-cols-2 gap-2">
        {data.headlineStats.map((stat) => (
          <div key={stat.label} className="flex justify-between">
            <span className="font-medium capitalize">{stat.label}</span>
            <span className="font-semibold">{stat.value}</span>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-3 gap-4">
        <div className="h-40">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.barData}>
              <XAxis dataKey="name" />
              <YAxis />
              <RechartsTooltip />
              <Bar dataKey="value" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="h-40">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart
              cx="50%"
              cy="50%"
              outerRadius="80%"
              data={data.radarData}
            >
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <PolarRadiusAxis />
              <Radar name="Metrics" dataKey="value" fillOpacity={0.6} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="h-40">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data.pieData}
                dataKey="value"
                nameKey="category"
                outerRadius={60}
              >
                {data.pieData.map((entry, idx) => (
                  <Cell key={`cell-${idx}`} fill={entry.color} />
                ))}
              </Pie>
              <RechartsTooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
