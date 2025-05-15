// frontend/src/components/ui/nba_game_features.panel.tsx

import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, Tooltip } from "recharts";
import { apiFetch } from "@/api/client";

export function NBAGameFeaturesPanel({ gameId }: { gameId: string }) {
  const [features, setFeatures] = useState<Record<string, number> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    apiFetch(`/api/v1/nba/games/${gameId}/features`)
      .then((r) => {
        if (!r.ok) throw new Error(`Bad response ${r.status}`);
        return r.json();
      })
      .then((d) => !cancelled && setFeatures(d.features ?? d.data?.features))
      .catch((e) => !cancelled && setError(e.message));

    return () => {
      cancelled = true;
    };
  }, [gameId]);

  if (error) return <div className="text-red-500">{error}</div>;
  if (!features) return <div>Loading…</div>;

  const data = Object.entries(features).map(([key, val]) => ({ key, val }));

  return (
    <BarChart width={600} height={300} data={data}>
      <XAxis dataKey="key" />
      <Tooltip />
      <Bar dataKey="val" />
    </BarChart>
  );
}