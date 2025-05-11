// frontend/src/components/ui/nba_game_features_panel.tsx

import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, Tooltip } from "recharts";

export function NBAGameFeaturesPanel({ gameId }: { gameId: string }) {
  const [features, setFeatures] = useState<Record<string, number> | null>(null);
  useEffect(() => {
    fetch(`/api/games/${gameId}/features`)
      .then((r) => r.json())
      .then((d) => setFeatures(d.features));
  }, [gameId]);
  if (!features) return <div>Loading…</div>;
  // Example: render a radar or bar chart of a subset
  const data = Object.entries(features).map(([key, val]) => ({ key, val }));
  return (
    <BarChart width={600} height={300} data={data}>
      <XAxis dataKey="key" />
      <Tooltip />
      <Bar dataKey="val" />
    </BarChart>
  );
}
