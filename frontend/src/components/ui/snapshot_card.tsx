// frontend/src/components/ui/SnapshotCard.tsx
import React, { useEffect, useState } from "react";

interface SnapshotCardProps { gameId: string; }

export default function SnapshotCard({ gameId }: SnapshotCardProps) {
  const [data, setData] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();

  // use our env-driven base URL
  const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/snapshots/${gameId}.json`)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((json) => setData(json))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [gameId, API_BASE]);

  if (loading) return <div>Loading snapshot…</div>;
  if (error)   return <div className="text-red-500">Error: {error}</div>;
  if (!data)  return <div>No snapshot available.</div>;

  return (
    <div className="grid grid-cols-2 gap-2 bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
      {Object.entries(data).map(([key, val]) => (
        <div key={key} className="flex justify-between">
          <span className="font-medium text-sm capitalize">
            {key.replace(/_/g, " ")}
          </span>
          <span className="text-sm">{val}</span>
        </div>
      ))}
    </div>
  );
}
