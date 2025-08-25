// frontend/src/components/games/value_badge.tsx
import React from "react";
import type { ValueEdge } from "@/types";
import { edgeFmt, pct } from "@/utils/edge";

/**
 * Plain-info pill:
 *   "Edge: Low (Home)"
 *   "Edge: Medium (Away)"
 *   "Edge: High (Home)"
 *
 * No arrows, no market type. Looks like "No Edge" pill.
 */
export default function ValueBadge({ edge }: { edge: ValueEdge }) {
  if (!edge) return null;

  const tierLabel =
    edge.tier === "HIGH" ? "High" : edge.tier === "MED" ? "Medium" : "Low";
  const sideLabel = edge.side === "HOME" ? "Home" : "Away";

  // keep the helpful explanation for hover/focus (cheap “tooltip” via title)
  const title = `Model vs market.
Edge: ${edgeFmt(edge.edgePct)}
Model: ${pct(edge.modelProb)}  |  Market: ${pct(edge.marketProb)}
Z-score: ${edge.z.toFixed(2)}`;

  return (
    <span
      title={title}
      className="text-sm text-[var(--color-text-secondary)]"
      style={{ lineHeight: 1.1 }}
    >
      Edge:{" "}
      <span className="text-[var(--color-brand-green)] opacity-70">
        {tierLabel} ({sideLabel})
      </span>
    </span>
  );
}
