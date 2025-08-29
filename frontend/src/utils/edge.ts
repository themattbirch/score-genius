// src/utils/edge.ts

import type { Sport, ValueEdge, EdgeTier } from "@/types";
/* ------------------------------------------------------------ */
/* Strategy toggle                                              */
/* ------------------------------------------------------------ */
// "EV"  → Expected-value dominant (more MEDIUM, fewer LOW)
// "DZ"  → Delta+Z combined (more NO EDGE, tight bands)
const EDGE_STRATEGY: "EV" | "DZ" =
  import.meta.env?.VITE_EDGE_STRATEGY === "EV" ? "EV" : "DZ";

/* ------------------------------------------------------------ */
/* Odds & Probability Utilities                                  */
/* ------------------------------------------------------------ */

/** American odds → implied prob (0..1). Returns null on bad input. */
export function americanToProb(
  odds: number | string | null | undefined
): number | null {
  if (odds === null || odds === undefined) return null;
  const n = Number(odds);
  if (!Number.isFinite(n) || n === 0) return null;
  if (n > 0) return 100 / (n + 100);
  return -n / (-n + 100);
}

/** Two-way de-vig: normalize raw probs so they sum to 1. */
export function devigTwoWay(
  pA: number | null,
  pB: number | null
): [number | null, number | null] {
  if (pA == null || pB == null) return [null, null];
  const s = pA + pB;
  if (s <= 0) return [null, null];
  return [pA / s, pB / s];
}

/** Standard normal CDF using numerical approximation. */
export function stdNormCDF(x: number): number {
  // Abramowitz & Stegun formula 7.1.26
  const b1 = 0.31938153;
  const b2 = -0.356563782;
  const b3 = 1.781477937;
  const b4 = -1.821255978;
  const b5 = 1.330274429;
  const p = 0.2316419;
  const c2 = 0.3989423;

  const a = Math.abs(x);
  const t = 1.0 / (1.0 + a * p);
  const b = c2 * Math.exp((-x * x) / 2.0);
  let n = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t;
  n = 1.0 - b * n;
  return x < 0.0 ? 1.0 - n : n;
}

/* ------------------------------------------------------------ */
/* Model Controls                                                */
/* ------------------------------------------------------------ */

export function marginSigma(sport: Sport): number {
  switch (sport) {
    case "NFL":
      return 13.0; // pts
    case "NBA":
      return 12.0; // pts
    case "MLB":
      return 3.8; // runs
    default:
      return 10.0;
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/** Net profit per $1 stake from American odds (e.g., +150 → 1.5, -120 → 0.8333). */
function payoutFromAmerican(
  odds: number | string | null | undefined
): number | null {
  if (odds === null || odds === undefined) return null;
  const n = Number(odds);
  if (!Number.isFinite(n) || n === 0) return null;
  return n > 0 ? n / 100 : 100 / Math.abs(n);
}

function deadZoneFor(sport: Sport) {
  // Sport-aware micro-filter. NFL gets a slightly looser EV floor.
  switch (sport) {
    case "NFL":
      return { edge: 0.5, ev: 0.0025 };
    case "NBA":
      return { edge: 0.55, ev: 0.0028 };
    case "MLB":
      return { edge: 0.6, ev: 0.003 };
    default:
      return { edge: 0.55, ev: 0.0028 };
  }
}

function isDeadZone(sport: Sport, edgePctAbs: number, ev: number): boolean {
  const t = deadZoneFor(sport);
  return edgePctAbs < t.edge || ev < t.ev;
}

/** Pick’em-aware confidence used for tiering (stricter near 0). */
function zForTiering(x: number): number {
  // Make near-pick’em less likely to pass gates by reducing the boost weight.
  const a = Math.abs(x);
  const pickemBoost = Math.exp(-0.5 * x * x); // (0,1], peaks at 1 when x≈0
  return 0.75 * a + 0.25 * pickemBoost; // previously 0.6/0.4
}

/** Tiering v2 – EV-dominant (promotes MED, trims LOW). */
function tierFromEV(
  sport: Sport,
  edgePct: number,
  ev: number
): EdgeTier | null {
  const e = Math.abs(edgePct);
  if (isDeadZone(sport, e, ev)) return null;
  // EV per $1 stake thresholds (sport-agnostic defaults)
  if (ev >= 0.04 && e >= 1.8) return "HIGH"; // ≥ +4.0% EV and solid delta
  if (ev >= 0.018 && e >= 1.0) return "MED"; // ≥ +1.8% EV
  // Low kept meaningful; many former LOWs fall to "No Edge"
  if (ev >= 0.006 && e >= 0.7) return "LOW"; // ≥ +0.6% EV
  return null;
}

/** Tiering v3 – Delta + Z (more No Edge, keeps High rare). */
function dzThresholds(sport: Sport) {
  // Sport-aware Z gates (z = |margin| / sigma). Deltas/EV are consistent.
  switch (sport) {
    case "MLB":
      return {
        lowZ: 0.18,
        medZ: 0.4,
        highZ: 0.6, // ≈0.7r, 1.5r, 2.3r
        lowΔ: 0.8,
        medΔ: 1.8,
        highΔ: 3.2, // % points vs market
        lowEV: 0.003,
        medEV: 0.012,
        highEV: 0.02,
      };
    case "NBA":
      return {
        lowZ: 0.3,
        medZ: 0.7,
        highZ: 1.2, // ≈3.6, 8.4, 14.4 pts
        lowΔ: 0.8,
        medΔ: 1.8,
        highΔ: 3.2,
        lowEV: 0.003,
        medEV: 0.012,
        highEV: 0.02,
      };
    case "NFL":
      return {
        lowZ: 0.35,
        medZ: 0.8,
        highZ: 1.3, // ≈4.6, 10.4, 16.9 pts
        lowΔ: 0.8,
        medΔ: 1.8,
        highΔ: 3.2,
        lowEV: 0.003,
        medEV: 0.012,
        highEV: 0.02,
      };
    default:
      return {
        lowZ: 0.3,
        medZ: 0.7,
        highZ: 1.1,
        lowΔ: 0.8,
        medΔ: 1.8,
        highΔ: 3.2,
        lowEV: 0.003,
        medEV: 0.012,
        highEV: 0.02,
      };
  }
}

function tierFromDeltaZ(
  sport: Sport,
  edgePct: number,
  zDisplay: number,
  ev: number
): EdgeTier | null {
  const e = Math.abs(edgePct);
  const z = Math.abs(zDisplay);
  if (isDeadZone(sport, e, ev)) return null;
  const t = dzThresholds(sport);
  // Require real separation from pick’em and sufficient delta/EV
  if (sport === "MLB") {
    if (e >= 3.2 && z >= 0.6 && ev >= 0.02) return "HIGH";
    if (e >= 1.8 && z >= 0.4 && ev >= 0.012) return "MED";
    if (e >= 0.8 && z >= 0.18 && ev >= 0.003) return "LOW";
  } else if (sport === "NBA") {
    if (e >= 3.2 && z >= 1.2 && ev >= 0.02) return "HIGH";
    if (e >= 1.8 && z >= 0.7 && ev >= 0.012) return "MED";
    if (e >= 0.8 && z >= 0.3 && ev >= 0.003) return "LOW";
  } else {
    // NFL + default
    if (e >= 3.2 && z >= 1.1 && ev >= 0.02) return "HIGH";
    if (e >= 1.6 && z >= 0.7 && ev >= 0.01) return "MED";
    if (e >= 0.7 && z >= 0.25 && ev >= 0.0025) return "LOW";
  }
  return null;
}

/** EV guardrails (sport-aware): keep tiers meaningful across books/sports. */
function applyEvGuard(
  tier: EdgeTier | null,
  ev: number,
  sport: Sport
): EdgeTier | null {
  if (!tier) return null;
  const g =
    sport === "NFL"
      ? { high: 0.018, med: 0.008, low: 0.0025 }
      : { high: 0.02, med: 0.01, low: 0.003 };
  if (tier === "HIGH" && ev < g.high) tier = "MED";
  if (tier === "MED" && ev < g.med) tier = "LOW";
  if (tier === "LOW" && ev < g.low) return null;
  return tier;
}

/* Helper: choose the stronger of two tiers */
const TIER_RANK: Record<Exclude<EdgeTier, never>, number> = {
  LOW: 1,
  MED: 2,
  HIGH: 3,
} as const;

function strongerTier(a: EdgeTier | null, b: EdgeTier | null): EdgeTier | null {
  if (!a) return b ?? null;
  if (!b) return a ?? null;
  return TIER_RANK[a] >= TIER_RANK[b] ? a : b;
}

/* ------------------------------------------------------------ */
/* Moneyline-Only Edge                                           */
/* ------------------------------------------------------------ */

type EdgeInputs = {
  sport: Sport;
  predHome: number | null | undefined;
  predAway: number | null | undefined;

  // moneyline odds
  mlHome?: number | string | null;
  mlAway?: number | string | null;

  // (ignored for ML MVP)
  spreadHomeLine?: number | null;
  spreadHomePrice?: number | string | null;
  spreadAwayPrice?: number | string | null;
};

/**
 * Compute Moneyline Edge only. Returns the stronger positive-EV side if it
 * meets tier thresholds (with EV guardrails); otherwise null (no badge).
 */
export function computeBestEdge({
  sport,
  predHome,
  predAway,
  mlHome,
  mlAway,
}: EdgeInputs): ValueEdge | null {
  if (predHome == null || predAway == null) return null;

  // Model margin & volatility
  const margin = Number(predHome) - Number(predAway);
  const sigma = marginSigma(sport);
  const x = margin / sigma;

  // Model win probabilities (clamped for stability)
  const rawModelPH = stdNormCDF(x);
  const modelPH = clamp(rawModelPH, 0.01, 0.99);
  const modelPA = 1 - modelPH;

  // Market implied (vig-free) probabilities
  const mlPH_raw = americanToProb(mlHome ?? null);
  const mlPA_raw = americanToProb(mlAway ?? null);
  const [mktPH, mktPA] = devigTwoWay(mlPH_raw, mlPA_raw);
  if (mktPH == null || mktPA == null) return null;

  // Payouts for EV calculation (use actual book odds)
  const payH = payoutFromAmerican(mlHome ?? null);
  const payA = payoutFromAmerican(mlAway ?? null);
  if (payH == null || payA == null) return null;

  // Edge % (model − market), in percentage points
  const homeEdgePct = (modelPH - mktPH) * 100;
  const awayEdgePct = (modelPA - mktPA) * 100;

  // z values
  const zDisplay = Math.abs(x); // shown in tooltip
  const zTier = zForTiering(x); // blended confidence for gating

  // Expected Value per $1 stake: EV = p * payout − (1 − p)
  const evHome = modelPH * payH - (1 - modelPH);
  const evAway = modelPA * payA - (1 - modelPA);

  type Candidate = {
    side: "HOME" | "AWAY";
    edgePct: number;
    modelProb: number;
    marketProb: number;
    zDisplay: number;
    zTier: number;
    ev: number;
  };

  const candidates: Candidate[] = [
    {
      side: "HOME",
      edgePct: homeEdgePct,
      modelProb: modelPH,
      marketProb: mktPH,
      zDisplay,
      zTier,
      ev: evHome,
    },
    {
      side: "AWAY",
      edgePct: awayEdgePct,
      modelProb: modelPA,
      marketProb: mktPA,
      zDisplay,
      zTier,
      ev: evAway,
    },
  ];

  // Choose the side with the highest EV; break ties by edgePct then zDisplay.
  candidates.sort((a, b) => {
    if (b.ev !== a.ev) return b.ev - a.ev;
    if (Math.abs(b.edgePct) !== Math.abs(a.edgePct))
      return Math.abs(b.edgePct) - Math.abs(a.edgePct);
    return Math.abs(b.zDisplay) - Math.abs(a.zDisplay);
  });

  const best = candidates[0];

  // Must be positive EV to even consider
  if (best.ev <= 0) return null;

  // Tiering: compute both, prefer env strategy, but fall back to the stronger.
  const tierEV = tierFromEV(sport, best.edgePct, best.ev);
  const tierDZ = tierFromDeltaZ(sport, best.edgePct, best.zDisplay, best.ev);
  const base = EDGE_STRATEGY === "EV" ? tierEV : tierDZ;
  const alt = EDGE_STRATEGY === "EV" ? tierDZ : tierEV;
  let tier = strongerTier(base, alt);
  tier = applyEvGuard(tier, best.ev, sport);
  if (!tier) return null;

  const result: ValueEdge = {
    market: "ML",
    side: best.side,
    edgePct: best.edgePct,
    modelProb: best.modelProb,
    marketProb: best.marketProb,
    z: best.zDisplay, // tooltip shows the pure standardized distance
    tier,
  };

  return result;
}

/* ------------------------------------------------------------ */
/* Format helpers for tooltips                                   */
/* ------------------------------------------------------------ */

export function pct(x: number): string {
  return `${(x * 100).toFixed(0)}%`;
}

export function edgeFmt(x: number): string {
  const s = x >= 0 ? "+" : "";
  return `${s}${x.toFixed(1)}%`;
}
