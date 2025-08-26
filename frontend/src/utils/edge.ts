// src/utils/edge.ts

import type { Sport, ValueEdge, EdgeTier } from "@/types";

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

/* ------------------------------------------------------------ */
/* Tiering & Confidence                                          */
/* ------------------------------------------------------------ */
/**
 * Thresholds:
 * - HIGH: edge ≥ 5.0% and zTier ≥ 0.90
 * - MED : edge ≥ 2.0% and zTier ≥ 0.60
 * - LOW : edge ≥ 0.5% and zTier ≥ 0.30
 */
export function tierFrom(edgePct: number, zTier: number): EdgeTier | null {
  const e = Math.abs(edgePct);
  const zz = Math.abs(zTier);
  if (e >= 5.0 && zz >= 0.9) return "HIGH";
  if (e >= 2.0 && zz >= 0.6) return "MED";
  if (e >= 0.5 && zz >= 0.3) return "LOW";
  return null;
}

/** Pick’em-aware confidence used for tiering (balanced). */
function zForTiering(x: number): number {
  // x = margin / sigma
  const zDisplay = Math.abs(x);
  const pickemBoost = Math.exp(-0.5 * x * x); // ∈ (0,1], peaks at 1 when x≈0
  // Blended confidence: coin-flip gets a nudge, but doesn't jump tiers easily.
  return 0.6 * zDisplay + 0.4 * pickemBoost;
}

/** Light EV guardrails: keep MED/HIGH meaningful, don't over-filter LOW. */
function applyEvGuard(tier: EdgeTier | null, ev: number): EdgeTier | null {
  if (!tier) return null;
  if (tier === "HIGH" && ev < 0.012) tier = "MED"; // < +1.2% EV → MED
  if (tier === "MED" && ev < 0.005) tier = "LOW"; // < +0.5%  EV → LOW
  if (tier === "LOW" && ev < 0.001) return null; // < +0.1%  EV → No Edge
  return tier;
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
  const zTier = zForTiering(x); // used for gating/tiers

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

  // Base tier by (edgePct, zTier), then apply EV guard
  let tier = tierFrom(best.edgePct, best.zTier);
  tier = applyEvGuard(tier, best.ev);
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
