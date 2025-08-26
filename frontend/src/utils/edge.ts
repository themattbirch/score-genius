// src/utils/edge.ts

import type { Sport, ValueEdge, EdgeTier } from "@/types";

/* ------------------------------------------------------------ */
/* Odds & Probability Utilities                                  */
/* ------------------------------------------------------------ */

/**
 * American odds → implied prob (0..1). Returns null on bad input.
 */
export function americanToProb(
  odds: number | string | null | undefined
): number | null {
  if (odds === null || odds === undefined) return null;
  const n = Number(odds);
  if (!Number.isFinite(n) || n === 0) return null;
  if (n > 0) return 100 / (n + 100);
  return -n / (-n + 100);
}

/**
 * Two-way de-vig: normalize raw probs so they sum to 1.
 * If either is missing, returns nulls.
 */
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

/**
 * Sport-specific margin standard deviation for a single game.
 * Tunable constants — adjust as you gather data.
 */
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

/** Clamp a number to a closed interval [lo, hi]. */
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
/* Edge Tiering (Moneyline MVP)                                  */
/* ------------------------------------------------------------ */
/**
 * Tighter thresholds to reduce noise:
 * - HIGH: edge ≥ 5.0% and z ≥ 0.90
 * - MED : edge ≥ 3.0% and z ≥ 0.60
 * - LOW : edge ≥ 1.0% and z ≥ 0.30
 */
export function tierFrom(edgePct: number, z: number): EdgeTier | null {
  const e = Math.abs(edgePct);
  const zz = Math.abs(z);
  if (e >= 5.0 && zz >= 0.9) return "HIGH";
  if (e >= 2.0 && zz >= 0.6) return "MED";
  if (e >= 0.5 && zz >= 0.3) return "LOW";
  return null;
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

  // spread fields may come through; ignored in ML MVP
  spreadHomeLine?: number | null;
  spreadHomePrice?: number | string | null;
  spreadAwayPrice?: number | string | null;
};

/**
 * Compute Moneyline Edge only. Returns the stronger positive-EV side if it
 * also meets tier thresholds; otherwise returns null (no badge).
 *
 * - Model prob: Φ(margin/σ), clamped to [0.01, 0.99] for stability.
 * - Market (vig-free) probs: devigTwoWay(americanToProb(H), americanToProb(A)).
 * - Selection: rank by EV per $1 stake, break ties with edgePct then z.
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

  // Model win probabilities (clamped for stability)
  const rawModelPH = stdNormCDF(margin / sigma);
  const modelPH = clamp(rawModelPH, 0.01, 0.99);
  const modelPA = 1 - modelPH;

  // Market implied (vig-free) probabilities
  const mlPH_raw = americanToProb(mlHome ?? null);
  const mlPA_raw = americanToProb(mlAway ?? null);
  const [mktPH, mktPA] = devigTwoWay(mlPH_raw, mlPA_raw);
  if (mktPH == null || mktPA == null) return null;

  // Payouts for EV calculation (use actual book odds, not vig-free)
  const payH = payoutFromAmerican(mlHome ?? null);
  const payA = payoutFromAmerican(mlAway ?? null);
  if (payH == null || payA == null) return null;

  // Edge % (model - market), in percentage points
  const homeEdgePct = (modelPH - mktPH) * 100;
  const awayEdgePct = (modelPA - mktPA) * 100;

  // Confidence proxy: standardized distance from coin-flip
  const zML = Math.abs(margin) / sigma;

  // Expected Value per $1 stake
  // EV = p * payout - (1 - p), where payout is net profit if the bet wins.
  const evHome = modelPH * payH - (1 - modelPH);
  const evAway = modelPA * payA - (1 - modelPA);

  type Candidate = {
    side: "HOME" | "AWAY";
    edgePct: number;
    modelProb: number;
    marketProb: number;
    z: number;
    ev: number;
  };

  const candidates: Candidate[] = [
    {
      side: "HOME",
      edgePct: homeEdgePct,
      modelProb: modelPH,
      marketProb: mktPH,
      z: zML,
      ev: evHome,
    },
    {
      side: "AWAY",
      edgePct: awayEdgePct,
      modelProb: modelPA,
      marketProb: mktPA,
      z: zML,
      ev: evAway,
    },
  ];

  // Choose the side with the highest EV; break ties by edgePct then z.
  candidates.sort((a, b) => {
    if (b.ev !== a.ev) return b.ev - a.ev;
    if (Math.abs(b.edgePct) !== Math.abs(a.edgePct))
      return Math.abs(b.edgePct) - Math.abs(a.edgePct);
    return Math.abs(b.z) - Math.abs(a.z);
  });

  const best = candidates[0];

  // Gate: must be positive EV AND meet tier thresholds to reduce noise
  if (best.ev <= 0) return null;

  const tier = tierFrom(best.edgePct, best.z);
  if (!tier) return null;

  const result: ValueEdge = {
    market: "ML",
    side: best.side,
    edgePct: best.edgePct,
    modelProb: best.modelProb,
    marketProb: best.marketProb,
    z: best.z,
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
