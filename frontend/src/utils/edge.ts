// src/utils/edge.ts

import type { Sport, ValueEdge, EdgeTier } from "@/types";

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

/** Edge tier thresholds */
export function tierFrom(edgePct: number, z: number): EdgeTier | null {
  const e = Math.abs(edgePct);
  const zz = Math.abs(z);
  if (e >= 5 && zz >= 0.75) return "HIGH";
  if (e >= 3 && zz >= 0.5) return "MED";
  if (e >= 0.5 && zz >= 0.25) return "LOW";
  return null;
}

type EdgeInputs = {
  sport: Sport;
  predHome: number | null | undefined;
  predAway: number | null | undefined;

  // moneyline odds
  mlHome?: number | string | null;
  mlAway?: number | string | null;

  // spread: home line (home favored is negative) + prices
  spreadHomeLine?: number | null;
  spreadHomePrice?: number | string | null;
  spreadAwayPrice?: number | string | null;
};

/**
 * Compute the stronger of ML or Spread edges (if any). Returns null if no value.
 */
export function computeBestEdge({
  sport,
  predHome,
  predAway,
  mlHome,
  mlAway,
  spreadHomeLine,
  spreadHomePrice,
  spreadAwayPrice,
}: EdgeInputs): ValueEdge | null {
  if (predHome == null || predAway == null) return null;
  const margin = Number(predHome) - Number(predAway);
  const sigma = marginSigma(sport);

  // ========= Moneyline =========
  let edgeML: ValueEdge | null = null;
  const mlPH = americanToProb(mlHome ?? null);
  const mlPA = americanToProb(mlAway ?? null);
  const [mktPH, mktPA] = devigTwoWay(mlPH, mlPA);

  if (mktPH != null && mktPA != null) {
    const modelPH = stdNormCDF(margin / sigma); // P(home wins)
    const modelPA = 1 - modelPH; // P(away wins)

    const homeEdgePct = (modelPH - mktPH) * 100;
    const awayEdgePct = (modelPA - mktPA) * 100;

    const zML = Math.abs(margin) / sigma; // distance from coin-flip

    // choose the better positive edge
    const isHome = homeEdgePct > awayEdgePct;
    const chosenPct = isHome ? homeEdgePct : awayEdgePct;
    const tier = tierFrom(chosenPct, zML);
    if (tier) {
      edgeML = {
        market: "ML",
        side: isHome ? "HOME" : "AWAY",
        edgePct: chosenPct,
        modelProb: isHome ? modelPH : modelPA,
        marketProb: isHome ? mktPH : mktPA,
        z: zML,
        tier,
      };
    }
  }

  // ========= Spread =========
  let edgeSpread: ValueEdge | null = null;

  const prH = americanToProb(spreadHomePrice ?? null);
  const prA = americanToProb(spreadAwayPrice ?? null);
  const [mktCoverH, mktCoverA] = devigTwoWay(prH, prA);

  if (
    spreadHomeLine != null &&
    mktCoverH != null &&
    mktCoverA != null &&
    Number.isFinite(spreadHomeLine)
  ) {
    // P(home covers): margin - line > 0  ⇒ Φ((margin - line)/σ)
    const modelCoverH = stdNormCDF((margin - Number(spreadHomeLine)) / sigma);
    const modelCoverA = 1 - modelCoverH; // at same line

    const homeEdgePct = (modelCoverH - mktCoverH) * 100;
    const awayEdgePct = (modelCoverA - mktCoverA) * 100;

    const zSpr = Math.abs(margin - Number(spreadHomeLine)) / sigma;

    const isHome = homeEdgePct > awayEdgePct;
    const chosenPct = isHome ? homeEdgePct : awayEdgePct;
    const tier = tierFrom(chosenPct, zSpr);
    if (tier) {
      edgeSpread = {
        market: "SPREAD",
        side: isHome ? "HOME" : "AWAY",
        edgePct: chosenPct,
        modelProb: isHome ? modelCoverH : modelCoverA,
        marketProb: isHome ? mktCoverH : mktCoverA,
        z: zSpr,
        tier,
      };
    }
  }

  // ========= Choose stronger (if any) =========
  if (edgeML && edgeSpread) {
    return Math.abs(edgeML.edgePct) >= Math.abs(edgeSpread.edgePct)
      ? edgeML
      : edgeSpread;
  }
  return edgeML ?? edgeSpread ?? null;
}

/** Format helpers for tooltips */
export function pct(x: number): string {
  return `${(x * 100).toFixed(0)}%`;
}
export function edgeFmt(x: number): string {
  const s = x >= 0 ? "+" : "";
  return `${s}${x.toFixed(1)}%`;
}
