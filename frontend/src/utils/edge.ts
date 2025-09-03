// src/utils/edge.ts
import type { Sport, ValueEdge, EdgeTier } from "@/types";

/* ------------------------------------------------------------ */
/* Strategy toggle                                              */
/* ------------------------------------------------------------ */
const EDGE_STRATEGY: "EV" | "DZ" =
  import.meta.env?.VITE_EDGE_STRATEGY === "EV" ? "EV" : "DZ";

/* ------------------------------------------------------------ */
/* Odds & Probability Utilities                                  */
/* ------------------------------------------------------------ */
export function americanToProb(
  odds: number | string | null | undefined
): number | null {
  if (odds == null) return null;
  const n = Number(odds);
  if (!Number.isFinite(n) || n === 0) return null;
  return n > 0 ? 100 / (n + 100) : -n / (-n + 100);
}

export function devigTwoWay(
  pA: number | null,
  pB: number | null
): [number | null, number | null] {
  if (pA == null || pB == null) return [null, null];
  const s = pA + pB;
  if (s <= 0) return [null, null];
  return [pA / s, pB / s];
}

/** Standard normal CDF (Abramowitz & Stegun 7.1.26). */
export function stdNormCDF(x: number): number {
  const b1 = 0.31938153,
    b2 = -0.356563782,
    b3 = 1.781477937;
  const b4 = -1.821255978,
    b5 = 1.330274429,
    p = 0.2316419,
    c2 = 0.3989423;
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
      return 13.0;
    case "NBA":
      return 12.0;
    case "MLB":
      return 3.8;
    default:
      return 10.0;
  }
}
function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
function payoutFromAmerican(
  odds: number | string | null | undefined
): number | null {
  if (odds == null) return null;
  const n = Number(odds);
  if (!Number.isFinite(n) || n === 0) return null;
  return n > 0 ? n / 100 : 100 / Math.abs(n);
}

/* ------------------------------------------------------------ */
/* Sport-aware gating                                            */
/* ------------------------------------------------------------ */
type DeadZone = { edge: number; ev: number };
type ZTierMins = { any: number; medHigh: number };

function deadZoneFor(sport: Sport): DeadZone {
  switch (sport) {
    case "MLB":
      return { edge: 0.45, ev: 0.0028 };
    case "NFL":
      return { edge: 0.7, ev: 0.0032 };
    case "NBA":
      return { edge: 1.0, ev: 0.0045 };
    default:
      return { edge: 1.0, ev: 0.0045 };
  }
}
function zTierMinimums(sport: Sport): ZTierMins {
  switch (sport) {
    case "MLB":
      return { any: 0.22, medHigh: 0.32 };
    case "NFL":
      return { any: 0.34, medHigh: 0.4 };
    case "NBA":
      return { any: 0.42, medHigh: 0.48 };
    default:
      return { any: 0.42, medHigh: 0.48 };
  }
}
function isDeadZone(sport: Sport, edgePctAbs: number, ev: number): boolean {
  const t = deadZoneFor(sport);
  return edgePctAbs < t.edge || ev < t.ev;
}

/** Pick’em-aware confidence (penalize |z|≈0). */
function zForTiering(x: number): number {
  const a = Math.abs(x);
  const pickemBoost = Math.exp(-0.5 * x * x);
  return 0.75 * a + 0.25 * pickemBoost;
}

/* ------------------------------------------------------------ */
/* Spread awareness                                              */
/* ------------------------------------------------------------ */
function expectedMarginFromSpreadHome(
  spreadHomeLine?: number | null
): number | null {
  if (spreadHomeLine == null) return null;
  const s = Number(spreadHomeLine);
  if (!Number.isFinite(s)) return null;
  // Home -3 → expected home margin +3; Home +3 → expected margin -3
  return -s;
}
function spreadNearThreshold(sport: Sport): number {
  switch (sport) {
    case "NFL":
      return 0.8; // points
    case "NBA":
      return 1.5; // points
    case "MLB":
      return 0.35; // runs
    default:
      return 1.0;
  }
}
function applySpreadConsistencyBrake(
  tier: EdgeTier | null,
  sport: Sport,
  modelMargin: number,
  spreadHomeLine?: number | null
): EdgeTier | null {
  if (!tier) return null;
  const exp = expectedMarginFromSpreadHome(spreadHomeLine);
  if (exp == null) return tier;

  const delta = Math.abs(modelMargin - exp);
  const thr = spreadNearThreshold(sport);

  // Very near the spread → nuke
  if (delta <= thr * 0.33) return null;

  // Near the spread → single demotion (keep LOW alive on MLB)
  if (delta <= thr) {
    if (tier === "HIGH") return "MED";
    if (tier === "MED") return sport === "MLB" ? "LOW" : null;
    return sport === "MLB" ? "LOW" : null;
  }
  return tier;
}

/** Spread Swing Boost: promote when model materially beats the spread (capped). */
function spreadSwingBoostParams(sport: Sport) {
  switch (sport) {
    case "NFL":
      return { oneTier: 3.0, twoTier: 5.5, minEV: 0.007, minEdge: 0.7 };
    case "MLB":
      return { oneTier: 0.6, twoTier: 1.0, minEV: 0.006, minEdge: 0.6 };
    case "NBA":
      return { oneTier: 2.0, twoTier: 4.0, minEV: 0.01, minEdge: 0.8 };
    default:
      return { oneTier: 2.0, twoTier: 4.0, minEV: 0.009, minEdge: 0.7 };
  }
}
function promoteOne(t: EdgeTier | null): EdgeTier | null {
  if (!t) return null;
  if (t === "LOW") return "MED";
  if (t === "MED") return "HIGH";
  return "HIGH";
}
function applySpreadSwingBoost(
  tier: EdgeTier | null,
  sport: Sport,
  modelMargin: number,
  spreadHomeLine: number | null | undefined,
  ev: number,
  edgePctAbs: number,
  zTier: number
): EdgeTier | null {
  if (!tier) return null;
  const exp = expectedMarginFromSpreadHome(spreadHomeLine);
  if (exp == null) return tier;

  const zMin = zTierMinimums(sport);
  const t = dzThresholds(sport);
  const p = spreadSwingBoostParams(sport);
  const delta = Math.abs(modelMargin - exp);

  // Quality floor
  if (ev < p.minEV || edgePctAbs < p.minEdge || zTier < zMin.any) return tier;

  // Cap: boost can never create HIGH unless true HIGH gates are met
  const qualifiesHigh =
    ev >= t.highEV && edgePctAbs >= t.highΔ * 0.95 && zTier >= zMin.medHigh;

  if (delta >= p.oneTier) {
    const boosted = promoteOne(tier);
    if (boosted === "HIGH" && !qualifiesHigh) return "MED"; // cap at MED
    return boosted;
  }
  return tier;
}

/* ------------------------------------------------------------ */
/* Tiering                                                       */
/* ------------------------------------------------------------ */
function tierFromEV(
  sport: Sport,
  edgePct: number,
  ev: number,
  zTier: number
): EdgeTier | null {
  const e = Math.abs(edgePct);
  if (isDeadZone(sport, e, ev)) return null;

  const zMin = zTierMinimums(sport);
  if (zTier < zMin.any) return null;

  // EV-led gates; med/high require stronger z
  if (ev >= 0.028 && e >= 1.3 && zTier >= zMin.medHigh) return "HIGH";
  if (ev >= 0.009 && e >= 0.7 && zTier >= zMin.medHigh) return "MED";
  if (ev >= 0.0033 && e >= 0.42) return "LOW";
  return null;
}

/** Per-sport DZ thresholds (loosened MLB/NFL to surface MED/LOW; NBA explicit). */
function dzThresholds(sport: Sport) {
  switch (sport) {
    case "MLB":
      return {
        lowΔ: 0.5,
        medΔ: 1.1,
        highΔ: 2.8,
        lowZ: 0.1,
        medZ: 0.28,
        highZ: 0.68,
        lowEV: 0.0028,
        medEV: 0.0075,
        highEV: 0.017,
      };
    case "NFL":
      return {
        lowΔ: 0.6,
        medΔ: 1.2,
        highΔ: 3.5,
        lowZ: 0.28,
        medZ: 0.7,
        highZ: 1.2,
        lowEV: 0.0032,
        medEV: 0.008,
        highEV: 0.021,
      };
    case "NBA":
      return {
        lowΔ: 1.0,
        medΔ: 2.0,
        highΔ: 3.8,
        lowZ: 0.35,
        medZ: 0.85,
        highZ: 1.35,
        lowEV: 0.0055,
        medEV: 0.014,
        highEV: 0.028,
      };
    default:
      return {
        lowΔ: 1.0,
        medΔ: 2.0,
        highΔ: 3.8,
        lowZ: 0.35,
        medZ: 0.8,
        highZ: 1.2,
        lowEV: 0.005,
        medEV: 0.014,
        highEV: 0.028,
      };
  }
}

/** Pick’em tightening—explicit NBA branch, MLB/NFL as tuned. */
function tightenForPickem(sport: Sport, zAbs: number) {
  if (sport === "MLB") {
    if (zAbs < 0.16) return 1.03;
    if (zAbs < 0.28) return 1.01;
    return 1.0;
  }
  if (sport === "NFL") {
    if (zAbs < 0.24) return 1.15;
    if (zAbs < 0.36) return 1.07;
    return 1.0;
  }
  if (sport === "NBA") {
    if (zAbs < 0.25) return 1.25;
    if (zAbs < 0.4) return 1.12;
    return 1.0;
  }
  if (zAbs < 0.25) return 1.25;
  if (zAbs < 0.4) return 1.12;
  return 1.0;
}

function tierFromDeltaZ(
  sport: Sport,
  edgePct: number,
  zDisplay: number,
  zTier: number,
  ev: number
): EdgeTier | null {
  const e = Math.abs(edgePct);
  const z = Math.abs(zDisplay);
  if (isDeadZone(sport, e, ev)) return null;

  const zMin = zTierMinimums(sport);
  if (zTier < zMin.any) return null;

  const t = dzThresholds(sport);
  const tighten = tightenForPickem(sport, z);

  const highΔ = t.highΔ * tighten,
    medΔ = t.medΔ * tighten,
    lowΔ = t.lowΔ * tighten;
  const highZ = t.highZ * tighten,
    medZ = t.medZ * tighten,
    lowZ = t.lowZ * tighten;
  const highEV = t.highEV * tighten,
    medEV = t.medEV * tighten,
    lowEV = t.lowEV * tighten;

  // HIGH: all three + zTier ≥ medHigh (keeps HIGH selective)
  if (e >= highΔ && z >= highZ && ev >= highEV && zTier >= zMin.medHigh)
    return "HIGH";

  // MED/LOW: 2-of-3 strong with lenient "near" for MLB/NFL; NBA uses default near
  const nearFactor = sport === "MLB" ? 0.75 : sport === "NFL" ? 0.8 : 0.85;
  const near = (x: number, thr: number) => x >= nearFactor * thr;

  const medStrong =
    (e >= medΔ && z >= medZ) ||
    (e >= medΔ && ev >= medEV) ||
    (z >= medZ && ev >= medEV);
  const medNear = near(e, medΔ) || near(z, medZ) || near(ev, medEV);
  if (
    zTier >= zMin.medHigh &&
    ((e >= medΔ && z >= medZ && ev >= medEV) || (medStrong && medNear))
  )
    return "MED";

  const lowStrong =
    (e >= lowΔ && z >= lowZ) ||
    (e >= lowΔ && ev >= lowEV) ||
    (z >= lowZ && ev >= lowEV);
  const lowNear = near(e, lowΔ) || near(z, lowZ) || near(ev, lowEV);
  if (lowStrong || (lowStrong && lowNear)) return "LOW";

  // Escape hatch: MLB/NFL — good EV+Δ with slightly light z → LOW
  if (
    (sport === "MLB" || sport === "NFL") &&
    e >= Math.max(lowΔ, sport === "MLB" ? 0.75 : 0.8) &&
    ev >= Math.max(lowEV, sport === "MLB" ? 0.0065 : 0.0075) &&
    z >= lowZ * 0.72
  ) {
    return "LOW";
  }
  return null;
}

/* ------------------------------------------------------------ */
/* Guardrails                                                    */
/* ------------------------------------------------------------ */
function applyEvGuard(
  tier: EdgeTier | null,
  ev: number,
  sport: Sport
): EdgeTier | null {
  if (!tier) return null;
  const g =
    sport === "MLB"
      ? { high: 0.017, med: 0.0075, low: 0.0028 }
      : sport === "NFL"
      ? { high: 0.02, med: 0.009, low: 0.003 }
      : /* NBA & default */ { high: 0.028, med: 0.014, low: 0.0055 };
  if (tier === "HIGH" && ev < g.high) tier = "MED";
  if (tier === "MED" && ev < g.med) tier = "LOW";
  if (tier === "LOW" && ev < g.low) return null;
  return tier;
}

function applyKellyGuard(
  tier: EdgeTier | null,
  kelly: number,
  sport: Sport
): EdgeTier | null {
  if (!tier) return null;

  if (sport === "MLB") {
    const minKelly = { HIGH: 0.008, MED: 0.0035 } as const;
    if (tier === "HIGH" && kelly < minKelly.HIGH) return "MED";
    if (tier === "MED" && kelly < minKelly.MED) return "LOW";
    return tier; // LOW not enforced
  }
  if (sport === "NFL") {
    const minKelly = { HIGH: 0.014, MED: 0.007, LOW: 0.002 } as const;
    if (tier === "HIGH" && kelly < minKelly.HIGH) return "MED";
    if (tier === "MED" && kelly < minKelly.MED) return "LOW";
    if (tier === "LOW" && kelly < minKelly.LOW) return null;
    return tier;
  }

  // NBA & default
  const minKelly = { HIGH: 0.018, MED: 0.009, LOW: 0.003 } as const;
  if (tier === "HIGH" && kelly < minKelly.HIGH) return "MED";
  if (tier === "MED" && kelly < minKelly.MED) return "LOW";
  if (tier === "LOW" && kelly < minKelly.LOW) return null;
  return tier;
}

function applyPriceBandBrake(
  tier: EdgeTier | null,
  chosenAmericanOdds: number | string | null | undefined,
  edgePctAbs: number,
  sport: Sport
): EdgeTier | null {
  if (!tier) return null;
  if (chosenAmericanOdds == null) return tier;
  const n = Number(chosenAmericanOdds);
  if (!Number.isFinite(n)) return tier;

  if (sport === "MLB") {
    if (n < -330 && edgePctAbs < 1.4) {
      if (tier === "HIGH") return "MED";
      if (tier === "MED") return "LOW";
      return "LOW"; // keep LOW instead of nuking
    }
    return tier;
  }
  if (sport === "NFL") {
    if (n < -240 && edgePctAbs < 2.0) {
      if (tier === "HIGH") return "MED";
      if (tier === "MED") return "LOW";
      return null;
    }
    return tier;
  }
  // NBA & default
  if (n < -200 && edgePctAbs < 2.3) {
    if (tier === "HIGH") return "MED";
    if (tier === "MED") return "LOW";
    return null;
  }
  return tier;
}

/* ------------------------------------------------------------ */
/* Helpers                                                       */
/* ------------------------------------------------------------ */
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

  // spread (home) for spread checks/boosts
  spreadHomeLine?: number | null;
  spreadHomePrice?: number | string | null;
  spreadAwayPrice?: number | string | null;
};

export function computeBestEdge({
  sport,
  predHome,
  predAway,
  mlHome,
  mlAway,
  spreadHomeLine,
}: EdgeInputs): ValueEdge | null {
  if (predHome == null || predAway == null) return null;

  // Model margin & volatility
  const margin = Number(predHome) - Number(predAway);
  const sigma = marginSigma(sport);
  const x = margin / sigma;

  // Model win probabilities (clamped)
  const rawModelPH = stdNormCDF(x);
  const modelPH = clamp(rawModelPH, 0.01, 0.99);
  const modelPA = 1 - modelPH;

  // Market implied (vig-free) probabilities
  const mlPH_raw = americanToProb(mlHome ?? null);
  const mlPA_raw = americanToProb(mlAway ?? null);
  const [mktPH, mktPA] = devigTwoWay(mlPH_raw, mlPA_raw);
  if (mktPH == null || mktPA == null) return null;

  // Payouts
  const payH = payoutFromAmerican(mlHome ?? null);
  const payA = payoutFromAmerican(mlAway ?? null);
  if (payH == null || payA == null) return null;

  // Edge % (model − market), in percentage points
  const homeEdgePct = (modelPH - mktPH) * 100;
  const awayEdgePct = (modelPA - mktPA) * 100;

  // z values
  const zDisplay = Math.abs(x);
  const zTier = zForTiering(x);

  // EV per $1 stake
  const evHome = modelPH * payH - (1 - modelPH);
  const evAway = modelPA * payA - (1 - modelPA);

  // Spread delta for fallback & transparency
  const exp = expectedMarginFromSpreadHome(spreadHomeLine);
  const spreadDelta =
    exp == null ? Number.POSITIVE_INFINITY : Math.abs(margin - exp);

  type Candidate = {
    side: "HOME" | "AWAY";
    edgePct: number;
    modelProb: number;
    marketProb: number;
    zDisplay: number;
    zTier: number;
    ev: number;
    payout: number;
    american: number | string | null | undefined;
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
      payout: payH,
      american: mlHome,
    },
    {
      side: "AWAY",
      edgePct: awayEdgePct,
      modelProb: modelPA,
      marketProb: mktPA,
      zDisplay,
      zTier,
      ev: evAway,
      payout: payA,
      american: mlAway,
    },
  ];
  candidates.sort((a, b) => {
    if (b.ev !== a.ev) return b.ev - a.ev;
    if (Math.abs(b.edgePct) !== Math.abs(a.edgePct))
      return Math.abs(b.edgePct) - Math.abs(a.edgePct);
    return Math.abs(b.zDisplay) - Math.abs(a.zDisplay);
  });

  const best = candidates[0];

  // Must be positive EV and pass base z-tier gate
  const zMin = zTierMinimums(sport);
  if (best.ev <= 0 || best.zTier < zMin.any) return null;

  // Tiering (EV/DZ), prefer env strategy
  const tierEV = tierFromEV(sport, best.edgePct, best.ev, best.zTier);
  const tierDZ = tierFromDeltaZ(
    sport,
    best.edgePct,
    best.zDisplay,
    best.zTier,
    best.ev
  );
  const base = EDGE_STRATEGY === "EV" ? tierEV : tierDZ;
  const alt = EDGE_STRATEGY === "EV" ? tierDZ : tierEV;
  let tier = strongerTier(base, alt);

  // EV guardrails
  tier = applyEvGuard(tier, best.ev, sport);

  // Spread-consistency brake (softened near-market)
  tier = applySpreadConsistencyBrake(tier, sport, margin, spreadHomeLine);

  // Spread swing boost (capped at MED unless true HIGH gates are met)
  tier = applySpreadSwingBoost(
    tier,
    sport,
    margin,
    spreadHomeLine ?? null,
    best.ev,
    Math.abs(best.edgePct),
    best.zTier
  );

  // Price-band brake (demotes; MLB keeps LOW)
  tier = applyPriceBandBrake(
    tier,
    best.american,
    Math.abs(best.edgePct),
    sport
  );

  // Kelly guard (sport-aware; MLB not enforced for LOW)
  const p = best.modelProb,
    q = 1 - p,
    b = best.payout;
  const kelly = (b * p - q) / b;
  tier = applyKellyGuard(tier, kelly, sport);

  // Fallback: if still null but close to LOW gates & not too near spread → surface LOW
  if (!tier) {
    const t = dzThresholds(sport);
    const nearLow =
      best.ev >= t.lowEV * 0.95 &&
      Math.abs(best.edgePct) >= t.lowΔ * 0.95 &&
      best.zTier >= zMin.any * 0.9;
    const notNearSpread = spreadDelta > spreadNearThreshold(sport) * 0.6;
    if (nearLow && notNearSpread) tier = "LOW";
  }

  if (!tier) return null;

  const result: ValueEdge = {
    market: "ML",
    side: best.side,
    edgePct: best.edgePct,
    modelProb: best.modelProb,
    marketProb: best.marketProb,
    z: best.zDisplay,
    tier,
  };
  return result;
}

/* ------------------------------------------------------------ */
/* Format helpers                                               */
/* ------------------------------------------------------------ */
export function pct(x: number): string {
  return `${(x * 100).toFixed(0)}%`;
}
export function edgeFmt(x: number): string {
  const s = x >= 0 ? "+" : "";
  return `${s}${x.toFixed(1)}%`;
}
