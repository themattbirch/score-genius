// backend/server/services/__tests__/nfl_sos_srs.test.js

/* ------------------------------------------------------------------ *
 * Hoisted mocks (Jest moves these above imports) to suppress noisy
 * console output & side-effects from the real modules during tests.
 * ------------------------------------------------------------------ */
jest.mock("../../utils/supabase_client.js", () => {
  const stub = { from: jest.fn() };
  return { __esModule: true, default: stub };
});

jest.mock("../../utils/cache.js", () => {
  const store = new Map();
  return {
    __esModule: true,
    default: {
      get: (k) => store.get(k),
      set: (k, v) => store.set(k, v),
      flushAll: () => store.clear(),
    },
  };
});

import supabase from "../../utils/supabase_client.js";
import cache from "../../utils/cache.js";
import { fetchNflSos, fetchNflSrs } from "../nfl_service.js";

/* ------------------------------------------------------------------ */
/* Helper: build a chain-able query-builder stub like supabase-js     */
/* ------------------------------------------------------------------ */
function makeQueryBuilder(returnPayload) {
  const qb = {};
  ["select", "eq", "in", "order"].forEach((m) => {
    qb[m] = jest.fn(() => qb);
  });
  // Make it await-able (`await qb`)
  qb.then = jest.fn((onFulfilled) => onFulfilled(returnPayload));
  return qb;
}

/* Default payload: empty but non-error; tests override data as needed */
const okResponse = { data: [], error: null, status: 200 };

/* Replace supabase.from so our stub is always returned               */
let builder;
supabase.from.mockImplementation(() => {
  builder = makeQueryBuilder(okResponse);
  return builder;
});

/* ------------------------------------------------------------------ */
/* Global hooks                                                       */
/* ------------------------------------------------------------------ */
beforeEach(() => {
  supabase.from.mockClear(); // reset spy counts
  cache.flushAll(); // clear LRU cache
  okResponse.data = []; // reset payload
});

/* ------------------------------------------------------------------ */
/* Tests for fetchNflSos                                              */
/* ------------------------------------------------------------------ */
describe("fetchNflSos", () => {
  it("fetches SOS data by season only", async () => {
    okResponse.data = [
      {
        season: 2021,
        team_id: 1,
        sos_wins: 10,
        sos_losses: 5,
        sos_ties: 0,
        sos_pct: 0.6667,
        sos_rank: 1,
      },
    ];

    const rows = await fetchNflSos({ season: 2021 });

    expect(supabase.from).toHaveBeenCalledWith("v_nfl_team_sos");
    expect(builder.select).toHaveBeenCalledWith("*");
    expect(builder.eq).toHaveBeenCalledWith("season", 2021);

    // Service normalises field names; assert key facts only.
    const r = rows[0];
    expect(r.season).toBe(2021);
    expect(r.teamId ?? r.team_id).toBe(1);
  });

  it("applies teamIds, conference & division filters", async () => {
    await fetchNflSos({
      season: 2022,
      teamIds: [2, 3],
      conference: "AFC",
      division: "East",
    });

    expect(supabase.from).toHaveBeenCalledWith("v_nfl_team_sos");
    expect(builder.eq).toHaveBeenCalledWith("season", 2022);
    expect(builder.in).toHaveBeenCalledWith("team_id", [2, 3]);
    expect(builder.eq).toHaveBeenCalledWith("conference", "AFC");
    expect(builder.eq).toHaveBeenCalledWith("division", "East");
  });

  it("returns cached data if present", async () => {
    // First call populates the cache
    okResponse.data = [{ team_id: 99, season: 2021 }];
    const prime = await fetchNflSos({ season: 2021 });

    supabase.from.mockClear(); // reset call count

    // Second call should hit cache, skipping supabase
    const cached = await fetchNflSos({ season: 2021 });
    expect(supabase.from).not.toHaveBeenCalled();
    expect(cached).toEqual(prime);
  });
});

/* ------------------------------------------------------------------ */
/* Tests for fetchNflSrs                                              */
/* ------------------------------------------------------------------ */
describe("fetchNflSrs", () => {
  it("fetches SRS data by season only", async () => {
    okResponse.data = [
      {
        team_id: 1,
        season: 2021,
        srs_lite: 4.2,
      },
    ];

    const rows = await fetchNflSrs({ season: 2021 });

    expect(supabase.from).toHaveBeenCalledWith("v_nfl_team_srs_lite");
    expect(builder.select).toHaveBeenCalledWith("*");
    expect(builder.eq).toHaveBeenCalledWith("season", 2021);

    const r = rows[0];
    expect(r.season).toBe(2021);
    expect(r.teamId ?? r.team_id).toBe(1);
  });

  it("applies teamIds, conference & division filters", async () => {
    await fetchNflSrs({
      season: 2021,
      teamIds: [1],
      conference: "NFC",
      division: "East",
    });

    expect(builder.in).toHaveBeenCalledWith("team_id", [1]);
    expect(builder.eq).toHaveBeenCalledWith("conference", "NFC");
    expect(builder.eq).toHaveBeenCalledWith("division", "East");
  });

  it("returns cached data if present", async () => {
    // Prime the cache
    okResponse.data = [{ team_id: 7, season: 2021 }];
    const prime = await fetchNflSrs({ season: 2021 });

    supabase.from.mockClear();

    const cached = await fetchNflSrs({ season: 2021 });
    expect(supabase.from).not.toHaveBeenCalled();
    expect(cached).toEqual(prime);
  });
});
