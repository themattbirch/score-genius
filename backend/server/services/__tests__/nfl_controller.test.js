import request from "supertest";
import express from "express";
import nflRoutes from "../../routes/nfl_routes.js";

// Mock all service methods
jest.mock("../../services/nfl_service.js", () => ({
  fetchNflScheduleData: jest.fn(),
  fetchNflSnapshotData: jest.fn(),
  fetchNflSnapshotsByIds: jest.fn(),
  fetchNflTeamSeasonFull: jest.fn(),
  fetchNflTeamSeasonRegOnly: jest.fn(),
  fetchNflSos: jest.fn(),
  fetchNflSrs: jest.fn(),
  fetchNflDashboardCards: jest.fn(),
  checkCronHealth: jest.fn(),
  validateTeamAgg: jest.fn(),
  buildCacheHeader: () => ({ "Cache-Control": "no-cache" }),
}));

function createApp() {
  const app = express();
  app.use(express.json());
  app.use("/api/v1/nfl", nflRoutes);
  // error handler
  app.use((err, _, res) =>
    res.status(err.status || 500).json({ message: err.message })
  );
  return app;
}

describe("GET /health/cron", () => {
  let app;
  beforeAll(() => {
    app = createApp();
    require("../../services/nfl_service.js").checkCronHealth.mockResolvedValue({
      lastRun: "2025-07-18T06:00:00Z",
      status: "ok",
    });
  });

  it("returns cron health data", async () => {
    const res = await request(app).get("/api/v1/nfl/health/cron").expect(200);
    expect(res.body).toEqual({ lastRun: "2025-07-18T06:00:00Z", status: "ok" });
  });
});

describe("GET /health/validate", () => {
  let app;
  beforeAll(() => {
    app = createApp();
    require("../../services/nfl_service.js").validateTeamAgg.mockResolvedValue({
      checks: [],
      errors: [],
    });
  });

  it("returns validation results", async () => {
    const res = await request(app)
      .get("/api/v1/nfl/health/validate")
      .expect(200);
    expect(res.body).toEqual({ checks: [], errors: [] });
  });
});
