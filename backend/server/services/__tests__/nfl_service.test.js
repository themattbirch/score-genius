import { mapScheduleRow } from "../nfl_service.js";

describe("mapScheduleRow", () => {
  it("maps a past game correctly", () => {
    const raw = {
      game_id: 123,
      game_date: "2025-01-05",
      status: "final",
      home_team_id: 1,
      away_team_id: 2,
      home_q1: 7,
      home_q2: 3,
      home_q3: 0,
      home_q4: 14,
      home_ot: null,
      away_q1: 0,
      away_q2: 7,
      away_q3: 10,
      away_q4: 7,
      away_ot: null,
      home_score: 24,
      away_score: 24,
    };
    expect(mapScheduleRow(raw, true)).toEqual({
      id: "123",
      gameDate: "2025-01-05",
      status: "final",
      homeTeamId: 1,
      awayTeamId: 2,
      dataType: "historical",
      finalHomeScore: 24,
      finalAwayScore: 24,
      homeQ: [7, 3, 0, 14],
      awayQ: [0, 7, 10, 7],
    });
  });

  it("maps a future game correctly", () => {
    const raw = {
      game_id: 456,
      game_date: "2025-11-20",
      status: "scheduled",
      home_team_id: 3,
      away_team_id: 4,
      scheduled_time: "2025-11-20T18:00:00Z",
      spread_clean: "-3.5 (-110)",
      total_clean: "48.5",
      predicted_home_score: 27.3,
      predicted_away_score: 24.1,
    };
    expect(mapScheduleRow(raw, false)).toEqual({
      id: "456",
      gameDate: "2025-11-20",
      status: "scheduled",
      homeTeamId: 3,
      awayTeamId: 4,
      dataType: "schedule",
      scheduledTimeUTC: "2025-11-20T18:00:00Z",
      spreadLine: -3.5,
      totalLine: 48.5,
      predictedHomeScore: 27.3,
      predictedAwayScore: 24.1,
    });
  });
});
