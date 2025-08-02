//backend/server/docs/swagger.js

import fs from "fs";
import path from "path";
import yaml from "js-yaml";
import swaggerUI from "swagger-ui-express";
import { fileURLToPath } from "url";

// ESM-safe __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Helper to safely load YAML if exists
function loadYamlSpec(specPath) {
  try {
    if (fs.existsSync(specPath)) {
      const contents = fs.readFileSync(specPath, "utf8");
      return yaml.load(contents);
    }
  } catch (err) {
    console.warn("Failed to load openapi.yaml:", err);
  }
  return null;
}

// Minimal definitions you can augment in openapi.yaml later
const nflTeamSummarySchema = {
  type: "object",
  properties: {
    season: { type: "integer", example: 2024 },
    retrieved: { type: "integer", example: 32 },
    data: {
      type: "array",
      items: {
        type: "object",
        properties: {
          teamId: { type: "string", example: "3" },
          teamName: { type: "string", example: "New England Patriots" },
          season: { type: "integer", example: 2024 },
          srs: { type: "number", example: -7.53 },
          sos: { type: "number", example: 1.2 },
          sosRank: { type: "integer", example: 22 },
          winPct: { type: "number", example: 0.235 },
          pythagoreanWinPct: { type: "number", example: 0.295 },
          avgThirdDownPct: { type: "number", example: 0.336 },
          avgRedZonePct: { type: "number", example: 0.447 },
          avgYardsPerDrive: { type: "number", example: 60.46 },
          avgTurnoversPerGame: { type: "number", example: 1.35 },
          avgTimeOfPossession: { type: "string", example: "29:20" },
        },
      },
    },
  },
};

const nflAdvancedSchema = {
  type: "object",
  properties: {
    season: { type: "integer", example: 2024 },
    retrieved: { type: "integer", example: 32 },
    data: {
      type: "array",
      items: {
        type: "object",
        properties: {
          season: { type: "integer", example: 2024 },
          team_id: { type: "integer", example: 11 },
          team_name: { type: "string", example: "Arizona Cardinals" },
          avg_third_down_pct: { type: "number", example: 0.423 },
          avg_red_zone_pct: { type: "number", example: 0.576 },
          avg_yards_per_drive: { type: "number", example: 60.88 },
          avg_turnovers_per_game: { type: "number", example: 1.17 },
          pythagorean_win_pct: { type: "number", example: 0.5319 },
          avg_time_of_possession: { type: "string", example: "29:45" },
        },
      },
    },
  },
};

export function setupSwagger(app) {
  const specPath = path.join(__dirname, "openapi.yaml");
  let swaggerDocument = loadYamlSpec(specPath);

  if (!swaggerDocument) {
    // Build a fallback swagger document if YAML missing
    swaggerDocument = {
      openapi: "3.1.0",
      info: {
        title: "Score Genius API",
        version: "1.0.0",
        description:
          "Fallback OpenAPI spec including key NFL team stats endpoints.",
      },
      servers: [
        {
          url: "/api/v1",
        },
      ],
      paths: {
        "/nfl/team-stats/summary": {
          get: {
            summary: "NFL team summary (SRS, SoS, Pythagorean, win % etc.)",
            description:
              "Consolidated team summary for NFL including advanced rating metrics and efficiency aggregates.",
            parameters: [
              {
                name: "season",
                in: "query",
                required: true,
                schema: { type: "integer", example: 2024 },
                description: "Season year (e.g., 2024).",
              },
            ],
            responses: {
              200: {
                description: "Team summary payload.",
                content: {
                  "application/json": {
                    schema: nflTeamSummarySchema,
                  },
                },
              },
            },
          },
        },
        "/nfl/team-stats/advanced": {
          get: {
            summary: "NFL advanced efficiency stats",
            description:
              "Efficiency-focused advanced stats for NFL teams (third down %, red zone %, yards per drive, etc.).",
            parameters: [
              {
                name: "season",
                in: "query",
                required: true,
                schema: { type: "integer", example: 2024 },
                description: "Season year (e.g., 2024).",
              },
            ],
            responses: {
              200: {
                description: "Advanced team stats payload.",
                content: {
                  "application/json": {
                    schema: nflAdvancedSchema,
                  },
                },
              },
            },
          },
        },
        "/nfl/team-stats": {
          get: {
            summary: "NFL generic team stats",
            description: "Fallback/generic team stats endpoint.",
            parameters: [
              {
                name: "season",
                in: "query",
                required: true,
                schema: { type: "integer", example: 2024 },
                description: "Season year.",
              },
            ],
            responses: {
              200: {
                description: "Generic team stats result.",
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        season: { type: "integer" },
                        retrieved: { type: "integer" },
                        data: { type: "array", items: { type: "object" } },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
      components: {
        schemas: {
          NflTeamSummary: nflTeamSummarySchema,
          NflAdvanced: nflAdvancedSchema,
        },
      },
    };
  } else {
    // If YAML exists, we can optionally ensure the key endpoints are present/up to date
    // (Could merge or validate here if desired)
  }

  // Swagger UI mount
  app.use(
    "/api-docs",
    swaggerUI.serve,
    swaggerUI.setup(swaggerDocument, {
      explorer: true,
      swaggerOptions: { docExpansion: "none" },
    })
  );

  // Raw JSON spec
  app.get("/api-docs.json", (_req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.send(swaggerDocument);
  });
}
