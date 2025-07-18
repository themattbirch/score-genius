import fs from "fs";
import path from "path";
import yaml from "js-yaml";
import swaggerUI from "swagger-ui-express";
import { fileURLToPath } from "url";

// ESM-safe __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load the YAML file
const specPath = path.join(__dirname, "openapi.yaml");
const swaggerDocument = yaml.load(fs.readFileSync(specPath, "utf8"));

export function setupSwagger(app) {
  app.use(
    "/api-docs",
    swaggerUI.serve,
    swaggerUI.setup(swaggerDocument, {
      explorer: true,
      swaggerOptions: { docExpansion: "none" },
    })
  );
  // Raw JSON endpoint for the spec
  app.get("/api-docs.json", (_req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.send(swaggerDocument);
  });
}
