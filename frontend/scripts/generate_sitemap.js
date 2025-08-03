// frontend/scripts/generate_sitemap.js
import { SitemapStream, streamToPromise } from "sitemap";
import { writeFileSync, readdirSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function buildSitemap() {
  const distDir = resolve(__dirname, "../dist");
  const backendStatic = resolve(__dirname, "../../backend/server/static");
  const hostname = "https://scoregenius.io";

  const pages = readdirSync(distDir)
    .filter((f) => f.endsWith(".html") && f !== "app.html")
    .map((f) => (f === "index.html" ? "/" : `/${f.replace(".html", "")}`));

  const smStream = new SitemapStream({ hostname });
  pages.forEach((url) => smStream.write({ url, changefreq: "weekly" }));
  smStream.end();
  const xml = (await streamToPromise(smStream)).toString();

  mkdirSync(backendStatic, { recursive: true }); // 👈 ensure static/ exists
  writeFileSync(resolve(distDir, "sitemap.xml"), xml, "utf-8");
  writeFileSync(resolve(backendStatic, "sitemap.xml"), xml, "utf-8");

  console.log("✅ sitemap.xml written with", pages.length, "routes");
}

buildSitemap().catch((err) => {
  console.error("❌ sitemap generation failed", err);
  process.exit(1);
});
