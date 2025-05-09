// frontend/scripts/generate_sitemap.js

import { SitemapStream, streamToPromise } from "sitemap";
import { createWriteStream, readdirSync } from "fs";
import { resolve } from "path";

async function buildSitemap() {
  const distDir = resolve("dist");
  const hostname = "https://scoregenius.io";

  /* ---------- collect routes ---------- */
  const pages = readdirSync(distDir)
    .filter((f) => f.endsWith(".html") && f !== "app.html") // ← exclude SPA shell
    .map((f) => (f === "index.html" ? "/" : `/${f.replace(".html", "")}`));

  /* ---------- stream → file ---------- */
  const smStream = new SitemapStream({ hostname });
  const write = createWriteStream(resolve(distDir, "sitemap.xml"));

  smStream.pipe(write); // 1) pipe first
  pages.forEach((url) => smStream.write({ url, changefreq: "weekly" }));
  smStream.end(); // 2) close the stream

  await streamToPromise(smStream); // 3) wait until fully flushed
  console.log("✅ sitemap.xml written with", pages.length, "routes");
}

buildSitemap().catch((err) => {
  console.error("❌ sitemap generation failed", err);
  process.exit(1);
});
