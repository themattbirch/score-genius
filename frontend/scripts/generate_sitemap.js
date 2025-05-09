import { SitemapStream, streamToPromise } from "sitemap";
import { createWriteStream, readdirSync } from "fs";
import { resolve } from "path";

const dist = resolve("dist");
const domain = "https://scoregenius.io";

const pages = readdirSync(dist)
  .filter((f) => f.endsWith(".html"))
  .map((f) => (f === "index.html" ? "/" : `/${f.replace(".html", "")}`));

const smStream = new SitemapStream({ hostname: domain });
const write = createWriteStream(resolve(dist, "sitemap.xml"));

streamToPromise(smStream).then(() => {
  console.log("✅ sitemap.xml written");
});

pages.forEach((p) => smStream.write({ url: p, changefreq: "weekly" }));
smStream.end();
smStream.pipe(write);
