import { SitemapStream, streamToPromise } from "sitemap";
import { createWriteStream } from "fs";
import { Readable } from "stream";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

// derive __dirname in ESM
const __dirname = dirname(fileURLToPath(import.meta.url));

// list your URLs...
const pages = [
  /* ... */
];

async function buildSitemap() {
  const host = "https://scoregenius.io";
  const smStream = new SitemapStream({ hostname: host });

  // write into frontend/dist, relative to this file
  const sitemapPath = resolve(__dirname, "../dist", "sitemap.xml");
  const writeStream = createWriteStream(sitemapPath);

  Readable.from(pages).pipe(smStream).pipe(writeStream);
  await streamToPromise(smStream);
  console.log(`✅ sitemap.xml written to ${sitemapPath}`);
}

buildSitemap().catch(console.error);
