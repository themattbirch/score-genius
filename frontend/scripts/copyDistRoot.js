// frontend/scripts/copyDistRoot.js
import { copyFileSync, existsSync, readdirSync } from "fs";
import { resolve, join } from "path";
import { fileURLToPath } from "url";

// Emulate __dirname in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, "..");

// Paths
const dist = resolve(__dirname, "../dist");
const pub = resolve(dist, "public");

// Make sure dist exists
if (!existsSync(dist)) {
  console.error("❌ dist folder not found at", dist);
  process.exit(1);
}

// Copy everything from dist/public up into dist/
if (existsSync(pub)) {
  for (const name of readdirSync(pub)) {
    const src = join(pub, name);
    const dst = join(dist, name);
    copyFileSync(src, dst);
    console.log(`📋 copied ${name} → dist/${name}`);
  }
} else {
  console.warn("⚠️  No public folder in dist, skipping copy-up");
}
