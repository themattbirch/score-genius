// scripts/prebuild_log.cjs

const fs = require("fs");
const path = require("path");

function walk(dir) {
  const files = fs.readdirSync(dir);
  files.forEach((file) => {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    if (stat.isDirectory()) {
      walk(fullPath);
    } else {
      console.log(`[SRC FILE] ${fullPath}`);
    }
  });
}

console.log(">>> [PREBUILD] Listing /app/frontend/src recursively:");
walk(path.resolve(__dirname, "../src"));

const swPath = path.resolve(__dirname, "../src/app-sw.ts");
console.log(">>> [PREBUILD] Does app-sw.ts exist?", fs.existsSync(swPath));
