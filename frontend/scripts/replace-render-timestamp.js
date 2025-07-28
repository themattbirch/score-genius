// scripts/inject-timestamp-postbuild.js
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distPath = path.resolve(__dirname, "../dist/support.html");

if (!fs.existsSync(distPath)) {
  console.error("❌ dist/support.html not found.");
  process.exit(1);
}

const timestamp = new Date().toISOString();

let html = fs.readFileSync(distPath, "utf-8");
html = html.replace(/%%RENDER_TIMESTAMP%%/, timestamp);
fs.writeFileSync(distPath, html);

console.log("✅ Injected timestamp into dist/support.html:", timestamp);
