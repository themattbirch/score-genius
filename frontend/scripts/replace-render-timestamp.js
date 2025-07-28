import fs from "fs";
const timestamp = new Date().toISOString();
const htmlPath = "./public/support.html";

let html = fs.readFileSync(htmlPath, "utf-8");
html = html.replace(/%%RENDER_TIMESTAMP%%/, timestamp);
fs.writeFileSync(htmlPath, html);
console.log("✅ Injected timestamp:", timestamp);
