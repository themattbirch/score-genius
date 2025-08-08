# ─── Stage 1: Build Frontend ──────────────────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app/frontend

# 1) Build-time env
ARG VITE_FIREBASE_API_KEY
ARG VITE_FIREBASE_AUTH_DOMAIN
ARG VITE_FIREBASE_PROJECT_ID
ARG VITE_FIREBASE_STORAGE_BUCKET
ARG VITE_FIREBASE_MESSAGING_SENDER_ID
ARG VITE_FIREBASE_APP_ID
ARG VITE_FIREBASE_MEASUREMENT_ID

# 2) Expose to Vite
ENV VITE_FIREBASE_API_KEY=${VITE_FIREBASE_API_KEY} \
    VITE_FIREBASE_AUTH_DOMAIN=${VITE_FIREBASE_AUTH_DOMAIN} \
    VITE_FIREBASE_PROJECT_ID=${VITE_FIREBASE_PROJECT_ID} \
    VITE_FIREBASE_STORAGE_BUCKET=${VITE_FIREBASE_STORAGE_BUCKET} \
    VITE_FIREBASE_MESSAGING_SENDER_ID=${VITE_FIREBASE_MESSAGING_SENDER_ID} \
    VITE_FIREBASE_APP_ID=${VITE_FIREBASE_APP_ID} \
    VITE_FIREBASE_MEASUREMENT_ID=${VITE_FIREBASE_MEASUREMENT_ID}

# 3) Install + build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
ARG CACHEBUST=1

# 4) Sanity + build
RUN test -f src/app/app-sw.ts || (echo "src/app/app-sw.ts missing" && ls -lR src/app && exit 1)
RUN npm run build && node scripts/generate_sitemap.js
RUN echo ">>> dist contents <<<" && ls -lR /app/frontend/dist

# ─── Stage 2: Assemble Backend + Static Assets ────────────────────────────────
FROM node:18-slim AS runner

WORKDIR /app

# Python env for your backend jobs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-distutils build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir pandas numpy python-dateutil supabase

# Backend deps
COPY backend/server/package*.json ./backend/server/
RUN cd backend/server && npm ci --production

# Backend source
COPY backend/ ./backend/

# Public marketing pages at /
RUN mkdir -p backend/server/static/public/.well-known
COPY frontend/public/*.html                               backend/server/static/public/
COPY frontend/public/robots.txt                           backend/server/static/public/robots.txt
COPY frontend/public/favicon.ico                          backend/server/static/public/favicon.ico
COPY frontend/public/.well-known/assetlinks.json          backend/server/static/public/.well-known/

# ─── /app (SPA + PWA + Windows installer artifacts) ───────────────────────────
RUN mkdir -p backend/server/static/app

# SPA from builder (note: base='/app/')
COPY --from=builder /app/frontend/dist/index.html                 backend/server/static/app/index.html
COPY --from=builder /app/frontend/dist/assets/                    backend/server/static/app/assets/
COPY --from=builder /app/frontend/dist/app-sw.js                  backend/server/static/app/app-sw.js
COPY --from=builder /app/frontend/dist/manifest.webmanifest       backend/server/static/app/manifest.webmanifest

# Optional asset folders produced by your build/copy
COPY --from=builder /app/frontend/dist/icons/                     backend/server/static/app/icons/
COPY --from=builder /app/frontend/dist/images/                    backend/server/static/app/images/
COPY --from=builder /app/frontend/dist/screenshots/               backend/server/static/app/screenshots/
# sitemap at site root (keep where you want it served)
COPY --from=builder /app/frontend/dist/sitemap.xml                backend/server/static/sitemap.xml

# Windows installer files (hosted, not built here)
# If these don't exist in the build context, comment these COPY lines or ensure CI drops them in place.
COPY frontend/app/ScoreGenius.appinstaller                        backend/server/static/app/ScoreGenius.appinstaller
COPY windows_package/out/ScoreGenius.appxbundle                   backend/server/static/app/ScoreGenius.appxbundle

# Final sanity: SW present
RUN test -f backend/server/static/app/app-sw.js || (echo "Service Worker missing!" && ls -lR backend/server/static/app && exit 1)

# ─── Run server ───────────────────────────────────────────────────────────────
WORKDIR /app/backend/server
EXPOSE 10000
CMD ["node", "server.js"]
