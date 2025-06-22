# ─── Stage 1: build frontend ───
FROM node:18-alpine AS builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ─── Stage 2: assemble backend + static ───
FROM node:18-slim AS runner
WORKDIR /app

# 1) Install backend deps
COPY backend/server/package*.json ./backend/server/
RUN cd backend/server && npm ci --production

# 2) Copy backend source
COPY backend/ ./backend/

# 3) Copy all your marketing pages (static HTML) into the image
#    Use a wildcard so the directory itself need not exist yet in build cache
RUN mkdir -p backend/server/static/public
COPY frontend/public/*.html \
     backend/server/static/public/

# 4) Overlay only the SPA’s index.html into that folder
COPY --from=builder /app/frontend/dist/public/index.html \
     backend/server/static/public/index.html

# 5) Copy the rest of the SPA build artifacts
COPY --from=builder /app/frontend/dist/app.html                    \
     backend/server/static/app.html
COPY --from=builder /app/frontend/dist/manifest.webmanifest       \
     backend/server/static/manifest.webmanifest
COPY --from=builder /app/frontend/dist/assets                      \
     backend/server/static/assets
COPY --from=builder /app/frontend/dist/media                       \
     backend/server/static/media
COPY --from=builder /app/frontend/dist/icons                       \
     backend/server/static/icons
COPY --from=builder /app/frontend/dist/app-sw.js                   \
     backend/server/static/app-sw.js

WORKDIR /app/backend/server
EXPOSE 10000
CMD ["node", "server.js"]
