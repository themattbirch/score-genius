# ─── Stage 1: build frontend ───
FROM node:18-alpine AS builder
WORKDIR /app/frontend

# Install frontend deps & build
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

# 2) Copy backend source (including all your marketing HTML in static/public)
COPY backend/ ./backend/

# 3) Copy only the new frontend SPA index.html into the marketing folder
#    (so we don’t clobber documentation.html, privacy.html, etc.)
COPY --from=builder /app/frontend/dist/public/index.html \
     ./backend/server/static/public/index.html

# 4) Copy the rest of your PWA build artifacts
COPY --from=builder /app/frontend/dist/app.html             ./backend/server/static/app.html
COPY --from=builder /app/frontend/dist/manifest.webmanifest  ./backend/server/static/manifest.webmanifest
COPY --from=builder /app/frontend/dist/assets               ./backend/server/static/assets
COPY --from=builder /app/frontend/dist/media                ./backend/server/static/media
COPY --from=builder /app/frontend/dist/app-sw.js            ./backend/server/static/app-sw.js

WORKDIR /app/backend/server
EXPOSE 10000
CMD ["node", "server.js"]
