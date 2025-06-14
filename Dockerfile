# ──────── STAGE 1: build frontend ────────
FROM node:18-alpine AS builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ──────── STAGE 2: install backend & assemble ────────
FROM node:18-slim AS runner
WORKDIR /app

# 1) install only server deps
COPY backend/server/package*.json ./backend/server/
RUN cd backend/server && npm ci --production

# 2) copy backend code + static assets
COPY backend/ ./backend/
COPY --from=builder /app/frontend/dist ./backend/server/static

WORKDIR /app/backend/server
EXPOSE 10000
CMD ["node", "server.js"]
