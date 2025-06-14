# ──────── STAGE 1: build frontend ────────
FROM node:18-alpine AS builder
WORKDIR /app/frontend

# 1) install & build React PWA
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build   # → generates dist/

# ──────── STAGE 2: minimal runtime ────────
FROM gcr.io/distroless/nodejs:18         AS runner
WORKDIR /app

# 1) Copy only what you need at runtime
COPY --from=builder /app/frontend/dist    ./static
COPY backend/server/package*.json         ./package*.json
RUN npm ci --production

COPY backend/server/.                      .

EXPOSE 10000
CMD ["server.js"]