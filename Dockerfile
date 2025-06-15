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

# 2) Copy backend source (including static/public/*)
COPY backend/ ./backend/

# 3) Copy all your built frontend artifacts INTO the server’s static folder:
COPY --from=builder /app/frontend/dist/app.html             ./backend/server/static/app.html
COPY --from=builder /app/frontend/dist/manifest.webmanifest  ./backend/server/static/manifest.webmanifest
COPY --from=builder /app/frontend/dist/assets               ./backend/server/static/assets
COPY --from=builder /app/frontend/dist/media                ./backend/server/static/media
COPY --from=builder /app/frontend/dist/app-sw.js            ./backend/server/static/app-sw.js

WORKDIR /app/backend/server
EXPOSE 10000
CMD ["node", "server.js"]
