FROM node:18-alpine AS builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM node:18-slim AS runner
WORKDIR /app

# Install only production dependencies for the server
COPY backend/server/package*.json ./backend/server/
RUN cd backend/server && npm ci --production

# Copy the backend source (routes, server.js, etc.)
COPY backend/ ./backend/

# Copy just the built frontend artifacts into the server’s static folder,
# without overwriting the marketing HTML in static/public:
COPY --from=builder /app/frontend/dist/app.html            ./backend/server/static/app.html
COPY --from=builder /app/frontend/dist/manifest.webmanifest ./backend/server/static/manifest.webmanifest
COPY --from=builder /app/frontend/dist/assets             ./backend/server/static/assets
COPY --from=builder /app/frontend/dist/media              ./backend/server/static/media
COPY --from=builder /app/frontend/dist/app-sw.js          ./backend/server/static/app-sw.js

WORKDIR /app/backend/server
EXPOSE 10000
CMD ["node", "server.js"]
