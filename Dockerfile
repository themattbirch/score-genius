# ─── Stage 1: Build Frontend ──────────────────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app/frontend

# 1. Build-time environment variables
ARG VITE_FIREBASE_API_KEY
ARG VITE_FIREBASE_AUTH_DOMAIN
ARG VITE_FIREBASE_PROJECT_ID
ARG VITE_FIREBASE_STORAGE_BUCKET
ARG VITE_FIREBASE_MESSAGING_SENDER_ID
ARG VITE_FIREBASE_APP_ID
ARG VITE_FIREBASE_MEASUREMENT_ID

# 2. Expose variables to the Vite build process
ENV VITE_FIREBASE_API_KEY=${VITE_FIREBASE_API_KEY} \
    VITE_FIREBASE_AUTH_DOMAIN=${VITE_FIREBASE_AUTH_DOMAIN} \
    VITE_FIREBASE_PROJECT_ID=${VITE_FIREBASE_PROJECT_ID} \
    VITE_FIREBASE_STORAGE_BUCKET=${VITE_FIREBASE_STORAGE_BUCKET} \
    VITE_FIREBASE_MESSAGING_SENDER_ID=${VITE_FIREBASE_MESSAGING_SENDER_ID} \
    VITE_FIREBASE_APP_ID=${VITE_FIREBASE_APP_ID} \
    VITE_FIREBASE_MEASUREMENT_ID=${VITE_FIREBASE_MEASUREMENT_ID}

# 3. Copy source and build the Single Page Application (SPA)
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ .
ARG CACHEBUST=1

# 4. Sanity checks and build execution
RUN test -f src/app/app-sw.ts || (echo "src/app/app-sw.ts missing" && ls -lR src/app && exit 1)
RUN echo ">>>> src tree <<<<" && ls -lR /app/frontend/src
RUN echo ">>> sanity: does src/app/app-sw.ts exist?" && test -f src/app/app-sw.ts && echo YES || (echo NO && ls -lR src/app && exit 1)
RUN npm run build
RUN echo ">>> dist contents <<<" && ls -lR /app/frontend/dist

# ─── Stage 2: Assemble Backend + Static Assets ──────────────────────────────
FROM node:18-slim AS runner

WORKDIR /app

# Install Python and create a virtual environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-distutils \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create and activate the virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    python-dateutil \
    supabase

# Install backend dependencies
COPY backend/server/package*.json ./backend/server/
RUN cd backend/server && npm ci --production

# Copy backend and frontend assets
COPY backend/ ./backend/
RUN mkdir -p backend/server/static/public/.well-known
COPY frontend/public/*.html backend/server/static/public/
COPY frontend/public/.well-known/assetlinks.json backend/server/static/public/.well-known/

# Copy SPA artifacts from the builder stage
COPY --from=builder /app/frontend/dist/public/index.html backend/server/static/public/index.html
COPY --from=builder /app/frontend/dist/app.html backend/server/static/app.html
COPY --from=builder /app/frontend/dist/manifest.webmanifest backend/server/static/manifest.webmanifest
COPY --from=builder /app/frontend/dist/assets/ backend/server/static/assets/
COPY --from=builder /app/frontend/dist/media/ backend/server/static/media/
COPY --from=builder /app/frontend/dist/images/ backend/server/static/images/
COPY --from=builder /app/frontend/dist/icons/ backend/server/static/icons/

# Copy generated PWA assets
RUN mkdir -p backend/server/static/app
COPY --from=builder /app/frontend/dist/offline.html backend/server/static/app/offline.html
COPY --from=builder /app/frontend/dist/app/app-sw.* backend/server/static/app/app-sw.js
COPY --from=builder /app/frontend/dist/workbox-*.js backend/server/static/app/

# Final sanity check for the service worker
RUN test -f backend/server/static/app/app-sw.js || (echo "Service Worker missing!" && ls -lR backend/server/static/app && exit 1)

# Configure and run the server
WORKDIR /app/backend/server
EXPOSE 10000
CMD ["node", "server.js"]