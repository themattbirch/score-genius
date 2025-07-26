# ─── Stage 1: build frontend ────────────────────────────────────────────────
FROM node:18-alpine AS builder

# 1) Build‑time env vars (Render injects these)
ARG VITE_FIREBASE_API_KEY
ARG VITE_FIREBASE_AUTH_DOMAIN
ARG VITE_FIREBASE_PROJECT_ID
ARG VITE_FIREBASE_STORAGE_BUCKET
ARG VITE_FIREBASE_MESSAGING_SENDER_ID
ARG VITE_FIREBASE_APP_ID
ARG VITE_FIREBASE_MEASUREMENT_ID

# 2) Expose them to Vite
ENV VITE_FIREBASE_API_KEY=${VITE_FIREBASE_API_KEY}
ENV VITE_FIREBASE_AUTH_DOMAIN=${VITE_FIREBASE_AUTH_DOMAIN}
ENV VITE_FIREBASE_PROJECT_ID=${VITE_FIREBASE_PROJECT_ID}
ENV VITE_FIREBASE_STORAGE_BUCKET=${VITE_FIREBASE_STORAGE_BUCKET}
ENV VITE_FIREBASE_MESSAGING_SENDER_ID=${VITE_FIREBASE_MESSAGING_SENDER_ID}
ENV VITE_FIREBASE_APP_ID=${VITE_FIREBASE_APP_ID}
ENV VITE_FIREBASE_MEASUREMENT_ID=${VITE_FIREBASE_MEASUREMENT_ID}

# 3) Build SPA
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build \
    && echo ">>> dist contents <<<" \
    && ls -lR dist

RUN echo ">>> dist contents <<<" && ls -lR /app/frontend/dist


# ─── Stage 2: assemble backend + static ─────────────────────────────────────
FROM node:18-slim AS runner
WORKDIR /app

# --- Install Python 3 and create a venv for snapshot scripts ---------
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-venv python3-distutils build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create a virtualenv and install Python dependencies there
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir \
      pandas \
      numpy \
      python-dateutil \
      supabase
# --------------------------------------------------------------------

# Install backend deps
COPY backend/server/package*.json ./backend/server/
RUN cd backend/server && npm ci --production

# Copy backend source
COPY backend/ ./backend/

# Copy marketing pages & SPA artifacts
RUN mkdir -p backend/server/static/public
COPY frontend/public/*.html backend/server/static/public/
RUN mkdir -p backend/server/static/public/.well-known
COPY frontend/public/.well-known/assetlinks.json \
     backend/server/static/public/.well-known/

# Overlay the SPA build
COPY --from=builder /app/frontend/dist/public/index.html \
     backend/server/static/public/index.html
COPY --from=builder /app/frontend/dist/app.html \
     backend/server/static/app.html
COPY --from=builder /app/frontend/dist/manifest.webmanifest \
     backend/server/static/manifest.webmanifest
COPY --from=builder /app/frontend/dist/assets \
     backend/server/static/assets
COPY --from=builder /app/frontend/dist/media \
     backend/server/static/media
COPY --from=builder /app/frontend/dist/images \
     backend/server/static/images
COPY --from=builder /app/frontend/dist/icons \
     backend/server/static/icons
# Copy the entire 'app' directory, including the SW and its Workbox dependency
RUN mkdir -p backend/server/static/app
# copy offline shell
COPY --from=builder /app/frontend/dist/app/offline.html \
     backend/server/static/app/

# copy the generated SW + runtime
COPY --from=builder /app/frontend/dist/app-sw.js \
     backend/server/static/app/app-sw.js
COPY --from=builder /app/frontend/dist/workbox-*.js \
     backend/server/static/app/

# sanity check
RUN test -f backend/server/static/app/app-sw.js \
    || (echo "SW missing!" && ls -lR backend/server/static/app && exit 1)

# Final runner
WORKDIR /app/backend/server
EXPOSE 10000
CMD ["node", "server.js"]
