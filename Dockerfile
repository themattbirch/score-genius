# ─── Stage 1: build frontend ────────────────────────────────────────────────
FROM node:18-alpine AS builder

# Set the working directory to the project root
WORKDIR /app

# 1) Build-time env vars (Render injects these)
ARG VITE_FIREBASE_API_KEY
ARG VITE_FIREBASE_AUTH_DOMAIN
ARG VITE_FIREBASE_PROJECT_ID
ARG VITE_FIREBASE_STORAGE_BUCKET
ARG VITE_FIREBASE_MESSAGING_SENDER_ID
ARG VITE_FIREBASE_APP_ID
ARG VITE_FIREBASE_MEASUREMENT_ID
ENV VITE_FIREBASE_API_KEY=${VITE_FIREBASE_API_KEY}
ENV VITE_FIREBASE_AUTH_DOMAIN=${VITE_FIREBASE_AUTH_DOMAIN}
ENV VITE_FIREBASE_PROJECT_ID=${VITE_FIREBASE_PROJECT_ID}
ENV VITE_FIREBASE_STORAGE_BUCKET=${VITE_FIREBASE_STORAGE_BUCKET}
ENV VITE_FIREBASE_MESSAGING_SENDER_ID=${VITE_FIREBASE_MESSAGING_SENDER_ID}
ENV VITE_FIREBASE_APP_ID=${VITE_FIREBASE_APP_ID}
ENV VITE_FIREBASE_MEASUREMENT_ID=${VITE_FIREBASE_MEASUREMENT_ID}

# 2) Copy all necessary source code for the build
COPY package*.json ./
COPY frontend/package*.json ./frontend/
COPY frontend/ ./frontend/
# FIX: Copy scripts into the frontend directory for the build context
COPY scripts/ ./frontend/scripts/

# 3) Install dependencies and run the build from the project root
RUN npm install --prefix frontend
RUN npm run build --prefix frontend

# ─── Stage 2: assemble backend + static ─────────────────────────────────────
FROM node:18-slim AS runner
WORKDIR /app

# --- Python installation remains the same ---
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  python3 python3-venv python3-distutils build-essential && \
  rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir \
  pandas \
  numpy \
  python-dateutil \
  supabase
# -------------------------------------------

# Install backend deps
COPY backend/server/package*.json ./backend/server/
RUN cd backend/server && npm ci --production

# Copy backend source
COPY backend/ ./backend/

# Create directories for static content
RUN mkdir -p backend/server/static/public/.well-known

# Copy marketing pages & SPA artifacts
COPY frontend/public/*.html backend/server/static/public/
COPY frontend/public/.well-known/assetlinks.json backend/server/static/public/.well-known/

# IMPORTANT: Paths are updated and fixed
COPY --from=builder /app/dist/index.html backend/server/static/app.html
COPY --from=builder /app/dist/manifest.webmanifest backend/server/static/manifest.webmanifest

# Copy asset directories. The trailing slash tells Docker it's a directory.
COPY --from=builder /app/dist/assets/ backend/server/static/assets/

# NOTE: If these directories are optional, the build will fail.
# If your build doesn't always create them, you can comment out these lines.
COPY --from=builder /app/dist/media/ backend/server/static/media/
COPY --from=builder /app/dist/images/ backend/server/static/images/
COPY --from=builder /app/dist/icons/ backend/server/static/icons/

# ─── Copy generated PWA assets ──────────────────────────────────────────────
# Copying to a directory (note the trailing slashes) resolves the linter errors.
COPY --from=builder /app/dist/support.html backend/server/static/public/
COPY --from=builder /app/dist/app-sw.js backend/server/static/
COPY --from=builder /app/dist/workbox-*.js backend/server/static/

# Final runner
WORKDIR /app/backend/server
EXPOSE 10000
CMD ["node", "server.js"]