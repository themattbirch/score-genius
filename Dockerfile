# ─── Stage 1: build frontend ────────────────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

# (ARG and ENV lines for Firebase remain the same)
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

# Copy only the necessary files for the build
COPY package*.json ./
COPY frontend/package*.json ./frontend/
COPY frontend/ ./frontend/
# Note: We are no longer copying the 'scripts' directory

# Install dependencies and run the correct, simplified build command
RUN npm install --prefix frontend
RUN npm run build --prefix frontend

# ─── Stage 2: assemble backend + static ─────────────────────────────────────
# (The rest of the Dockerfile remains exactly the same as the last correct version)
FROM node:18-slim AS runner
WORKDIR /app

# --- Python installation ---
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

# --- Backend and Static File Assembly ---
COPY backend/server/package*.json ./backend/server/
RUN cd backend/server && npm ci --production
COPY backend/ ./backend/
RUN mkdir -p backend/server/static/public/.well-known
COPY frontend/public/*.html backend/server/static/public/
COPY frontend/public/.well-known/assetlinks.json backend/server/static/public/.well-known/
COPY --from=builder /app/dist/index.html backend/server/static/app.html
COPY --from=builder /app/dist/manifest.webmanifest backend/server/static/manifest.webmanifest
COPY --from=builder /app/dist/assets/ backend/server/static/assets/
COPY --from=builder /app/dist/media/ backend/server/static/media/
COPY --from=builder /app/dist/images/ backend/server/static/images/
COPY --from=builder /app/dist/icons/ backend/server/static/icons/
COPY --from=builder /app/dist/support.html backend/server/static/public/
COPY --from=builder /app/dist/app-sw.js backend/server/static/
COPY --from=builder /app/dist/workbox-*.js backend/server/static/

# --- Final runner ---
WORKDIR /app/backend/server
EXPOSE 10000
CMD ["node", "server.js"]