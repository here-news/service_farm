# Multi-stage build: Frontend + Backend in one image

# Stage 1: Build frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /frontend

# Copy package files and install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy source and build
COPY frontend/ ./
RUN npm run build

# Stage 2: Python app with built frontend
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

# Set environment variable to persist Playwright browsers location
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers in persistent location
RUN playwright install chromium --with-deps

# Copy backend application code
COPY backend/ .

# Copy built frontend from stage 1 to /static (outside /app)
# This prevents frontend from being overwritten when mounting ./backend:/app for dev
COPY --from=frontend-builder /output /static

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
