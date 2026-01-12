#!/bin/bash
# Rebuild and restart APP container (API + Frontend)
# Usage: ./rebuild_app.sh

set -e

# Clear shell env vars that override .env file
unset POSTGRES_HOST NEO4J_URI NEO4J_PASSWORD POSTGRES_PASSWORD 2>/dev/null || true

echo "=== Rebuilding APP (API + Frontend) ==="

echo "Building app container..."
docker-compose build app

echo "Removing stale app containers..."
docker ps -a --filter "name=herenews-app" -q | xargs -r docker rm -f 2>/dev/null || true

echo "Starting app..."
docker-compose up -d app

echo ""
echo "Done! App is starting on port 7272..."
docker ps --filter "name=herenews-app" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
