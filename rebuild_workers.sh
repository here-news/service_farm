#!/bin/bash
# Rebuild and restart WORKERS container (extraction, knowledge, weaver, inquiry)
# Usage: ./rebuild_workers.sh

set -e

# Clear shell env vars that override .env file
unset POSTGRES_HOST NEO4J_URI NEO4J_PASSWORD POSTGRES_PASSWORD 2>/dev/null || true

echo "=== Rebuilding WORKERS ==="

echo "Building workers container..."
docker-compose build workers

echo "Removing stale worker containers..."
docker ps -a --filter "name=herenews-workers" -q | xargs -r docker rm -f 2>/dev/null || true

echo "Starting workers..."
docker-compose up -d workers

echo ""
echo "Done! Workers are starting..."
docker ps --filter "name=herenews-workers" --format "table {{.Names}}\t{{.Status}}"
echo ""
echo "View logs: docker-compose logs -f workers"
