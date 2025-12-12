#!/bin/bash
# Quick frontend build and restart script
# Avoids docker-compose ContainerConfig errors by force-removing stale containers

set -e

echo "Building frontend..."
docker-compose build app

echo "Removing stale app containers..."
docker ps -a --filter "name=app" -q | xargs -r docker rm -f 2>/dev/null || true

echo "Starting app..."
docker-compose up -d app

echo "Done! App is starting..."
docker ps --filter "name=herenews-app" --format "table {{.Names}}\t{{.Status}}"
