#!/bin/bash

echo "Stopping Breathing KB Demo..."

cd "$(dirname "$0")"

docker-compose -f docker-compose.demo.yml down

echo "✓ All services stopped"
echo ""
echo "Data preserved in Docker volume."
echo "To completely remove data: docker volume rm demo_demo_pg_data"
