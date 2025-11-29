#!/bin/bash

echo "🌊 Breathing Knowledge Base - Demo Startup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables from parent .env
if [ -f "../.env" ]; then
    set -a
    source ../.env
    set +a
    echo -e "${GREEN}✓${NC} Loaded environment from ../.env"
else
    echo -e "${YELLOW}⚠${NC}  No ../.env file found"
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠${NC}  OPENAI_API_KEY not set. Semantic worker will fail."
    echo "   Add it to ../.env"
    exit 1
else
    echo -e "${GREEN}✓${NC} OPENAI_API_KEY loaded (${OPENAI_API_KEY:0:8}...)"
fi

# Stop and clean up any existing containers
echo ""
echo -e "${BLUE}Stopping any existing demo containers...${NC}"
cd "$(dirname "$0")"
docker-compose -f docker-compose.demo.yml down 2>/dev/null || true

# Start all services with docker-compose
echo ""
echo -e "${BLUE}Starting all services with Docker Compose...${NC}"

docker-compose -f docker-compose.demo.yml up --build -d

echo ""
echo "Waiting for services to be ready..."
sleep 5

# Copy latest frontend (bypass Docker cache)
echo "Copying latest frontend files..."
docker cp frontend/index.html demo-frontend:/app/index.html 2>/dev/null || true

# Check service status
echo ""
echo -e "${GREEN}Services started:${NC}"
docker-compose -f docker-compose.demo.yml ps

echo ""
echo "=========================================="
echo -e "${GREEN}Demo is ready!${NC}"
echo ""
echo "Open the demo webapp:"
echo "  http://localhost:8002"
echo ""
echo "API Endpoints:"
echo "  Health:  http://localhost:8001/health"
echo "  Submit:  POST http://localhost:8001/api/artifacts/draft?url=..."
echo "  Status:  GET http://localhost:8001/api/artifacts/draft/{id}"
echo ""
echo "View logs:"
echo "  All:              docker-compose -f docker-compose.demo.yml logs -f"
echo "  Backend:          docker logs -f demo-backend"
echo "  Frontend:         docker logs -f demo-frontend"
echo "  Extraction:       docker logs -f demo-worker-extraction"
echo "  Semantic:         docker logs -f demo-worker-semantic"
echo "  Event:            docker logs -f demo-worker-event"
echo ""
echo "Stop demo:"
echo "  ./stop_demo.sh"
echo "=========================================="
