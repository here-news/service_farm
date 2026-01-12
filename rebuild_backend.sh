#!/bin/bash
# Rebuild and restart WORKERS container (REEE kernel + workers)
# Usage: ./rebuild_backend.sh
#        ./rebuild_backend.sh --skip-tests

set -e

SKIP_TESTS=false
if [[ "$1" == "--skip-tests" ]]; then
    SKIP_TESTS=true
fi

# Clear shell env vars that override .env file
unset POSTGRES_HOST NEO4J_URI NEO4J_PASSWORD POSTGRES_PASSWORD 2>/dev/null || true

echo "=== Rebuilding WORKERS ==="

echo "Building workers container..."
docker-compose build workers

echo "Removing stale worker containers..."
docker ps -a --filter "name=herenews-workers" -q | xargs -r docker rm -f 2>/dev/null || true

echo "Starting workers..."
docker-compose up -d workers

# Wait for container to be ready
echo "Waiting for container to start..."
sleep 2

# Run kernel tests
if [[ "$SKIP_TESTS" == "false" ]]; then
    echo ""
    echo "=== Running REEE Kernel Tests ==="
    # Note: paths are relative to /app since ./backend is mounted to /app
    # Run core kernel tests (excluding integration/db tests)
    if docker exec herenews-workers python -m pytest \
        reee/tests/test_canonical_stability.py \
        reee/tests/test_surface_identity.py \
        reee/tests/test_types.py \
        reee/tests/test_golden_trace.py \
        reee/tests/test_scoped_surface.py \
        reee/tests/test_sparse_entity_claims.py \
        reee/tests/test_semantic_antitrap.py \
        reee/tests/test_public_api.py \
        reee/tests/test_story_builder.py \
        reee/tests/test_membrane.py \
        reee/tests/test_kernel_isolation.py \
        reee/tests/test_no_deprecated_imports.py \
        reee/tests/test_no_structure_from_deprecated.py \
        -v --tb=short; then
        echo ""
        echo "✓ All kernel tests passed!"
    else
        echo ""
        echo "✗ Kernel tests FAILED!"
        echo "  Check the output above for details."
        exit 1
    fi
else
    echo ""
    echo "(Skipping tests - use without --skip-tests to run)"
fi

echo ""
echo "Done! Workers are running."
docker ps --filter "name=herenews-workers" --format "table {{.Names}}\t{{.Status}}"
echo ""
echo "View logs: docker-compose logs -f workers"
