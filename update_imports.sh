#!/bin/bash
# Script to update imports from app.* to backend.* across all API routers

echo "🔧 Updating imports in backend/api/ routers..."

# Update all files in backend/api/
for file in backend/api/*.py; do
    if [ -f "$file" ]; then
        echo "  Processing $file..."

        # Update imports
        sed -i 's/from app\.auth\.google_oauth/from backend.middleware.google_oauth/g' "$file"
        sed -i 's/from app\.auth\.session/from backend.middleware.jwt_session/g' "$file"
        sed -i 's/from app\.auth\.middleware/from backend.middleware.auth/g' "$file"
        sed -i 's/from app\.database\.connection/from backend.repositories/g' "$file"
        sed -i 's/from app\.database\.repositories\.\([a-z_]*\)_repo/from backend.repositories.\1_repository/g' "$file"
        sed -i 's/from app\.models\./from backend.models.api./g' "$file"
        sed -i 's/from app\.services\./from backend.services./g' "$file"
        sed -i 's/from app\.config/from backend.config/g' "$file"

        # Remove get_db import and usage
        sed -i '/from sqlalchemy.ext.asyncio import AsyncSession/d' "$file"
        sed -i '/db: AsyncSession = Depends(get_db)/d' "$file"

        echo "    ✅ Updated $file"
    fi
done

echo ""
echo "✅ All imports updated!"
echo ""
echo "⚠️  Manual review needed for:"
echo "   - Endpoints using get_db dependency"
echo "   - UserRepository/CommentRepository/ChatSessionRepository instantiation"
echo "   - Add 'from backend.repositories import db_pool' where needed"
