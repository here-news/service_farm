"""
URL Preview API
Calls Cloud Run service to extract metadata and semantic data from URLs
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
import httpx
from typing import Optional
import re
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/preview", tags=["preview"])

# Get settings for service farm URL
from config import get_settings
settings = get_settings()


class URLPreviewRequest(BaseModel):
    url: HttpUrl


class URLPreviewResponse(BaseModel):
    task_id: str
    url: str
    status: str = "fetching"  # fetching, processing, completed, failed
    title: Optional[str] = None
    description: Optional[str] = None
    image: Optional[str] = None
    domain: Optional[str] = None
    favicon: Optional[str] = None
    ontology: Optional[dict] = None
    entities: Optional[dict] = None  # Changed from list to dict


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # fetching, processing, completed, failed
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    image: Optional[str] = None
    domain: Optional[str] = None
    favicon: Optional[str] = None
    # Additional semantic data from Cloud Run
    ontology: Optional[dict] = None
    entities: Optional[dict] = None  # Changed from list to dict


def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    match = re.match(r'https?://([^/]+)', url)
    return match.group(1) if match else url


@router.post("", response_model=URLPreviewResponse)
async def submit_url_preview(request: URLPreviewRequest):
    """
    Submit URL to Cloud Run service for preview

    Uses the unified /preview endpoint which automatically:
    - Returns cached preview (50-400ms)
    - OR fetches iFramely preview while extracting (500-1000ms)
    - OR returns task_id for polling (3000ms timeout)

    Example responses:

    Cached/iFramely:
    {
        "task_id": "abc123",
        "url": "https://example.com/article",
        "status": "completed"  // or "extracting"
    }

    Extraction started:
    {
        "task_id": "abc123",
        "url": "https://example.com/article",
        "status": "fetching"
    }
    """
    url = str(request.url)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Call unified /api/preview endpoint (handles everything)
            service_url = f"{settings.service_farm_url}/api/preview"
            logger.info(f"🔍 Calling service_farm at: {service_url} with url={url}")
            preview_response = await client.get(
                service_url,
                params={"url": url},
                timeout=15.0  # Fail faster to avoid gateway timeouts
            )
            logger.info(f"📡 Service_farm response: status={preview_response.status_code}")

            # Handle response codes
            if preview_response.status_code == 200:
                result = preview_response.json()

                # Case 1: New service_farm format (direct fields)
                if result.get("title") and not result.get("found"):
                    # New format from service_farm on port 8080
                    return URLPreviewResponse(
                        task_id="preview-cached",
                        url=url,
                        status="completed",
                        title=result.get("title"),
                        description=result.get("description"),
                        image=result.get("image"),
                        domain=extract_domain(url),
                        favicon=None,
                        ontology=None,
                        entities=None
                    )

                # Case 2: Old format - Cached/iFramely preview available
                elif result.get("found") and result.get("preview"):
                    preview_data = result["preview"]
                    task_id = preview_data.get("task_id", "preview-cached")

                    # Check if still extracting
                    is_extracting = preview_data.get("status") == "extracting"

                    # Return full preview data to avoid extra status API call
                    return URLPreviewResponse(
                        task_id=task_id,
                        url=url,
                        status="extracting" if is_extracting else "completed",
                        title=preview_data.get("title"),
                        description=preview_data.get("description"),
                        image=preview_data.get("image") or preview_data.get("thumbnail_url"),
                        domain=extract_domain(url),
                        favicon=preview_data.get("favicon"),
                        ontology=preview_data.get("ontology"),
                        entities=preview_data.get("entities")
                    )

                # Case 3: Not found, but extraction started (found: false, extraction_started: true)
                elif result.get("extraction_started"):
                    task_id = result.get("task_id")

                    if not task_id:
                        raise HTTPException(status_code=500, detail="No task_id returned for extraction")

                    return URLPreviewResponse(
                        task_id=task_id,
                        url=url,
                        status="fetching"
                    )

                # Unexpected 200 response
                logger.error(f"Unexpected 200 response from service farm for {url}: {result}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected response format: {result}"
                )

            elif preview_response.status_code == 202:
                # Extraction started, need to poll
                result = preview_response.json()
                task_id = result.get("task_id")

                if not task_id:
                    raise HTTPException(status_code=500, detail="No task_id returned")

                return URLPreviewResponse(
                    task_id=task_id,
                    url=url,
                    status="fetching"
                )

            # Handle other status codes (4xx, 5xx)
            # Log the actual response for debugging
            try:
                error_detail = preview_response.json()
                logger.error(f"Service farm error {preview_response.status_code} for {url}: {error_detail}")
            except:
                error_text = preview_response.text
                logger.error(f"Service farm error {preview_response.status_code} for {url}: {error_text}")

            # For 500 errors from service farm, return a failed status instead of propagating error
            if preview_response.status_code >= 500:
                # Service farm itself had an error - return a task response with failed status
                return URLPreviewResponse(
                    task_id="failed",
                    url=url,
                    status="failed",
                    domain=extract_domain(url)
                )

            # For other unexpected codes, raise exception
            raise HTTPException(
                status_code=preview_response.status_code,
                detail="Unexpected response from preview service"
            )

    except httpx.TimeoutException as e:
        logger.warning(f"Preview timeout for URL {url}: {e}")
        # Return failed status instead of error to allow graceful fallback
        return URLPreviewResponse(
            task_id="timeout",
            url=url,
            status="failed",
            domain=extract_domain(url)
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error for URL {url}: {e.response.status_code} - {e}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Preview service error: {e}")
    except httpx.RequestError as e:
        logger.error(f"Request error for URL {url}: {e}")
        raise HTTPException(status_code=503, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error fetching preview for URL {url}")
        raise HTTPException(status_code=500, detail=f"Error fetching preview: {str(e)}")


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Poll for URL extraction task status

    Handles both fast preview responses and full extraction tasks

    Example response:
    {
        "task_id": "abc123",
        "status": "completed",
        "url": "https://example.com/article",
        "title": "Article Title",
        "description": "Article description...",
        "image": "https://example.com/image.jpg",
        "domain": "example.com",
        "favicon": "https://example.com/favicon.ico",
        "ontology": {...},
        "entities": [...]
    }
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check if this is a cached preview (instant response)
            if task_id == "cached":
                raise HTTPException(status_code=400, detail="Invalid task_id")

            # Check task status from service farm
            response = await client.get(
                f"{settings.service_farm_url}/api/task/{task_id}"
            )
            response.raise_for_status()

            result = response.json()

            # Extract relevant fields from service farm response
            status = result.get("status", "unknown")
            url = result.get("url", "")

            # Build response
            task_response = {
                "task_id": task_id,
                "status": status,
                "url": url,
                "domain": extract_domain(url) if url else None
            }

            # Extract metadata for completed, extracting, or blocked content
            if status in ["completed", "extracting", "blocked"]:
                # Try preview_meta first (fast preview data - available even for blocked content)
                preview_meta = result.get("preview_meta", {})
                if preview_meta:
                    task_response.update({
                        "title": preview_meta.get("title"),
                        "description": preview_meta.get("description"),
                        "image": preview_meta.get("thumbnail_url"),
                        "favicon": None,
                        "ontology": None,
                        "entities": None
                    })
                elif "result" in result:
                    # Use full result data if available
                    res = result["result"]
                    resolved_entities = res.get("resolved_entities", {})
                    task_response.update({
                        "title": res.get("title") or res.get("meta_description", ""),
                        "description": res.get("meta_description"),
                        "image": None,
                        "favicon": None,
                        "ontology": res.get("og_metadata"),
                        "entities": resolved_entities
                    })

            return TaskStatusResponse(**task_response)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Service farm error: {e}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Network error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking task status: {str(e)}")
