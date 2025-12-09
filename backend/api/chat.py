"""
Chat session API router for premium story chat
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List, Dict, Any
import re

from openai import OpenAI

from middleware.auth import get_current_user_optional
from repositories import db_pool
from repositories.chat_session_repository import ChatSessionRepository
from repositories.user_repository import UserRepository
from models.api.user import UserPublic, UserResponse
from models.api.chat import (
    ChatSessionResponse,
    UnlockChatRequest,
    SendMessageRequest,
    SendMessageResponse
)
from services.neo4j_client import neo4j_client
from config import get_settings

router = APIRouter(prefix="/api/chat", tags=["chat"])
settings = get_settings()

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

MAX_MESSAGES_PER_SESSION = 100
CHAT_UNLOCK_COST = 10


def _strip_entity_markup(text: str) -> str:
    """Remove [[Entity]] markup from text"""
    return re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)


def _get_story_claims(story_id: str) -> List[Dict[str, Any]]:
    """Fetch verified claims for a story from Neo4j"""
    try:
        with neo4j_client.driver.session(database=neo4j_client.database) as session:
            result = session.run('''
                MATCH (s:Story {id: $story_id})<-[:PART_OF]-(a:Page)-[:HAS_CLAIM]->(c:Claim)
                WHERE c.verified = true OR c.confidence >= 0.7
                RETURN c.text as text,
                       c.confidence as confidence,
                       c.type as type,
                       c.timestamp as timestamp,
                       a.url as source_url
                ORDER BY c.confidence DESC
                LIMIT 50
            ''', story_id=story_id)

            claims = []
            for record in result:
                claims.append({
                    'text': record['text'],
                    'confidence': record['confidence'],
                    'type': record['type'],
                    'timestamp': str(record['timestamp']) if record['timestamp'] else None,
                    'source_url': record['source_url']
                })
            return claims
    except Exception as e:
        print(f"Error fetching claims: {e}")
        return []


def _build_story_context(story_id: str) -> str:
    """Build rich story context for OpenAI"""
    try:
        with neo4j_client.driver.session(database=neo4j_client.database) as session:
            result = session.run('''
                MATCH (s:Story {id: $story_id})
                OPTIONAL MATCH (s)-[:MENTIONS_ENTITY]->(e)
                WITH s, collect(DISTINCT e.name) as entities
                RETURN s.title as title,
                       s.gist as description,
                       s.content as content,
                       s.created_at as created_at,
                       entities
            ''', story_id=story_id)

            record = result.single()
            if not record:
                return "Story not found."

            # Build context
            context_parts = []

            if record['title']:
                context_parts.append(f"Title: {record['title']}")

            if record['description']:
                context_parts.append(f"Summary: {_strip_entity_markup(record['description'])}")

            if record['content']:
                content = _strip_entity_markup(record['content'])
                # Truncate to 4000 chars
                if len(content) > 4000:
                    content = content[:4000] + "..."
                context_parts.append(f"Full Content:\n{content}")

            if record['entities']:
                context_parts.append(f"Key Entities: {', '.join(record['entities'][:20])}")

            # Add claims
            claims = _get_story_claims(story_id)
            if claims:
                claims_text = "\n".join([
                    f"{i+1}. {claim['text']} (confidence: {claim.get('confidence', 'N/A')})"
                    for i, claim in enumerate(claims[:50])
                ])
                context_parts.append(f"Verified Claims:\n{claims_text}")

            return "\n\n".join(context_parts)

    except Exception as e:
        print(f"Error building story context: {e}")
        return "Error loading story context."


@router.post("/unlock", response_model=ChatSessionResponse)
async def unlock_chat(
    request: UnlockChatRequest,
    current_user: Optional[UserPublic] = Depends(get_current_user_optional)
):
    """
    Unlock premium chat for a story (costs 10 credits)

    Creates a ChatSession if not already unlocked
    Deducts credits from user account
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    chat_repo = ChatSessionRepository(db)

    # Check if already unlocked
    existing_session = await chat_repo.get_by_story_and_user(
        request.story_id,
        current_user.user_id
    )

    if existing_session:
        return ChatSessionResponse.model_validate(existing_session)

    # Create new session and deduct credits
    try:
        session = await chat_repo.create_session(
            story_id=request.story_id,
            user_id=current_user.user_id,
            cost=CHAT_UNLOCK_COST
        )
        await db.commit()

        return ChatSessionResponse.model_validate(session)

    except ValueError as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/session/{story_id}", response_model=Optional[ChatSessionResponse])
async def get_session(
    story_id: str,
    current_user: Optional[UserPublic] = Depends(get_current_user_optional)
):
    """
    Get chat session status for a story

    Returns None if not unlocked, session details if unlocked
    """
    if not current_user:
        return None

    chat_repo = ChatSessionRepository(db)
    session = await chat_repo.get_by_story_and_user(story_id, current_user.user_id)

    if not session:
        return None

    return ChatSessionResponse.model_validate(session)


@router.post("/message", response_model=SendMessageResponse)
async def send_message(
    request: SendMessageRequest,
    current_user: Optional[UserPublic] = Depends(get_current_user_optional)
):
    """
    Send a message in the chat session

    Increments message count and checks if session is exhausted
    Returns AI response
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not openai_client:
        raise HTTPException(status_code=503, detail="Chat service unavailable (OpenAI not configured)")

    chat_repo = ChatSessionRepository(db)

    # Get session
    session = await chat_repo.get_by_story_and_user(
        request.story_id,
        current_user.user_id
    )

    if not session:
        raise HTTPException(
            status_code=403,
            detail="Chat not unlocked. Please unlock chat first."
        )

    # Increment message count
    try:
        session = await chat_repo.increment_message_count(
            session.id,
            max_messages=MAX_MESSAGES_PER_SESSION
        )
        await db.commit()

    except ValueError as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

    # Build story context
    context = _build_story_context(request.story_id)

    # Build messages for OpenAI
    system_prompt = f"""You are a concise assistant that answers questions about news stories based on verified information.

Story Context:
{context}

Guidelines:
- Answer in 2-3 sentences maximum
- Use only the provided story information
- Be direct and factual - no filler words
- If information isn't available, say so in one sentence
- When citing claims, reference them by number (e.g., "Claim 5 states..." or "Claims 3 and 7 confirm...")
- **ENTITY MARKUP**: When mentioning key entities (people, organizations, locations), wrap ONLY the FIRST mention in double brackets:
  Format: [[Entity Name]]
  Example: "[[Chuck Schumer]] proposed the bill. Schumer also mentioned..." (only first "Chuck Schumer" is marked)
- Maintain a neutral, journalistic tone"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last 10 messages for context)
    for msg in request.conversation_history[-10:]:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })

    # Add current user message
    messages.append({"role": "user", "content": request.message})

    # Call OpenAI
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=200
        )

        ai_response = response.choices[0].message.content or "I apologize, I couldn't generate a response."

    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate AI response")

    remaining = await chat_repo.get_remaining_messages(
        session.id,
        max_messages=MAX_MESSAGES_PER_SESSION
    )

    return SendMessageResponse(
        message=ai_response,
        message_count=session.message_count,
        remaining_messages=remaining,
        session_status=session.status
    )


@router.get("/credits", response_model=dict)
async def get_credits(
    current_user: Optional[UserPublic] = Depends(get_current_user_optional)
):
    """
    Get current user's credit balance
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(current_user.user_id)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "credits": user.credits_balance,
        "reputation": user.reputation
    }
