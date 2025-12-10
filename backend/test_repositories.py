"""
Test script to validate repositories with actual database
Run: python test_repositories.py
"""
import sys
import os
import asyncio
import asyncpg

# Add backend to path
sys.path.insert(0, 'backend')

from repositories.user_repository import UserRepository
from repositories.comment_repository import CommentRepository
from repositories.chat_session_repository import ChatSessionRepository
from models.domain.user import User
from models.domain.comment import Comment, ReactionType
from models.domain.chat_session import ChatSession


async def create_db_pool():
    """Create database connection pool"""
    return await asyncpg.create_pool(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'herenews_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'herenews_pass'),
        database=os.getenv('POSTGRES_DB', 'herenews'),
        min_size=1,
        max_size=5
    )


async def run_migrations(pool):
    """Run database migrations"""
    print("\n📋 Running database migrations...")

    # Read migration file
    migration_path = 'migrations/001_create_user_tables.sql'
    with open(migration_path, 'r') as f:
        migration_sql = f.read()

    async with pool.acquire() as conn:
        await conn.execute(migration_sql)

    print("✅ Migrations complete")


async def test_user_repository(pool):
    """Test UserRepository"""
    print("\n" + "=" * 60)
    print("Testing UserRepository")
    print("=" * 60)

    repo = UserRepository(pool)

    # Create user
    user = User(
        user_id="",
        email="test@example.com",
        google_id="google_test_123",
        name="Test User",
        credits_balance=1000,
        reputation=50
    )

    created_user = await repo.create(user)
    print(f"✅ Created user: {created_user.user_id}")
    print(f"   Email: {created_user.email}")
    print(f"   Credits: {created_user.credits_balance}")

    # Get by ID
    fetched = await repo.get_by_id(created_user.user_id)
    assert fetched is not None, "User not found by ID"
    print(f"✅ Fetched by ID: {fetched.email}")

    # Get by email
    fetched_email = await repo.get_by_email("test@example.com")
    assert fetched_email is not None, "User not found by email"
    print(f"✅ Fetched by email: {fetched_email.user_id}")

    # Get by Google ID
    fetched_google = await repo.get_by_google_id("google_test_123")
    assert fetched_google is not None, "User not found by Google ID"
    print(f"✅ Fetched by Google ID: {fetched_google.user_id}")

    # Deduct credits
    success = await repo.deduct_credits(created_user.user_id, 10)
    assert success, "Credit deduction failed"
    print(f"✅ Deducted 10 credits")

    # Verify new balance
    updated_user = await repo.get_by_id(created_user.user_id)
    assert updated_user.credits_balance == 990, f"Expected 990 credits, got {updated_user.credits_balance}"
    print(f"✅ New balance: {updated_user.credits_balance}")

    # Try to deduct more than available
    success = await repo.deduct_credits(created_user.user_id, 10000)
    assert not success, "Should have failed with insufficient credits"
    print(f"✅ Correctly rejected insufficient credits")

    # Update last login
    await repo.update_last_login(created_user.user_id)
    print(f"✅ Updated last login")

    return created_user


async def test_comment_repository(pool, user_id):
    """Test CommentRepository"""
    print("\n" + "=" * 60)
    print("Testing CommentRepository")
    print("=" * 60)

    repo = CommentRepository(pool)

    # Create comment on event
    comment = Comment(
        id="",
        user_id=user_id,
        event_id="ev_test1234",
        text="This is a test comment on an event",
        reaction_type=ReactionType.SUPPORT
    )

    created_comment = await repo.create(comment)
    print(f"✅ Created comment: {created_comment.id}")
    print(f"   Text: {created_comment.text}")
    print(f"   Event ID: {created_comment.event_id}")

    # Get by ID
    fetched = await repo.get_by_id(created_comment.id)
    assert fetched is not None, "Comment not found"
    print(f"✅ Fetched by ID: {fetched.id}")

    # Create reply
    reply = Comment(
        id="",
        user_id=user_id,
        event_id="ev_test1234",
        text="This is a reply to the comment",
        parent_comment_id=created_comment.id
    )

    created_reply = await repo.create(reply)
    print(f"✅ Created reply: {created_reply.id}")
    print(f"   Parent: {created_reply.parent_comment_id}")

    # Get comments by event
    event_comments = await repo.get_by_event("ev_test1234")
    assert len(event_comments) == 2, f"Expected 2 comments, got {len(event_comments)}"
    print(f"✅ Found {len(event_comments)} comments for event")

    # Get replies
    replies = await repo.get_replies(created_comment.id)
    assert len(replies) == 1, f"Expected 1 reply, got {len(replies)}"
    print(f"✅ Found {len(replies)} reply to comment")

    # Count comments
    count = await repo.count_by_event("ev_test1234")
    assert count == 2, f"Expected count 2, got {count}"
    print(f"✅ Count by event: {count}")

    # Update text
    await repo.update_text(created_comment.id, "Updated comment text")
    updated = await repo.get_by_id(created_comment.id)
    assert updated.text == "Updated comment text", "Text not updated"
    print(f"✅ Updated comment text")

    # Delete comment (wrong user should fail)
    deleted = await repo.delete(created_comment.id, "00000000-0000-0000-0000-000000000000")
    assert not deleted, "Should have failed to delete with wrong user"
    print(f"✅ Correctly rejected delete by wrong user")

    # Delete comment (correct user)
    deleted = await repo.delete(created_reply.id, user_id)
    assert deleted, "Failed to delete comment"
    print(f"✅ Deleted reply by correct user")

    return created_comment


async def test_chat_session_repository(pool, user_id):
    """Test ChatSessionRepository"""
    print("\n" + "=" * 60)
    print("Testing ChatSessionRepository")
    print("=" * 60)

    repo = ChatSessionRepository(pool)

    # Create chat session (should deduct credits)
    session = ChatSession(
        id="",
        event_id="ev_test5678",
        user_id=user_id,
        cost=10
    )

    created_session = await repo.create(session, deduct_credits_from_user=user_id)
    print(f"✅ Created chat session: {created_session.id}")
    print(f"   Event ID: {created_session.event_id}")
    print(f"   User ID: {created_session.user_id}")
    print(f"   Cost: {created_session.cost} credits")

    # Verify credits were deducted
    user_repo = UserRepository(pool)
    user = await user_repo.get_by_id(user_id)
    expected_balance = 990 - 10  # 990 from previous test - 10 for chat
    assert user.credits_balance == expected_balance, f"Expected {expected_balance} credits, got {user.credits_balance}"
    print(f"✅ Credits deducted, new balance: {user.credits_balance}")

    # Get by ID
    fetched = await repo.get_by_id(created_session.id)
    assert fetched is not None, "Session not found"
    print(f"✅ Fetched by ID: {fetched.id}")

    # Get by event and user
    fetched_event = await repo.get_by_event_and_user("ev_test5678", user_id)
    assert fetched_event is not None, "Session not found by event and user"
    print(f"✅ Fetched by event and user: {fetched_event.id}")

    # Increment message count
    print(f"\n📨 Testing message increment...")
    for i in range(3):
        updated = await repo.increment_message_count(created_session.id, max_messages=5)
        print(f"   Message {i+1}: count={updated.message_count}, status={updated.status.value}")

    # Check remaining messages
    remaining = await repo.get_remaining_messages(created_session.id, max_messages=5)
    assert remaining == 2, f"Expected 2 remaining, got {remaining}"
    print(f"✅ Remaining messages: {remaining}")

    # Exhaust messages
    await repo.increment_message_count(created_session.id, max_messages=5)
    await repo.increment_message_count(created_session.id, max_messages=5)
    exhausted_session = await repo.get_by_id(created_session.id)
    assert exhausted_session.status.value == "exhausted", "Session should be exhausted"
    assert exhausted_session.message_count == 5, f"Expected 5 messages, got {exhausted_session.message_count}"
    print(f"✅ Session exhausted after 5 messages")

    # Try to send another message (should fail)
    try:
        await repo.increment_message_count(created_session.id, max_messages=5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Correctly rejected message on exhausted session: {e}")

    return created_session


async def cleanup(pool):
    """Clean up test data"""
    print("\n🧹 Cleaning up test data...")

    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM chat_sessions WHERE event_id LIKE 'ev_test%'")
        await conn.execute("DELETE FROM comments WHERE event_id LIKE 'ev_test%'")
        await conn.execute("DELETE FROM users WHERE email = 'test@example.com'")

    print("✅ Cleanup complete")


async def main():
    """Run all repository tests"""
    print("\n🧪 Testing Repositories with Database")
    print("=" * 60)

    pool = None
    try:
        # Create database pool
        print("🔌 Connecting to database...")
        pool = await create_db_pool()
        print("✅ Connected")

        # Run migrations
        await run_migrations(pool)

        # Clean up any previous test data
        await cleanup(pool)

        # Test repositories
        user = await test_user_repository(pool)
        comment = await test_comment_repository(pool, user.user_id)
        session = await test_chat_session_repository(pool, user.user_id)

        # Cleanup
        await cleanup(pool)

        print("\n" + "=" * 60)
        print("✅ ALL REPOSITORY TESTS PASSED")
        print("=" * 60)
        print("\n✨ Repositories are ready for API integration!")
        print("\nNext steps:")
        print("  1. Add auth middleware (Google OAuth, JWT)")
        print("  2. Create API routes (auth, comments, chat)")
        print("  3. Test end-to-end authentication flow")

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if pool:
            await pool.close()
            print("\n🔌 Database connection closed")


if __name__ == "__main__":
    asyncio.run(main())
