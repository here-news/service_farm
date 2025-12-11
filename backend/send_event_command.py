#!/usr/bin/env python3
"""
Send commands to living events via Redis queue.

Usage:
    python send_event_command.py <event_id> <command> [params_json]

Examples:
    python send_event_command.py ev_pth3a8dc /retopologize
    python send_event_command.py ev_pth3a8dc /status
    python send_event_command.py ev_pth3a8dc /regenerate '{"force": true}'
    python send_event_command.py ev_pth3a8dc /rehydrate

Available commands:
    /retopologize - Re-run full Bayesian topology analysis
    /regenerate   - Regenerate narrative (with optional force param)
    /status       - Get current event status
    /rehydrate    - Reload claims from storage
"""
import asyncio
import json
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.job_queue import JobQueue


async def send_command(event_id: str, command: str, params: dict = None):
    """Send a command to an event via Redis queue."""
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

    queue = JobQueue(redis_url)
    await queue.connect()

    try:
        await queue.send_event_command(event_id, command, params)
        print(f"✅ Command sent: {command} -> {event_id}")
        if params:
            print(f"   Params: {json.dumps(params)}")
        print(f"\n📡 Check event worker logs for result")
    finally:
        await queue.close()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    event_id = sys.argv[1]
    command = sys.argv[2]

    # Ensure command starts with /
    if not command.startswith('/'):
        command = '/' + command

    # Parse optional params
    params = None
    if len(sys.argv) > 3:
        try:
            params = json.loads(sys.argv[3])
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON params: {e}")
            sys.exit(1)

    asyncio.run(send_command(event_id, command, params))


if __name__ == '__main__':
    main()
