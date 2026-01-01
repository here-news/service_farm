#!/usr/bin/env python3
"""
API Integration Test
====================

Tests the inquiry REST API endpoints end-to-end.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
from typing import Optional

BASE_URL = "http://localhost:7272/api"


def log(msg: str):
    print(msg, flush=True)


def section(title: str):
    log(f"\n{'='*70}")
    log(f"  {title}")
    log(f"{'='*70}\n")


async def test_api_endpoints():
    """Test all inquiry API endpoints."""

    section("API INTEGRATION TEST")

    async with httpx.AsyncClient(timeout=30.0) as client:
        errors = []

        # 1. Health check
        log("  1. Testing health endpoint...")
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                log(f"     ✓ Health OK: {response.json()}")
            else:
                errors.append(f"Health check failed: {response.status_code}")
        except Exception as e:
            errors.append(f"Health check error: {e}")

        # 2. Create inquiry
        log("\n  2. Creating test inquiry...")
        inquiry_id: Optional[str] = None
        try:
            response = await client.post(
                f"{BASE_URL}/inquiry",
                json={
                    "title": "Test inquiry: How many widgets were sold in Q4?",
                    "description": "Testing the API",
                    "schema": {
                        "schema_type": "monotone_count",
                        "count_scale": "medium",
                        "count_max": 1000,
                        "count_monotone": False,
                        "rigor": "B"
                    },
                    "scope_entities": ["Widgets Inc", "Q4 2025"],
                    "scope_keywords": ["sales", "widgets", "q4"],
                    "initial_stake": 50.0
                }
            )
            if response.status_code == 200:
                data = response.json()
                inquiry_id = data.get("id")
                log(f"     ✓ Created inquiry: {inquiry_id}")
                log(f"       Title: {data.get('title')}")
                log(f"       Stake: ${data.get('stake', 0):.2f}")
            else:
                errors.append(f"Create inquiry failed: {response.status_code} - {response.text}")
        except Exception as e:
            errors.append(f"Create inquiry error: {e}")

        if not inquiry_id:
            log("\n  ❌ Cannot continue without inquiry ID")
            return errors

        # 3. List inquiries
        log("\n  3. Listing inquiries...")
        try:
            response = await client.get(f"{BASE_URL}/inquiry")
            if response.status_code == 200:
                data = response.json()
                log(f"     ✓ Found {len(data)} inquiries")
                for inq in data[:3]:
                    log(f"       - {inq.get('id')}: {inq.get('title')[:40]}...")
            else:
                errors.append(f"List inquiries failed: {response.status_code}")
        except Exception as e:
            errors.append(f"List inquiries error: {e}")

        # 4. Get inquiry detail
        log("\n  4. Getting inquiry detail...")
        try:
            response = await client.get(f"{BASE_URL}/inquiry/{inquiry_id}")
            if response.status_code == 200:
                data = response.json()
                log(f"     ✓ Got detail for {inquiry_id}")
                log(f"       Status: {data.get('status')}")
                log(f"       Posterior MAP: {data.get('posterior', {}).get('map')}")
            else:
                errors.append(f"Get inquiry failed: {response.status_code}")
        except Exception as e:
            errors.append(f"Get inquiry error: {e}")

        # 5. Add contribution
        log("\n  5. Adding contribution...")
        try:
            response = await client.post(
                f"{BASE_URL}/inquiry/{inquiry_id}/contribute",
                json={
                    "type": "evidence",
                    "text": "Q4 sales report shows 500 widgets sold",
                    "source_url": "https://widgets.inc/q4-report",
                    "source_name": "Widgets Inc",
                    "extracted_value": 500,
                    "observation_kind": "point"
                }
            )
            if response.status_code == 200:
                data = response.json()
                log(f"     ✓ Added contribution")
                log(f"       ID: {data.get('contribution', {}).get('id')}")
                log(f"       Impact: {data.get('contribution', {}).get('impact', 0)*100:.1f}%")
                log(f"       New MAP: {data.get('updated_posterior', {}).get('map')}")
            else:
                errors.append(f"Add contribution failed: {response.status_code} - {response.text}")
        except Exception as e:
            errors.append(f"Add contribution error: {e}")

        # 6. Add second contribution
        log("\n  6. Adding corroborating contribution...")
        try:
            response = await client.post(
                f"{BASE_URL}/inquiry/{inquiry_id}/contribute",
                json={
                    "type": "evidence",
                    "text": "Industry report confirms ~500 widget sales",
                    "source_url": "https://industry.org/report",
                    "source_name": "Industry Analysts",
                    "extracted_value": 500,
                    "observation_kind": "approximate"
                }
            )
            if response.status_code == 200:
                data = response.json()
                log(f"     ✓ Added corroborating evidence")
                log(f"       New probability: {data.get('updated_posterior', {}).get('probability', 0)*100:.1f}%")
            else:
                errors.append(f"Add second contribution failed: {response.status_code}")
        except Exception as e:
            errors.append(f"Add second contribution error: {e}")

        # 7. Add stake
        log("\n  7. Adding stake...")
        try:
            response = await client.post(
                f"{BASE_URL}/inquiry/{inquiry_id}/stake",
                json={"amount": 25.0}
            )
            if response.status_code == 200:
                data = response.json()
                log(f"     ✓ Added $25 stake")
                log(f"       New total: ${data.get('total_stake', 0):.2f}")
            else:
                errors.append(f"Add stake failed: {response.status_code}")
        except Exception as e:
            errors.append(f"Add stake error: {e}")

        # 8. Get trace
        log("\n  8. Getting epistemic trace...")
        try:
            response = await client.get(f"{BASE_URL}/inquiry/{inquiry_id}/trace")
            if response.status_code == 200:
                data = response.json()
                log(f"     ✓ Got trace")
                log(f"       Observations: {data.get('belief_state', {}).get('observation_count', 0)}")
                log(f"       Contributions: {len(data.get('contributions', []))}")
                log(f"       Tasks: {len(data.get('tasks', []))}")

                if data.get('posterior_top_10'):
                    log(f"       Top hypotheses:")
                    for item in data['posterior_top_10'][:3]:
                        log(f"         {item['value']}: {item['probability']*100:.1f}%")
            else:
                errors.append(f"Get trace failed: {response.status_code}")
        except Exception as e:
            errors.append(f"Get trace error: {e}")

        # 9. Get tasks
        log("\n  9. Getting tasks...")
        try:
            response = await client.get(f"{BASE_URL}/inquiry/{inquiry_id}/tasks")
            if response.status_code == 200:
                data = response.json()
                log(f"     ✓ Got {len(data)} tasks")
                for task in data:
                    log(f"       - [{task.get('type')}] ${task.get('bounty', 0):.2f}")
            else:
                errors.append(f"Get tasks failed: {response.status_code}")
        except Exception as e:
            errors.append(f"Get tasks error: {e}")

        # Summary
        section("RESULTS")

        if not errors:
            log("  ✓ All API tests PASSED")
        else:
            log(f"  ✗ {len(errors)} errors:")
            for e in errors:
                log(f"    - {e}")

        return errors


async def main():
    errors = await test_api_endpoints()

    section("API ENDPOINT SUMMARY")

    log("""
  WORKING ENDPOINTS:
    POST /api/inquiry              - Create inquiry
    GET  /api/inquiry              - List inquiries
    GET  /api/inquiry/{id}         - Get inquiry detail
    POST /api/inquiry/{id}/contribute - Add contribution
    POST /api/inquiry/{id}/stake   - Add stake
    GET  /api/inquiry/{id}/trace   - Get epistemic trace
    GET  /api/inquiry/{id}/tasks   - Get open tasks

  PENDING ENDPOINTS:
    POST /api/inquiry/{id}/tasks/{task_id}/claim   - Claim task
    POST /api/inquiry/{id}/tasks/{task_id}/complete - Complete task
    POST /api/inquiry/{id}/resolve - Force resolution (admin)
    DELETE /api/inquiry/{id}       - Close inquiry
    """)


if __name__ == "__main__":
    asyncio.run(main())
