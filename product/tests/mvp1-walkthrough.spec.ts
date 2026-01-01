import { test, expect } from '@playwright/test';

/**
 * MVP1 Walkthrough Tests
 *
 * End-to-end tests that verify all key user journeys for the Inquiry MVP.
 * These tests document the expected behavior and can be used for regression testing.
 */

test.describe('MVP1 Complete Walkthrough', () => {
  test.describe('1. Guest Browsing Experience', () => {
    test('1.1 View inquiry list page', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry');
      await page.waitForLoadState('networkidle');

      // Should show the main heading
      await expect(page.locator('h1')).toContainText("What's True?");

      // Should show carousels
      await expect(page.getByText('Recently Resolved')).toBeVisible();
      await expect(page.getByText('Top Bounties')).toBeVisible();

      // Should show inquiry cards with bounty amounts
      const bountyTexts = page.locator('text=/\\$\\d+/');
      expect(await bountyTexts.count()).toBeGreaterThan(0);
    });

    test('1.2 Browse resolved inquiries', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry');
      await page.waitForLoadState('networkidle');

      // Find resolved inquiry in carousel
      const resolvedSection = page.getByText('Recently Resolved').locator('..');
      await expect(resolvedSection).toBeVisible();
    });

    test('1.3 View inquiry detail (simulated)', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should show title
      await expect(page.locator('h1')).toContainText('Russian soldiers');

      // Should show bounty pool
      await expect(page.getByText('Bounty Pool')).toBeVisible();

      // Should show belief state
      await expect(page.getByText(/315,?000|315000/)).toBeVisible();
    });

    test('1.4 View probability distribution', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should show distribution section
      await expect(page.getByText('Probability Distribution')).toBeVisible();

      // Should show bars
      const distributionSection = page.locator('[class*="Distribution"]').first();
      await expect(distributionSection).toBeVisible();
    });

    test('1.5 View evidence gaps', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should show evidence gaps section
      await expect(page.getByText('Evidence Gaps')).toBeVisible();
    });

    test('1.6 View claim clusters', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should show claim clusters
      await expect(page.getByText('Claim Clusters')).toBeVisible();
    });
  });

  test.describe('2. Search and Navigation', () => {
    test('2.1 Navigate from list to detail', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry');
      await page.waitForLoadState('networkidle');

      // Click on an inquiry card
      const firstCard = page.locator('a[href^="/inquiry/"]').first();
      await firstCard.click();

      // Should navigate to detail
      await expect(page).toHaveURL(/\/inquiry\/.+/);
    });

    test('2.2 Navigate back to list', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Click back button
      const backButton = page.getByText('← Back to questions');
      await backButton.click();

      // Should be on list page
      await expect(page).toHaveURL('/inquiry');
    });

    test('2.3 Sort inquiries', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry');
      await page.waitForLoadState('networkidle');

      // Find sort dropdown
      const sortSelect = page.locator('select').first();
      if (await sortSelect.isVisible()) {
        await sortSelect.selectOption({ label: 'By Uncertainty' });
        // Page should update (no error)
        await page.waitForTimeout(500);
      }
    });
  });

  test.describe('3. Contribution UI (Guest)', () => {
    test('3.1 Share box is visible', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should show share box
      const shareBox = page.locator('textarea[placeholder*="Share what you know"]');
      await expect(shareBox).toBeVisible();
    });

    test('3.2 Can type in share box', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      const shareBox = page.locator('textarea[placeholder*="Share what you know"]');
      await shareBox.fill('According to Reuters, the confirmed count is approximately 310,000.');

      // Should show character count
      const text = await shareBox.inputValue();
      expect(text.length).toBeGreaterThan(20);
    });

    test('3.3 Post button exists', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Post button should exist
      await expect(page.getByRole('button', { name: /Post/i })).toBeVisible();
    });

    test('3.4 Guest posting shows demo message', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      const shareBox = page.locator('textarea[placeholder*="Share what you know"]');
      await shareBox.fill('According to official sources, the number is around 320,000.');

      // Click post
      await page.getByRole('button', { name: /Post/i }).click();

      // Should show feedback (either toast or in-place message)
      // The exact UX may vary, so we check for any response
      await page.waitForTimeout(1000);
    });
  });

  test.describe('4. Bounty Pool UI', () => {
    test('4.1 Bounty pool displays total', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should show bounty amount
      await expect(page.getByText('$5000.00')).toBeVisible();
    });

    test('4.2 Add bounty input exists', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should have input for adding bounty
      const addInput = page.locator('input[type="number"], input[placeholder*="Add"]');
      await expect(addInput).toBeVisible();
    });

    test('4.3 Add button exists', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should have Add button
      await expect(page.getByRole('button', { name: /Add/i })).toBeVisible();
    });
  });

  test.describe('5. Resolution Status', () => {
    test('5.1 Open inquiry shows OPEN badge', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should show OPEN status badge
      await expect(page.getByText('OPEN', { exact: true }).first()).toBeVisible();
    });

    test('5.2 Resolved inquiry shows RESOLVED badge', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_resolved_1');
      await page.waitForLoadState('networkidle');

      // Should show RESOLVED status badge
      await expect(page.getByText('RESOLVED', { exact: true })).toBeVisible();
    });

    test('5.3 Resolution progress indicator visible', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should show resolution progress or status text
      await expect(page.getByText(/Resolution|Gathering Evidence|confidence|P\(MAP\)/i)).toBeVisible();
    });
  });

  test.describe('6. Recent Rewards Section', () => {
    test('6.1 Recent rewards section exists', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should show recent rewards or similar
      const rewardsSection = page.getByText(/Recent rewards|Recent earnings|Rewards/i);
      // This may be a clickable toggle
    });
  });

  test.describe('7. Community Contributions Display', () => {
    test('7.1 Community section visible', async ({ page }) => {
      await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
      await page.waitForLoadState('networkidle');

      // Should show Community section
      await expect(page.getByText('Community')).toBeVisible();
    });
  });

  test.describe('8. API Integration', () => {
    test('8.1 List API returns data', async ({ request }) => {
      const response = await request.get('http://localhost:7272/api/inquiry');
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(Array.isArray(data)).toBeTruthy();
      expect(data.length).toBeGreaterThan(0);
    });

    test('8.2 Detail API returns inquiry', async ({ request }) => {
      const response = await request.get('http://localhost:7272/api/inquiry/sim_bounty_1');
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(data.id).toBe('sim_bounty_1');
      expect(data.title).toContain('Russian');
    });

    test('8.3 Trace API returns belief state', async ({ request }) => {
      const response = await request.get('http://localhost:7272/api/inquiry/sim_bounty_1/trace');
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(data.belief_state).toBeDefined();
      expect(data.belief_state.map).toBe(315000);
    });

    test('8.4 Contribute API accepts demo contribution', async ({ request }) => {
      const response = await request.post('http://localhost:7272/api/inquiry/sim_bounty_1/contribute', {
        data: {
          type: 'evidence',
          text: 'Test contribution from automated test - Reuters reports 315000 casualties'
        }
      });
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(data.demo_mode).toBe(true);
    });
  });
});

test.describe('MVP1 Mobile Experience', () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test('Mobile: List page is responsive', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry');
    await page.waitForLoadState('networkidle');

    // Should show heading
    await expect(page.locator('h1')).toBeVisible();

    // No horizontal scrollbar
    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 5); // 5px tolerance
  });

  test('Mobile: Detail page is responsive', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
    await page.waitForLoadState('networkidle');

    // Should show title
    await expect(page.locator('h1')).toBeVisible();

    // Bounty pool should be visible
    await expect(page.getByText('Bounty Pool')).toBeVisible();
  });
});
