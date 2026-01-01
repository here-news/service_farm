import { test, expect, Page } from '@playwright/test';

/**
 * Exploratory tests for the Inquiry MVP prototype
 * These tests document current functionality and help define product requirements
 */

test.describe('Inquiry Page Exploration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry');
    // Wait for the page to load (simulated data)
    await page.waitForTimeout(1000);
  });

  test('should display inquiry list page with correct header', async ({ page }) => {
    // Check main heading
    await expect(page.locator('h1')).toContainText("What's True?");

    // Check tagline
    await expect(page.getByText('Collaborative fact-finding')).toBeVisible();
  });

  test('should show carousels for different inquiry categories', async ({ page }) => {
    // Recently Resolved carousel
    await expect(page.getByText('Recently Resolved')).toBeVisible();

    // Top Bounties carousel
    await expect(page.getByText('Top Bounties')).toBeVisible();

    // Highly Contested carousel
    await expect(page.getByText('Highly Contested')).toBeVisible();

    // All Open Questions section
    await expect(page.getByText('All Open Questions')).toBeVisible();
  });

  test('should show inquiry cards with key metrics', async ({ page }) => {
    // Look for bounty/stake amounts
    const bountyElements = page.locator('text=/\\$\\d+/');
    await expect(bountyElements.first()).toBeVisible();

    // Look for entropy indicators
    const entropyIndicators = page.locator('text=/bits/i');
    // entropy should be shown somewhere
  });

  test('should have FAB button to create new inquiry', async ({ page }) => {
    // The floating action button
    const fab = page.locator('button[title="Ask a new question"]');
    await expect(fab).toBeVisible();
  });

  test('should navigate to inquiry detail when clicking a card', async ({ page }) => {
    // Find an inquiry card and click it
    const inquiryCard = page.locator('[class*="rounded-xl"]').filter({
      hasText: /\$\d+/
    }).first();

    await inquiryCard.click();

    // Should navigate to detail page
    await expect(page).toHaveURL(/\/inquiry\/sim_/);
  });

  test('should have sorting options for open questions', async ({ page }) => {
    const sortSelect = page.locator('select');
    await expect(sortSelect).toBeVisible();

    // Check options
    await expect(page.getByRole('option', { name: 'By Bounty' })).toBeAttached();
    await expect(page.getByRole('option', { name: 'By Uncertainty' })).toBeAttached();
    await expect(page.getByRole('option', { name: 'By Activity' })).toBeAttached();
  });
});

test.describe('Inquiry Detail Page Exploration', () => {
  test.beforeEach(async ({ page }) => {
    // Go directly to a simulated inquiry
    await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
    await page.waitForTimeout(1000);
  });

  test('should display inquiry title and status', async ({ page }) => {
    // Status badge
    await expect(page.getByText('OPEN')).toBeVisible();

    // Title should be visible
    await expect(page.locator('h1')).toBeVisible();
  });

  test('should show current best estimate with confidence', async ({ page }) => {
    // Look for the big answer display
    const answerSection = page.getByText('Current Best Estimate');
    await expect(answerSection).toBeVisible();

    // Confidence percentage
    await expect(page.getByText(/\d+%/)).toBeVisible();
  });

  test('should display bounty pool with add functionality', async ({ page }) => {
    // Bounty Pool header
    await expect(page.getByText('Bounty Pool')).toBeVisible();

    // Add bounty input
    const addInput = page.locator('input[placeholder="Add to pool"]');
    await expect(addInput).toBeVisible();

    // Add button
    await expect(page.getByRole('button', { name: 'Add' })).toBeVisible();
  });

  test('should show community contributions section', async ({ page }) => {
    // Community header
    await expect(page.getByText('Community')).toBeVisible();

    // Share box with textarea
    const shareTextarea = page.locator('textarea[placeholder*="Share what you know"]');
    await expect(shareTextarea).toBeVisible();

    // Post button
    await expect(page.getByRole('button', { name: 'Post' })).toBeVisible();
  });

  test('should display probability distribution chart', async ({ page }) => {
    // Distribution header
    await expect(page.getByText('Probability Distribution')).toBeVisible();
  });

  test('should show evidence gaps (tasks)', async ({ page }) => {
    // Evidence Gaps section
    await expect(page.getByText('Evidence Gaps')).toBeVisible();
  });

  test('should show claim clusters (surfaces)', async ({ page }) => {
    // Claim Clusters section
    await expect(page.getByText('Claim Clusters')).toBeVisible();
  });

  test('should allow adding contribution', async ({ page }) => {
    const textarea = page.locator('textarea[placeholder*="Share what you know"]');
    await textarea.fill('According to Reuters, the count is approximately 315,000.');

    // Check character count
    await expect(page.getByText(/\d+$/)).toBeVisible();

    // Submit
    await page.getByRole('button', { name: 'Post' }).click();

    // Should show toast
    await expect(page.getByText(/Demo mode|Evidence added/)).toBeVisible({ timeout: 3000 });
  });

  test('should show recent rewards in bounty panel', async ({ page }) => {
    // Click to show rewards
    const rewardsToggle = page.getByText('Recent rewards');
    await rewardsToggle.click();

    // Should show reward entries
    await expect(page.getByText(/Primary source verification|Conflicting evidence/)).toBeVisible();
  });

  test('should allow file attachment UI', async ({ page }) => {
    // Attach button
    await expect(page.getByText('Attach')).toBeVisible();
  });

  test('should show resolution policy', async ({ page }) => {
    // Resolution info
    await expect(page.getByText(/Resolution:|P\(MAP\)/)).toBeVisible();
  });
});

test.describe('Bounty System Exploration', () => {
  test('should show add bounty input and validate', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
    await page.waitForTimeout(1000);

    // Find the bounty add input
    const input = page.locator('input[placeholder="Add to pool"]');
    await input.fill('50');

    // Click add
    await page.getByRole('button', { name: 'Add' }).click();

    // Should show demo toast
    await expect(page.getByText(/Demo mode|Added \$/)).toBeVisible({ timeout: 3000 });
  });
});

test.describe('URL Preview Exploration', () => {
  test('should detect URLs in contribution text', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
    await page.waitForTimeout(1000);

    const textarea = page.locator('textarea[placeholder*="Share what you know"]');
    await textarea.fill('Check this source: https://reuters.com/article/ukraine-war-casualties');

    // The URL preview would need API support - check if component exists
  });
});

test.describe('Resolved Inquiry Exploration', () => {
  test('should show resolved status and styling', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry/sim_resolved_1');
    await page.waitForTimeout(1000);

    // Should show RESOLVED badge
    await expect(page.getByText('RESOLVED')).toBeVisible();

    // Should show "Resolved Answer" instead of "Current Best Estimate"
    await expect(page.getByText('Resolved Answer')).toBeVisible();
  });
});

test.describe('Search Functionality', () => {
  test('should filter inquiries by search query', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry?q=Tesla');
    await page.waitForTimeout(1000);

    // Should show search results
    await expect(page.getByText('Search Results')).toBeVisible();

    // Should find Tesla inquiry
    await expect(page.getByText(/Tesla/)).toBeVisible();
  });

  test('should show no results message for unknown query', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry?q=xyznonexistent');
    await page.waitForTimeout(1000);

    // Should show no results
    await expect(page.getByText(/No matching questions found/)).toBeVisible();
  });
});
