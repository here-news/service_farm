import { test, expect } from '@playwright/test';

/**
 * UI Analysis Tests - Capture screenshots and analyze layout
 */

test.describe('UI Analysis - Capture Current State', () => {
  test('Capture inquiry list page - full', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'screenshots/01-inquiry-list-full.png', fullPage: true });
  });

  test('Capture inquiry list page - above fold', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'screenshots/02-inquiry-list-fold.png' });
  });

  test('Capture inquiry detail - bounty inquiry', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'screenshots/03-detail-bounty-full.png', fullPage: true });
  });

  test('Capture inquiry detail - above fold', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'screenshots/04-detail-bounty-fold.png' });
  });

  test('Capture resolved inquiry', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry/sim_resolved_1');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'screenshots/05-detail-resolved.png', fullPage: true });
  });

  test('Capture mobile list view', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('http://localhost:7272/inquiry');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'screenshots/06-mobile-list.png', fullPage: true });
  });

  test('Capture mobile detail view', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'screenshots/07-mobile-detail.png', fullPage: true });
  });

  test('Capture full contested inquiry page', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry/sim_contested_1');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1500);
    // Full page screenshot to see everything
    await page.screenshot({ path: 'screenshots/08-contested-full.png', fullPage: true });
  });

  test('Capture claim clusters with relationship symbols', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry/sim_bounty_1');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1500);
    // Scroll down to see claim clusters
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(500);
    await page.screenshot({ path: 'screenshots/09-claim-clusters.png' });
  });
});
