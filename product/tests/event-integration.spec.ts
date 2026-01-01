import { test, expect } from '@playwright/test';

/**
 * Event-Inquiry Integration Tests
 */

test.describe('Event-Inquiry Integration', () => {
  test('Homepage shows Active Events section with Wang Fuk Fire', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1500);

    // Check for Active Events section
    await expect(page.getByText('Active Events')).toBeVisible();
    await expect(page.getByText('Wang Fuk Court Fire')).toBeVisible();

    await page.screenshot({ path: 'screenshots/10-homepage-events.png', fullPage: true });
  });

  test('Navigate from homepage to event page', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Click on the event card
    await page.click('text=Wang Fuk Court Fire');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Verify we're on the event page
    await expect(page.url()).toContain('event-inquiry/wang-fuk-court-fire');

    await page.screenshot({ path: 'screenshots/11-event-page-full.png', fullPage: true });
  });

  test('Event page shows narrative sections with inquiry chips', async ({ page }) => {
    await page.goto('http://localhost:7272/event-inquiry/wang-fuk-court-fire');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1500);

    // Check for key sections
    await expect(page.getByText('The Incident')).toBeVisible();
    await expect(page.getByText('Casualties')).toBeVisible();
    await expect(page.getByText('Investigation & Arrests')).toBeVisible();

    // Check for embedded inquiry
    await expect(page.getByText('What is the confirmed death toll')).toBeVisible();

    await page.screenshot({ path: 'screenshots/12-event-page-above-fold.png' });
  });

  test('Event page casualties section with contested inquiry', async ({ page }) => {
    await page.goto('http://localhost:7272/event-inquiry/wang-fuk-court-fire');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Scroll to casualties section
    const casualtiesSection = page.getByText('Casualties').first();
    await casualtiesSection.scrollIntoViewIfNeeded();
    await page.waitForTimeout(500);

    await page.screenshot({ path: 'screenshots/13-event-casualties-section.png' });
  });

  test('Click inquiry chip navigates to inquiry detail', async ({ page }) => {
    await page.goto('http://localhost:7272/event-inquiry/wang-fuk-court-fire');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Click on the death toll inquiry
    await page.click('text=What is the confirmed death toll');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Verify navigation to inquiry detail
    await expect(page.url()).toContain('/inquiry/inq_death_toll');

    await page.screenshot({ path: 'screenshots/14-inquiry-from-event.png', fullPage: true });
  });

  test('Event page mobile view', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('http://localhost:7272/event-inquiry/wang-fuk-court-fire');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1500);

    await page.screenshot({ path: 'screenshots/15-event-mobile.png', fullPage: true });
  });
});
