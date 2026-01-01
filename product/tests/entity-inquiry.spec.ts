import { test, expect } from '@playwright/test';

/**
 * Entity Inquiry Page Tests
 */

test.describe('Entity Inquiry Pages', () => {
  test('Elon Musk entity page loads with inquiries', async ({ page }) => {
    await page.goto('http://localhost:7272/entity-inquiry/elon-musk');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1500);

    // Check entity header
    await expect(page.getByText('Elon Musk')).toBeVisible();
    await expect(page.getByText('PERSON')).toBeVisible();

    // Check inquiries
    await expect(page.getByText('How many children does Elon Musk have?')).toBeVisible();

    await page.screenshot({ path: 'screenshots/20-entity-elon-musk.png', fullPage: true });
  });

  test('Japan entity page loads with inquiries', async ({ page }) => {
    await page.goto('http://localhost:7272/entity-inquiry/japan');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1500);

    // Check entity header
    await expect(page.getByText('Japan')).toBeVisible();
    await expect(page.getByText('LOCATION')).toBeVisible();

    // Check inquiries
    await expect(page.getByText('Is there a tsunami warning in Japan today?')).toBeVisible();

    await page.screenshot({ path: 'screenshots/21-entity-japan.png', fullPage: true });
  });

  test('Inquiry wizard opens and shows steps', async ({ page }) => {
    await page.goto('http://localhost:7272/entity-inquiry/elon-musk');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Click Ask Question button
    await page.click('text=Ask Question');
    await page.waitForTimeout(500);

    // Check wizard appeared
    await expect(page.getByText('Create Inquiry')).toBeVisible();
    await expect(page.getByText('Your Question')).toBeVisible();

    await page.screenshot({ path: 'screenshots/22-inquiry-wizard-step1.png' });
  });

  test('Inquiry wizard - answer type step', async ({ page }) => {
    await page.goto('http://localhost:7272/entity-inquiry/elon-musk');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Open wizard
    await page.click('text=Ask Question');
    await page.waitForTimeout(500);

    // Type a question
    await page.fill('textarea', 'How many companies does Elon Musk currently lead?');
    await page.waitForTimeout(500);

    // Continue to next step
    await page.click('text=Continue');
    await page.waitForTimeout(300);

    // Check we're on answer type step
    await expect(page.getByText('Answer Type')).toBeVisible();
    await expect(page.getByText('Number')).toBeVisible();

    await page.screenshot({ path: 'screenshots/23-inquiry-wizard-step2.png' });
  });

  test('Inquiry wizard - context step', async ({ page }) => {
    await page.goto('http://localhost:7272/entity-inquiry/japan');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Open wizard
    await page.click('text=Ask Question');
    await page.waitForTimeout(500);

    // Fill question
    await page.fill('textarea', 'What was the magnitude of the largest earthquake in Japan this year?');

    // Continue through steps
    await page.click('text=Continue');
    await page.waitForTimeout(300);
    await page.click('text=Continue'); // Skip type selection
    await page.waitForTimeout(300);

    // Check context step
    await expect(page.getByText('Time reference')).toBeVisible();
    await expect(page.getByText('Today')).toBeVisible();

    await page.screenshot({ path: 'screenshots/24-inquiry-wizard-step3.png' });
  });

  test('Similar inquiries detection', async ({ page }) => {
    await page.goto('http://localhost:7272/entity-inquiry/elon-musk');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);

    // Open wizard
    await page.click('text=Ask Question');
    await page.waitForTimeout(500);

    // Type a question similar to existing one
    await page.fill('textarea', 'How many children does Musk have?');
    await page.waitForTimeout(1000);

    // Check similar inquiry warning appears
    await expect(page.getByText('Similar questions already exist')).toBeVisible();

    await page.screenshot({ path: 'screenshots/25-similar-inquiries-warning.png' });
  });

  test('Entity page mobile view', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('http://localhost:7272/entity-inquiry/elon-musk');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1500);

    await page.screenshot({ path: 'screenshots/26-entity-mobile.png', fullPage: true });
  });
});
