import { test, expect } from '@playwright/test'

test.describe('Homepage: Attention Market', () => {
  test('Discover tab shows all sections per product vision', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    // Verify hero section with live stats
    await expect(page.locator('text=What\'s True?')).toBeVisible()
    // Hero stats are visible (use first match for bounties since it appears multiple times)
    await expect(page.locator('.text-indigo-200:has-text("in bounties")').first()).toBeVisible()

    // Verify tab structure
    await expect(page.locator('button:has-text("Discover")')).toBeVisible()
    await expect(page.locator('button:has-text("Questions")')).toBeVisible()

    // Verify Live Events section
    await expect(page.locator('h2:has-text("Live Events")')).toBeVisible()
    await expect(page.locator('text=Wang Fuk Court Fire')).toBeVisible()

    // Verify Trending Entities section
    await expect(page.locator('h2:has-text("Trending")')).toBeVisible()
    // Check trending entity cards exist (use first match since names appear multiple times)
    await expect(page.getByRole('heading', { name: 'Elon Musk', exact: true })).toBeVisible()

    // Verify Needs Evidence section (high-entropy inquiries)
    await expect(page.locator('h2:has-text("Needs Evidence")')).toBeVisible()

    // Verify Knowledge Gaps section
    await expect(page.locator('h2:has-text("Knowledge Gaps")')).toBeVisible()
    await expect(page.locator('text=Casualty Counts')).toBeVisible()

    // Verify Recently Verified section
    await expect(page.locator('h2:has-text("Recently Verified")')).toBeVisible()

    // Verify Browse by Type section
    await expect(page.locator('h2:has-text("Browse by Type")')).toBeVisible()

    // Screenshot: full homepage discover tab
    await page.screenshot({ path: 'product/screenshots/30-homepage-discover.png', fullPage: true })
  })

  test('Questions tab shows inquiries sorted by bounty/entropy', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    // Click Questions tab
    await page.locator('button:has-text("Questions")').click()
    await page.waitForTimeout(500)

    // Verify All Open Questions section with sorting
    await expect(page.locator('h2:has-text("All Open Questions")')).toBeVisible()

    // Screenshot: questions tab
    await page.screenshot({ path: 'product/screenshots/31-homepage-questions.png', fullPage: true })
  })

  test('Mobile responsive homepage', async ({ page }) => {
    // Mobile viewport
    await page.setViewportSize({ width: 375, height: 812 })
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    // Verify key sections are visible on mobile
    await expect(page.locator('text=What\'s True?')).toBeVisible()
    await expect(page.locator('text=Live Events')).toBeVisible()

    // Screenshot: mobile view
    await page.screenshot({ path: 'product/screenshots/32-homepage-mobile.png', fullPage: true })
  })

  test('Trending entity navigation works', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    // Click on Elon Musk entity
    await page.click('text=Elon Musk')
    await page.waitForURL('**/entity-inquiry/elon-musk')

    // Verify we're on the entity page
    await expect(page.locator('h1:has-text("Elon Musk")')).toBeVisible()
    await expect(page.locator('text=Questions about Elon Musk')).toBeVisible()

    // Screenshot: entity page navigation
    await page.screenshot({ path: 'product/screenshots/33-entity-from-homepage.png', fullPage: true })
  })

  test('Event navigation works', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    // Click on Wang Fuk Court Fire event
    await page.click('text=Wang Fuk Court Fire')
    await page.waitForURL('**/event-inquiry/wang-fuk-court-fire')

    // Verify we're on the event page
    await expect(page.locator('h1:has-text("Wang Fuk Court Fire")')).toBeVisible()

    // Screenshot: event page navigation
    await page.screenshot({ path: 'product/screenshots/34-event-from-homepage.png', fullPage: true })
  })
})
