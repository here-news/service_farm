import { test, expect } from '@playwright/test'

test.describe('Homepage', () => {
  test('Shows entities and breaking events at a glance', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    // Entities are showing (use heading to be specific)
    await expect(page.getByRole('heading', { name: 'Elon Musk' })).toBeVisible()

    // Breaking event
    await expect(page.locator('text=Assad Regime Collapses')).toBeVisible()

    // Screenshot
    await page.screenshot({ path: 'screenshots/42-homepage-entities.png', fullPage: true })
  })

  test('Mobile YOLO dashboard', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 })
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    await expect(page.locator('h1:has-text("φ HERE")')).toBeVisible()
    await expect(page.locator('text=Now').first()).toBeVisible()

    await page.screenshot({ path: 'screenshots/41-yolo-mobile.png', fullPage: true })
  })
})
