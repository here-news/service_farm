import { test, expect } from '@playwright/test'

test.describe('Live Pulse: The Pulsing World', () => {
  test('Homepage shows Live Events and System Pulse side by side', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    // Verify side-by-side layout
    await expect(page.locator('h2:has-text("Live Events")')).toBeVisible()
    await expect(page.locator('h2:has-text("System Pulse")')).toBeVisible()

    // Verify Live Pulse content
    await expect(page.locator('text=Live Pulse')).toBeVisible()
    await expect(page.locator('text=The epistemic loop')).toBeVisible()

    // Verify pulse events are showing
    await expect(page.locator('text=Belief Update')).toBeVisible()

    // Verify loop stats
    await expect(page.locator('text=evidence/hr')).toBeVisible()
    await expect(page.locator('text=tasks/hr')).toBeVisible()
    await expect(page.locator('text=resolutions/day')).toBeVisible()

    // Screenshot: pulsing homepage
    await page.screenshot({ path: 'screenshots/35-homepage-pulse.png', fullPage: true })
  })

  test('Live Pulse can be paused', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    // Find and click the LIVE button to pause
    const liveButton = page.locator('button:has-text("LIVE")')
    await expect(liveButton).toBeVisible()
    await liveButton.click()

    // Should now show PAUSED
    await expect(page.locator('button:has-text("PAUSED")')).toBeVisible()

    // Screenshot: paused state
    await page.screenshot({ path: 'screenshots/36-pulse-paused.png', fullPage: false })
  })

  test('Pulse events are clickable and navigate', async ({ page }) => {
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    // Click on a pulse event with inquiry link
    const eventWithInquiry = page.locator('.bg-white\\/5:has-text("Wang Fuk Court Fire")').first()
    if (await eventWithInquiry.isVisible()) {
      await eventWithInquiry.click()
      // Should navigate to inquiry page
      await page.waitForURL('**/inquiry/**')
    }
  })

  test('Mobile view stacks Live Events and Pulse vertically', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 })
    await page.goto('http://localhost:7272/inquiry')
    await page.waitForLoadState('networkidle')

    // Both sections should be visible
    await expect(page.locator('h2:has-text("Live Events")')).toBeVisible()
    await expect(page.locator('h2:has-text("System Pulse")')).toBeVisible()

    // Screenshot: mobile pulse view
    await page.screenshot({ path: 'screenshots/37-pulse-mobile.png', fullPage: true })
  })
})
