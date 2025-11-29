/**
 * Background Service Worker - Polls PostgreSQL (via API) for rogue extraction tasks
 *
 * UPDATED FOR POSTGRESQL MIGRATION - NO LONGER USES FIRESTORE
 *
 * Flow:
 * 1. Poll service-farm API every 3 seconds for pending tasks
 * 2. Pick up task (API atomically marks as "processing")
 * 3. Open URL in background tab with mobile User-Agent
 * 4. Content script extracts Open Graph metadata + full article text
 * 5. Upload metadata back to PostgreSQL via API
 * 6. Close tab and repeat
 */

// Configuration - UPDATE THIS WITH YOUR SERVICE-FARM URL
const SERVICE_FARM_URL = 'http://localhost:8080'; // Update for production

// Extension state
let isProcessing = false;
let currentTaskId = null;
let pollingInterval = null;

// Configuration
const POLL_INTERVAL_MS = 3000; // 3 seconds
const MAX_PROCESSING_TIME_MS = 60000; // 60 seconds timeout

/**
 * Start polling for tasks
 */
function startPolling() {
  console.log('🔄 Starting PostgreSQL task polling...');

  // Poll immediately on startup
  pollForTasks();

  // Then poll every 3 seconds
  pollingInterval = setInterval(pollForTasks, POLL_INTERVAL_MS);
}

/**
 * Stop polling
 */
function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval);
    pollingInterval = null;
    console.log('⏸️  Polling stopped');
  }
}

/**
 * Poll PostgreSQL (via API) for pending rogue extraction tasks
 */
async function pollForTasks() {
  // Skip if already processing a task
  if (isProcessing) {
    console.log('⏭️  Skipping poll - already processing task', currentTaskId);
    return;
  }

  try {
    console.log('🔍 Polling PostgreSQL for pending tasks...');

    // Poll API endpoint (atomically claims task)
    const response = await fetch(`${SERVICE_FARM_URL}/api/rogue/poll`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (response.status === 404) {
      // No pending tasks
      console.log('✅ No pending tasks');
      updateBadge(0);
      return;
    }

    if (!response.ok) {
      throw new Error(`Poll failed: ${response.status} ${response.statusText}`);
    }

    const task = await response.json();
    currentTaskId = task.task_id;

    console.log(`📋 Found task: ${currentTaskId}`, task.url);

    // Process the task
    await processTask(task.url, currentTaskId);

  } catch (error) {
    console.error('❌ Polling error:', error);

    // Reset state on error
    if (currentTaskId) {
      try {
        // Mark task as failed so it can be retried
        const formData = new FormData();
        formData.append('status', 'failed');
        formData.append('error_message', error.message);

        await fetch(`${SERVICE_FARM_URL}/api/rogue/${currentTaskId}`, {
          method: 'PATCH',
          body: formData
        });
      } catch (updateError) {
        console.error('❌ Failed to update task status:', updateError);
      }
    }

    isProcessing = false;
    currentTaskId = null;
  }
}

/**
 * Process a rogue URL extraction task
 */
async function processTask(url, taskId) {
  isProcessing = true;
  updateBadge('⚙️');
  let tab = null;
  let ruleId = null;

  try {
    console.log(`🌐 Opening URL: ${url}`);

    // Create tab directly with target URL
    tab = await chrome.tabs.create({
      url: url,
      active: false // Don't switch to the tab
    });

    // Set mobile User-Agent for this tab (matches Playwright: iPhone 14 Pro Max)
    ruleId = Math.floor(Math.random() * 1000000) + 1;
    await chrome.declarativeNetRequest.updateSessionRules({
      addRules: [{
        id: ruleId,
        priority: 1,
        action: {
          type: 'modifyHeaders',
          requestHeaders: [
            {
              header: 'User-Agent',
              operation: 'set',
              value: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1'
            }
          ]
        },
        condition: {
          tabIds: [tab.id],
          resourceTypes: ['main_frame', 'sub_frame', 'xmlhttprequest', 'script', 'stylesheet', 'image']
        }
      }],
      removeRuleIds: []
    });

    console.log(`📱 Mobile User-Agent set for tab ${tab.id} (iPhone 14 Pro Max)`);

    // Wait for page to load and content script to extract metadata
    const metadata = await waitForMetadata(tab.id, taskId);

    if (metadata) {
      console.log(`✅ Metadata extracted:`, metadata);

      // Save metadata to PostgreSQL via API
      const formData = new FormData();
      formData.append('status', 'completed');
      formData.append('metadata', JSON.stringify(metadata));

      const response = await fetch(`${SERVICE_FARM_URL}/api/rogue/${taskId}`, {
        method: 'PATCH',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Failed to update task: ${response.status} ${response.statusText}`);
      }

      console.log(`✅ Task completed: ${taskId}`);
    } else {
      throw new Error('No metadata received from content script');
    }

  } catch (error) {
    console.error(`❌ Task failed: ${taskId}`, error);

    // Mark task as failed
    const formData = new FormData();
    formData.append('status', 'failed');
    formData.append('error_message', error.message);

    await fetch(`${SERVICE_FARM_URL}/api/rogue/${taskId}`, {
      method: 'PATCH',
      body: formData
    });

  } finally {
    // Cleanup: Remove the User-Agent rule
    if (ruleId) {
      try {
        await chrome.declarativeNetRequest.updateSessionRules({
          addRules: [],
          removeRuleIds: [ruleId]
        });
        console.log(`🗑️  Removed User-Agent rule ${ruleId}`);
      } catch (ruleError) {
        console.warn(`⚠️  Failed to remove rule ${ruleId}:`, ruleError.message);
      }
    }

    // Cleanup: ALWAYS close the tab (success or failure)
    if (tab) {
      try {
        await chrome.tabs.remove(tab.id);
        console.log(`🗑️  Closed tab ${tab.id}`);
      } catch (closeError) {
        console.warn(`⚠️  Failed to close tab ${tab.id}:`, closeError.message);
      }
    }

    isProcessing = false;
    currentTaskId = null;
    updateBadge(0);
  }
}

/**
 * Wait for content script to extract and send metadata
 */
function waitForMetadata(tabId, taskId) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error('Metadata extraction timeout (60s) - content script did not respond'));
    }, MAX_PROCESSING_TIME_MS);

    // Listen for message from content script
    const messageListener = (message, sender) => {
      if (sender.tab?.id === tabId) {
        if (message.type === 'METADATA_EXTRACTED') {
          cleanup();

          // Validate metadata has useful content
          if (!message.metadata || (!message.metadata.title && !message.metadata.content_text)) {
            reject(new Error('Extracted metadata is empty (no title or content)'));
            return;
          }

          resolve(message.metadata);
        } else if (message.type === 'METADATA_EXTRACTION_ERROR') {
          cleanup();
          reject(new Error(`Content script error: ${message.error}`));
        }
      }
    };

    const cleanup = () => {
      clearTimeout(timeout);
      chrome.runtime.onMessage.removeListener(messageListener);
    };

    chrome.runtime.onMessage.addListener(messageListener);

    // Inject Readability library first, then content script
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      files: ['scripts/Readability.js']
    }).then(() => {
      console.log('✅ Readability.js injected');

      // Then inject content script
      return chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ['scripts/content.js']
      });
    }).then(() => {
      console.log('✅ content.js injected');
    }).catch(error => {
      cleanup();
      console.error('❌ Script injection failed:', error);

      // Provide more helpful error messages
      if (error.message.includes('Cannot access')) {
        reject(new Error(`Cannot inject script on this page (likely CSP/permissions issue): ${error.message}`));
      } else {
        reject(new Error(`Script injection failed: ${error.message}`));
      }
    });
  });
}

/**
 * Update extension badge to show status
 */
function updateBadge(text) {
  chrome.action.setBadgeText({ text: String(text) });
  chrome.action.setBadgeBackgroundColor({
    color: text === '⚙️' ? '#FF9800' : '#4CAF50'
  });
}

/**
 * Listen for messages from popup
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'GET_STATUS') {
    sendResponse({
      isProcessing,
      currentTaskId,
      isPolling: pollingInterval !== null
    });
  } else if (message.type === 'START_POLLING') {
    startPolling();
    sendResponse({ success: true });
  } else if (message.type === 'STOP_POLLING') {
    stopPolling();
    sendResponse({ success: true });
  }
  return true; // Keep channel open for async response
});

// Start polling when extension loads
chrome.runtime.onStartup.addListener(startPolling);
chrome.runtime.onInstalled.addListener(startPolling);

console.log('🚀 HERE News Rogue URL Extractor - PostgreSQL Mode - Background worker initialized');
