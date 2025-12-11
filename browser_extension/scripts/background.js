/**
 * Background Service Worker - Polls PostgreSQL API for rogue extraction tasks
 *
 * Flow:
 * 1. Poll API every 3 seconds for pending tasks
 * 2. Pick up task (API marks as "processing")
 * 3. Open URL in background tab with mobile User-Agent
 * 4. Content script extracts Open Graph metadata + full article text
 * 5. Upload metadata back via API
 * 6. Close tab and repeat
 */

// API Configuration - set this to your server
const API_BASE_URL = 'http://localhost:7272/api';

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
  console.log('🔄 Starting task polling...');

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
 * Poll API for pending rogue extraction tasks
 */
async function pollForTasks() {
  // Skip if already processing a task
  if (isProcessing) {
    console.log('⏭️  Skipping poll - already processing task', currentTaskId);
    return;
  }

  try {
    console.log('🔍 Polling for pending tasks...');

    // GET /api/rogue/tasks returns pending tasks and marks them as processing
    const response = await fetch(`${API_BASE_URL}/rogue/tasks?limit=1`);

    if (!response.ok) {
      console.error('❌ API error:', response.status);
      return;
    }

    const tasks = await response.json();

    if (!tasks || tasks.length === 0) {
      console.log('✅ No pending tasks');
      updateBadge(0);
      return;
    }

    // Pick up the first task (already marked as processing by API)
    const task = tasks[0];
    currentTaskId = task.id;

    console.log(`📋 Found task: ${currentTaskId}`, task.url);

    // Process the task
    await processTask(task.url, currentTaskId);

  } catch (error) {
    console.error('❌ Polling error:', error);
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

    // Create tab with target URL
    tab = await chrome.tabs.create({
      url: url,
      active: false // Don't switch to the tab
    });

    // Set mobile User-Agent for this tab using declarativeNetRequest
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

      // POST metadata to API
      const completeResponse = await fetch(`${API_BASE_URL}/rogue/tasks/${taskId}/complete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(metadata)
      });

      if (!completeResponse.ok) {
        throw new Error(`Failed to complete task: ${completeResponse.status}`);
      }

      console.log(`✅ Task completed: ${taskId}`);
    } else {
      throw new Error('No metadata received from content script');
    }

  } catch (error) {
    console.error(`❌ Task failed: ${taskId}`, error);

    // Mark task as failed via API
    try {
      await fetch(`${API_BASE_URL}/rogue/tasks/${taskId}/fail`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ error_message: error.message })
      });
    } catch (failError) {
      console.error('❌ Failed to mark task as failed:', failError);
    }

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

    // Cleanup: ALWAYS close the tab
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
      reject(new Error('Metadata extraction timeout'));
    }, MAX_PROCESSING_TIME_MS);

    // Listen for message from content script
    const messageListener = (message, sender) => {
      if (sender.tab?.id === tabId && message.type === 'METADATA_EXTRACTED') {
        cleanup();
        resolve(message.metadata);
      }
    };

    const cleanup = () => {
      clearTimeout(timeout);
      chrome.runtime.onMessage.removeListener(messageListener);
    };

    chrome.runtime.onMessage.addListener(messageListener);

    // Inject content script to extract metadata
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      files: ['scripts/content.js']
    }).catch(error => {
      cleanup();
      reject(error);
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
      isPolling: pollingInterval !== null,
      apiBaseUrl: API_BASE_URL
    });
  } else if (message.type === 'START_POLLING') {
    startPolling();
    sendResponse({ success: true });
  } else if (message.type === 'STOP_POLLING') {
    stopPolling();
    sendResponse({ success: true });
  } else if (message.type === 'SET_API_URL') {
    // Allow popup to configure API URL
    // Note: This won't persist across restarts - need storage API for that
    sendResponse({ success: true, note: 'URL change not persisted' });
  }
  return true; // Keep channel open for async response
});

// Start polling when extension loads
chrome.runtime.onStartup.addListener(startPolling);
chrome.runtime.onInstalled.addListener(startPolling);

console.log('🚀 HERE News Rogue URL Extractor - Background worker initialized');
console.log(`📡 API endpoint: ${API_BASE_URL}`);
