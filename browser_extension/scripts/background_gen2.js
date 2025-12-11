/**
 * Background Service Worker - Gen2 PostgreSQL/REST API version
 *
 * Polls REST API instead of Firestore for rogue extraction tasks
 *
 * Flow:
 * 1. Poll GET /api/rogue/tasks every 3 seconds
 * 2. Pick up task and open URL in background tab
 * 3. Content script extracts metadata + article text
 * 4. POST /api/rogue/tasks/{id}/complete with metadata
 * 5. Close tab and repeat
 */

// Configuration
const API_BASE_URL = 'http://localhost:7272';  // Update for production
const POLL_INTERVAL_MS = 3000; // 3 seconds
const MAX_PROCESSING_TIME_MS = 60000; // 60 seconds timeout

// Extension state
let isProcessing = false;
let currentTaskId = null;
let currentTabId = null;
let pollingInterval = null;

/**
 * Start polling for tasks
 */
function startPolling() {
  console.log('🔄 Starting task polling (Gen2 REST API)...');

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
 * Poll REST API for pending rogue extraction tasks
 */
async function pollForTasks() {
  // Skip if already processing
  if (isProcessing) {
    console.log('⏭️  Skipping poll - already processing task', currentTaskId);
    return;
  }

  try {
    console.log('🔍 Polling for pending tasks...');

    // GET /api/rogue/tasks
    const response = await fetch(`${API_BASE_URL}/api/rogue/tasks?limit=1`);

    if (!response.ok) {
      console.error('❌ API error:', response.status, await response.text());
      updateBadge(0, true);
      return;
    }

    const tasks = await response.json();

    if (!tasks || tasks.length === 0) {
      console.log('✅ No pending tasks');
      updateBadge(0);
      return;
    }

    // Pick up the first task
    const task = tasks[0];
    currentTaskId = task.id;

    console.log(`📋 Found task: ${currentTaskId}`, task.url);

    // Process task
    await processTask(task);

  } catch (error) {
    console.error('❌ Polling error:', error);
    updateBadge(0, true);
  }
}

/**
 * Process a rogue extraction task
 */
async function processTask(task) {
  isProcessing = true;
  updateBadge(1);

  try {
    console.log(`🌐 Opening URL: ${task.url}`);

    // Open URL in background tab
    const tab = await chrome.tabs.create({
      url: task.url,
      active: false
    });

    currentTabId = tab.id;

    // Wait for content script to extract metadata
    // Content script will send message when done
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Timeout waiting for extraction')), MAX_PROCESSING_TIME_MS);
    });

    const extractionPromise = new Promise((resolve, reject) => {
      chrome.runtime.onMessage.addListener(function listener(message, sender, sendResponse) {
        if (message.type === 'METADATA_EXTRACTED' && sender.tab?.id === currentTabId) {
          chrome.runtime.onMessage.removeListener(listener);
          sendResponse({ received: true });
          resolve(message.metadata);
        } else if (message.type === 'METADATA_EXTRACTION_ERROR' && sender.tab?.id === currentTabId) {
          chrome.runtime.onMessage.removeListener(listener);
          sendResponse({ received: true });
          reject(new Error(message.error));
        }
      });
    });

    const metadata = await Promise.race([extractionPromise, timeoutPromise]);

    console.log('✅ Metadata extracted:', metadata);

    // Send metadata back to API
    await completeTask(task.id, metadata);

    // Close tab
    await chrome.tabs.remove(currentTabId);
    console.log('🗑️  Tab closed');

  } catch (error) {
    console.error('❌ Task processing failed:', error);

    // Mark task as failed
    await failTask(task.id, error.message);

    // Close tab if still open
    if (currentTabId) {
      try {
        await chrome.tabs.remove(currentTabId);
      } catch (e) {
        // Tab might already be closed
      }
    }
  } finally {
    isProcessing = false;
    currentTaskId = null;
    currentTabId = null;
    updateBadge(0);
  }
}

/**
 * Complete task by sending metadata to API
 */
async function completeTask(taskId, metadata) {
  console.log(`📤 Completing task ${taskId}`);

  const response = await fetch(`${API_BASE_URL}/api/rogue/tasks/${taskId}/complete`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(metadata)
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${await response.text()}`);
  }

  const result = await response.json();
  console.log('✅ Task completed:', result);
}

/**
 * Mark task as failed
 */
async function failTask(taskId, errorMessage) {
  console.log(`❌ Failing task ${taskId}:`, errorMessage);

  try {
    const response = await fetch(`${API_BASE_URL}/api/rogue/tasks/${taskId}/fail`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ error_message: errorMessage })
    });

    if (response.ok) {
      console.log('✅ Task marked as failed');
    }
  } catch (error) {
    console.error('Failed to mark task as failed:', error);
  }
}

/**
 * Update extension badge
 */
function updateBadge(count, error = false) {
  chrome.action.setBadgeText({ text: count > 0 ? count.toString() : '' });
  chrome.action.setBadgeBackgroundColor({
    color: error ? '#FF0000' : count > 0 ? '#4CAF50' : '#999999'
  });
}

/**
 * Handle messages from popup
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Ignore stray metadata messages (e.g., content script running on non-task tabs)
  if (message.type === 'METADATA_EXTRACTED' || message.type === 'METADATA_EXTRACTION_ERROR') {
    if (!currentTabId || sender.tab?.id !== currentTabId) {
      console.log('ℹ️  Received metadata message from unrelated tab, ignoring:', sender.tab?.id);
      sendResponse({ ignored: true });
      return true;
    }
  }

  if (message.type === 'GET_STATUS') {
    sendResponse({
      isPolling: pollingInterval !== null,
      isProcessing: isProcessing,
      currentTaskId: currentTaskId
    });
    return true;
  }

  if (message.type === 'START_POLLING') {
    if (!pollingInterval) {
      startPolling();
    }
    sendResponse({ success: true });
    return true;
  }

  if (message.type === 'STOP_POLLING') {
    stopPolling();
    sendResponse({ success: true });
    return true;
  }
});

// Start polling when extension loads
chrome.runtime.onStartup.addListener(startPolling);
chrome.runtime.onInstalled.addListener(startPolling);

// Immediately start polling (for development reload)
startPolling();

console.log('🚀 HERE News Rogue Extractor Gen2 - Ready');
