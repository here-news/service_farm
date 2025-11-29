/**
 * Background Service Worker - Polls Firestore for rogue extraction tasks
 *
 * Flow:
 * 1. Poll Firestore every 3 seconds for pending tasks (fast pickup!)
 * 2. Pick up task and mark as "processing"
 * 3. Open URL in background tab (single navigation, no debugger overhead)
 * 4. Content script extracts Open Graph metadata + full article text
 * 5. Upload metadata back to Firestore
 * 6. Close tab and repeat
 */

import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js';
import { getFirestore, collection, query, where, orderBy, limit, getDocs, doc, updateDoc, Timestamp } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js';

// Firebase configuration (replace with your project config)
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "here2-474221.firebaseapp.com",
  projectId: "here2-474221",
  storageBucket: "here2-474221.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// Extension state
let isProcessing = false;
let currentTaskId = null;
let pollingInterval = null;

// Configuration
const POLL_INTERVAL_MS = 3000; // 3 seconds (10x faster!)
const MAX_PROCESSING_TIME_MS = 60000; // 60 seconds timeout

/**
 * Start polling for tasks
 */
function startPolling() {
  console.log('🔄 Starting task polling...');

  // Poll immediately on startup
  pollForTasks();

  // Then poll every 3 seconds (fast pickup for instant response)
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
 * Poll Firestore for pending rogue extraction tasks
 */
async function pollForTasks() {
  // Skip if already processing a task
  if (isProcessing) {
    console.log('⏭️  Skipping poll - already processing task', currentTaskId);
    return;
  }

  try {
    console.log('🔍 Polling for pending tasks...');

    // Query for pending tasks (oldest first)
    const tasksRef = collection(db, 'rogue_extraction_tasks');
    const q = query(
      tasksRef,
      where('status', '==', 'pending'),
      orderBy('created_at', 'asc'),
      limit(1)
    );

    const snapshot = await getDocs(q);

    if (snapshot.empty) {
      console.log('✅ No pending tasks');
      updateBadge(0);
      return;
    }

    // Pick up the first pending task
    const taskDoc = snapshot.docs[0];
    const task = taskDoc.data();
    currentTaskId = taskDoc.id;

    console.log(`📋 Found task: ${currentTaskId}`, task.url);

    // Mark task as processing
    await updateDoc(doc(db, 'rogue_extraction_tasks', currentTaskId), {
      status: 'processing',
      assigned_to: chrome.runtime.id,
      processing_started_at: Timestamp.now()
    });

    // Process the task
    await processTask(task.url, currentTaskId);

  } catch (error) {
    console.error('❌ Polling error:', error);

    // Reset state on error
    if (currentTaskId) {
      try {
        await updateDoc(doc(db, 'rogue_extraction_tasks', currentTaskId), {
          status: 'pending', // Reset to pending so another poll can pick it up
          error_message: error.message
        });
      } catch (updateError) {
        console.error('❌ Failed to reset task status:', updateError);
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

    // Create tab directly with target URL (avoid about:blank permission issue)
    tab = await chrome.tabs.create({
      url: url,
      active: false // Don't switch to the tab
    });

    // Set mobile User-Agent for this tab using declarativeNetRequest (fast, no debugger!)
    // This matches our Playwright setup: iPhone 14 Pro Max
    // Use simple incremental ID (Chrome API expects strict integer, not timestamp)
    ruleId = Math.floor(Math.random() * 1000000) + 1; // Random integer 1-1000000
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

      // Save metadata to Firestore
      await updateDoc(doc(db, 'rogue_extraction_tasks', taskId), {
        status: 'completed',
        metadata: metadata,
        completed_at: Timestamp.now()
      });

      console.log(`✅ Task completed: ${taskId}`);
    } else {
      throw new Error('No metadata received from content script');
    }

  } catch (error) {
    console.error(`❌ Task failed: ${taskId}`, error);

    // Mark task as failed
    await updateDoc(doc(db, 'rogue_extraction_tasks', taskId), {
      status: 'failed',
      error_message: error.message,
      failed_at: Timestamp.now()
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

console.log('🚀 HERE News Rogue URL Extractor - Background worker initialized');
