/**
 * Popup UI Script - Controls and displays extension status
 * UPDATED FOR POSTGRESQL MIGRATION
 */

// Configuration - UPDATE THIS WITH YOUR SERVICE-FARM URL
const SERVICE_FARM_URL = 'http://localhost:7272'; // Gen2 API

// DOM elements
const pendingCountEl = document.getElementById('pendingCount');
const completedCountEl = document.getElementById('completedCount');
const currentTaskSection = document.getElementById('currentTaskSection');
const currentTaskUrlEl = document.getElementById('currentTaskUrl');
const recentTasksListEl = document.getElementById('recentTasksList');
const extractCurrentBtn = document.getElementById('extractCurrentBtn');
const refreshBtn = document.getElementById('refreshBtn');
const togglePollingBtn = document.getElementById('togglePollingBtn');

// State
let isPolling = false;
let isProcessing = false;

/**
 * Update UI with current status
 */
async function updateStatus() {
  try {
    // Get status from background script
    const response = await chrome.runtime.sendMessage({ type: 'GET_STATUS' });

    isPolling = response.isPolling;
    isProcessing = response.isProcessing;

    // Update current task section
    if (isProcessing && response.currentTaskId) {
      currentTaskSection.style.display = 'block';
      currentTaskUrlEl.textContent = 'Task ID: ' + response.currentTaskId;
    } else {
      currentTaskSection.style.display = 'none';
    }

    // Update toggle button
    togglePollingBtn.textContent = isPolling ? '⏸️ Stop Polling' : '▶️ Start Polling';
    togglePollingBtn.className = 'button ' + (isPolling ? 'button-secondary' : 'button-primary');

    // Get task counts from Firestore (if available)
    await updateTaskCounts();

    // Update recent tasks list
    await updateRecentTasks();

  } catch (error) {
    console.error('Failed to update status:', error);
  }
}

/**
 * Update task counts from PostgreSQL (via API)
 */
async function updateTaskCounts() {
  try {
    // Use service-farm API to get stats
    const response = await fetch(`${SERVICE_FARM_URL}/api/rogue/stats`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const stats = await response.json();

    pendingCountEl.textContent = stats.by_status?.pending || 0;
    completedCountEl.textContent = stats.by_status?.completed || 0;

  } catch (error) {
    console.error('Failed to get task counts:', error);
    pendingCountEl.textContent = '?';
    completedCountEl.textContent = '?';
  }
}

/**
 * Update recent tasks list from PostgreSQL (via API)
 */
async function updateRecentTasks() {
  try {
    const response = await fetch(`${SERVICE_FARM_URL}/api/rogue/tasks?limit=5&recent=true`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const tasks = await response.json();

    if (!tasks || tasks.length === 0) {
      recentTasksListEl.innerHTML = '<div style="padding: 8px; color: #999; font-size: 11px;">No recent tasks</div>';
      return;
    }

    // Build task list HTML
    const html = tasks.map(task => {
      const urlHost = new URL(task.url).hostname.replace('www.', '');
      const apiUrl = `${SERVICE_FARM_URL}/api/artifacts/${task.page_id}`;

      return `
        <div class="task-item">
          <div class="task-url" title="${task.url}">${urlHost}</div>
          <a href="${apiUrl}" target="_blank" class="task-link">View</a>
          <span class="task-status ${task.status}">${task.status}</span>
        </div>
      `;
    }).join('');

    recentTasksListEl.innerHTML = html;

  } catch (error) {
    console.error('Failed to get recent tasks:', error);
    recentTasksListEl.innerHTML = '<div style="padding: 8px; color: #999; font-size: 11px;">Failed to load tasks</div>';
  }
}

/**
 * Format timestamp as "X ago"
 */
function formatTimeAgo(date) {
  const seconds = Math.floor((new Date() - date) / 1000);

  if (seconds < 60) return `${seconds}s ago`;

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

/**
 * Extract metadata from current active tab
 */
async function extractCurrentPage() {
  try {
    extractCurrentBtn.disabled = true;
    extractCurrentBtn.textContent = '⏳ Extracting...';

    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab?.url) {
      alert('No active tab found');
      return;
    }

    // Submit URL to service farm
    const response = await fetch(`${SERVICE_FARM_URL}/api/artifacts?url=${encodeURIComponent(tab.url)}`, {
      method: 'POST'
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();

    // Show success
    alert(`✅ Submitted! Page ID: ${result.page_id}`);

    // Refresh status
    await updateStatus();

  } catch (error) {
    console.error('Failed to extract current page:', error);
    const errorMsg = error.message || JSON.stringify(error) || 'Unknown error';
    alert('Failed to extract: ' + errorMsg);
  } finally {
    extractCurrentBtn.disabled = false;
    extractCurrentBtn.textContent = '📄 Extract Current Page';
  }
}

/**
 * Toggle polling on/off
 */
async function togglePolling() {
  try {
    const message = isPolling ? { type: 'STOP_POLLING' } : { type: 'START_POLLING' };
    await chrome.runtime.sendMessage(message);

    // Update UI
    await updateStatus();
  } catch (error) {
    console.error('Failed to toggle polling:', error);
    alert('Failed to toggle polling: ' + error.message);
  }
}

// Event listeners
extractCurrentBtn.addEventListener('click', extractCurrentPage);
refreshBtn.addEventListener('click', updateStatus);
togglePollingBtn.addEventListener('click', togglePolling);

// Initial status update
updateStatus();

// Auto-refresh every 5 seconds
setInterval(updateStatus, 5000);
