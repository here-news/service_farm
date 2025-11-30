/**
 * Popup UI Script - Controls and displays extension status
 * UPDATED FOR POSTGRESQL MIGRATION
 */

// Configuration - UPDATE THIS WITH YOUR SERVICE-FARM URL
const SERVICE_FARM_URL = 'http://localhost:8000'; // Gen2 API (port 8000)

// DOM elements
const pendingCountEl = document.getElementById('pendingCount');
const completedCountEl = document.getElementById('completedCount');
const currentTaskSection = document.getElementById('currentTaskSection');
const currentTaskUrlEl = document.getElementById('currentTaskUrl');
const recentTasksListEl = document.getElementById('recentTasksList');
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
    const response = await fetch(`${SERVICE_FARM_URL}/api/v2/rogue/stats`);

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
  // TODO: Implement /api/v2/rogue/recent endpoint for recent tasks
  // For now, just show a simple message
  if (recentTasksListEl) {
    recentTasksListEl.innerHTML = '<div style="padding: 8px; color: #999; font-size: 11px;">Recent tasks view coming soon</div>';
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
refreshBtn.addEventListener('click', updateStatus);
togglePollingBtn.addEventListener('click', togglePolling);

// Initial status update
updateStatus();

// Auto-refresh every 5 seconds
setInterval(updateStatus, 5000);
