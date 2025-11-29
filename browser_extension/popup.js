/**
 * Popup UI Script - Controls and displays extension status
 * UPDATED FOR POSTGRESQL MIGRATION
 */

// Configuration - UPDATE THIS WITH YOUR SERVICE-FARM URL
const SERVICE_FARM_URL = 'http://localhost:8080'; // Update for production

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
    const response = await fetch(`${SERVICE_FARM_URL}/api/rogue/stats`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const stats = await response.json();

    pendingCountEl.textContent = stats.pending || 0;
    completedCountEl.textContent = stats.completed || 0;

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
    // Use service-farm API to get recent completed tasks
    const response = await fetch(`${SERVICE_FARM_URL}/api/rogue/recent?limit=5`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const completedTasks = await response.json();

    if (completedTasks.length === 0) {
      recentTasksListEl.innerHTML = '<div style="padding: 8px; color: #999;">No completed tasks yet</div>';
      return;
    }

    // Build HTML for recent tasks
    const tasksHtml = completedTasks.map(task => {
      const urlShort = task.url.length > 50 ? task.url.substring(0, 50) + '...' : task.url;
      const timeAgo = task.completed_at ? formatTimeAgo(new Date(task.completed_at)) : 'Unknown time';
      const wordCountClass = task.word_count >= 200 ? 'active' : 'inactive';

      return `
        <div class="status-row task-row-clickable" data-task-id="${task.task_id}" style="cursor: pointer;" title="Click to copy task ID: ${task.task_id}">
          <div style="flex: 1; min-width: 0;">
            <div style="font-size: 11px; color: #333; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${task.url}">${urlShort}</div>
            <div style="font-size: 10px; color: #999; margin-top: 2px;">${timeAgo}</div>
          </div>
          <div class="status-value ${wordCountClass}" style="font-size: 12px; margin-left: 10px;">
            ${task.word_count} words
          </div>
        </div>
      `;
    }).join('');

    recentTasksListEl.innerHTML = tasksHtml;

    // Add click handlers to copy task ID
    document.querySelectorAll('.task-row-clickable').forEach(row => {
      row.addEventListener('click', async function() {
        const taskId = this.getAttribute('data-task-id');
        try {
          await navigator.clipboard.writeText(taskId);

          // Visual feedback
          const originalBg = this.style.backgroundColor;
          this.style.backgroundColor = '#d4edda';
          setTimeout(() => {
            this.style.backgroundColor = originalBg;
          }, 300);

          console.log('Copied task ID:', taskId);
        } catch (err) {
          console.error('Failed to copy task ID:', err);
        }
      });
    });

  } catch (error) {
    console.error('Failed to get recent tasks:', error);
    recentTasksListEl.innerHTML = '<div style="padding: 8px; color: #999;">Error loading tasks</div>';
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
