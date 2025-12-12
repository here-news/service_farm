import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { formatTime } from '../../utils/timeFormat';

interface Thread {
  id: string;
  event_id: string;
  user_id: string;
  user_name: string;
  user_picture?: string;
  title: string;
  content: string;
  endowment: number;
  reply_count: number;
  created_at: string;
}

interface Comment {
  id: string;
  thread_id: string;
  user_id: string;
  user_name: string;
  user_picture?: string;
  text: string;
  parent_comment_id?: string;
  created_at: string;
}

interface DebateThreadProps {
  eventId: string;
}

// Thread Starter Box with Credit Endowment
function ThreadStarterBox({
  onSubmit,
  submitting
}: {
  onSubmit: (title: string, content: string, endowment: number) => Promise<void>;
  submitting: boolean;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [endowment, setEndowment] = useState(10);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = async () => {
    const trimmedTitle = title.trim();
    const trimmedContent = content.trim();
    if (!trimmedTitle || !trimmedContent || submitting) return;

    await onSubmit(trimmedTitle, trimmedContent, endowment);
    setTitle('');
    setContent('');
    setEndowment(10);
    setIsExpanded(false);
  };

  const endowmentOptions = [5, 10, 25, 50, 100];

  if (!isExpanded) {
    return (
      <button
        onClick={() => setIsExpanded(true)}
        className="w-full bg-gradient-to-r from-indigo-50 to-purple-50 border-2 border-dashed border-indigo-200 rounded-xl p-6 text-center hover:border-indigo-400 hover:from-indigo-100 hover:to-purple-100 transition-all group"
      >
        <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">💬</div>
        <div className="text-lg font-semibold text-slate-700 mb-1">Start a Discussion Thread</div>
        <div className="text-sm text-slate-500">
          Endow credits to kickstart a focused debate on this event
        </div>
      </button>
    );
  }

  return (
    <div className="bg-white rounded-xl border-2 border-indigo-200 shadow-lg overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-indigo-100">
        <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
          <span>💬</span> Start a New Thread
        </h3>
        <p className="text-sm text-slate-600 mt-1">
          Seed the discussion with credits to attract quality contributions
        </p>
      </div>

      <div className="p-6">
        {/* Title Input */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Thread Title
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="What's the main question or topic?"
            disabled={submitting}
            className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:bg-slate-100 text-sm"
            maxLength={200}
          />
          <div className="text-xs text-slate-400 mt-1 text-right">{title.length}/200</div>
        </div>

        {/* Content Input */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Opening Statement
          </label>
          <textarea
            ref={textareaRef}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Provide context, share your perspective, or pose questions for discussion..."
            disabled={submitting}
            rows={4}
            className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none disabled:bg-slate-100 text-sm leading-relaxed"
            maxLength={2000}
          />
          <div className="text-xs text-slate-400 mt-1 text-right">{content.length}/2000</div>
        </div>

        {/* Credit Endowment Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-slate-700 mb-3">
            <span className="flex items-center gap-2">
              <span className="text-amber-500">⭐</span>
              Credit Endowment
              <span className="text-xs font-normal text-slate-500">(rewards quality contributors)</span>
            </span>
          </label>

          <div className="flex flex-wrap gap-2 mb-3">
            {endowmentOptions.map((amount) => (
              <button
                key={amount}
                onClick={() => setEndowment(amount)}
                disabled={submitting}
                className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${
                  endowment === amount
                    ? 'bg-gradient-to-r from-amber-500 to-yellow-500 text-white shadow-md scale-105'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {amount} ⭐
              </button>
            ))}
            <div className="flex items-center gap-2 ml-2">
              <span className="text-slate-400">or</span>
              <input
                type="number"
                value={endowment}
                onChange={(e) => setEndowment(Math.max(5, Math.min(1000, parseInt(e.target.value) || 5)))}
                disabled={submitting}
                className="w-20 px-3 py-2 border border-slate-300 rounded-lg text-center text-sm focus:outline-none focus:ring-2 focus:ring-amber-500"
                min={5}
                max={1000}
              />
            </div>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm">
            <div className="flex items-start gap-2">
              <span className="text-amber-600">💡</span>
              <div className="text-amber-800">
                <strong>How endowment works:</strong> Your {endowment} credits will be distributed to
                contributors who provide valuable insights, fact-checks, or sources. Higher endowment
                attracts more engagement.
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between">
          <button
            onClick={() => setIsExpanded(false)}
            disabled={submitting}
            className="px-4 py-2 text-slate-600 hover:text-slate-800 text-sm"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting || !title.trim() || !content.trim()}
            className="px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-medium hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
          >
            {submitting ? (
              <>
                <span className="animate-spin">⏳</span>
                Creating...
              </>
            ) : (
              <>
                Start Thread
                <span className="text-xs opacity-75 bg-white/20 px-2 py-0.5 rounded">
                  {endowment} ⭐
                </span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// Single Thread Card
function ThreadCard({
  thread,
  onClick
}: {
  thread: Thread;
  onClick: () => void;
}) {
  const timeInfo = formatTime(thread.created_at);

  return (
    <div
      onClick={onClick}
      className="bg-white rounded-lg border border-slate-200 p-4 hover:border-indigo-300 hover:shadow-md transition-all cursor-pointer"
    >
      <div className="flex items-start gap-3">
        {/* Avatar */}
        {thread.user_picture ? (
          <img
            src={thread.user_picture}
            alt={thread.user_name}
            className="w-10 h-10 rounded-full flex-shrink-0"
          />
        ) : (
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-400 to-purple-400 flex items-center justify-center text-white font-semibold flex-shrink-0">
            {thread.user_name.charAt(0).toUpperCase()}
          </div>
        )}

        <div className="flex-1 min-w-0">
          {/* Title */}
          <h4 className="font-semibold text-slate-900 mb-1 line-clamp-1">
            {thread.title}
          </h4>

          {/* Content preview */}
          <p className="text-sm text-slate-600 line-clamp-2 mb-2">
            {thread.content}
          </p>

          {/* Meta info */}
          <div className="flex items-center gap-4 text-xs text-slate-400">
            <span>{thread.user_name}</span>
            <span title={timeInfo.fullDateTime}>{timeInfo.relative}</span>
            <span className="flex items-center gap-1">
              <span>💬</span> {thread.reply_count} replies
            </span>
          </div>
        </div>

        {/* Endowment badge */}
        <div className="flex-shrink-0 text-right">
          <div className="bg-gradient-to-r from-amber-100 to-yellow-100 text-amber-700 px-3 py-1.5 rounded-lg font-semibold text-sm">
            {thread.endowment} ⭐
          </div>
          <div className="text-xs text-slate-400 mt-1">endowed</div>
        </div>
      </div>
    </div>
  );
}

// Comment Item with nested replies
function CommentItem({
  comment,
  allComments,
  depth,
  onReply
}: {
  comment: Comment;
  allComments: Comment[];
  depth: number;
  onReply: (commentId: string, userName: string) => void;
}) {
  const replies = allComments.filter(c => c.parent_comment_id === comment.id);
  const timeInfo = formatTime(comment.created_at);

  return (
    <div className={`${depth > 0 ? 'ml-8 pl-4 border-l-2 border-slate-200' : ''}`}>
      <div className="flex gap-3 py-3">
        {/* Avatar */}
        {comment.user_picture ? (
          <img
            src={comment.user_picture}
            alt={comment.user_name}
            className="w-8 h-8 rounded-full flex-shrink-0"
          />
        ) : (
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-400 to-purple-400 flex items-center justify-center text-white text-sm font-semibold flex-shrink-0">
            {comment.user_name.charAt(0).toUpperCase()}
          </div>
        )}

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold text-slate-900 text-sm">{comment.user_name}</span>
            <span className="text-xs text-slate-400" title={timeInfo.fullDateTime}>
              {timeInfo.relative}
            </span>
          </div>
          <p className="text-slate-700 text-sm leading-relaxed whitespace-pre-wrap">
            {comment.text}
          </p>
          <button
            onClick={() => onReply(comment.id, comment.user_name)}
            className="text-xs text-indigo-600 hover:text-indigo-800 font-medium mt-1"
          >
            Reply
          </button>
        </div>
      </div>

      {/* Nested replies */}
      {replies.length > 0 && (
        <div>
          {replies.map(reply => (
            <CommentItem
              key={reply.id}
              comment={reply}
              allComments={allComments}
              depth={depth + 1}
              onReply={onReply}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// Thread Detail View
function ThreadDetailView({
  thread,
  onBack
}: {
  thread: Thread;
  onBack: () => void;
}) {
  const [comments, setComments] = useState<Comment[]>([]);
  const [input, setInput] = useState('');
  const [replyingTo, setReplyingTo] = useState<{ id: string; name: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    loadComments();
  }, [thread.id]);

  const loadComments = async () => {
    // Mock data for now - replace with actual API
    setLoading(false);
    setComments([
      {
        id: 'c1',
        thread_id: thread.id,
        user_id: 'u1',
        user_name: 'Alice Chen',
        text: 'Great question! I think we need to consider the source credibility here.',
        created_at: new Date(Date.now() - 3600000).toISOString()
      },
      {
        id: 'c2',
        thread_id: thread.id,
        user_id: 'u2',
        user_name: 'Bob Smith',
        text: 'I found additional sources that support this claim.',
        parent_comment_id: 'c1',
        created_at: new Date(Date.now() - 1800000).toISOString()
      }
    ]);
  };

  const handleSubmit = async () => {
    if (!input.trim() || submitting) return;
    setSubmitting(true);
    // TODO: Actual API call
    setTimeout(() => {
      const newComment: Comment = {
        id: `c${Date.now()}`,
        thread_id: thread.id,
        user_id: 'current',
        user_name: 'You',
        text: input.trim(),
        parent_comment_id: replyingTo?.id,
        created_at: new Date().toISOString()
      };
      setComments([...comments, newComment]);
      setInput('');
      setReplyingTo(null);
      setSubmitting(false);
    }, 500);
  };

  const handleReply = (commentId: string, userName: string) => {
    setReplyingTo({ id: commentId, name: userName });
  };

  const timeInfo = formatTime(thread.created_at);
  const topLevelComments = comments.filter(c => !c.parent_comment_id);

  return (
    <div>
      {/* Back button */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-slate-600 hover:text-slate-800 mb-4"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to threads
      </button>

      {/* Thread Header */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 mb-4">
        <div className="flex items-start gap-4">
          {thread.user_picture ? (
            <img
              src={thread.user_picture}
              alt={thread.user_name}
              className="w-12 h-12 rounded-full"
            />
          ) : (
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-400 to-purple-400 flex items-center justify-center text-white text-lg font-semibold">
              {thread.user_name.charAt(0).toUpperCase()}
            </div>
          )}
          <div className="flex-1">
            <h2 className="text-xl font-bold text-slate-900 mb-2">{thread.title}</h2>
            <p className="text-slate-700 leading-relaxed mb-3">{thread.content}</p>
            <div className="flex items-center gap-4 text-sm text-slate-500">
              <span>{thread.user_name}</span>
              <span title={timeInfo.fullDateTime}>{timeInfo.relative}</span>
              <span className="bg-amber-100 text-amber-700 px-2 py-0.5 rounded font-medium">
                {thread.endowment} ⭐ endowed
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Comment Input */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 mb-4">
        {replyingTo && (
          <div className="text-sm text-slate-600 mb-2 flex items-center justify-between">
            <span>Replying to <strong>{replyingTo.name}</strong></span>
            <button
              onClick={() => setReplyingTo(null)}
              className="text-slate-400 hover:text-slate-600"
            >
              ✕
            </button>
          </div>
        )}
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
          placeholder="Share your thoughts, add evidence, or challenge claims..."
          disabled={submitting}
          rows={3}
          className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none text-sm"
        />
        <div className="flex justify-between items-center mt-2">
          <span className="text-xs text-slate-500">
            Press <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded">Enter</kbd> to post
          </span>
          <button
            onClick={handleSubmit}
            disabled={submitting || !input.trim()}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
          >
            {submitting ? 'Posting...' : 'Post'}
            <span className="text-xs opacity-75">(2⭐)</span>
          </button>
        </div>
      </div>

      {/* Comments List */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="font-semibold text-slate-800 mb-4">
          {comments.length} {comments.length === 1 ? 'Reply' : 'Replies'}
        </h3>
        {loading ? (
          <div className="text-center py-8 text-slate-500">Loading replies...</div>
        ) : comments.length === 0 ? (
          <div className="text-center py-8 text-slate-400">
            No replies yet. Be the first to contribute!
          </div>
        ) : (
          <div>
            {topLevelComments.map(comment => (
              <CommentItem
                key={comment.id}
                comment={comment}
                allComments={comments}
                depth={0}
                onReply={handleReply}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Main DebateThread Component
function DebateThread({ eventId }: DebateThreadProps) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [selectedThread, setSelectedThread] = useState<Thread | null>(null);

  useEffect(() => {
    loadThreads();
  }, [eventId]);

  const loadThreads = async () => {
    // Mock data for now - replace with actual API
    setLoading(false);
    setThreads([
      {
        id: 't1',
        event_id: eventId,
        user_id: 'u1',
        user_name: 'David Kim',
        title: 'Are the casualty numbers accurate?',
        content: 'I\'ve seen conflicting reports from different sources. Can we verify the official figures?',
        endowment: 50,
        reply_count: 12,
        created_at: new Date(Date.now() - 7200000).toISOString()
      },
      {
        id: 't2',
        event_id: eventId,
        user_id: 'u2',
        user_name: 'Sarah Lee',
        title: 'Timeline discrepancies between sources',
        content: 'The BBC reports the event started at 3pm, but Reuters says 4:30pm. Which is correct?',
        endowment: 25,
        reply_count: 5,
        created_at: new Date(Date.now() - 14400000).toISOString()
      }
    ]);
  };

  const handleCreateThread = async (title: string, content: string, endowment: number) => {
    setSubmitting(true);
    // TODO: Actual API call
    setTimeout(() => {
      const newThread: Thread = {
        id: `t${Date.now()}`,
        event_id: eventId,
        user_id: 'current',
        user_name: 'You',
        title,
        content,
        endowment,
        reply_count: 0,
        created_at: new Date().toISOString()
      };
      setThreads([newThread, ...threads]);
      setSubmitting(false);
    }, 500);
  };

  if (selectedThread) {
    return (
      <ThreadDetailView
        thread={selectedThread}
        onBack={() => setSelectedThread(null)}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Thread Starter */}
      <ThreadStarterBox onSubmit={handleCreateThread} submitting={submitting} />

      {/* Threads List */}
      <div>
        <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
          <span>🔥</span> Active Discussions
          <span className="text-sm font-normal text-slate-500">({threads.length})</span>
        </h3>

        {loading ? (
          <div className="text-center py-8 text-slate-500">Loading discussions...</div>
        ) : threads.length === 0 ? (
          <div className="bg-slate-50 rounded-lg p-8 text-center">
            <div className="text-4xl mb-4">💬</div>
            <h4 className="text-lg font-semibold text-slate-700 mb-2">No discussions yet</h4>
            <p className="text-slate-500">
              Be the first to start a thread and earn credits for quality contributions!
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {threads.map(thread => (
              <ThreadCard
                key={thread.id}
                thread={thread}
                onClick={() => setSelectedThread(thread)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default DebateThread;
