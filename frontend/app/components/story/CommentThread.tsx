import React, { useState, useRef, KeyboardEvent, useEffect } from 'react'
import URLPreview from '../comment/URLPreview'
import { extractUrls } from '../../utils/urlExtractor'
import { formatTime } from '../../utils/timeFormat'

interface Comment {
  id: string
  user_id: string
  user_name: string
  user_picture?: string
  user_email: string
  text: string
  story_id: string
  parent_comment_id?: string
  reaction_type?: string
  created_at: string
  updated_at?: string
}

interface CommentThreadProps {
  storyId: string
}

interface CommentItemProps {
  comment: Comment
  allComments: Comment[]
  depth: number
  onReply: (commentId: string, parentName: string) => void
}

interface CommentItemInternalProps extends CommentItemProps {
  replyingTo: { id: string; name: string } | null
  replyInput: string
  onReplyInputChange: (value: string) => void
  onSubmitReply: () => void
  onCancelReply: () => void
  submitting: boolean
}

function CommentItem({ comment, allComments, depth, onReply, replyingTo, replyInput, onReplyInputChange, onSubmitReply, onCancelReply, submitting }: CommentItemInternalProps) {
  // Find replies to this comment
  const replies = allComments.filter(c => c.parent_comment_id === comment.id)
  const isReplying = replyingTo?.id === comment.id

  // Extract URLs from comment text
  const urls = extractUrls(comment.text)

  // Format timestamp
  const timeInfo = formatTime(comment.created_at)

  return (
    <div className={`${depth > 0 ? 'ml-8 pl-4 border-l-2 border-slate-200' : ''}`}>
      <div className="flex gap-3 py-3">
        {/* Avatar */}
        {comment.user_picture ? (
          <img
            src={comment.user_picture}
            alt={comment.user_name}
            className="w-10 h-10 rounded-full flex-shrink-0"
          />
        ) : (
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-400 to-purple-400 flex items-center justify-center text-white font-semibold flex-shrink-0">
            {comment.user_name.charAt(0).toUpperCase()}
          </div>
        )}

        <div className="flex-1 min-w-0">
          {/* User info */}
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold text-slate-900">{comment.user_name}</span>
            <span
              className="text-xs text-slate-400 cursor-help"
              title={timeInfo.fullDateTime}
            >
              {timeInfo.relative}
            </span>
          </div>

          {/* Comment text */}
          <p className="text-slate-700 leading-relaxed whitespace-pre-wrap text-sm mb-2">
            {comment.text}
          </p>

          {/* URL Previews */}
          {urls.length > 0 && (
            <div className="space-y-2 mb-2">
              {urls.slice(0, 2).map((url, index) => (
                <URLPreview key={index} url={url} />
              ))}
              {urls.length > 2 && (
                <div className="text-xs text-slate-500">
                  +{urls.length - 2} more link{urls.length - 2 > 1 ? 's' : ''}
                </div>
              )}
            </div>
          )}

          {/* Reply button */}
          <button
            onClick={() => onReply(comment.id, comment.user_name)}
            className="text-xs text-indigo-600 hover:text-indigo-800 font-medium flex items-center gap-1"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
            </svg>
            Reply
          </button>
        </div>
      </div>

      {/* Inline reply input */}
      {isReplying && (
        <div className="ml-13 mr-3 mb-3 bg-slate-50 rounded-lg p-3 border border-slate-200">
          <div className="text-xs text-slate-600 mb-2">
            Replying to <span className="font-semibold">{replyingTo.name}</span>
          </div>
          <textarea
            value={replyInput}
            onChange={(e) => onReplyInputChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                onSubmitReply()
              }
            }}
            placeholder="Write your reply..."
            disabled={submitting}
            rows={2}
            className="w-full resize-none px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
          />
          <div className="flex justify-end gap-2 mt-2">
            <button
              onClick={onCancelReply}
              className="px-3 py-1.5 text-sm text-slate-600 hover:text-slate-800"
            >
              Cancel
            </button>
            <button
              onClick={onSubmitReply}
              disabled={submitting || !replyInput.trim()}
              className="px-4 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
            >
              {submitting ? 'Posting...' : 'Reply'}
              <span className="text-xs opacity-75">(2★)</span>
            </button>
          </div>
        </div>
      )}

      {/* Recursive rendering of replies */}
      {replies.length > 0 && (
        <div className="space-y-0">
          {replies.map(reply => (
            <CommentItem
              key={reply.id}
              comment={reply}
              allComments={allComments}
              depth={depth + 1}
              onReply={onReply}
              replyingTo={replyingTo}
              replyInput={replyInput}
              onReplyInputChange={onReplyInputChange}
              onSubmitReply={onSubmitReply}
              onCancelReply={onCancelReply}
              submitting={submitting}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function CommentThread({ storyId }: CommentThreadProps) {
  const [comments, setComments] = useState<Comment[]>([])
  const [input, setInput] = useState('')
  const [replyInput, setReplyInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [replyingTo, setReplyingTo] = useState<{ id: string; name: string } | null>(null)
  const [inputUrls, setInputUrls] = useState<string[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    loadComments()
  }, [storyId])

  useEffect(() => {
    // Extract URLs from input text
    const extractedUrls = extractUrls(input)
    setInputUrls(extractedUrls)
  }, [input])

  const loadComments = async () => {
    try {
      setLoading(true)
      const response = await fetch(`/api/comments/story/${storyId}`)
      if (response.ok) {
        const data = await response.json()
        setComments(data || [])
      }
    } catch (error) {
      console.error('Error loading comments:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async () => {
    const trimmed = input.trim()
    if (!trimmed || submitting) return

    try {
      setSubmitting(true)
      const response = await fetch('/api/comments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          story_id: storyId,
          text: trimmed
        })
      })

      if (response.ok) {
        const data = await response.json()
        setInput('')
        if (textareaRef.current) {
          textareaRef.current.style.height = 'auto'
        }
        await loadComments()

        // Show success with remaining credits
        if (data.credits_remaining !== undefined) {
          console.log(`Comment posted! ${data.credits_remaining} credits remaining`)
        }
      } else if (response.status === 402) {
        // Insufficient credits
        const error = await response.json()
        alert(error.detail || 'Insufficient credits to post comment')
      } else {
        const error = await response.json()
        alert(error.detail || 'Failed to post comment')
      }
    } catch (error) {
      console.error('Error posting comment:', error)
      alert('Error posting comment')
    } finally {
      setSubmitting(false)
    }
  }

  const handleSubmitReply = async () => {
    const trimmed = replyInput.trim()
    if (!trimmed || submitting || !replyingTo) return

    try {
      setSubmitting(true)
      const response = await fetch('/api/comments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          story_id: storyId,
          text: trimmed,
          parent_comment_id: replyingTo.id
        })
      })

      if (response.ok) {
        const data = await response.json()
        setReplyInput('')
        setReplyingTo(null)
        await loadComments()

        // Show success with remaining credits
        if (data.credits_remaining !== undefined) {
          console.log(`Reply posted! ${data.credits_remaining} credits remaining`)
        }
      } else if (response.status === 402) {
        // Insufficient credits
        const error = await response.json()
        alert(error.detail || 'Insufficient credits to post reply')
      } else {
        const error = await response.json()
        alert(error.detail || 'Failed to post reply')
      }
    } catch (error) {
      console.error('Error posting reply:', error)
      alert('Error posting reply')
    } finally {
      setSubmitting(false)
    }
  }

  const handleReply = (commentId: string, parentName: string) => {
    setReplyingTo({ id: commentId, name: parentName })
    setReplyInput('')
  }

  const cancelReply = () => {
    setReplyingTo(null)
    setReplyInput('')
  }

  // IME-SAFE: Use onKeyDown, check e.key === 'Enter'
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)

    // Auto-resize
    const textarea = e.target
    textarea.style.height = 'auto'
    const newHeight = Math.max(Math.min(textarea.scrollHeight, 200), 80)
    textarea.style.height = `${newHeight}px`
  }

  // Get top-level comments (no parent)
  const topLevelComments = comments.filter(c => !c.parent_comment_id)

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-slate-900 mb-6">Comments</h2>

      {/* Top-level Comment Input - Hidden when replying to a specific comment */}
      {!replyingTo && (
        <div className="mb-8">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Share your thoughts... (supports Chinese, Japanese, Korean)"
            disabled={submitting}
            rows={3}
            className="w-full resize-none px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-slate-100 disabled:text-slate-400 text-sm leading-relaxed"
            style={{ minHeight: '80px', maxHeight: '200px' }}
          />

          {/* URL Previews */}
          {inputUrls.length > 0 && (
            <div className="space-y-2 mt-3">
              {inputUrls.slice(0, 2).map((url, index) => (
                <URLPreview key={index} url={url} />
              ))}
              {inputUrls.length > 2 && (
                <div className="text-xs text-slate-500">
                  +{inputUrls.length - 2} more link{inputUrls.length - 2 > 1 ? 's' : ''}
                </div>
              )}
            </div>
          )}

          <div className="flex justify-between items-center mt-2">
            <p className="text-xs text-slate-500">
              Press <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded">Enter</kbd> to post,
              <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded ml-1">Shift+Enter</kbd> for new line
            </p>
            <button
              onClick={handleSubmit}
              disabled={submitting || !input.trim()}
              className="px-6 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-1.5"
            >
              {submitting ? (
                'Posting...'
              ) : (
                <>
                  Post Comment
                  <span className="text-xs opacity-75">(2★)</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Comments List */}
      {loading ? (
        <div className="text-center py-8 text-slate-500">Loading comments...</div>
      ) : comments.length === 0 ? (
        <div className="text-center py-8 text-slate-400">No comments yet. Be the first to comment!</div>
      ) : (
        <div className="space-y-2">
          {topLevelComments.map((comment) => (
            <CommentItem
              key={comment.id}
              comment={comment}
              allComments={comments}
              depth={0}
              onReply={handleReply}
              replyingTo={replyingTo}
              replyInput={replyInput}
              onReplyInputChange={setReplyInput}
              onSubmitReply={handleSubmitReply}
              onCancelReply={cancelReply}
              submitting={submitting}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default CommentThread
