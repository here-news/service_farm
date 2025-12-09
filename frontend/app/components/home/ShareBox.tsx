import React, { useState, useRef, useEffect } from 'react'
import URLPreview from '../comment/URLPreview'
import { extractUrls } from '../../utils/urlExtractor'

interface ShareBoxProps {
  onSubmit: (content: string, urls: string[]) => Promise<void>
  onCancel: () => void
}

function ShareBox({ onSubmit, onCancel }: ShareBoxProps) {
  const [content, setContent] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [urls, setUrls] = useState<string[]>([])

  useEffect(() => {
    // Focus textarea when component mounts
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }, [])

  useEffect(() => {
    // Extract URLs from content
    const extractedUrls = extractUrls(content)
    setUrls(extractedUrls)
  }, [content])

  const handleSubmit = async () => {
    const trimmed = content.trim()
    if (!trimmed || submitting) return

    try {
      setSubmitting(true)
      await onSubmit(trimmed, urls)
      setContent('')
    } catch (error) {
      console.error('Failed to submit event:', error)
      alert(error instanceof Error ? error.message : 'Failed to submit event')
    } finally {
      setSubmitting(false)
    }
  }

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(e.target.value)

    // Auto-grow textarea
    const textarea = e.target
    textarea.style.height = 'auto'
    const newHeight = Math.max(textarea.scrollHeight, 120)
    textarea.style.height = `${newHeight}px`
  }

  const charCount = content.length
  const maxChars = 1000

  return (
    <div className="bg-white rounded-xl shadow-lg border-2 border-indigo-200 p-6 mb-6">
      <label className="block text-sm font-medium text-slate-700 mb-2">
        🔍 What event do you want to report?
      </label>

      <textarea
        ref={textareaRef}
        value={content}
        onChange={handleTextareaChange}
        placeholder="Describe the event and paste any relevant URLs...

Example:
New data suggests climate tipping point may be closer than expected
https://example.com/article
https://example.com/research-paper"
        disabled={submitting}
        maxLength={maxChars}
        className="w-full p-3 border-2 border-slate-300 rounded-lg focus:outline-none focus:border-indigo-500 resize-none text-sm overflow-hidden disabled:bg-slate-100"
        style={{ minHeight: '120px' }}
      />

      <div className="flex items-center justify-between mt-1 mb-3">
        <span className={`text-xs ${charCount > 900 ? 'text-red-600 font-semibold' : charCount > 800 ? 'text-amber-600' : 'text-slate-500'}`}>
          {charCount} / {maxChars}
        </span>
        {urls.length > 0 && (
          <span className="text-xs text-indigo-600">
            📎 {urls.length} URL{urls.length > 1 ? 's' : ''} detected
          </span>
        )}
      </div>

      <div className="p-3 bg-blue-50 rounded-lg border border-blue-200 mb-3">
        <p className="text-xs text-blue-800">
          <strong>💡 Tips:</strong>
          {' • '}Describe the event clearly
          {' • '}Paste URLs directly (we'll extract them automatically)
          {' • '}Include source links for verification
        </p>
      </div>

      {/* URL Previews */}
      {urls.length > 0 && (
        <div className="space-y-2 mb-3">
          {urls.map((url, index) => (
            <URLPreview key={index} url={url} />
          ))}
        </div>
      )}

      <div className="flex gap-2 justify-between items-center">
        <div className="text-sm text-slate-600">
          Submission is free
        </div>
        <div className="flex gap-2">
          <button
            onClick={onCancel}
            disabled={submitting}
            className="px-4 py-2 text-sm border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting || !content.trim()}
            className="px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {submitting ? 'Submitting...' : 'Submit Event'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default ShareBox
