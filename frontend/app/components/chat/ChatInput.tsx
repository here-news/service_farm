import React, { useState, useRef, KeyboardEvent } from 'react'

interface ChatInputProps {
  onSubmit: (input: string) => void
  disabled?: boolean
  placeholder?: string
}

function ChatInput({
  onSubmit,
  disabled = false,
  placeholder = "Ask a question..."
}: ChatInputProps) {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = () => {
    const trimmed = input.trim()
    if (!trimmed || disabled) return

    onSubmit(trimmed)
    setInput('')

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  // IME-SAFE: Use onKeyDown, check e.key === 'Enter'
  // This works correctly with Chinese/Japanese/Korean IME composition
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)

    // Auto-resize textarea
    const textarea = e.target
    textarea.style.height = 'auto'
    const newHeight = Math.max(Math.min(textarea.scrollHeight, 200), 56)
    textarea.style.height = `${newHeight}px`
  }

  return (
    <div className="border-t border-slate-200 bg-white p-4">
      <div className="flex gap-3">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={2}
          className="flex-1 resize-none px-4 py-3 border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-slate-100 disabled:text-slate-400 text-sm leading-relaxed"
          style={{ maxHeight: '200px', minHeight: '56px', height: '56px' }}
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || !input.trim()}
          className="self-end px-4 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition"
        >
          Send
        </button>
      </div>
      <p className="text-xs text-slate-400 mt-2">
        Press <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded">Enter</kbd> to send,
        <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded ml-1">Shift+Enter</kbd> for new line
      </p>
    </div>
  )
}

export default ChatInput
