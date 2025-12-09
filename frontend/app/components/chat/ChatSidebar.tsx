import React, { useState, useEffect, useRef } from 'react'
import { ChatMessage, ChatSession, User } from '../../types/chat'
import ChatInput from './ChatInput'
import ChatUnlockModal from './ChatUnlockModal'

interface ChatSidebarProps {
  storyId: string
  storyTitle?: string
  isOpen: boolean
  onClose: () => void
}

function ChatSidebar({ storyId, storyTitle = 'this story', isOpen, onClose }: ChatSidebarProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isSending, setIsSending] = useState(false)
  const [session, setSession] = useState<ChatSession | null>(null)
  const [remainingMessages, setRemainingMessages] = useState(100)
  const [showUnlockModal, setShowUnlockModal] = useState(false)
  const [isUnlocking, setIsUnlocking] = useState(false)
  const [user, setUser] = useState<User | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (storyId && isOpen) {
      loadUser()
      checkUnlockStatus()
    }
  }, [storyId, isOpen])

  const loadUser = async () => {
    try {
      const response = await fetch('/api/auth/status', { credentials: 'include' })
      if (response.ok) {
        const data = await response.json()
        if (data.authenticated) {
          setUser(data.user)
        }
      }
    } catch (error) {
      console.error('Failed to load user:', error)
    }
  }

  const checkUnlockStatus = async () => {
    loadConversationHistory()
    const sessionData = await loadSession()

    // If no session found, show unlock modal
    if (!sessionData) {
      setShowUnlockModal(true)
    }
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const loadConversationHistory = () => {
    try {
      const stored = localStorage.getItem(`chat_${storyId}`)
      if (stored) {
        setMessages(JSON.parse(stored))
      }
    } catch (error) {
      console.error('Failed to load conversation history:', error)
    }
  }

  const saveConversationHistory = (newMessages: ChatMessage[]) => {
    try {
      localStorage.setItem(`chat_${storyId}`, JSON.stringify(newMessages))
    } catch (error) {
      console.error('Failed to save conversation history:', error)
    }
  }

  const loadSession = async (): Promise<ChatSession | null> => {
    try {
      const response = await fetch(`/api/chat/session/${storyId}`, {
        credentials: 'include'
      })

      if (response.ok) {
        const data = await response.json()
        setSession(data)
        if (data) {
          setRemainingMessages(Math.max(0, 100 - data.message_count))
        }
        return data
      }
      return null
    } catch (error) {
      console.error('Failed to load session:', error)
      return null
    }
  }

  const handleSendMessage = async (userMessage: string) => {
    if (!userMessage || isSending) return

    // Add user message to UI
    const newMessages = [...messages, { role: 'user' as const, content: userMessage }]
    setMessages(newMessages)
    setIsSending(true)

    try {
      const response = await fetch('/api/chat/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          story_id: storyId,
          message: userMessage,
          conversation_history: messages.slice(-10)
        })
      })

      if (!response.ok) {
        const error = await response.json()

        // Handle insufficient credits
        if (response.status === 402) {
          alert(error.detail || 'Insufficient credits to continue chatting')
          // Remove the optimistic user message
          setMessages(messages)
          return
        }

        throw new Error(error.detail || 'Failed to send message')
      }

      const data = await response.json()

      // Add AI response
      const updatedMessages = [...newMessages, { role: 'assistant' as const, content: data.message }]
      setMessages(updatedMessages)
      saveConversationHistory(updatedMessages)
      setRemainingMessages(data.remaining_messages)

      if (data.session_status === 'exhausted') {
        setSession({ ...session!, status: 'exhausted' })
      }

      // Show credits info if provided
      if (data.credits_remaining !== undefined) {
        console.log(`Message sent! ${data.credits_remaining} credits remaining`)
      }
    } catch (error) {
      console.error('Error sending message:', error)
      alert(error instanceof Error ? error.message : 'Failed to send message')
      // Remove the optimistic user message on error
      setMessages(messages)
    } finally {
      setIsSending(false)
    }
  }

  const handleUnlock = async () => {
    try {
      setIsUnlocking(true)
      const response = await fetch('/api/chat/unlock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ story_id: storyId })
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to unlock chat')
      }

      const data = await response.json()
      setSession(data)
      setShowUnlockModal(false)

      // Refresh user credits
      await loadUser()
    } catch (error) {
      console.error('Unlock failed:', error)
      alert(error instanceof Error ? error.message : 'Failed to unlock chat')
      throw error
    } finally {
      setIsUnlocking(false)
    }
  }

  const scrollToBottom = () => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }

  const isExhausted = session?.status === 'exhausted'

  if (!isOpen) return null

  // Show unlock modal if chat is not unlocked
  if (showUnlockModal && user) {
    return (
      <ChatUnlockModal
        storyTitle={storyTitle}
        userCredits={user.credits}
        onUnlock={handleUnlock}
        onCancel={onClose}
      />
    )
  }

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black bg-opacity-30 z-40"
        onClick={onClose}
      />

      {/* Sidebar */}
      <div className="fixed top-0 right-0 bottom-0 w-96 bg-white shadow-2xl z-50 flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6 flex-shrink-0">
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-xl font-bold">Story Chat</h2>
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-full bg-white bg-opacity-20 hover:bg-opacity-30 flex items-center justify-center transition"
            >
              ×
            </button>
          </div>
          <div className="text-sm opacity-90">
            {remainingMessages} / 100 messages remaining
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="text-center text-slate-400 mt-16">
              <div className="text-4xl mb-4">💬</div>
              <p className="leading-relaxed">
                Start a conversation about this story.<br />
                Ask questions, explore connections, or get AI-powered insights.
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white font-bold flex-shrink-0 mr-2">
                    φ
                  </div>
                )}
                <div
                  className={`max-w-[80%] px-4 py-3 rounded-xl ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
                      : 'bg-slate-100 text-slate-800'
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))
          )}

          {isSending && (
            <div className="flex justify-start">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white font-bold flex-shrink-0 mr-2">
                φ
              </div>
              <div className="bg-slate-100 px-4 py-3 rounded-xl">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        {isExhausted ? (
          <div className="p-6 bg-red-50 border-t border-red-200">
            <p className="text-sm text-red-800">
              You've reached the 100-message limit for this story. Start a new conversation by visiting another story.
            </p>
          </div>
        ) : (
          <ChatInput
            onSubmit={handleSendMessage}
            disabled={isSending}
            placeholder="Ask a question about this story..."
          />
        )}
      </div>
    </>
  )
}

export default ChatSidebar
