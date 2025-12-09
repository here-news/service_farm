export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatSession {
  id: string
  message_count: number
  status: string
}

export interface User {
  id: string
  name: string
  email: string
  picture?: string
  credits: number
}
