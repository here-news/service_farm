import React, { useState, useRef, useEffect } from 'react'
import { User } from '../../types/chat'

interface UserProfileProps {
  user: User
}

function UserProfile({ user }: UserProfileProps) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleLogout = () => {
    window.location.href = '/api/auth/logout'
  }

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 hover:opacity-80 transition"
      >
        {user.picture ? (
          <img
            src={user.picture}
            alt={user.name}
            className="w-8 h-8 rounded-full border-2 border-white shadow-sm"
          />
        ) : (
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white font-semibold text-sm">
            {user.name.charAt(0).toUpperCase()}
          </div>
        )}
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-xl border border-slate-200 py-2 z-50">
          {/* User Info */}
          <div className="px-4 py-3 border-b border-slate-200">
            <div className="font-semibold text-slate-900">{user.name}</div>
            <div className="text-sm text-slate-500">{user.email}</div>
            <div className="text-xs text-indigo-600 mt-1">
              Credits: {user.credits}
            </div>
          </div>

          {/* Menu Items */}
          <div className="py-1">
            <button
              onClick={() => {
                setIsOpen(false)
                // Navigate to profile (placeholder)
                console.log('Navigate to profile')
              }}
              className="w-full text-left px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 transition"
            >
              Profile Settings
            </button>
            <button
              onClick={() => {
                setIsOpen(false)
                // Navigate to credits (placeholder)
                console.log('Navigate to credits')
              }}
              className="w-full text-left px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 transition"
            >
              Manage Credits
            </button>
          </div>

          <div className="border-t border-slate-200 py-1">
            <button
              onClick={handleLogout}
              className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition"
            >
              Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default UserProfile
