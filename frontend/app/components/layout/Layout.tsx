import React from 'react'
import Header from './Header'

interface LayoutProps {
  children: React.ReactNode
}

function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-slate-50">
      <Header />
      <main>
        {children}
      </main>
    </div>
  )
}

export default Layout
