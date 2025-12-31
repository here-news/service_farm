import React, { useRef, useState } from 'react'
import InquiryCard, { InquirySummary } from './InquiryCard'

interface InquiryCarouselProps {
  title: string
  icon: string
  badge?: { text: string; color: string }
  inquiries: InquirySummary[]
  variant?: 'default' | 'resolved' | 'bounty' | 'contested'
  onSelect: (id: string) => void
}

function InquiryCarousel({
  title,
  icon,
  badge,
  inquiries,
  variant = 'default',
  onSelect
}: InquiryCarouselProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [showLeftArrow, setShowLeftArrow] = useState(false)
  const [showRightArrow, setShowRightArrow] = useState(true)

  const handleScroll = () => {
    if (!scrollRef.current) return
    const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current
    setShowLeftArrow(scrollLeft > 10)
    setShowRightArrow(scrollLeft < scrollWidth - clientWidth - 10)
  }

  const scroll = (direction: 'left' | 'right') => {
    if (!scrollRef.current) return
    const scrollAmount = 320 // Card width + gap
    scrollRef.current.scrollBy({
      left: direction === 'left' ? -scrollAmount : scrollAmount,
      behavior: 'smooth'
    })
  }

  if (inquiries.length === 0) return null

  return (
    <section className="mb-10">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <span className="text-xl">{icon}</span>
        <h2 className="text-lg font-semibold text-slate-700">{title}</h2>
        {badge && (
          <span className={`text-xs px-2 py-0.5 rounded-full ${badge.color}`}>
            {badge.text}
          </span>
        )}
      </div>

      {/* Carousel container */}
      <div className="relative group">
        {/* Left arrow */}
        {showLeftArrow && (
          <button
            onClick={() => scroll('left')}
            className="absolute left-0 top-1/2 -translate-y-1/2 z-10 w-10 h-10 bg-white/90 hover:bg-white shadow-lg rounded-full flex items-center justify-center text-slate-600 hover:text-indigo-600 transition-all opacity-0 group-hover:opacity-100"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
        )}

        {/* Scrollable container */}
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="flex gap-4 overflow-x-auto scrollbar-hide pb-2 -mx-4 px-4 snap-x snap-mandatory"
          style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          {inquiries.map((inquiry) => (
            <div
              key={inquiry.id}
              className="flex-shrink-0 w-80 snap-start"
            >
              <InquiryCard
                inquiry={inquiry}
                variant={variant}
                onClick={() => onSelect(inquiry.id)}
              />
            </div>
          ))}
        </div>

        {/* Right arrow */}
        {showRightArrow && inquiries.length > 3 && (
          <button
            onClick={() => scroll('right')}
            className="absolute right-0 top-1/2 -translate-y-1/2 z-10 w-10 h-10 bg-white/90 hover:bg-white shadow-lg rounded-full flex items-center justify-center text-slate-600 hover:text-indigo-600 transition-all opacity-0 group-hover:opacity-100"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        )}

        {/* Gradient overlays */}
        {showLeftArrow && (
          <div className="absolute left-0 top-0 bottom-0 w-8 bg-gradient-to-r from-slate-50 to-transparent pointer-events-none" />
        )}
        {showRightArrow && inquiries.length > 3 && (
          <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-slate-50 to-transparent pointer-events-none" />
        )}
      </div>
    </section>
  )
}

export default InquiryCarousel
