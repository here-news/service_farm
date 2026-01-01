import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/layout/Layout'
import StoryPage from './StoryPage'
import EventPage from './EventPage'
import EventInquiryPage from './EventInquiryPage'
import EntityPage from './EntityPage'
import EntityInquiryPage from './EntityInquiryPage'
import PagePage from './PagePage'
import ArchivePage from './ArchivePage'
import GraphPage from './GraphPage'
import MapPage from './MapPage'
import InquiryPage from './InquiryPage'
import InquiryDetailPage from './InquiryDetailPage'

function App() {
  return (
    <Router>
      <Routes>
        {/* Home - YOLO dashboard */}
        <Route path="/" element={<Layout><InquiryPage /></Layout>} />

        {/* Core inquiry routes */}
        <Route path="/inquiry" element={<Layout><InquiryPage /></Layout>} />
        <Route path="/inquiry/:inquiryId" element={<Layout><InquiryDetailPage /></Layout>} />

        {/* Event pages */}
        <Route path="/event/:eventSlug" element={<Layout><EventPage /></Layout>} />
        <Route path="/event-inquiry/:eventSlug" element={<EventInquiryPage />} />

        {/* Entity pages */}
        <Route path="/entity/:entityId" element={<Layout><EntityPage /></Layout>} />
        <Route path="/entity-inquiry/:entitySlug" element={<EntityInquiryPage />} />

        {/* Supporting pages */}
        <Route path="/story/:storyId" element={<Layout><StoryPage /></Layout>} />
        <Route path="/story/:storyId/:slug" element={<Layout><StoryPage /></Layout>} />
        <Route path="/page/:pageId" element={<Layout><PagePage /></Layout>} />
        <Route path="/archive" element={<Layout><ArchivePage /></Layout>} />
        <Route path="/graph" element={<Layout><GraphPage /></Layout>} />
        <Route path="/map" element={<Layout><MapPage /></Layout>} />

        {/* Legacy redirects */}
        <Route path="/app" element={<Navigate to="/" replace />} />
        <Route path="/app/*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  )
}

export default App
