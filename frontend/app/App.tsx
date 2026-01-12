import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/layout/Layout'
import LandingPage from './LandingPage'
import TempHomePage from './TempHomePage'
import EventInquiryPage from './EventInquiryPage'
import EntityInquiryPage from './EntityInquiryPage'
import PagePage from './PagePage'
import ArchivePage from './ArchivePage'
import GraphPage from './GraphPage'
import MapPage from './MapPage'
import InquiryPage from './InquiryPage'
import InquiryDetailPage from './InquiryDetailPage'
import ProfilePage from './ProfilePage'

function App() {
  return (
    <Router>
      <Routes>
        {/* Landing page - full screen visualization */}
        <Route path="/" element={<LandingPage />} />

        {/* Browse page - data listing */}
        <Route path="/browse" element={<Layout><TempHomePage /></Layout>} />

        {/* Core inquiry routes */}
        <Route path="/inquiry" element={<Layout><InquiryPage /></Layout>} />
        <Route path="/inquiry/:inquiryId" element={<Layout><InquiryDetailPage /></Layout>} />

        {/* Story pages - unified L3/L4 (incident/case) */}
        <Route path="/story/:storyId" element={<Layout><EventInquiryPage /></Layout>} />

        {/* Event pages - legacy, redirects to story */}
        <Route path="/event/:eventId" element={<Layout><EventInquiryPage /></Layout>} />

        {/* Entity pages - people, places, orgs */}
        <Route path="/entity/:entityId" element={<Layout><EntityInquiryPage /></Layout>} />

        {/* User profile - credits, stakes, contributions */}
        <Route path="/profile" element={<Layout><ProfilePage /></Layout>} />

        {/* Supporting pages */}
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
