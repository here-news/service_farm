import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
import LandingPage from './LandingPage'
import HomePage from './HomePage'
import StoryPage from './StoryPage'
import EventPage from './EventPage'
import EntityPage from './EntityPage'
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
        {/* Landing page - no layout, full screen */}
        <Route path="/" element={<LandingPage />} />

        {/* Direct routes (without /app prefix) */}
        <Route path="/inquiry" element={<Layout><InquiryPage /></Layout>} />
        <Route path="/inquiry/:inquiryId" element={<Layout><InquiryDetailPage /></Layout>} />
        <Route path="/graph" element={<Layout><GraphPage /></Layout>} />
        <Route path="/map" element={<Layout><MapPage /></Layout>} />
        <Route path="/archive" element={<Layout><ArchivePage /></Layout>} />
        <Route path="/story/:storyId" element={<Layout><StoryPage /></Layout>} />
        <Route path="/story/:storyId/:slug" element={<Layout><StoryPage /></Layout>} />
        <Route path="/event/:eventSlug" element={<Layout><EventPage /></Layout>} />
        <Route path="/entity/:entityId" element={<Layout><EntityPage /></Layout>} />
        <Route path="/page/:pageId" element={<Layout><PagePage /></Layout>} />

        {/* App routes - with /app prefix (legacy support) */}
        <Route path="/app" element={<Layout><HomePage /></Layout>} />
        <Route path="/app/archive" element={<Layout><ArchivePage /></Layout>} />
        <Route path="/app/graph" element={<Layout><GraphPage /></Layout>} />
        <Route path="/app/map" element={<Layout><MapPage /></Layout>} />
        <Route path="/app/story/:storyId" element={<Layout><StoryPage /></Layout>} />
        <Route path="/app/story/:storyId/:slug" element={<Layout><StoryPage /></Layout>} />
        <Route path="/app/event/:eventSlug" element={<Layout><EventPage /></Layout>} />
        <Route path="/app/entity/:entityId" element={<Layout><EntityPage /></Layout>} />
        <Route path="/app/page/:pageId" element={<Layout><PagePage /></Layout>} />
      </Routes>
    </Router>
  )
}

export default App
