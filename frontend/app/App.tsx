import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
import HomePage from './HomePage'
import StoryPage from './StoryPage'
import EventPage from './EventPage'
import EntityPage from './EntityPage'
import PagePage from './PagePage'
import ArchivePage from './ArchivePage'
import GraphPage from './GraphPage'
import MapPage from './MapPage'

function App() {
  return (
    <Router basename="/app">
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/archive" element={<ArchivePage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/map" element={<MapPage />} />
          <Route path="/story/:storyId" element={<StoryPage />} />
          <Route path="/story/:storyId/:slug" element={<StoryPage />} />
          <Route path="/event/:eventSlug" element={<EventPage />} />
          <Route path="/entity/:entityId" element={<EntityPage />} />
          <Route path="/page/:pageId" element={<PagePage />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App
