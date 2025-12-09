import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import HomePage from './HomePage'
import StoryPage from './StoryPage'
import EventPage from './EventPage'
import ArchivePage from './ArchivePage'
import GraphPage from './GraphPage'
import MapPage from './MapPage'

function App() {
  return (
    <Router basename="/app">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/archive" element={<ArchivePage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/map" element={<MapPage />} />
        <Route path="/story/:storyId" element={<StoryPage />} />
        <Route path="/story/:storyId/:slug" element={<StoryPage />} />
        <Route path="/event/:eventSlug" element={<EventPage />} />
      </Routes>
    </Router>
  )
}

export default App
