import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, CircleMarker, useMap } from 'react-leaflet';
import { useNavigate } from 'react-router-dom';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface Entity {
  id: string;
  canonical_name: string;
  entity_type: string;
  mention_count?: number;
}

interface EpicenterMapCardProps {
  entities: Entity[];
  eventName: string;
}

interface GeocodedLocation {
  name: string;
  lat: number;
  lon: number;
  mentionCount: number;
  country?: string;
}

// Create epicenter marker icon
const createEpicenterIcon = () => {
  return L.divIcon({
    className: 'epicenter-marker',
    html: `
      <div style="position: relative;">
        <div style="
          background: radial-gradient(circle, #ef4444 0%, #dc2626 100%);
          width: 14px;
          height: 14px;
          border-radius: 50%;
          border: 2px solid #fff;
          box-shadow: 0 0 8px rgba(239, 68, 68, 0.6), 0 2px 4px rgba(0,0,0,0.3);
          animation: epicenterPulse 2s infinite;
        "></div>
      </div>
    `,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });
};

// Component to set map view
const SetView: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
};

const EpicenterMapCard: React.FC<EpicenterMapCardProps> = ({ entities }) => {
  const [epicenter, setEpicenter] = useState<GeocodedLocation | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  const handleMapClick = () => {
    if (epicenter) {
      const params = new URLSearchParams({
        lat: epicenter.lat.toString(),
        lon: epicenter.lon.toString(),
        zoom: '10',
        name: epicenter.name,
      });
      navigate(`/map?${params.toString()}`);
    }
  };

  useEffect(() => {
    findEpicenter();
  }, [entities]);

  const findEpicenter = async () => {
    setLoading(true);

    // Filter location entities and sort by mention count
    const locationEntities = entities
      .filter(e => e.entity_type === 'LOCATION' || e.entity_type === 'GPE')
      .sort((a, b) => (b.mention_count || 0) - (a.mention_count || 0));

    if (locationEntities.length === 0) {
      setLoading(false);
      return;
    }

    // Try to geocode the top location (epicenter)
    const topLocation = locationEntities[0];
    const coords = await geocodeLocation(topLocation.canonical_name);

    if (coords) {
      setEpicenter({
        name: topLocation.canonical_name,
        lat: coords.lat,
        lon: coords.lon,
        mentionCount: topLocation.mention_count || 0,
        country: coords.country,
      });
    }

    setLoading(false);
  };

  const geocodeLocation = async (location: string): Promise<{ lat: number; lon: number; country?: string } | null> => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(location)}&limit=1&addressdetails=1`,
        {
          headers: {
            'User-Agent': 'HereNews/1.0',
          },
        }
      );

      if (!response.ok) return null;

      const results = await response.json();
      if (results.length === 0) return null;

      return {
        lat: parseFloat(results[0].lat),
        lon: parseFloat(results[0].lon),
        country: results[0].address?.country,
      };
    } catch (error) {
      console.error('Geocoding error:', error);
      return null;
    }
  };

  // No location data
  if (!loading && !epicenter) {
    return null;
  }

  // Loading state
  if (loading) {
    return (
      <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden w-64">
        <div className="p-3 flex items-center gap-2">
          <div className="animate-pulse w-3 h-3 bg-slate-300 rounded-full"></div>
          <span className="text-slate-500 text-sm">Locating...</span>
        </div>
      </div>
    );
  }

  const regionalCenter: [number, number] = [epicenter!.lat, epicenter!.lon];

  return (
    <div
      className="bg-white rounded-lg border border-slate-200 shadow-md overflow-hidden w-64 cursor-pointer hover:shadow-lg transition-shadow group"
      onClick={handleMapClick}
      title="Click to view on full map"
    >
      {/* Main Regional Map with Inset */}
      <div className="relative h-48">
        {/* Main map (regional view) */}
        <MapContainer
          center={regionalCenter}
          zoom={9}
          style={{ height: '100%', width: '100%' }}
          zoomControl={false}
          attributionControl={false}
          dragging={false}
          scrollWheelZoom={false}
          doubleClickZoom={false}
          touchZoom={false}
        >
          {/* Light tile layer for better contrast */}
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            maxZoom={20}
          />
          <Marker
            position={regionalCenter}
            icon={createEpicenterIcon()}
          />
          {/* Subtle radius circle */}
          <CircleMarker
            center={regionalCenter}
            radius={20}
            pathOptions={{
              color: '#ef4444',
              fillColor: '#ef4444',
              fillOpacity: 0.08,
              weight: 1,
              dashArray: '4,4',
            }}
          />
          <SetView center={regionalCenter} zoom={9} />
        </MapContainer>

        {/* Inset global map (bottom-right corner) */}
        <div className="absolute bottom-2 right-2 w-20 h-16 rounded border-2 border-white shadow-lg overflow-hidden z-[1000]">
          <MapContainer
            center={regionalCenter}
            zoom={1}
            style={{ height: '100%', width: '100%' }}
            zoomControl={false}
            attributionControl={false}
            dragging={false}
            scrollWheelZoom={false}
            doubleClickZoom={false}
            touchZoom={false}
          >
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
              maxZoom={20}
            />
            {/* Red dot for location on globe */}
            <CircleMarker
              center={regionalCenter}
              radius={4}
              pathOptions={{
                color: '#fff',
                fillColor: '#ef4444',
                fillOpacity: 1,
                weight: 1,
              }}
            />
            <SetView center={regionalCenter} zoom={1} />
          </MapContainer>
          {/* Viewport indicator box */}
          <div className="absolute inset-0 border border-red-400/50 pointer-events-none" />
        </div>

        {/* Location label overlay */}
        <div className="absolute top-2 left-2 z-[1000]">
          <div className="bg-white/95 backdrop-blur-sm px-2.5 py-1.5 rounded shadow-sm border border-slate-200">
            <div className="flex items-center gap-1.5">
              <span className="text-red-500 text-sm">●</span>
              <span className="text-slate-800 text-sm font-medium">{epicenter!.name}</span>
            </div>
            {epicenter!.country && epicenter!.name !== epicenter!.country && (
              <div className="text-slate-500 text-xs ml-4">{epicenter!.country}</div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="px-3 py-2 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs">
        <span className="text-slate-400">
          {epicenter!.lat.toFixed(2)}°, {epicenter!.lon.toFixed(2)}°
        </span>
        <div className="flex items-center gap-2">
          {epicenter!.mentionCount > 0 && (
            <span className="text-slate-500">
              {epicenter!.mentionCount} mentions
            </span>
          )}
          <span className="text-slate-400 group-hover:text-indigo-500 transition-colors">
            View map →
          </span>
        </div>
      </div>

      {/* CSS for pulse animation */}
      <style>{`
        @keyframes epicenterPulse {
          0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5), 0 2px 4px rgba(0,0,0,0.3); }
          70% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0), 0 2px 4px rgba(0,0,0,0.3); }
          100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0), 0 2px 4px rgba(0,0,0,0.3); }
        }
      `}</style>
    </div>
  );
};

export default EpicenterMapCard;
