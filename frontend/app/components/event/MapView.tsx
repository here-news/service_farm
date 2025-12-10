import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface Entity {
  id: string;
  canonical_name: string;
  entity_type: string;
  mention_count?: number;
}

interface Claim {
  id: string;
  text: string;
}

interface MapViewProps {
  entities: Entity[];
  claims: Claim[];
  eventName: string;
}

interface LocationWithCoords {
  entity: Entity;
  lat: number;
  lon: number;
}

// Fix Leaflet default marker icons
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

// Custom marker icons
const createMainMarkerIcon = () => {
  return L.divIcon({
    className: 'custom-marker',
    html: '<div style="background: #667eea; width: 30px; height: 30px; border-radius: 50%; border: 3px solid #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.5);"></div>',
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  });
};

const createLocationMarkerIcon = () => {
  return L.divIcon({
    className: 'custom-marker',
    html: '<div style="background: #10b981; width: 18px; height: 18px; border-radius: 50%; border: 2px solid #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  });
};

// Component to fit map bounds
const FitBounds: React.FC<{ locations: LocationWithCoords[] }> = ({ locations }) => {
  const map = useMap();

  useEffect(() => {
    if (locations.length > 0) {
      const bounds = L.latLngBounds(locations.map(loc => [loc.lat, loc.lon]));
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [locations, map]);

  return null;
};

const MapView: React.FC<MapViewProps> = ({ entities, claims, eventName }) => {
  const [locations, setLocations] = useState<LocationWithCoords[]>([]);
  const [loading, setLoading] = useState(true);
  const [primaryLocation, setPrimaryLocation] = useState<LocationWithCoords | null>(null);

  useEffect(() => {
    geocodeLocations();
  }, [entities]);

  const geocodeLocations = async () => {
    setLoading(true);

    // Filter location entities
    const locationEntities = entities.filter(
      e => e.entity_type === 'LOCATION' || e.entity_type === 'GPE'
    );

    if (locationEntities.length === 0) {
      setLoading(false);
      return;
    }

    // Sort by mention count to prioritize important locations
    const sortedLocations = [...locationEntities].sort(
      (a, b) => (b.mention_count || 0) - (a.mention_count || 0)
    );

    // Geocode locations (limit to first 10 to avoid API rate limits)
    const geocoded: LocationWithCoords[] = [];

    for (const entity of sortedLocations.slice(0, 10)) {
      const coords = await geocodeLocation(entity.canonical_name);
      if (coords) {
        geocoded.push({
          entity,
          lat: coords.lat,
          lon: coords.lon,
        });
      }
      // Add delay to respect Nominatim rate limits (1 request per second)
      await new Promise(resolve => setTimeout(resolve, 1100));
    }

    setLocations(geocoded);
    if (geocoded.length > 0) {
      setPrimaryLocation(geocoded[0]);
    }
    setLoading(false);
  };

  const geocodeLocation = async (location: string): Promise<{ lat: number; lon: number } | null> => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(location)}&limit=1`,
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
      };
    } catch (error) {
      console.error('Geocoding error:', error);
      return null;
    }
  };

  const getLocationMentions = (locationName: string): number => {
    return claims.filter(c =>
      c.text.toLowerCase().includes(locationName.toLowerCase())
    ).length;
  };

  if (loading) {
    return (
      <div className="h-[600px] bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Geocoding locations...</p>
          <p className="text-gray-500 text-sm mt-2">This may take a few moments</p>
        </div>
      </div>
    );
  }

  if (locations.length === 0) {
    return (
      <div className="h-[600px] bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-center text-gray-400">
          <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
          </svg>
          <p className="text-lg font-medium">Geographic Map</p>
          <p className="text-sm mt-2">No geographic locations found or could not geocode locations</p>
        </div>
      </div>
    );
  }

  const defaultCenter: [number, number] = primaryLocation
    ? [primaryLocation.lat, primaryLocation.lon]
    : [22.3193, 114.1694]; // Default to Hong Kong

  return (
    <div className="h-[600px] bg-gray-900 rounded-lg overflow-hidden relative">
      <MapContainer
        center={defaultCenter}
        zoom={10}
        style={{ height: '100%', width: '100%' }}
        zoomControl={true}
      >
        {/* Dark tile layer */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          maxZoom={20}
        />

        {/* Primary location marker */}
        {primaryLocation && (
          <Marker
            position={[primaryLocation.lat, primaryLocation.lon]}
            icon={createMainMarkerIcon()}
          >
            <Popup>
              <div className="text-sm">
                <div className="text-blue-400 font-semibold text-xs uppercase mb-1">Main Location</div>
                <div className="font-bold text-base mb-1">{primaryLocation.entity.canonical_name}</div>
                <div className="text-gray-600 text-xs">
                  {getLocationMentions(primaryLocation.entity.canonical_name)} mention(s) in claims
                </div>
              </div>
            </Popup>
          </Marker>
        )}

        {/* Other location markers */}
        {locations.slice(1).map((loc) => (
          <Marker
            key={loc.entity.id}
            position={[loc.lat, loc.lon]}
            icon={createLocationMarkerIcon()}
          >
            <Popup>
              <div className="text-sm">
                <div className="text-green-400 font-semibold text-xs uppercase mb-1">Location Entity</div>
                <div className="font-bold text-base mb-1">{loc.entity.canonical_name}</div>
                <div className="text-gray-600 text-xs mb-1">
                  {getLocationMentions(loc.entity.canonical_name)} mention(s) in claims
                </div>
                {loc.entity.mention_count && (
                  <div className="text-gray-600 text-xs">
                    {loc.entity.mention_count} total mentions
                  </div>
                )}
              </div>
            </Popup>
          </Marker>
        ))}

        {/* Fit bounds to show all locations */}
        <FitBounds locations={locations} />
      </MapContainer>

      {/* Info panel */}
      <div className="absolute bottom-4 left-4 bg-gray-800/95 backdrop-blur-sm rounded-lg p-4 border border-gray-700 z-[1000] max-w-xs">
        <h3 className="text-sm font-semibold text-white mb-2">Geographic Distribution</h3>
        <div className="flex gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">{locations.length}</div>
            <div className="text-xs text-gray-400">Locations</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">{claims.length}</div>
            <div className="text-xs text-gray-400">Claims</div>
          </div>
        </div>
        <div className="mt-2 text-xs text-gray-500">
          {eventName}
        </div>
      </div>
    </div>
  );
};

export default MapView;
