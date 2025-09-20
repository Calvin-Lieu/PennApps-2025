import React, { useState, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, useMapEvents, Pane, Popup } from 'react-leaflet';
import L, { LatLngTuple } from 'leaflet';
import 'leaflet/dist/leaflet.css';
import AdvancedSettings from './AdvancedSettings';

// Fix default icon path issues in Vite by using import.meta.url
const iconRetinaUrl = new URL('leaflet/dist/images/marker-icon-2x.png', import.meta.url).href
const iconUrl = new URL('leaflet/dist/images/marker-icon.png', import.meta.url).href
const shadowUrl = new URL('leaflet/dist/images/marker-shadow.png', import.meta.url).href

delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl,
  iconUrl,
  shadowUrl
})

// Icons
const startIcon = new L.Icon({
  iconUrl: new URL('leaflet/dist/images/marker-icon.png', import.meta.url).href,
  iconRetinaUrl: new URL('leaflet/dist/images/marker-icon-2x.png', import.meta.url).href,
  shadowUrl: new URL('leaflet/dist/images/marker-shadow.png', import.meta.url).href,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
  className: 'start-marker'
});

const endIcon = new L.Icon({
    iconUrl: new URL('leaflet/dist/images/marker-icon.png', import.meta.url).href,
    iconRetinaUrl: new URL('leaflet/dist/images/marker-icon-2x.png', import.meta.url).href,
    shadowUrl: new URL('leaflet/dist/images/marker-shadow.png', import.meta.url).href,
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
    className: 'end-marker'
  });
  
  const shopIcon = new L.Icon({
    iconUrl: new URL('leaflet/dist/images/marker-icon.png', import.meta.url).href,
    iconRetinaUrl: new URL('leaflet/dist/images/marker-icon-2x.png', import.meta.url).href,
    shadowUrl: new URL('leaflet/dist/images/marker-shadow.png', import.meta.url).href,
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
    className: 'shop-marker'
  });

interface PathState {
  startPoint: LatLngTuple | null;
  endPoint: LatLngTuple | null;
  path: LatLngTuple[];
  viaPath: LatLngTuple[];
  shopPoints: { point: LatLngTuple; label: string }[];
  loading: boolean;
  error: string | null;
}

function MapClickHandler({ onMapClick }: { onMapClick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click: (e) => onMapClick(e.latlng.lat, e.latlng.lng),
  });
  return null;
}

export default function Map() {
  const [includeShop, setIncludeShop] = useState<boolean>(false);
  const [maxShopDist, setMaxShopDist] = useState<number>(1000); // Default search radius in meters
  const [selectedTags, setSelectedTags] = useState<{ [key: string]: number }>({});
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [tagMatchRule, setTagMatchRule] = useState<'any' | 'all'>('any');

  const [pathState, setPathState] = useState<PathState>({
    startPoint: null,
    endPoint: null,
    path: [],
    viaPath: [],
    shopPoints: [],
    loading: false,
    error: null,
  });

  const handleMapClick = useCallback(
    async (lat: number, lng: number) => {
      if (pathState.loading) return;

      if (!pathState.startPoint) {
        setPathState((prev) => ({
          ...prev,
          startPoint: [lat, lng],
          endPoint: null,
          path: [],
          viaPath: [],
          shopPoints: [],
          error: null,
        }));
      } else if (!pathState.endPoint) {
        const start = pathState.startPoint;
        setPathState((prev) => ({
          ...prev,
          endPoint: [lat, lng],
          loading: true,
          error: null,
        }));

        try {
          const url = 'http://localhost:8000/route_via_shop';
          
          let tags_to_send = {};
          if (includeShop) {
            if (showAdvanced) {
              tags_to_send = selectedTags;
            } else {
              tags_to_send = { convenience: 1 };
            }
          }

          const body = {
            start_lat: start[0],
            start_lng: start[1],
            end_lat: lat,
            end_lng: lng,
            shop_tags: tags_to_send,
            tag_match_rule: tagMatchRule,
            max_dist: maxShopDist, // Pass the distance to the backend
          };

          const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          });

          const data = await response.json();
          console.log('route response:', data);

          if (!response.ok || data.error) {
            throw new Error(data.error || 'Routing failed');
          }

          setPathState((prev) => ({
            ...prev,
            path: (data.base_path || []) as LatLngTuple[],
            viaPath: (data.via_path || []) as LatLngTuple[],
            shopPoints: (data.shop_points || []) as { point: LatLngTuple; label: string }[],
            loading: false,
            error: null,
          }));
        } catch (err: any) {
          setPathState((prev) => ({
            ...prev,
            loading: false,
            error: err?.message || 'Failed to compute path',
          }));
        }
      } else {
        setPathState({
          startPoint: [lat, lng],
          endPoint: null,
          path: [],
          viaPath: [],
          shopPoints: [],
          loading: false,
          error: null,
        });
      }
    },
    [pathState.loading, pathState.startPoint, pathState.endPoint, selectedTags, tagMatchRule, includeShop, showAdvanced, maxShopDist]
  );

  return (
    <div style={{ position: 'relative', height: '100%', width: '100%' }}>
      <MapContainer center={[39.9526, -75.1652]} zoom={13} style={{ height: '100%', width: '100%' }}>
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
        />
        <MapClickHandler onMapClick={handleMapClick} />

        {pathState.startPoint && <Marker position={pathState.startPoint} icon={startIcon} />}
        {pathState.endPoint && <Marker position={pathState.endPoint} icon={endIcon} />}

        {pathState.shopPoints.map((shop, index) => (
          <Marker key={`shop-${index}`} position={shop.point} icon={shopIcon}>
            <Popup>{shop.label}</Popup>
          </Marker>
        ))}

        <Pane name="route-under" style={{ zIndex: 399 }}>
          {pathState.viaPath.length > 0 && pathState.path.length > 0 && (
            <Polyline
              pane="route-under"
              positions={pathState.path}
              pathOptions={{ color: 'gray', weight: 3, opacity: 0.6, dashArray: '6 8' }}
            />
          )}
        </Pane>

        <Pane name="route-over" style={{ zIndex: 401 }}>
          {(pathState.viaPath.length > 0 || pathState.path.length > 0) && (
            <Polyline
              pane="route-over"
              positions={pathState.viaPath.length > 0 ? pathState.viaPath : pathState.path}
              pathOptions={{ color: 'blue', weight: 6, opacity: 0.95 }}
            />
          )}
        </Pane>
      </MapContainer>

      <div
        style={{
          position: 'absolute',
          top: 10,
          left: 10,
          background: 'white',
          padding: '10px',
          borderRadius: '5px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
          zIndex: 1000,
          maxWidth: 320,
          maxHeight: '80vh',
          overflowY: 'auto',
        }}
      >
        <div style={{ marginBottom: 8 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input
              type="checkbox"
              checked={includeShop}
              onChange={(e) => setIncludeShop(e.target.checked)}
            />
            Include a shop stop
          </label>

          {includeShop && !showAdvanced && (
            <div style={{ marginTop: 8 }}>
              <label style={{ fontSize: 12 }}>
                Search Radius: {maxShopDist} m
              </label>
              <input
                type="range"
                min={200}
                max={5000}
                step={100}
                value={maxShopDist}
                onChange={(e) => setMaxShopDist(parseInt(e.target.value, 10))}
                style={{ width: '100%' }}
              />
            </div>
          )}

          <button onClick={() => setShowAdvanced(!showAdvanced)} style={{ width: '100%', marginBottom: '10px', marginTop: '10px' }}>
            {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
          </button>
          {showAdvanced && (
            <AdvancedSettings
              selectedTags={selectedTags}
              setSelectedTags={setSelectedTags}
              tagMatchRule={tagMatchRule}
              setTagMatchRule={setTagMatchRule}
            />
          )}
        </div>

        <div className="path-info">
          {!pathState.startPoint && <p>Click on the map to set a start point.</p>}
          {pathState.startPoint && !pathState.endPoint && <p>Click on the map to set an end point.</p>}
          {pathState.loading && <p>Loading...</p>}
          {pathState.error && <p style={{ color: 'red' }}>Error: {pathState.error}</p>}
          {pathState.viaPath.length > 0 && <p>Route ready! Click anywhere to start over.</p>}
          
          {pathState.shopPoints.length > 0 && (
            <div>
              <p>Waypoints:</p>
              <ul>
                {pathState.shopPoints.map((shop, i) => (
                  <li key={i}>{shop.label}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
