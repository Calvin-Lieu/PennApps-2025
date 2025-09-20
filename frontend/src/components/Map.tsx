import React, { useState, useCallback } from 'react'
import { MapContainer, TileLayer, Marker, Polyline, useMapEvents, Pane } from 'react-leaflet'
import L, { LatLngTuple } from 'leaflet'
import 'leaflet/dist/leaflet.css'

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
  iconUrl,
  iconRetinaUrl,
  shadowUrl,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
  className: 'start-marker'
})

const endIcon = new L.Icon({
  iconUrl,
  iconRetinaUrl,
  shadowUrl,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
  className: 'end-marker'
})

const shopIcon = new L.Icon({
  iconUrl,
  iconRetinaUrl,
  shadowUrl,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
  className: 'shop-marker'
})

interface PathState {
  startPoint: LatLngTuple | null
  endPoint: LatLngTuple | null
  path: LatLngTuple[]                // direct path (optional display)
  viaPath: LatLngTuple[]             // start→shop→end (if toggle on)
  shopPoint: LatLngTuple | null
  shopLabel?: string | null
  loading: boolean
  error: string | null
}

function MapClickHandler({ onMapClick }: { onMapClick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click: (e) => onMapClick(e.latlng.lat, e.latlng.lng)
  })
  return null
}

export default function Map() {
  const [includeShop, setIncludeShop] = useState<boolean>(true)
  const [maxShopDist, setMaxShopDist] = useState<number>(120) // meters

  const [pathState, setPathState] = useState<PathState>({
    startPoint: null,
    endPoint: null,
    path: [],
    viaPath: [],
    shopPoint: null,
    shopLabel: null,
    loading: false,
    error: null
  })

  const handleMapClick = useCallback(async (lat: number, lng: number) => {
    if (pathState.loading) return

    if (!pathState.startPoint) {
      setPathState(prev => ({
        ...prev,
        startPoint: [lat, lng],
        error: null
      }))
    } else if (!pathState.endPoint) {
      // finalize end and fetch route
      const start = pathState.startPoint
      setPathState(prev => ({
        ...prev,
        endPoint: [lat, lng],
        loading: true,
        error: null
      }))

      try {
        const url = includeShop ? 'http://localhost:8000/route_via_shop' : 'http://localhost:8000/shortest_path'
        const body: any = includeShop
          ? {
              start_lat: start[0],
              start_lng: start[1],
              end_lat: lat,
              end_lng: lng,
              max_shop_dist_m: maxShopDist
            }
          : {
              start_lat: start[0],
              start_lng: start[1],
              end_lat: lat,
              end_lng: lng
            }

        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        })

        const data = await response.json()
        // Debug: log lengths to help diagnose empty via_path issues
        console.log('route response lengths -> via_path:', (data.via_path || []).length, 'base_path:', (data.base_path || []).length, data)

        if (!response.ok || data.error) {
          throw new Error(data.error || 'Routing failed')
        }

        if (includeShop) {
          // Expecting the backend to return: via_path, base_path, shop_point, shop_label
          const base = (data.base_path || []) as LatLngTuple[]
          const via = (data.via_path || []) as LatLngTuple[]
          // If via_path is empty, fall back to base_path so the UI can still show a highlighted route
          if ((via || []).length === 0 && base.length > 0) {
            console.warn('route_via_shop returned empty via_path; falling back to base_path for viaPath')
          }
          setPathState(prev => ({
            ...prev,
            path: base,
            viaPath: via.length > 0 ? via : base,
            shopPoint: (data.shop_point || null) as LatLngTuple | null,
            shopLabel: data.shop_label || null,
            loading: false,
            error: null
          }))
        } else {
          setPathState(prev => ({
            ...prev,
            path: (data.path || []) as LatLngTuple[],
            viaPath: [],
            shopPoint: null,
            shopLabel: null,
            loading: false,
            error: null
          }))
        }
      } catch (err: any) {
        setPathState(prev => ({
          ...prev,
          loading: false,
          error: err?.message || 'Failed to compute path'
        }))
      }
    } else {
      // Reset and start over with new start
      setPathState({
        startPoint: [lat, lng],
        endPoint: null,
        path: [],
        viaPath: [],
        shopPoint: null,
        shopLabel: null,
        loading: false,
        error: null
      })
    }
  }, [pathState.startPoint, pathState.endPoint, pathState.loading, includeShop, maxShopDist])

  return (
    <div style={{ position: 'relative', height: '100%', width: '100%' }}>
      <MapContainer center={[39.9526, -75.1652]} zoom={13} style={{ height: '100%', width: '100%' }}>
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
        />
        <MapClickHandler onMapClick={handleMapClick} />

        {/* Start / End markers */}
        {pathState.startPoint && <Marker position={pathState.startPoint} icon={startIcon} />}
        {pathState.endPoint && <Marker position={pathState.endPoint} icon={endIcon} />}

        {/* Shop waypoint marker */}
        {pathState.shopPoint && <Marker position={pathState.shopPoint} icon={shopIcon} />}

        {/* Original direct path (dashed) - render in an underlay pane only if we have a viaPath */}
        <Pane name="route-under" style={{ zIndex: 399 }}>
          {pathState.viaPath.length > 0 && pathState.path.length > 0 && (
            <Polyline
              pane="route-under"
              positions={pathState.path}
              pathOptions={{ color: 'gray', weight: 3, opacity: 0.6, dashArray: '6 8' }}
            />
          )}
        </Pane>

        {/* Primary route (blue): show viaPath if present, else fallback to direct path */}
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

      {/* Controls + Status */}
      <div style={{
        position: 'absolute',
        top: 10,
        left: 10,
        background: 'white',
        padding: '10px',
        borderRadius: '5px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
        zIndex: 1000,
        maxWidth: 320
      }}>
        <div style={{ marginBottom: 8 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input
              type="checkbox"
              checked={includeShop}
              onChange={(e) => setIncludeShop(e.target.checked)}
            />
            Include shop stop
          </label>
          {includeShop && (
            <div style={{ marginTop: 8 }}>
              <label style={{ fontSize: 12 }}>
                Max shop distance from path: {maxShopDist} m
              </label>
              <input
                type="range"
                min={20}
                max={300}
                step={10}
                value={maxShopDist}
                onChange={(e) => setMaxShopDist(parseInt(e.target.value, 10))}
                style={{ width: '100%' }}
              />
            </div>
          )}
        </div>

        {!pathState.startPoint && <div>Click on the map to set start point</div>}
        {pathState.startPoint && !pathState.endPoint && <div>Click on the map to set end point</div>}
        {pathState.loading && <div>Computing path...</div>}
        {pathState.error && <div style={{ color: 'red' }}>Error: {pathState.error}</div>}

        {(pathState.viaPath.length > 0 || pathState.path.length > 0) && (
          <div>
            Route ready! Click anywhere to start over.
            <br />
            {includeShop && pathState.shopPoint ? (
              <>
                Via-shop segments: {Math.max(0, pathState.viaPath.length - 1)}<br />
                {pathState.shopLabel ? <>Shop: {pathState.shopLabel}<br /></> : null}
              </>
            ) : (
              <>Segments: {Math.max(0, pathState.path.length - 1)}<br /></>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
