// src/components/Map.tsx
import { useEffect, useRef, useState, useCallback } from "react";
import * as L from "leaflet";
import "leaflet/dist/leaflet.css";

// ShadeMap → adds L.shadeMap(...)
import "leaflet-shadow-simulator";
// @ts-ignore – local shim in src/types
import osmtogeojson from "osmtogeojson";

import AddressSearch from "./AddressSearch";

// ---------- Icons fix for Vite ----------
const iconRetinaUrl = new URL("leaflet/dist/images/marker-icon-2x.png", import.meta.url).href;
const iconUrl = new URL("leaflet/dist/images/marker-icon.png", import.meta.url).href;
const shadowUrl = new URL("leaflet/dist/images/marker-shadow.png", import.meta.url).href;
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({ iconRetinaUrl, iconUrl, shadowUrl });

const startIcon = new L.Icon({
  iconUrl, iconRetinaUrl, shadowUrl,
  iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41],
  className: "start-marker",
});
const endIcon = new L.Icon({
  iconUrl, iconRetinaUrl, shadowUrl,
  iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41],
  className: "end-marker",
});

// Water fountain icon
const waterIcon = new L.Icon({
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [20, 32], iconAnchor: [10, 32], popupAnchor: [1, -24], shadowSize: [32, 32],
  className: "water-marker",
});

// ---------- Types ----------
type Pt = { lat: number; lng: number };
export type Edge = { id: string; a: Pt; b: Pt };
export type EdgeResult = { id: string; shadePct: number; shaded: boolean; nSamples: number };

interface Waypoint {
  id: string;
  type: string;
  name: string;
  coordinates: [number, number]; // [lat, lng]
  longitude: number;
  latitude: number;
  amenity: string;
  shop: string;
  opening_hours: string;
  website: string;
  phone: string;
}

function metersToLatDeg(m: number) { return m / 110540; }
function metersToLngDeg(m: number, lat: number) { return m / (111320 * Math.cos(lat * Math.PI / 180)); }
function isShadowRGBA(arr: Uint8ClampedArray, alphaThreshold = 16) { return arr[3] >= alphaThreshold; }
function lerp(a: Pt, b: Pt, t: number): Pt { return { lat: a.lat + (b.lat - a.lat) * t, lng: a.lng + (b.lng - a.lng) * t }; }
function jitterMeters(p: Pt, r: number): Pt {
  if (!r) return p;
  const ang = Math.random() * 2 * Math.PI;
  const rad = Math.random() * r;
  const dx = rad * Math.cos(ang), dy = rad * Math.sin(ang);
  return { lat: p.lat + metersToLatDeg(dy), lng: p.lng + metersToLngDeg(dx, p.lat) };
}
function colorForPct(p: number) { 
  // Gradient from red (0% shade = sunny/hot) to green (100% shade = cool)
  // 0% shade = red (sunny/hot), 100% shade = green (shaded/cool)
  const red = Math.round((1 - p) * 255);
  const green = Math.round(p * 255);
  return `rgb(${red}, ${green}, 0)`;
}

interface PathState {
  startPoint: [number, number] | null;
  endPoint: [number, number] | null;
  path: [number, number][];
  loading: boolean;
  error: string | null;
  routeStats?: {
    originalDistance: number;
    shadeAwareDistance: number;
    shadePenalty: number;
    analysisTime: string;
    shadeMode: string;
    numSegments: number;
    shadedSegments: number;
    shadePercentage: number;
    totalShadeLength: number;
    shadePenaltyAdded: number;
  };
  waypoints?: Waypoint[]; // Add waypoints to path state
}

export default function Map({
  edges = [],
  date = new Date(),
  onResults,
}: {
  edges?: Edge[];
  date?: Date;
  onResults?: (r: EdgeResult[]) => void;
} = {}) {
  const mapRef = useRef<L.Map | null>(null);
  const shadeRef = useRef<any>(null);
  const edgeLayerRef = useRef<L.LayerGroup | null>(null);
  const pathLayerRef = useRef<L.LayerGroup | null>(null);
  const waypointLayerRef = useRef<L.LayerGroup | null>(null); // New layer for waypoints
  const markersRef = useRef<L.Marker[]>([]);
  const [testToggle, setTestToggle] = useState(false);
  const [ready, setReady] = useState(false);
  const [currentHour, setCurrentHour] = useState(9);
  const [shadePenalty, setShadePenalty] = useState(1.0);
  const [useShadeRouting, setUseShadeRouting] = useState(true);
  const [includeWaterStops, setIncludeWaterStops] = useState(false); // New state for water stops
  const [maxDetourDistance, setMaxDetourDistance] = useState(100); // Max detour in meters
  const fetchTokenRef = useRef(0);

  // Use refs instead of state to avoid re-renders for pathfinding
  const pathStateRef = useRef<PathState>({
    startPoint: null,
    endPoint: null,
    path: [],
    loading: false,
    error: null,
    routeStats: undefined,
    waypoints: []
  });
  const [pathUIState, setPathUIState] = useState<PathState>({
    startPoint: null,
    endPoint: null,
    path: [],
    loading: false,
    error: null,
    routeStats: undefined,
    waypoints: []
  });

  // Refs for reactive recomputation system
  const penaltyUpdateTimeoutRef = useRef<number | null>(null);
  const prevShadeRoutingRef = useRef(useShadeRouting);
  const prevCurrentHourRef = useRef(currentHour);
  const prevIncludeWaterStopsRef = useRef(includeWaterStops); // Track water stop changes
  const lastDateRef = useRef(new Date());

  // Building data caching system
  const buildingDataCacheRef = useRef<any[]>([]);
  const lastBoundsRef = useRef<string>('');
  const shadeOptionsRef = useRef<any>(null);

  // Build ShadeMap options using correct API (with comprehensive caching)
  const buildShadeOptions = (when: Date) => {
    // Check if we can reuse cached options
    if (shadeOptionsRef.current) {
      return {
        ...shadeOptionsRef.current,
        date: when, // Only update the date
      };
    }

    const options = {
      date: when,
      color: "#01112f",
      opacity: 0.7,
      apiKey: (import.meta as any).env.VITE_SHADEMAP_KEY,
      terrainSource: {
        tileSize: 256,
        maxZoom: 15,
        getSourceUrl: ({ x, y, z }: any) =>
          `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`,
        getElevation: ({ r, g, b, a }: any) => (r * 256 + g + b / 256) - 32768,
        _overzoom: 19,
      },
      getFeatures: async () => {
        if (!mapRef.current || mapRef.current.getZoom() < 15) return [];
        
        const my = ++fetchTokenRef.current;
        await new Promise((r) => setTimeout(r, 200)); // debounce small pans
        
        if (my !== fetchTokenRef.current) {
          console.log("🢠 Fetch cancelled due to newer request");
          return [];
        }

        const b = mapRef.current.getBounds();
        const north = b.getNorth(), south = b.getSouth(), east = b.getEast(), west = b.getWest();
        
        // Create a bounds key to check if we need to refetch
        const boundsKey = `${north.toFixed(4)},${south.toFixed(4)},${east.toFixed(4)},${west.toFixed(4)}`;
        
        console.log("🢠 Current bounds:", boundsKey);
        console.log("🢠 Last bounds:", lastBoundsRef.current);
        console.log("🢠 Cached buildings count:", buildingDataCacheRef.current.length);
        
        // Return cached data if bounds haven't changed significantly
        if (lastBoundsRef.current === boundsKey && buildingDataCacheRef.current.length > 0) {
          console.log("✅ Using cached building data");
          return buildingDataCacheRef.current;
        }

        console.log("🔄 Fetching new building data for bounds:", boundsKey);

        const query = `
          [out:json][timeout:25];
          (
            way["building"](${south},${west},${north},${east});
            relation["building"](${south},${west},${north},${east});
          );
          (._;>;);
          out body;
        `;
        const overpass = "https://overpass-api.de/api/interpreter";
        const url = `${overpass}?data=${encodeURIComponent(query)}`;

        try {
          console.log("🌐 Starting Overpass API request...");
          const resp = await fetch(url);
          if (!resp.ok) {
            console.log("❌ Overpass API request failed, using cached data");
            return buildingDataCacheRef.current; // Return cached data on error
          }
          
          console.log("🌐 Overpass API response received, parsing...");
          const data = await resp.json();
          const gj = osmtogeojson(data);

          console.log("🏗️ Processing building features...");
          for (const f of gj.features) {
            const props = (f.properties ||= {});
            let h: number | undefined;
            if (props.height) {
              const m = String(props.height).match(/[\d.]+/);
              if (m) h = parseFloat(m[0]);
            }
            if (!h && props["building:levels"]) {
              const lv = parseFloat(String(props["building:levels"]));
              if (!Number.isNaN(lv)) h = lv * 3;
            }
            if (!h || !Number.isFinite(h)) h = 10;
            props.height = h;
            props.render_height = h;
          }
          
          // Cache the building data and bounds
          buildingDataCacheRef.current = gj.features;
          lastBoundsRef.current = boundsKey;
          
          console.log("✅ Building data cached. Features count:", gj.features.length);
          return gj.features;
        } catch (e) {
          console.warn("❌ Error fetching building data:", e);
          console.log("🔄 Falling back to cached data");
          return buildingDataCacheRef.current; // Return cached data on error
        }
      },
    };

    // Cache the options (excluding the date which changes)
    shadeOptionsRef.current = {
      color: options.color,
      opacity: options.opacity,
      apiKey: options.apiKey,
      terrainSource: options.terrainSource,
      getFeatures: options.getFeatures,
    };

    return options;;
  };

  // Helper to create the ShadeMap layer
  const createShadeLayer = (map: L.Map, when: Date) => {
    console.log("Creating shade layer for time:", when);
    const layer = (L as any).shadeMap(buildShadeOptions(when));

    layer.once("idle", () => {
      console.log("Shade layer ready");
      setReady(true);
    });

    layer.addTo(map);
    shadeRef.current = layer;

    // Prevent the shade layer from responding to map events
    if (layer._container || layer.getContainer?.()) {
      const container = layer._container || layer.getContainer();
      if (container) {
        container.style.pointerEvents = 'none';
      }
    }
  };

  // Function to find nearby waypoints along a path
  const findNearbyWaypoints = useCallback(async (pathCoords: [number, number][], maxDetour: number = 100): Promise<Waypoint[]> => {
    if (pathCoords.length === 0) return [];

    try {
      console.log("🚰 Fetching nearby water stops...");
  const response = await fetch('http://localhost:8000/waypoints/near_path', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          path: pathCoords,
          max_detour_meters: maxDetour,
          waypoint_types: ['water'] // Nearby fountains as fallback visualization
        })
      });

      if (!response.ok) {
        console.warn("Failed to fetch waypoints:", response.statusText);
        return [];
      }

      const data = await response.json();
      console.log("🚰 Found waypoints:", data.waypoints?.length || 0);
      
      return data.waypoints || [];
    } catch (error) {
      console.error("Error fetching waypoints:", error);
      return [];
    }
  }, []);

  // Function to display waypoints on the map
  const displayWaypoints = useCallback((waypoints: Waypoint[]) => {
    const waypointLayer = waypointLayerRef.current!;
    waypointLayer.clearLayers();

    waypoints.forEach((waypoint) => {
      const [lat, lng] = waypoint.coordinates;

      // Humanize names like "drinking_water" → "Drinking water"
      const humanize = (s?: string) => {
        if (!s) return 'Water stop';
        const t = String(s).replace(/_/g, ' ').trim();
        if (t.toLowerCase() === 'drinking water' || t.toLowerCase() === 'drinking_water') return 'Drinking water';
        // Title-case basic words
        return t.replace(/\w\S*/g, (w) => w.charAt(0).toUpperCase() + w.slice(1));
      };
      const displayName = humanize(waypoint.name);
      
      // Create custom icon for water fountains
      const icon = L.divIcon({
        html: '💧',
        iconSize: [32, 32],
        iconAnchor: [16, 28],
        popupAnchor: [0, -28],
        className: 'water-waypoint-icon'
      });

      const marker = L.marker([lat, lng], { icon })
        .bindPopup(`
          <div style="font-size: 12px; line-height: 1.4;">
            <b>💧 ${displayName}</b><br>
            <small>${humanize(waypoint.amenity) || 'Water stop'}</small><br>
            ${waypoint.opening_hours ? `<small>Hours: ${waypoint.opening_hours}</small><br>` : ''}
            <small>Lat: ${lat.toFixed(6)}, Lng: ${lng.toFixed(6)}</small>
          </div>
        `)
        .bindTooltip(`💧 ${displayName}`, { 
          permanent: false, 
          direction: 'top',
          offset: [0, -20]
        })
        .addTo(waypointLayer);
    });
  }, []);

  // Function to display path with gradient shade analysis
  const displayPathWithShadeAnalysis = useCallback(async (pathCoords: [number, number][]) => {
    if (!ready || !shadeRef.current || !mapRef.current) return;
    console.log("hello");
    // Convert path to edges for analysis
    const pathEdges: Edge[] = pathCoords.slice(0, -1).map((point, i) => ({
      id: `path-${i}`,
      a: { lat: point[0], lng: point[1] },
      b: { lat: pathCoords[i + 1][0], lng: pathCoords[i + 1][1] }
    }));

    // Analyze each path segment
    const map = mapRef.current;
    const shade = shadeRef.current;
    const rect = map.getContainer().getBoundingClientRect();
    const pathResults: EdgeResult[] = [];

    for (const edge of pathEdges) {
      const lenM = L.latLng(edge.a).distanceTo(L.latLng(edge.b));
      const steps = Math.min(Math.max(1, Math.ceil(lenM / 10)), 20);
      let hits = 0, total = 0;

      for (let j = 0; j <= steps; j++) {
        const t = steps === 0 ? 0.5 : j / steps;
        const base = lerp(edge.a, edge.b, t);

        for (let s = 0; s < 3; s++) {
          const p = jitterMeters(base, 1.5);
          const cp = map.latLngToContainerPoint([p.lat, p.lng]);

          if (cp.x < 0 || cp.y < 0 || cp.x >= rect.width || cp.y >= rect.height) {
            continue;
          }

          const xWin = rect.left + cp.x;
          const yWin = window.innerHeight - (rect.top + cp.y);

          try {
            const rgba: Uint8ClampedArray = shade.readPixel(xWin, yWin);
            if (rgba && isShadowRGBA(rgba, 16)) hits++;
            total++;
          } catch (e) {
            console.warn('Error reading pixel for path analysis:', e);
          }
        }
      }

      const shadePct = total ? hits / total : 0;
      pathResults.push({
        id: edge.id,
        shadePct,
        shaded: shadePct >= 0.5,
        nSamples: total
      });
    }

    // Display path with gradient colors and preserve markers
    const pathLayer = pathLayerRef.current!;
    const markers: L.Marker[] = [];
    pathLayer.eachLayer((layer) => {
      if (layer instanceof L.Marker) {
        markers.push(layer);
      }
    });
    pathLayer.clearLayers();
    markers.forEach(marker => pathLayer.addLayer(marker));

    // Draw path segments with gradient colors
    for (let i = 0; i < pathEdges.length; i++) {
      const edge = pathEdges[i];
      const result = pathResults.find(r => r.id === edge.id);
      const pct = result?.shadePct ?? 0;

      L.polyline(
        [[edge.a.lat, edge.a.lng], [edge.b.lat, edge.b.lng]],
        { color: colorForPct(pct), weight: 6, opacity: 0.8 }
      )
        .bindTooltip(`Segment ${i + 1}: ${(pct * 100).toFixed(0)}% shaded (${result?.nSamples || 0} samples)`)
        .addTo(pathLayer);
    }

    // Add waypoints if enabled
    if (includeWaterStops) {
      // Prefer backend-provided chosen waypoint if present; otherwise optionally show nearby fountains
      const existing = pathStateRef.current.waypoints || [];
      if (existing.length > 0) {
        displayWaypoints(existing);
      } else {
        const waypoints = await findNearbyWaypoints(pathCoords, maxDetourDistance);
        pathStateRef.current.waypoints = waypoints;
        setPathUIState(prev => ({ ...prev, waypoints }));
        displayWaypoints(waypoints);
      }
    }
  }, [ready, includeWaterStops, maxDetourDistance, findNearbyWaypoints, displayWaypoints]);

  // Unified function to compute and display path with backend API calls
  const computeAndDisplayPath = useCallback(async () => {
    if (!pathStateRef.current.startPoint || !pathStateRef.current.endPoint) {
      return;
    }

    const startPoint = pathStateRef.current.startPoint;
    const endPoint = pathStateRef.current.endPoint;

    console.log("🚀 Computing path from backend API");
    pathStateRef.current = {
      ...pathStateRef.current,
      loading: true,
      error: null
    };
    setPathUIState({ ...pathStateRef.current });

    try {
      // Choose endpoint based on routing mode
      const endpoint = useShadeRouting ? 'shortest_path_shade' : 'shortest_path';
      const basePayload = {
        start_lat: startPoint[0],
        start_lng: startPoint[1],
        end_lat: endPoint[0],
        end_lng: endPoint[1]
      };

      const payload: any = useShadeRouting ? {
        ...basePayload,
        time: currentHour,
        shade_penalty: shadePenalty
      } : { ...basePayload };

      // Ask backend to route via a water stop/store when enabled
      if (includeWaterStops) {
        payload.add_water_stop = true;
      }

      console.log("📡 Calling backend API:", endpoint, payload);
      const response = await fetch(`http://localhost:8000/${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      console.log("📥 Backend response:", data);

      if (data.error) {
        pathStateRef.current = {
          ...pathStateRef.current,
          loading: false,
          error: data.error
        };
        setPathUIState({ ...pathStateRef.current });
        return;
      }

      const pathCoords: [number, number][] = data.path || [];
      
      // Extract route statistics 
      let routeStats = undefined;
      if (data.original_distance_m !== undefined || data.total_distance_m !== undefined) {
        routeStats = {
          originalDistance: data.original_distance_m || data.total_distance_m,
          shadeAwareDistance: data.shade_aware_distance_m || data.total_distance_m,
          shadePenalty: data.shade_penalty_applied || 1.0,
          analysisTime: data.analysis_time || "9:00",
          shadeMode: data.shade_mode || "standard",
          numSegments: data.num_segments || 0,
          shadedSegments: data.shaded_segments || 0,
          shadePercentage: data.shade_percentage || 0,
          totalShadeLength: data.total_shade_length_m || 0,
          shadePenaltyAdded: data.shade_penalty_added_m || 0
        };
      }

      // Normalize backend-provided waypoint (if any) into our Waypoint shape
      let backendWaypoints: Waypoint[] = [];
      if (includeWaterStops && Array.isArray(data.waypoints) && data.waypoints.length > 0) {
        backendWaypoints = data.waypoints.map((w: any, idx: number) => {
          const lat = w.lat ?? w.latitude ?? (Array.isArray(w.coordinates) ? w.coordinates[0] : undefined);
          const lon = w.lon ?? w.lng ?? w.longitude ?? (Array.isArray(w.coordinates) ? w.coordinates[1] : undefined);
          const name = w.name || 'Water Stop';
          const id = w.id || `${name}-${lat}-${lon}-${idx}`;
          return {
            id,
            type: w.type || 'water',
            name,
            coordinates: [lat, lon],
            longitude: lon,
            latitude: lat,
            amenity: w.amenity || '',
            shop: w.shop || '',
            opening_hours: w.opening_hours || '',
            website: w.website || '',
            phone: w.phone || ''
          } as Waypoint;
        }).filter((wp: any) => Number.isFinite(wp.coordinates?.[0]) && Number.isFinite(wp.coordinates?.[1]));
      }

      pathStateRef.current = {
        ...pathStateRef.current,
        path: pathCoords,
        loading: false,
        error: null,
        routeStats,
        waypoints: backendWaypoints // Waypoints from backend if any
      };
      setPathUIState({ ...pathStateRef.current });

      console.log("✅ Path computed, displaying on map with shade analysis");
      // Display path with shade analysis
      if (pathCoords.length > 0) {
        // If backend chose a waypoint, render it immediately (always show it since route uses it)
        if (backendWaypoints.length > 0) {
          console.log("📍 Rendering backend-chosen waypoint(s):", backendWaypoints.length);
          displayWaypoints(backendWaypoints);
        }
        await displayPathWithShadeAnalysis(pathCoords);
      }

    } catch (err) {
      console.error("❌ Backend API error:", err);
      pathStateRef.current = {
        ...pathStateRef.current,
        loading: false,
        error: 'Failed to compute path'
      };
      setPathUIState({ ...pathStateRef.current });
    }
  }, [useShadeRouting, currentHour, shadePenalty, displayPathWithShadeAnalysis]);

  // Handle map clicks for pathfinding
  const handleMapClick = useCallback(async (e: L.LeafletMouseEvent) => {
    if (pathStateRef.current.loading) return;
    const { lat, lng } = e.latlng;
    console.log("Map clicked for pathfinding at:", lat, lng);

    if (!pathStateRef.current.startPoint) {
      // Set start point
      console.log("Setting start point");
      pathStateRef.current = {
        ...pathStateRef.current,
        startPoint: [lat, lng],
        error: null
      };

      // Update UI state
      setPathUIState({ ...pathStateRef.current });

      // Add start marker with custom icon to path layer
      const pathLayer = pathLayerRef.current!;
      const marker = L.marker([lat, lng], { icon: startIcon }).addTo(pathLayer);
      marker.bindPopup("Start Point");
      markersRef.current.push(marker);
      
    } else if (!pathStateRef.current.endPoint) {
      // Set end point
      console.log("Setting end point");
      pathStateRef.current = {
        ...pathStateRef.current,
        endPoint: [lat, lng],
        error: null
      };

      // Update UI state
      setPathUIState({ ...pathStateRef.current });

      // Add end marker with custom icon to path layer
      const pathLayer = pathLayerRef.current!;
      const marker = L.marker([lat, lng], { icon: endIcon }).addTo(pathLayer);
      marker.bindPopup("End Point");
      markersRef.current.push(marker);

      // Compute path using backend API
      console.log("🔄 Both points set, calling backend API");
      await computeAndDisplayPath();
      
    } else {
      // Reset and start over
      console.log("Resetting pathfinding");
      
      // Clear path layer (which includes all pathfinding markers)
      const pathLayer = pathLayerRef.current!;
      const waypointLayer = waypointLayerRef.current!;
      pathLayer.clearLayers();
      waypointLayer.clearLayers(); // Clear waypoints too
      markersRef.current = [];

      pathStateRef.current = {
        startPoint: [lat, lng],
        endPoint: null,
        path: [],
        loading: false,
        error: null,
        routeStats: undefined,
        waypoints: []
      };

      // Update UI state
      setPathUIState({ ...pathStateRef.current });

      // Add new start marker with custom icon to path layer
      const marker = L.marker([lat, lng], { icon: startIcon }).addTo(pathLayer);
      marker.bindPopup("Start Point");
      markersRef.current.push(marker);
    }
  }, [computeAndDisplayPath]);

  // attach click handler after map created
  useEffect(() => {
    console.log("TestMap useEffect triggered - Map setup");
    
    // Map setup
    const mapContainer = document.getElementById("test-map");
    if (!mapContainer) return;

    const map = L.map(mapContainer, {
      zoomControl: true,
    }).setView([39.9526, -75.1652], 16);

    mapRef.current = map;

    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OSM",
      maxZoom: 19,
    }).addTo(map);

    // Create edge layer for shadow classification
    edgeLayerRef.current = L.layerGroup().addTo(map);

    // Create path layer for pathfinding
    pathLayerRef.current = L.layerGroup().addTo(map);

    // Create waypoint layer for water stops
    waypointLayerRef.current = L.layerGroup().addTo(map);

    // Add click handler for placing markers (now supports pathfinding)
    map.on('click', handleMapClick);

    // Create shade layer
    map.whenReady(() => {
      setTimeout(() => {
        const shadeDate = new Date();
        shadeDate.setHours(currentHour, 0, 0, 0);
        createShadeLayer(map, shadeDate);
      }, 100);
    });

    return () => {
      console.log("TestMap cleanup - removing map");
      
      map.off('click', handleMapClick);
      
      // Clear layers
      if (edgeLayerRef.current) {
        try {
          map.removeLayer(edgeLayerRef.current);
        } catch (e) {
          console.warn('Error removing edge layer:', e);
        }
      }
      if (pathLayerRef.current) {
        try {
          map.removeLayer(pathLayerRef.current);
        } catch (e) {
          console.warn('Error removing path layer:', e);
        }
      }
      if (waypointLayerRef.current) {
        try {
          map.removeLayer(waypointLayerRef.current);
        } catch (e) {
          console.warn('Error removing waypoint layer:', e);
        }
      }
      
      // Clear markers
      markersRef.current.forEach(marker => {
        try {
          map.removeLayer(marker);
        } catch (e) {
          console.warn('Error removing marker:', e);
        }
      });
      markersRef.current = [];
      
      // Remove shade layer
      if (shadeRef.current) {
        try {
          map.removeLayer(shadeRef.current);
        } catch (e) {
          console.warn('Error removing shade layer:', e);
        }
      }
      
      // Clear all caches
      buildingDataCacheRef.current = [];
      lastBoundsRef.current = '';
      shadeOptionsRef.current = null;
      console.log("🗑️ Cleared all building data and shade option caches");
      
      // Remove map
      map.remove();
    };
  }, [handleMapClick]);

  // Update shade time when hour changes
  useEffect(() => {
    console.log("TestMap useEffect triggered - Hour changed:", currentHour);
    if (shadeRef.current?.setDate) {
      setReady(false);
      const newDate = new Date();
      newDate.setHours(currentHour, 0, 0, 0);
      shadeRef.current.setDate(newDate);
      shadeRef.current.once("idle", () => {
        console.log("Shade layer updated for hour:", currentHour);
        setReady(true);
      });
    }
  }, [currentHour]);

  // Separate useEffect for toggle changes to see if this causes refresh
  useEffect(() => {
    console.log("TestMap useEffect triggered - Toggle changed:", testToggle);
  }, [testToggle]);

  // Reactive recomputation when shade settings change
  useEffect(() => {
    console.log("🔄 Reactive recomputation useEffect triggered");
    if (pathStateRef.current.startPoint && pathStateRef.current.endPoint) {
      // Debounce the recomputation to avoid rapid API calls
      if (penaltyUpdateTimeoutRef.current) {
        clearTimeout(penaltyUpdateTimeoutRef.current);
      }
      
      // Check if shade routing mode, time, or water stops changed
      const shadeRoutingChanged = prevShadeRoutingRef.current !== useShadeRouting;
      const timeChanged = prevCurrentHourRef.current !== currentHour;
      const waterStopsChanged = prevIncludeWaterStopsRef.current !== includeWaterStops;
      
      console.log("🔄 Change detection:", { shadeRoutingChanged, timeChanged, waterStopsChanged });
      
      prevShadeRoutingRef.current = useShadeRouting;
      prevCurrentHourRef.current = currentHour;
      prevIncludeWaterStopsRef.current = includeWaterStops;
      
      // Longer delay when shade routing toggles or time changes to allow shadow recomputation,
      // shorter delay for penalty adjustments or water stop changes
      const delay = (shadeRoutingChanged || timeChanged) ? 800 : 150;
      
      console.log("🔄 Setting recomputation timer with delay:", delay + "ms");
      penaltyUpdateTimeoutRef.current = window.setTimeout(() => {
        console.log("🔄 Executing debounced recomputation");
        computeAndDisplayPath();
      }, delay);
    }
  }, [useShadeRouting, shadePenalty, currentHour, includeWaterStops, computeAndDisplayPath]);

  // Classify edges by sampling the ShadeMap canvas
  async function classify({
    stepMeters = 15,
    samplesPerPoint = 3,
    jitterRadius = 1.5,
    alphaThreshold = 16,
    maxSteps = 20,
    earlyExit = true,
  } = {}): Promise<EdgeResult[]> {
    const map = mapRef.current, shade = shadeRef.current;
    if (!map || !shade || !ready) return [];

    const rect = map.getContainer().getBoundingClientRect();
    const out: EdgeResult[] = [];
    const BATCH = 300;

    for (let i = 0; i < edges.length; i += BATCH) {
      const chunk = edges.slice(i, i + BATCH);
      const part = await Promise.all(
        chunk.map(async (e) => {
          const lenM = L.latLng(e.a).distanceTo(L.latLng(e.b));
          const steps = Math.min(Math.max(1, Math.ceil(lenM / stepMeters)), maxSteps);
          let hits = 0, total = 0;

          for (let j = 0; j <= steps; j++) {
            const t = steps === 0 ? 0.5 : j / steps;
            const base = lerp(e.a, e.b, t);
            for (let s = 0; s < samplesPerPoint; s++) {
              const p = jitterMeters(base, jitterRadius);
              const cp = map.latLngToContainerPoint([p.lat, p.lng]);
              if (cp.x < 0 || cp.y < 0 || cp.x >= rect.width || cp.y >= rect.height) continue;
              const xWin = rect.left + cp.x;
              const yWin = window.innerHeight - (rect.top + cp.y);
              try {
                const rgba: Uint8ClampedArray = shade.readPixel(xWin, yWin);
                if (rgba && isShadowRGBA(rgba, alphaThreshold)) hits++;
                total++;
              } catch { /* ignore */ }
            }
            if (earlyExit && total >= 6) {
              const remaining = (steps - j) * samplesPerPoint;
              if (hits === total && remaining < total / 2) break;
              if (hits === 0 && remaining < total / 2) break;
            }
          }
          const shadePct = total ? hits / total : 0;
          return { id: e.id, shadePct, shaded: shadePct >= 0.5, nSamples: total };
        })
      );
      out.push(...part);
      await new Promise((r) => requestAnimationFrame(r));
    }

    onResults?.(out);
    return out;
  }

  async function classifyAndDraw() {
    if (!ready || edges.length === 0) return;
    const results = await classify();
    const layer = edgeLayerRef.current!;
    layer.clearLayers();
    for (const e of edges) {
      const r = results.find((x) => x.id === e.id);
      const pct = r?.shadePct ?? 0;
      L.polyline([[e.a.lat, e.a.lng], [e.b.lat, e.b.lng]], {
        color: colorForPct(pct), weight: 6, opacity: 0.9,
      })
        .bindTooltip(`shade: ${(pct * 100).toFixed(0)}% (${r?.nSamples || 0} samples)`)
        .addTo(layer);
    }
  }

  return (
    <div style={{ height: "100%", position: "relative" }}>
      <div id="test-map" style={{ height: "100%" }} />

      {/* Settings Panel - Top Right */}
      <div style={{
        position: "absolute", right: 12, top: 12, zIndex: 1000,
        background: "rgba(255,255,255,0.92)", color: "#333", padding: 12, borderRadius: 8,
        font: "14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        minWidth: 200,
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
      }}>
        <div style={{ marginBottom: 8 }}>
          <div style={{ marginBottom: 4, fontSize: 12 }}>Time: {`${currentHour.toString().padStart(2, '0')}:00`}</div>
          <input
            type="range" min={0} max={23} step={1} value={currentHour}
            onChange={(e) => {
              const hours = parseInt((e.target as HTMLInputElement).value, 10);
              setCurrentHour(hours);
            }}
            style={{ width: '100%' }}
          />
        </div>
        
        {/* Pathfinding controls */}
        <div style={{ borderTop: '1px solid #ddd', paddingTop: 8 }}>
          <div style={{ marginBottom: 6 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 12 }}>
              <input
                type="checkbox"
                checked={useShadeRouting}
                onChange={(e) => {
                  setUseShadeRouting(e.target.checked);
                }}
              />
              Shade-aware routing
            </label>
          </div>
          
          {useShadeRouting && (
            <div style={{ fontSize: 12 }}>
              <div style={{ marginBottom: 4 }}>Shade penalty: {shadePenalty.toFixed(1)}x</div>
              <input
                type="range"
                min={0.5}
                max={3.0}
                step={0.1}
                value={shadePenalty}
                onChange={(e) => {
                  const newPenalty = parseFloat(e.target.value);
                  setShadePenalty(newPenalty);
                }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Water stops controls */}
          <div style={{ borderTop: '1px solid #ddd', paddingTop: 8, marginTop: 8 }}>
            <div style={{ marginBottom: 6 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 12 }}>
                <input
                  type="checkbox"
                  checked={includeWaterStops}
                  onChange={(e) => {
                    setIncludeWaterStops(e.target.checked);
                  }}
                />
                Add water stop waypoint
              </label>
            </div>
            
            {includeWaterStops && (
              <div style={{ fontSize: 12 }}>
                <div style={{ marginBottom: 4 }}>Max detour: {maxDetourDistance}m</div>
                <input
                  type="range"
                  min={50}
                  max={500}
                  step={25}
                  value={maxDetourDistance}
                  onChange={(e) => {
                    const newDistance = parseInt(e.target.value, 10);
                    setMaxDetourDistance(newDistance);
                  }}
                  style={{ width: '100%' }}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Compact Info Panel - Top Center (Hover to Expand) */}
      <div style={{
        position: 'absolute',
        top: 20,
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(255,255,255,0.92)',
        padding: '12px',
        borderRadius: '8px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        zIndex: 1000,
        font: "14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        transition: 'all 0.3s ease',
        cursor: pathUIState.path.length > 0 ? 'pointer' : 'default'
      }}
      className="info-panel"
      onMouseEnter={(e) => {
        if (pathUIState.path.length > 0) {
          e.currentTarget.style.maxWidth = '350px';
          e.currentTarget.style.padding = '16px';
        }
      }}
      onMouseLeave={(e) => {
        if (pathUIState.path.length > 0) {
          e.currentTarget.style.maxWidth = '200px';
          e.currentTarget.style.padding = '12px';
        }
      }}>
        {!ready && (
          <div style={{ textAlign: 'center', color: '#007cba' }}>Loading shadows...</div>
        )}
        {ready && !pathUIState.startPoint && (
          <div style={{ textAlign: 'center', color: '#666' }}>Click to set start</div>
        )}
        {ready && pathUIState.startPoint && !pathUIState.endPoint && (
          <div style={{ textAlign: 'center', color: '#666' }}>Click to set destination</div>
        )}
        {pathUIState.loading && (
          <div style={{ textAlign: 'center', color: '#007cba' }}>Computing path...</div>
        )}
        {pathUIState.error && (
          <div style={{ color: 'red', textAlign: 'center' }}>{pathUIState.error}</div>
        )}
        {pathUIState.path.length > 0 && ready && (
          <div>
            <div style={{ fontWeight: 'bold', marginBottom: 8, textAlign: 'center' }}>
              Path Found
            </div>
            
            {/* Compact view */}
            <div className="compact-info">
              <div style={{ fontSize: 12, color: '#666', textAlign: 'center' }}>
                {pathUIState.routeStats ? 
                  `${pathUIState.routeStats.shadeAwareDistance.toFixed(0)}m • ${currentHour.toString().padStart(2, '0')}:00` :
                  `${pathUIState.path.length - 1} segments`
                }
                {includeWaterStops && pathUIState.waypoints && pathUIState.waypoints.length > 0 && (
                  <span> • {pathUIState.waypoints.length} water stops</span>
                )}
              </div>
            </div>

            {/* Expanded view (shown on hover) */}
            <div className="expanded-info" style={{ 
              display: 'none',
              marginTop: 8,
              fontSize: 12,
              lineHeight: 1.4 
            }}>
              {pathUIState.routeStats ? (
                <>
                  <div>Distance: {pathUIState.routeStats.shadeAwareDistance.toFixed(0)}m</div>
                  
                  {pathUIState.routeStats.shadeMode === 'daylight' ? (
                    <>
                      <div>Shaded: {pathUIState.routeStats.totalShadeLength.toFixed(0)}m</div>
                      <div>Unshaded: {(pathUIState.routeStats.shadeAwareDistance - pathUIState.routeStats.totalShadeLength).toFixed(0)}m</div>
                      <div>Shortest Path: {pathUIState.routeStats.originalDistance.toFixed(0)}m</div>
                      <div>Time: {pathUIState.routeStats.analysisTime} ({pathUIState.routeStats.shadeMode})</div>
                      <div>Penalty: +{pathUIState.routeStats.shadePenaltyAdded.toFixed(0)}m ({pathUIState.routeStats.shadePenalty}x)</div>
                    </>
                  ) : pathUIState.routeStats.shadeMode === 'standard' ? (
                    <>
                      <div>Shaded: {pathUIState.routeStats.totalShadeLength ? pathUIState.routeStats.totalShadeLength.toFixed(0) : '0'}m</div>
                      <div>Unshaded: {pathUIState.routeStats.totalShadeLength ? (pathUIState.routeStats.shadeAwareDistance - pathUIState.routeStats.totalShadeLength).toFixed(0) : pathUIState.routeStats.shadeAwareDistance.toFixed(0)}m</div>
                      <div>Time: {currentHour.toString().padStart(2, '0')}:00 (standard)</div>
                    </>
                  ) : (
                    <>
                      <div>Shortest Path: {pathUIState.routeStats.originalDistance.toFixed(0)}m</div>
                      <div>Time: {pathUIState.routeStats.analysisTime} ({pathUIState.routeStats.shadeMode})</div>
                      <div>Night mode - no shade penalties</div>
                    </>
                  )}
                  
                  {includeWaterStops && pathUIState.waypoints && pathUIState.waypoints.length > 0 && (
                    <div style={{ marginTop: 4, paddingTop: 4, borderTop: '1px solid #eee' }}>
                      Water stops: {pathUIState.waypoints.length}
                    </div>
                  )}
                  
                  <div style={{ marginTop: 8, fontSize: 11, color: '#999', textAlign: 'center' }}>
                    Click anywhere to start over
                  </div>
                </>
              ) : (
                <div>Segments: {pathUIState.path.length - 1}</div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Legend - Bottom Right */}
      <div style={{
        position: 'absolute',
        bottom: 20,
        right: 20,
        background: 'rgba(255,255,255,0.92)',
        padding: '12px',
        borderRadius: '8px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        zIndex: 1000,
        font: "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: 8, textAlign: 'center' }}>
          Route Shade Legend
        </div>
        
        {/* Gradient bar */}
        <div style={{
          height: 20,
          width: 200,
          background: 'linear-gradient(to right, #c62828 0%, #ff8f00 25%, #ffc107 50%, #8bc34a 75%, #1a7f37 100%)',
          borderRadius: 4,
          border: '1px solid #ddd',
          marginBottom: 6
        }} />
        
        {/* Labels */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          width: 200,
          fontSize: 10,
          color: '#666'
        }}>
          <span>Unshaded (Hot)</span>
          <span>Shaded (Cool)</span>
        </div>
        
        <div style={{ 
          marginTop: 6, 
          fontSize: 10, 
          color: '#999', 
          textAlign: 'center' 
        }}>
          Paths colored by shade coverage
          {includeWaterStops && (
            <div style={{ marginTop: 4 }}>Water stops shown as droplet icons</div>
          )}
        </div>
      </div>

      {/* Style for hover effects and water waypoints */}
      <style dangerouslySetInnerHTML={{
        __html: `
          .info-panel:hover .compact-info {
            display: none;
          }
          .info-panel:hover .expanded-info {
            display: block !important;
          }
          .water-waypoint-icon {
            background: transparent !important;
            border: none !important;
            font-size: 28px;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
          }
        `
      }} />
    </div>
  );
}