import { useEffect, useRef, useState, useCallback } from "react";
import * as L from "leaflet";
// ShadeMap is attached to L as L.shadeMap()
import "leaflet-shadow-simulator";
// @ts-ignore – shim in src/types if needed
import osmtogeojson from "osmtogeojson";
import AddressSearch from "./AddressSearch";

// ---- Leaflet marker icon fixes (keep) ----
const iconRetinaUrl = new URL('leaflet/dist/images/marker-icon-2x.png', import.meta.url).href;
const iconUrl = new URL('leaflet/dist/images/marker-icon.png', import.meta.url).href;
const shadowUrl = new URL('leaflet/dist/images/marker-shadow.png', import.meta.url).href;

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({ iconRetinaUrl, iconUrl, shadowUrl });

const startIcon = new L.Icon({
  iconUrl, iconRetinaUrl, shadowUrl,
  iconSize: [25, 41], iconAnchor: [12, 41],
  popupAnchor: [1, -34], shadowSize: [41, 41],
  className: "start-marker",
});
const endIcon = new L.Icon({
  iconUrl, iconRetinaUrl, shadowUrl,
  iconSize: [25, 41], iconAnchor: [12, 41],
  popupAnchor: [1, -34], shadowSize: [41, 41],
  className: "end-marker",
});

// ---- Types (keep) ----
type Pt = { lat: number; lng: number };
export type Edge = { id: string; a: Pt; b: Pt };
export type EdgeResult = { id: string; shadePct: number; shaded: boolean; nSamples: number };

interface PathState {
  startPoint: [number, number] | null;
  endPoint: [number, number] | null;
  path: [number, number][];
  loading: boolean;
  error: string | null;
}

// ---- Small helpers (keep) ----
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
function colorForPct(p: number) { return p >= 0.5 ? "#1a7f37" : "#c62828"; }

// ---- Route options with NEW uneven terrain option ----
type RouteOpts = {
  avoid_stairs: boolean;
  prefer_smooth: boolean;
  avoid_rough: boolean;
  wheelchair: boolean;
  avoid_uneven: boolean;  // NEW
};
const defaultRouteOpts: RouteOpts = {
  avoid_stairs: false,
  prefer_smooth: false,
  avoid_rough: false,
  wheelchair: false,
  avoid_uneven: false,  // NEW
};

export default function ShadeClassifier({
  edges,
  date,
  onResults,
}: {
  edges: Edge[];
  date: Date;
  onResults?: (r: EdgeResult[]) => void;
}) {
  const mapRef = useRef<L.Map | null>(null);
  const shadeRef = useRef<any>(null);
  const edgeLayerRef = useRef<L.LayerGroup | null>(null);
  const pathLayerRef = useRef<L.LayerGroup | null>(null);
  const [ready, setReady] = useState(false);
  const lastDateRef = useRef<Date>(date);
  const fetchTokenRef = useRef(0);

  const [currentTime, setCurrentTime] = useState(540); // 09:00
  const pathStateRef = useRef<PathState>({
    startPoint: null, endPoint: null, path: [], loading: false, error: null
  });
  const [pathUIState, setPathUIState] = useState<PathState>(pathStateRef.current);

  // Route options with NEW uneven terrain option
  const [routeOpts, setRouteOpts] = useState<RouteOpts>(defaultRouteOpts);

  // Auto re-route when accessibility options change and we have a path
  useEffect(() => {
    const autoReRoute = async () => {
      // Only re-route if we have both start and end points
      if (!pathStateRef.current.startPoint || !pathStateRef.current.endPoint || pathStateRef.current.loading) {
        return;
      }

      const [startLat, startLng] = pathStateRef.current.startPoint;
      const [endLat, endLng] = pathStateRef.current.endPoint;

      pathStateRef.current = { ...pathStateRef.current, loading: true, error: null };
      setPathUIState({ ...pathStateRef.current });

      try {
        const resp = await fetch("http://localhost:8000/route/shortest_path_accessible", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            start_lat: startLat,
            start_lng: startLng,
            end_lat: endLat,
            end_lng: endLng,
            prefer_paved: routeOpts.avoid_rough || routeOpts.prefer_smooth,
            prefer_smoothness: routeOpts.prefer_smooth,
            avoid_steps: routeOpts.avoid_stairs,
            avoid_uneven: routeOpts.avoid_uneven,
          }),
        });
        const data = await resp.json();

        if (data.error) {
          pathStateRef.current = { ...pathStateRef.current, loading: false, error: data.error };
        } else {
          const pathCoords: [number, number][] = data.path || [];
          pathStateRef.current = { ...pathStateRef.current, path: pathCoords, loading: false, error: null };

          // Redraw the route on the map
          const pathLayer = pathLayerRef.current!;
          const markers: L.Marker[] = [];
          pathLayer.eachLayer((l) => { if (l instanceof L.Marker) markers.push(l); });
          pathLayer.clearLayers();
          markers.forEach(m => pathLayer.addLayer(m));

          if (pathCoords.length > 0) {
            L.polyline(pathCoords, { color: "blue", weight: 4, opacity: 0.7 }).addTo(pathLayer);
          }
        }
      } catch (err) {
        pathStateRef.current = { ...pathStateRef.current, loading: false, error: "Failed to re-compute path" };
      }
      setPathUIState({ ...pathStateRef.current });
    };

    autoReRoute();
  }, [routeOpts]); // Re-run whenever routeOpts changes

  // ---- Shade options (keep) ----
  const buildShadeOptions = (when: Date) => ({
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
    // Buildings via Overpass (keep)
    getFeatures: async () => {
      if (!mapRef.current || mapRef.current.getZoom() < 15) return [];
      const my = ++fetchTokenRef.current;
      await new Promise((r) => setTimeout(r, 200));
      if (my !== fetchTokenRef.current) return [];

      const b = mapRef.current.getBounds();
      const north = b.getNorth(), south = b.getSouth(), east = b.getEast(), west = b.getWest();

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
      try {
        const resp = await fetch(`${overpass}?data=${encodeURIComponent(query)}`);
        if (!resp.ok) return [];
        const data = await resp.json();
        const gj = osmtogeojson(data);

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
        return gj.features;
      } catch {
        return [];
      }
    },
  });

  const createShadeLayer = (map: L.Map, when: Date) => {
    const layer = (L as any).shadeMap(buildShadeOptions(when));
    layer.once("idle", () => setReady(true));
    layer.addTo(map);
    shadeRef.current = layer;

    // Keep clicks going to map (don't steal events)
    if (layer._container || layer.getContainer?.()) {
      const c = layer._container || layer.getContainer();
      if (c) c.style.pointerEvents = "none";
    }
  };

  // ---- Map setup (keep) ----
  useEffect(() => {
    const mapEl = document.getElementById("map");
    if (!mapEl) return;

    const map = L.map(mapEl, { zoomControl: true }).setView([39.9526, -75.1652], 16);
    mapRef.current = map;
    lastDateRef.current = date;

    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OSM",
      maxZoom: 19,
    }).addTo(map);

    // layers
    edgeLayerRef.current = L.layerGroup().addTo(map);
    pathLayerRef.current = L.layerGroup().addTo(map);

    // legend (keep)
    const legend: any = (L as any).control({ position: "bottomleft" });
    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "legend");
      div.innerHTML = `
        <div style="padding:8px;background:#0008;color:#fff;border-radius:8px;font:14px system-ui,-apple-system,Segoe UI,Roboto,sans-serif">
          <div><span style="color:#1a7f37">■■</span> mostly shaded</div>
          <div><span style="color:#c62828">■■</span> mostly sunny</div>
        </div>`;
      return div;
    };
    legend.addTo(map);

    // Shade layer
    map.whenReady(() => setTimeout(() => createShadeLayer(map, lastDateRef.current), 100));

    // Manual click pathfinding (keep)
    map.on("click", handleMapClick);

    // Keep shade canvas aligned on move/zoom (safety)
    const sync = () => shadeRef.current?.redraw?.();
    map.on("move", sync);
    map.on("zoom", sync);
    map.on("resize", sync);

    return () => {
      map.off("click", handleMapClick);
      map.off("move", sync);
      map.off("zoom", sync);
      map.off("resize", sync);
      try { shadeRef.current && map.removeLayer(shadeRef.current); } catch { }
      try { edgeLayerRef.current && map.removeLayer(edgeLayerRef.current); } catch { }
      try { pathLayerRef.current && map.removeLayer(pathLayerRef.current); } catch { }
      map.remove();
    };
  }, []);

  // Keep shade date in sync (keep)
  useEffect(() => {
    lastDateRef.current = date;
    if (shadeRef.current?.setDate) {
      setReady(false);
      shadeRef.current.setDate(date);
      shadeRef.current.once("idle", () => setReady(true));
    }
  }, [date]);

  // ---- Manual click pathfinding (UPDATED to include avoid_uneven) ----
  const handleMapClick = useCallback(async (e: L.LeafletMouseEvent) => {
    if (pathStateRef.current.loading) return;

    const { lat, lng } = e.latlng;

    const pathLayer = pathLayerRef.current!;
    if (!pathStateRef.current.startPoint) {
      pathLayer.clearLayers();
      L.marker([lat, lng], { icon: startIcon }).addTo(pathLayer);
      pathStateRef.current = { startPoint: [lat, lng], endPoint: null, path: [], loading: false, error: null };
      setPathUIState({ ...pathStateRef.current });
      return;
    }

    if (!pathStateRef.current.endPoint) {
      L.marker([lat, lng], { icon: endIcon }).addTo(pathLayer);
      pathStateRef.current = { ...pathStateRef.current, endPoint: [lat, lng], loading: true, error: null };
      setPathUIState({ ...pathStateRef.current });

      try {
        // call accessible endpoint with mapped options INCLUDING avoid_uneven
        const resp = await fetch("http://localhost:8000/route/shortest_path_accessible", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            start_lat: pathStateRef.current.startPoint![0],
            start_lng: pathStateRef.current.startPoint![1],
            end_lat: lat,
            end_lng: lng,
            prefer_paved: routeOpts.avoid_rough || routeOpts.prefer_smooth,
            prefer_smoothness: routeOpts.prefer_smooth,
            avoid_steps: routeOpts.avoid_stairs,
            avoid_uneven: routeOpts.avoid_uneven,  // NEW
          }),
        });
        const data = await resp.json();
        if (data.error) {
          pathStateRef.current = { ...pathStateRef.current, loading: false, error: data.error };
        } else {
          const pathCoords: [number, number][] = data.path || [];
          pathStateRef.current = { ...pathStateRef.current, path: pathCoords, loading: false, error: null };
          if (pathCoords.length > 0) {
            L.polyline(pathCoords, { color: "blue", weight: 4, opacity: 0.7 }).addTo(pathLayer);
          }
        }
      } catch (err) {
        pathStateRef.current = { ...pathStateRef.current, loading: false, error: "Failed to compute path" };
      }
      setPathUIState({ ...pathStateRef.current });
      return;
    }

    // third click resets with new start
    pathLayer.clearLayers();
    L.marker([lat, lng], { icon: startIcon }).addTo(pathLayer);
    pathStateRef.current = { startPoint: [lat, lng], endPoint: null, path: [], loading: false, error: null };
    setPathUIState({ ...pathStateRef.current });
  }, [routeOpts]);

  // ---- Classification (keep) ----
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
              const rgba: Uint8ClampedArray = shade.readPixel(xWin, yWin);
              if (rgba && isShadowRGBA(rgba, alphaThreshold)) hits++;
              total++;
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

  // keep global helper for quick console testing
  useEffect(() => {
    // @ts-ignore
    window.__classifyEdges = classify;
  }, []);

  async function classifyAndDraw() {
    if (!ready) return;
    const results = await classify();
    const layer = edgeLayerRef.current!;
    layer.clearLayers();
    for (const e of edges) {
      const r = results.find((x) => x.id === e.id);
      const pct = r?.shadePct ?? 0;
      L.polyline([[e.a.lat, e.a.lng], [e.b.lat, e.b.lng]], {
        color: colorForPct(pct), weight: 6, opacity: 0.9
      })
        .bindTooltip(`shade: ${(pct * 100).toFixed(0)}% (${r?.nSamples || 0} samples)`)
        .addTo(layer);
    }
  }

  // Convert computed path to edges and analyze shade (keep)
  const pathToEdges = useCallback((): Edge[] => {
    if (pathStateRef.current.path.length < 2) return [];
    return pathStateRef.current.path.slice(0, -1).map((p, i) => ({
      id: `path-${i}`,
      a: { lat: p[0], lng: p[1] },
      b: { lat: pathStateRef.current.path[i + 1][0], lng: pathStateRef.current.path[i + 1][1] },
    }));
  }, []);

  const analyzePathShade = useCallback(async () => {
    if (!ready || !shadeRef.current) return;
    const map = mapRef.current!;
    const shade = shadeRef.current!;
    const rect = map.getContainer().getBoundingClientRect();
    const segs = pathToEdges();
    const results: EdgeResult[] = [];

    for (const e of segs) {
      const lenM = L.latLng(e.a).distanceTo(L.latLng(e.b));
      const steps = Math.min(Math.max(1, Math.ceil(lenM / 10)), 20);
      let hits = 0, total = 0;
      for (let j = 0; j <= steps; j++) {
        const t = steps === 0 ? 0.5 : j / steps;
        const base = lerp(e.a, e.b, t);
        for (let s = 0; s < 3; s++) {
          const p = jitterMeters(base, 1.5);
          const cp = map.latLngToContainerPoint([p.lat, p.lng]);
          if (cp.x < 0 || cp.y < 0 || cp.x >= rect.width || cp.y >= rect.height) continue;
          const xWin = rect.left + cp.x;
          const yWin = window.innerHeight - (rect.top + cp.y);
          const rgba: Uint8ClampedArray = shade.readPixel(xWin, yWin);
          if (rgba && isShadowRGBA(rgba, 16)) hits++;
          total++;
        }
      }
      const shadePct = total ? hits / total : 0;
      results.push({ id: e.id, shadePct, shaded: shadePct >= 0.5, nSamples: total });
    }

    // redraw path with colored segments
    const pathLayer = pathLayerRef.current!;
    const markers: L.Marker[] = [];
    pathLayer.eachLayer((l) => { if (l instanceof L.Marker) markers.push(l); });
    pathLayer.clearLayers();
    markers.forEach(m => pathLayer.addLayer(m));

    for (let i = 0; i < segs.length; i++) {
      const e = segs[i];
      const r = results.find(x => x.id === e.id);
      const pct = r?.shadePct ?? 0;
      L.polyline([[e.a.lat, e.a.lng], [e.b.lat, e.b.lng]], {
        color: colorForPct(pct), weight: 6, opacity: 0.8
      })
        .bindTooltip(`Path segment ${i + 1}: ${(pct * 100).toFixed(0)}% shaded (${r?.nSamples || 0})`)
        .addTo(pathLayer);
    }
  }, [pathToEdges, ready]);

  // ---- AddressSearch integration (UPDATED to include avoid_uneven) ----
  const handleRouteSearch = useCallback(async (coord1: { lat: number; lng: number }, coord2: { lat: number; lng: number }) => {
    const pathLayer = pathLayerRef.current!;
    pathLayer.clearLayers();

    // markers
    L.marker([coord1.lat, coord1.lng], { icon: startIcon }).addTo(pathLayer);
    L.marker([coord2.lat, coord2.lng], { icon: endIcon }).addTo(pathLayer);

    // view
    const bounds = L.latLngBounds([coord1.lat, coord1.lng], [coord2.lat, coord2.lng]);
    mapRef.current?.fitBounds(bounds, { padding: [50, 50] });

    // fetch route (mapped options)
    pathStateRef.current = {
      startPoint: [coord1.lat, coord1.lng],
      endPoint: [coord2.lat, coord2.lng],
      path: [], loading: true, error: null
    };
    setPathUIState({ ...pathStateRef.current });

    try {
      const resp = await fetch("http://localhost:8000/route/shortest_path_accessible", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_lat: coord1.lat, start_lng: coord1.lng,
          end_lat: coord2.lat, end_lng: coord2.lng,
          prefer_paved: routeOpts.avoid_rough || routeOpts.prefer_smooth,
          prefer_smoothness: routeOpts.prefer_smooth,
          avoid_steps: routeOpts.avoid_stairs,
          avoid_uneven: routeOpts.avoid_uneven,  // NEW
        }),
      });
      const data = await resp.json();
      if (data.error) {
        pathStateRef.current = { ...pathStateRef.current, loading: false, error: data.error };
      } else {
        const coords: [number, number][] = data.path || [];
        pathStateRef.current = { ...pathStateRef.current, path: coords, loading: false, error: null };
        if (coords.length > 0) {
          L.polyline(coords, { color: "blue", weight: 4, opacity: 0.7 }).addTo(pathLayer);
        }
      }
    } catch (e) {
      pathStateRef.current = { ...pathStateRef.current, loading: false, error: "Failed to compute path from addresses" };
    }
    setPathUIState({ ...pathStateRef.current });
  }, [routeOpts]);

  // ---- UI ----
  return (
    <div style={{ height: "100%", position: "relative" }}>
      <div id="map" style={{ height: "100%" }} />

      {/* LEFT: Shadow controls (KEPT EXACT STYLING) */}
      <div style={{
        position: "absolute", left: 12, top: 12, zIndex: 1000,
        background: "rgba(0,0,0,0.6)", color: "#fff", padding: 8, borderRadius: 8,
        font: "14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
      }}>
        {ready ? "Shadows ready" : "Rendering shadows…"}
        {/* Styled time slider (kept) */}
        <div style={{ marginTop: 8, width: '400px' }}>
          <div style={{ position: 'relative', height: '40px' }}>
            <div style={{
              display: 'flex', justifyContent: 'space-between', position: 'absolute',
              width: '100%', top: '20px', fontSize: '10px', color: '#ccc'
            }}>
              {Array.from({ length: 13 }, (_, i) => {
                const hour = i * 2;
                return <div key={hour} style={{ textAlign: 'center', width: 20 }}>{hour.toString().padStart(2, '0')}</div>;
              })}
            </div>
            <input
              type="range"
              min={0} max={1440} step={5} value={currentTime}
              style={{
                width: '100%', position: 'absolute', top: 0 as number,
                WebkitAppearance: 'none', height: 4,
                background: 'linear-gradient(to right, #1a1a1a 0%, #1a1a1a 25%, #ffd700 50%, #ff6b35 75%, #1a1a1a 100%)',
                borderRadius: 2, outline: 'none'
              }}
              onChange={async (e) => {
                const mins = parseInt((e.target as HTMLInputElement).value, 10);
                setCurrentTime(mins);
                const d = new Date(); d.setHours(0, 0, 0, 0); d.setMinutes(mins);
                lastDateRef.current = d;
                if (shadeRef.current?.setDate) {
                  setReady(false);
                  shadeRef.current.setDate(d);
                  await new Promise<void>((res) => shadeRef.current.once("idle", () => { setReady(true); res(); }));
                  await classifyAndDraw();
                }
              }}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, justifyContent: 'space-between' }}>
            <div style={{ background: 'rgba(255,255,255,0.1)', padding: '4px 8px', borderRadius: 4, fontSize: 12 }}>
              {(() => {
                const h = Math.floor(currentTime / 60);
                const m = currentTime % 60;
                return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
              })()}
            </div>
            <button
              onClick={classifyAndDraw}
              disabled={!ready}
              style={{
                padding: '4px 12px', fontSize: 12,
                backgroundColor: ready ? '#007cba' : '#666',
                color: 'white', border: 'none', borderRadius: 4, cursor: ready ? 'pointer' : 'not-allowed'
              }}
            >
              {ready ? "Classify edges" : "Wait..."}
            </button>
          </div>
        </div>
      </div>

      {/* RIGHT: Pathfinding panel (UPDATED with uneven terrain option) */}
      <div style={{
        position: 'absolute', top: 10, right: 10, zIndex: 1000,
        background: 'rgba(255,255,255,0.92)', padding: 10, borderRadius: 6,
        boxShadow: '0 2px 4px rgba(0,0,0,0.2)', maxWidth: 360,
        font: "14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
      }}>
        {/* Address search component */}
        <AddressSearch
          onRouteSearch={handleRouteSearch}
          disabled={pathUIState.loading}
        />

        {/* Accessibility / surface toggles (UPDATED with uneven terrain) */}
        <div style={{ borderTop: '1px solid #e5e5e5', paddingTop: 10, marginTop: 10 }}>
          <div style={{ fontWeight: 600, marginBottom: 6, fontSize: 13 }}>Accessibility & Surface</div>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
            <input
              type="checkbox"
              checked={routeOpts.avoid_stairs}
              onChange={(e) => setRouteOpts(o => ({ ...o, avoid_stairs: e.target.checked }))}
            />
            Avoid stairs/steps
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
            <input
              type="checkbox"
              checked={routeOpts.prefer_smooth}
              onChange={(e) => setRouteOpts(o => ({ ...o, prefer_smooth: e.target.checked }))}
            />
            Prefer smooth/level surfaces
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
            <input
              type="checkbox"
              checked={routeOpts.avoid_rough}
              onChange={(e) => setRouteOpts(o => ({ ...o, avoid_rough: e.target.checked }))}
            />
            Avoid rough (cobblestone, unpaved)
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
            <input
              type="checkbox"
              checked={routeOpts.wheelchair}
              onChange={(e) => setRouteOpts(o => ({ ...o, wheelchair: e.target.checked }))}
            />
            Wheelchair-friendly (where data allows)
          </label>
          {/* NEW: Uneven terrain option */}
          <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input
              type="checkbox"
              checked={routeOpts.avoid_uneven}
              onChange={(e) => setRouteOpts(o => ({ ...o, avoid_uneven: e.target.checked }))}
            />
            Avoid uneven terrain (cobblestone, rocks)
          </label>
        </div>

        {/* Manual click instructions (kept) */}
        <div style={{ fontSize: 13, color: '#666', marginTop: 10 }}>
          Or click on the map to pick start and end.
        </div>

        {!pathUIState.startPoint && <div>Click on map to set start point</div>}
        {pathUIState.startPoint && !pathUIState.endPoint && <div>Click on map to set end point</div>}
        {pathUIState.loading && <div>Computing path...</div>}
        {pathUIState.error && <div style={{ color: 'red' }}>Error: {pathUIState.error}</div>}

        {pathUIState.path.length > 0 && (
          <div style={{ marginTop: 8 }}>
            Path found! Click anywhere to start over.
            <br />Segments: {pathUIState.path.length - 1}
            <br />
            <button
              onClick={analyzePathShade}
              disabled={!ready}
              style={{
                marginTop: 6, padding: '4px 8px', fontSize: 12,
                backgroundColor: ready ? '#007cba' : '#ccc',
                color: 'white', border: 'none', borderRadius: 4,
                cursor: ready ? 'pointer' : 'not-allowed'
              }}
            >
              Analyze Path Shade
            </button>
          </div>
        )}
      </div>
    </div>
  );
}