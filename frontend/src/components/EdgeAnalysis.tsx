import { useEffect, useRef, useState, useCallback } from "react";
import * as L from "leaflet";
// ShadeMap should be available as L.shadeMap() after import
import "leaflet-shadow-simulator";
// @ts-ignore – shim in src/types
import osmtogeojson from "osmtogeojson";

// Fix default icon path issues in Vite
const iconRetinaUrl = new URL('leaflet/dist/images/marker-icon-2x.png', import.meta.url).href;
const iconUrl = new URL('leaflet/dist/images/marker-icon.png', import.meta.url).href;
const shadowUrl = new URL('leaflet/dist/images/marker-shadow.png', import.meta.url).href;

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl,
  iconUrl,
  shadowUrl
});

type Pt = { lat: number; lng: number };
export type Edge = { id: string; a: Pt; b: Pt; weight?: number };
export type EdgeResult = { id: string; shadePct: number; shaded: boolean; nSamples: number };

interface AnalysisProgress {
  processed: number;
  total: number;
  currentEdge: string | null;
  startTime: number;
  errors: number;
}

interface ClassificationResult {
  timestamp: string;
  analysisTime: Date;
  totalEdges: number;
  processedEdges: number;
  errors: number;
  processingTimeMs: number;
  edges: EdgeResult[];
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
function colorForPct(p: number) { return p >= 0.5 ? "#1a7f37" : "#c62828"; }

export default function EdgeAnalysis({ onBack }: { onBack?: () => void }) {
  const mapRef = useRef<L.Map | null>(null);
  const shadeRef = useRef<any>(null);
  const edgeLayerRef = useRef<L.LayerGroup | null>(null);
  const [ready, setReady] = useState(false);
  const [currentHour, setCurrentHour] = useState(9);
  const fetchTokenRef = useRef(0);
  
  // Analysis state
  const [edges, setEdges] = useState<Edge[]>([]);
  const [loadingEdges, setLoadingEdges] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState<AnalysisProgress>({
    processed: 0,
    total: 0,
    currentEdge: null,
    startTime: 0,
    errors: 0
  });
  const [edgeLimit, setEdgeLimit] = useState<number>(100); // Default limit for testing

  // Build ShadeMap options using correct API
  const buildShadeOptions = (when: Date) => {
    return {
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
        const url = `${overpass}?data=${encodeURIComponent(query)}`;

        try {
          const resp = await fetch(url);
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
    };
  };

  // Helper to create the ShadeMap layer
  const createShadeLayer = (map: L.Map, when: Date) => {
    const layer = (L as any).shadeMap(buildShadeOptions(when));

    layer.once("idle", () => {
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

  // Load edges from backend
  const loadEdges = useCallback(async () => {
    setLoadingEdges(true);
    try {
      const url = edgeLimit > 0 
        ? `http://localhost:8000/graph/edges?limit=${edgeLimit}`
        : 'http://localhost:8000/graph/edges';
      
      console.log(`Loading edges from: ${url}`);
      const response = await fetch(url);
      const data = await response.json();
      
      if (data.error) {
        alert(`Error loading edges: ${data.error}`);
        return;
      }
      
      setEdges(data.edges);
      console.log(`Loaded ${data.count} edges (${data.total_available} total available)`);
      
      if (data.limited) {
        console.log(`Note: Limited to ${data.count} of ${data.total_available} total edges`);
      }
    } catch (error) {
      console.error('Failed to load edges:', error);
      alert('Failed to load edges from backend');
    } finally {
      setLoadingEdges(false);
    }
  }, [edgeLimit]);

  // Classify edges by sampling the ShadeMap canvas
  async function classifyEdges({
    stepMeters = 15,
    samplesPerPoint = 3,
    jitterRadius = 1.5,
    alphaThreshold = 16,
    maxSteps = 20,
    earlyExit = true,
  } = {}): Promise<EdgeResult[]> {
    const map = mapRef.current;
    const shade = shadeRef.current;

    if (!map || !shade || !ready) {
      console.warn('Map or shade layer not ready for classification');
      return [];
    }

    if (edges.length === 0) {
      console.warn('No edges loaded for classification');
      return [];
    }

    console.log(`Starting classification of ${edges.length} edges at ${new Date().toLocaleTimeString()}`);
    
    const rect = map.getContainer().getBoundingClientRect();
    const out: EdgeResult[] = [];
    const startTime = Date.now();
    
    setProgress({
      processed: 0,
      total: edges.length,
      currentEdge: null,
      startTime,
      errors: 0
    });

    let errors = 0;
    const BATCH = 50; // Process in smaller batches for progress updates
    
    for (let i = 0; i < edges.length; i += BATCH) {
      const chunk = edges.slice(i, i + BATCH);
      
      const part = await Promise.all(
        chunk.map(async (e, chunkIndex) => {
          const globalIndex = i + chunkIndex;
          
          try {
            const lenM = L.latLng(e.a).distanceTo(L.latLng(e.b));
            const steps = Math.min(Math.max(1, Math.ceil(lenM / stepMeters)), maxSteps);
            let hits = 0, total = 0;

            for (let j = 0; j <= steps; j++) {
              const t = steps === 0 ? 0.5 : j / steps;
              const base = lerp(e.a, e.b, t);

              for (let s = 0; s < samplesPerPoint; s++) {
                const p = jitterMeters(base, jitterRadius);
                const cp = map.latLngToContainerPoint([p.lat, p.lng]);

                if (cp.x < 0 || cp.y < 0 || cp.x >= rect.width || cp.y >= rect.height) {
                  continue;
                }

                const xWin = rect.left + cp.x;
                const yWin = window.innerHeight - (rect.top + cp.y);

                try {
                  const rgba: Uint8ClampedArray = shade.readPixel(xWin, yWin);
                  if (rgba && isShadowRGBA(rgba, alphaThreshold)) hits++;
                  total++;
                } catch (e) {
                  // Pixel read error, continue
                }
              }

              if (earlyExit && total >= 6) {
                const remaining = (steps - j) * samplesPerPoint;
                if (hits === total && remaining < total / 2) break;
                if (hits === 0 && remaining < total / 2) break;
              }
            }

            const shadePct = total ? hits / total : 0;
            return { id: e.id, shadePct, shaded: shadePct >= 0.5, nSamples: total };
          } catch (error) {
            console.warn(`Error processing edge ${e.id}:`, error);
            errors++;
            return { id: e.id, shadePct: 0, shaded: false, nSamples: 0 };
          }
        })
      );
      
      out.push(...part);
      
      // Update progress after batch completion
      const processed = Math.min(i + BATCH, edges.length);
      const elapsed = Date.now() - startTime;
      const rate = processed / (elapsed / 1000);
      const remaining = edges.length - processed;
      const eta = remaining / rate;
      
      // Update progress state with current batch completion
      setProgress(prev => {
        const newProgress = {
          ...prev,
          processed: processed,
          currentEdge: processed < edges.length ? `Batch ${Math.floor(i/BATCH) + 1}/${Math.ceil(edges.length/BATCH)}` : null,
          errors: errors
        };
        console.log(`Progress update:`, newProgress);
        return newProgress;
      });
      
      console.log(`Progress: ${processed}/${edges.length} (${(processed/edges.length*100).toFixed(1)}%) - Rate: ${rate.toFixed(1)} edges/sec - ETA: ${eta.toFixed(0)}s`);
      
      // Allow UI to update
      await new Promise((r) => requestAnimationFrame(r));
    }

    const totalTime = Date.now() - startTime;
    console.log(`Classification complete: ${out.length} edges processed in ${(totalTime/1000).toFixed(1)}s (${errors} errors)`);
    
    // Final progress update to show completion
    setProgress(prev => ({
      ...prev,
      processed: edges.length,
      currentEdge: null,
      errors: errors
    }));

    return out;
  }

  // Run full analysis and download results
  const runAnalysis = useCallback(async () => {
    if (!ready || analyzing) return;
    
    setAnalyzing(true);
    try {
      console.log('Starting edge analysis...');
      const results = await classifyEdges();
      
      const analysisResult: ClassificationResult = {
        timestamp: new Date().toISOString(),
        analysisTime: new Date(),
        totalEdges: edges.length,
        processedEdges: results.length,
        errors: progress.errors,
        processingTimeMs: Date.now() - progress.startTime,
        edges: results
      };
      
      // Create filename with timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const filename = `edge_classification_${currentHour}h_${timestamp}.json`;
      
      // Download as JSON file
      const blob = new Blob([JSON.stringify(analysisResult, null, 2)], { 
        type: 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      // Show success message with download info
      alert(`Analysis complete!\n\nResults downloaded as:\n${filename}\n\nCheck your browser's Downloads folder.`);
      console.log(`Analysis complete and downloaded as: ${filename}`);
      
      // Draw results on map
      const layer = edgeLayerRef.current!;
      layer.clearLayers();

      for (const e of edges) {
        const r = results.find((x) => x.id === e.id);
        const pct = r?.shadePct ?? 0;
        L.polyline(
          [[e.a.lat, e.a.lng], [e.b.lat, e.b.lng]],
          { color: colorForPct(pct), weight: 3, opacity: 0.8 }
        )
          .bindTooltip(`${e.id}: ${(pct * 100).toFixed(0)}% shaded (${r?.nSamples || 0} samples)`)
          .addTo(layer);
      }
      
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed: ' + error);
    } finally {
      setAnalyzing(false);
    }
  }, [ready, analyzing, edges, currentHour, progress]);

  useEffect(() => {
    // Map setup
    const mapContainer = document.getElementById("analysis-map");
    if (!mapContainer) return;

    const map = L.map(mapContainer, {
      zoomControl: true,
    }).setView([39.9526, -75.1652], 16);

    mapRef.current = map;

    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OSM",
      maxZoom: 19,
    }).addTo(map);

    // Create edge layer for visualization
    edgeLayerRef.current = L.layerGroup().addTo(map);

    // Create shade layer
    map.whenReady(() => {
      setTimeout(() => {
        const analysisTime = new Date();
        analysisTime.setHours(currentHour, 0, 0, 0);
        createShadeLayer(map, analysisTime);
      }, 100);
    });

    return () => {
      if (edgeLayerRef.current) {
        try {
          map.removeLayer(edgeLayerRef.current);
        } catch (e) {
          console.warn('Error removing edge layer:', e);
        }
      }
      if (shadeRef.current) {
        try {
          map.removeLayer(shadeRef.current);
        } catch (e) {
          console.warn('Error removing shade layer:', e);
        }
      }
      map.remove();
    };
  }, [currentHour]);

  // Update shade time when hour changes
  useEffect(() => {
    if (shadeRef.current?.setDate) {
      setReady(false);
      const analysisTime = new Date();
      analysisTime.setHours(currentHour, 0, 0, 0);
      shadeRef.current.setDate(analysisTime);
      shadeRef.current.once("idle", () => setReady(true));
    }
  }, [currentHour]);

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  };

  const estimatedTimeRemaining = () => {
    if (progress.processed === 0) return 'Unknown';
    const elapsed = Date.now() - progress.startTime;
    const rate = progress.processed / elapsed;
    const remaining = progress.total - progress.processed;
    return formatTime(remaining / rate);
  };

  return (
    <div style={{ height: "100%", position: "relative" }}>
      <div id="analysis-map" style={{ height: "100%" }} />

      {/* Main Control Panel */}
      <div style={{
        position: "absolute", right: 12, top: 12, zIndex: 1000,
        background: "rgba(255,255,255,0.95)", padding: 16, borderRadius: 8,
        boxShadow: "0 4px 8px rgba(0,0,0,0.2)", minWidth: 300,
        font: "14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
      }}>
        <h3 style={{ margin: '0 0 12px 0', color: '#333' }}>Edge Shadow Analysis</h3>
        <div style={{ 
          marginBottom: 12, 
          padding: 8, 
          backgroundColor: '#e8f4f8', 
          borderRadius: 4, 
          fontSize: 12,
          border: '1px solid #bee5eb'
        }}>
          <strong>💡 Access this page directly at:</strong><br/>
          <code style={{ backgroundColor: 'white', padding: '2px 4px', borderRadius: 2 }}>
            /analysis
          </code>
        </div>
        
        {/* Time Control */}
        <div style={{ marginBottom: 12 }}>
          <label style={{ display: 'block', marginBottom: 4, fontWeight: 'bold' }}>
            Analysis Time: {currentHour.toString().padStart(2, '0')}:00
          </label>
          <input
            type="range" min={0} max={23} step={1} value={currentHour}
            onChange={(e) => setCurrentHour(parseInt(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>

        {/* Edge Limit Control */}
        <div style={{ marginBottom: 12 }}>
          <label style={{ display: 'block', marginBottom: 4, fontWeight: 'bold' }}>
            Edge Limit (0 = all):
          </label>
          <input
            type="number"
            value={edgeLimit}
            onChange={(e) => setEdgeLimit(parseInt(e.target.value) || 0)}
            min={0}
            max={10000}
            style={{ width: '100%', padding: '4px' }}
          />
        </div>

        {/* Load Edges */}
        <button
          onClick={loadEdges}
          disabled={loadingEdges}
          style={{
            width: '100%',
            padding: '8px',
            marginBottom: 8,
            backgroundColor: loadingEdges ? '#ccc' : '#007cba',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loadingEdges ? 'not-allowed' : 'pointer'
          }}
        >
          {loadingEdges ? 'Loading...' : `Load Edges ${edgeLimit > 0 ? `(${edgeLimit})` : '(All)'}`}
        </button>

        {/* Edge Status */}
        {edges.length > 0 && (
          <div style={{ marginBottom: 12, padding: 8, backgroundColor: '#f0f0f0', borderRadius: 4 }}>
            <strong>{edges.length} edges loaded</strong>
          </div>
        )}

        {/* Analysis Button */}
        <button
          onClick={runAnalysis}
          disabled={!ready || analyzing || edges.length === 0}
          style={{
            width: '100%',
            padding: '12px',
            marginBottom: 8,
            backgroundColor: (!ready || analyzing || edges.length === 0) ? '#ccc' : '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: (!ready || analyzing || edges.length === 0) ? 'not-allowed' : 'pointer',
            fontWeight: 'bold'
          }}
        >
          {analyzing ? 'Analyzing...' : ready ? 'Start Analysis & Download' : 'Waiting for shadows...'}
        </button>

        {/* Progress Display */}
        {analyzing && (
          <div style={{ marginTop: 12, padding: 8, backgroundColor: '#e3f2fd', borderRadius: 4 }}>
            <div style={{ marginBottom: 4 }}>
              <strong>Progress: {progress.processed}/{progress.total} ({progress.total > 0 ? ((progress.processed / progress.total) * 100).toFixed(1) : '0.0'}%)</strong>
            </div>
            <div style={{ fontSize: 12 }}>
              {progress.currentEdge && <div>Status: {progress.currentEdge}</div>}
              {progress.processed > 0 && <div>ETA: {estimatedTimeRemaining()}</div>}
              {progress.errors > 0 && <div style={{ color: 'red' }}>Errors: {progress.errors}</div>}
            </div>
          </div>
        )}

        {/* Shadow Status */}
        <div style={{ 
          marginTop: 12, 
          padding: 8, 
          backgroundColor: ready ? '#d4edda' : '#fff3cd', 
          borderRadius: 4,
          fontSize: 12
        }}>
          {ready ? "✅ Shadows ready" : "⏳ Rendering shadows..."}
        </div>
      </div>

      {/* Back Button */}
      <div style={{
        position: 'absolute',
        top: 12,
        left: 12,
        zIndex: 1000
      }}>
        <button
          onClick={() => onBack?.() || window.history.back()}
          style={{
            padding: '8px 12px',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          ← Back to Map
        </button>
      </div>
    </div>
  );
}