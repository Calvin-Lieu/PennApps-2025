from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import os
from pyproj import Geod
import pickle
from contextlib import asynccontextmanager
# Add these imports to the top of app.py
import json
from geopy.distance import geodesic
from typing import List, Tuple, Optional, Dict, Any
import os
from pathlib import Path
from math import radians, cos
from importlib import import_module

# Lazy loader for optional helper
def _get_find_path_with_water_stop():
    try:
        return globals().get('find_path_with_water_stop_cached')
    except Exception:
        pass
    func = None
    try:
        mod = import_module('backend.find_shop_waypoint')
        func = getattr(mod, 'find_path_with_water_stop', None)
    except Exception:
        try:
            mod = import_module('find_shop_waypoint')
            func = getattr(mod, 'find_path_with_water_stop', None)
        except Exception:
            func = None
    globals()['find_path_with_water_stop_cached'] = func
    return func

# Initialize FastAPI early so route decorators below can bind to it
app = FastAPI(title="PennApps Demo Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global graph and geodesic helper
G: Optional[nx.Graph] = None
geod = Geod(ellps="WGS84")

# Helper: compute a rough centroid of the loaded graph (lat, lon)
def _graph_centroid() -> Optional[Tuple[float, float]]:
    global G
    if G is None:
        return None
    try:
        # Sample up to first 2000 nodes for speed
        total = 0
        sum_lat = 0.0
        sum_lon = 0.0
        for i, node in enumerate(G.nodes()):
            lon, lat = node  # nodes are (lon, lat)
            sum_lat += float(lat)
            sum_lon += float(lon)
            total += 1
            if i >= 1999:
                break
        if total > 0:
            return (sum_lat / total, sum_lon / total)
    except Exception:
        pass
    return None

# Normalize a path to [[lat, lng]] even if input was [[lng, lat]].
def _normalize_path_latlng(path: List[List[float]]) -> List[List[float]]:
    if not path:
        return path
    # If any point clearly violates lat/lon ranges, try swapping
    def swap(p):
        return [p[1], p[0]]

    # Build two interpretations
    as_is = path
    swapped = [swap(p) for p in path]

    # Prefer the interpretation whose points are closer to the graph's centroid
    centroid = _graph_centroid()
    if centroid is None:
        # Fallback heuristic: if majority have abs(lat) > abs(lng), it's likely swapped
        votes_swapped = sum(1 for p in as_is if abs(p[0]) > abs(p[1]))
        return swapped if votes_swapped > len(as_is) / 2 else as_is

    def mean_distance_m(points: List[List[float]]) -> float:
        try:
            dists = [geodesic((p[0], p[1]), centroid).meters for p in points]
            return sum(dists) / max(1, len(dists))
        except Exception:
            return float('inf')

    d_as_is = mean_distance_m(as_is)
    d_swapped = mean_distance_m(swapped)
    if d_swapped + 1e-6 < d_as_is:
        # Debug logging for analysis
        try:
            print(f"Normalized path by swapping coordinate order; mean dist to centroid improved {d_as_is:.1f}m -> {d_swapped:.1f}m")
        except Exception:
            pass
        return swapped
    return as_is

@app.on_event("startup")
async def _startup_load_data():
    """Load graph and waypoint data at startup."""
    global G
    enhanced_graph_path = os.path.join(os.path.dirname(__file__), "data", "graph_segments_with_shade.gpickle")
    original_graph_path = os.path.join(os.path.dirname(__file__), "data", "graph_segments.gpickle")
    try:
        if os.path.exists(enhanced_graph_path):
            with open(enhanced_graph_path, "rb") as f:
                G = pickle.load(f)
            print(f"Loaded enhanced graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            sample_edge = next(iter(G.edges(data=True)), None)
            if sample_edge and 'shade_fraction_9' in sample_edge[2]:
                print("✅ Shade data available for enhanced pathfinding")
            else:
                print("⚠️ No shade data found in enhanced graph")
        else:
            with open(original_graph_path, "rb") as f:
                G = pickle.load(f)
            print(f"Loaded original graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            print("⚠️ Enhanced graph not found - shade-aware pathfinding not available")
    except Exception as e:
        print(f"Failed to load graph: {e}")
        G = None
    # Always attempt to load waypoint data
    load_waypoints_data()

# Add this after the existing models in app.py

class WaypointsNearPathRequest(BaseModel):
    path: List[List[float]]  # List of [lat, lng] coordinates
    max_detour_meters: Optional[int] = 100
    waypoint_types: Optional[List[str]] = ['water', 'store']  # Types to include


# Global waypoints cache
waypoints_cache: Optional[List[Dict[str, Any]]] = None

def load_waypoints_data():
    """Load waypoints from JSON, supporting multiple filenames/locations and structures."""
    global waypoints_cache

    if waypoints_cache is not None:
        return waypoints_cache

    base_dir = os.path.dirname(__file__)
    # Support both singular/plural filenames and common locations
    candidate_paths = [
        # relative to CWD
        "waypoint_data.json",
        "waypoints_data.json",
        "backend/waypoint_data.json",
        "backend/waypoints_data.json",
        "data/waypoint_data.json",
        "data/waypoints_data.json",
        # relative to backend module dir
        os.path.join(base_dir, "waypoint_data.json"),
        os.path.join(base_dir, "waypoints_data.json"),
        os.path.join(base_dir, "data", "waypoint_data.json"),
        os.path.join(base_dir, "data", "waypoints_data.json"),
    ]

    tried = []
    for path in candidate_paths:
        tried.append(path)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Accept both list-of-waypoints or dict with lists
            waypoints: list = []
            if isinstance(data, list):
                waypoints = data
                print(f"ℹ️ Waypoints JSON is a list with {len(waypoints)} items")
            elif isinstance(data, dict):
                if "waypoints" in data and isinstance(data["waypoints"], list):
                    waypoints = data["waypoints"]
                    print(f"ℹ️ Using 'waypoints' key with {len(waypoints)} items")
                else:
                    # Merge all list values across keys
                    merged: list = []
                    for k, v in data.items():
                        if isinstance(v, list):
                            merged.extend(v)
                    waypoints = merged
                    print(f"ℹ️ Merged {len(waypoints)} items from dict keys: {', '.join([k for k,v in data.items() if isinstance(v, list)])}")

            waypoints_cache = waypoints if waypoints is not None else []
            print(f"📍 Loaded {len(waypoints_cache)} waypoints from {path}")
            return waypoints_cache
        except Exception as e:
            print(f"Error loading waypoints from {path}: {e}")
            continue

    print("⚠️ No waypoints data found. Checked:")
    for p in tried:
        print(f"  - {p}")
    print("Run waypoint_pathfinding.py to generate data or place waypoint_data.json in backend/.")
    waypoints_cache = []
    return waypoints_cache


@app.get("/waypoints/reload")
async def reload_waypoints() -> Dict[str, Any]:
    """Clear the in-memory cache and reload waypoints from disk."""
    global waypoints_cache
    waypoints_cache = None
    data = load_waypoints_data()
    # Return a quick summary
    types = {}
    for wp in data:
        t = str(wp.get('type', 'unknown')).lower()
        types[t] = types.get(t, 0) + 1
    return {
        "reloaded": True,
        "total": len(data),
        "type_counts": types,
    }


def point_to_line_distance(point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
    """
    Calculate the shortest distance from a point to a line segment using geodesic distance.
    
    Args:
        point: (lat, lng) of the point
        line_start: (lat, lng) of line segment start
        line_end: (lat, lng) of line segment end
        
    Returns:
        Distance in meters
    """
    # Convert to meters using geodesic distance
    def distance_m(p1, p2):
        return geodesic(p1, p2).meters
    
    # If line segment is actually a point
    if line_start == line_end:
        return distance_m(point, line_start)
    
    # Calculate distances
    d_start_end = distance_m(line_start, line_end)
    d_start_point = distance_m(line_start, point)
    d_end_point = distance_m(line_end, point)
    
    # Use dot product to find projection
    # Convert lat/lng to approximate meters for calculation
    lat_to_m = 111320  # meters per degree latitude (approximate)
    lng_to_m = 111320 * abs(cos(radians((line_start[0] + line_end[0]) / 2)))
    
    # Vector from start to end
    dx = (line_end[1] - line_start[1]) * lng_to_m
    dy = (line_end[0] - line_start[0]) * lat_to_m
    
    # Vector from start to point
    px = (point[1] - line_start[1]) * lng_to_m
    py = (point[0] - line_start[0]) * lat_to_m
    
    # Calculate parameter t for projection
    if d_start_end == 0:
        return d_start_point
        
    t = max(0, min(1, (px * dx + py * dy) / (dx * dx + dy * dy)))
    
    # Calculate projection point
    proj_lng = line_start[1] + t * (line_end[1] - line_start[1])
    proj_lat = line_start[0] + t * (line_end[0] - line_start[0])
    
    # Return distance from point to projection
    return distance_m(point, (proj_lat, proj_lng))


def find_waypoints_near_path(path: List[List[float]], max_detour: int, waypoint_types: List[str]) -> List[Dict[str, Any]]:
    """
    Find waypoints within max_detour meters of the path.
    
    Args:
        path: List of [lat, lng] coordinates defining the path
        max_detour: Maximum distance in meters from the path
        waypoint_types: List of waypoint types to include ('water', 'store')
        
    Returns:
        List of nearby waypoints with distance information
    """
    # Ensure path is in [lat, lng]
    path = _normalize_path_latlng(path)

    waypoints = load_waypoints_data()
    if not waypoints:
        return []
    
    nearby_waypoints = []
    
    def waypoint_matches_types(waypoint: Dict[str, Any], requested: List[str]) -> bool:
        if not requested:
            return True
        req = {str(x).lower() for x in requested}

        # Expand umbrella categories into concrete OSM-like types
        store_like = {
            'convenience','supermarket','grocery','general','kiosk','marketplace','shop',
            'beverages','variety_store','chemist','pharmacy'
        }
        food_like = {
            'cafe','fast_food','restaurant','pub','bar','vending_machine','ice_cream'
        }
        water_like = {'drinking_water','water','water_point'}

        allowed: set = set()
        for r in req:
            if r in ('store','shop','market'):
                allowed |= store_like
            elif r in ('food','drink','eat'):
                allowed |= food_like
            elif r in ('water','drinking_water'):
                allowed |= water_like
            else:
                allowed.add(r)

        # Candidate values to check (common fields)
        typ = str(waypoint.get('type','')).lower()
        amenity = str(waypoint.get('amenity','')).lower()
        shop = str(waypoint.get('shop','')).lower()
        tags = waypoint.get('tags') or {}
        tag_amenity = str(tags.get('amenity','')).lower()
        tag_shop = str(tags.get('shop','')).lower()
        name = str(waypoint.get('name','')).lower()

        candidates = {typ, amenity, shop, tag_amenity, tag_shop}
        if any(c in allowed for c in candidates if c):
            return True

        # Heuristic: if requesting store-like and name suggests a store/market
        store_keywords = ('store','shop','market','mart','7-eleven','7 eleven','bodega')
        if (('store' in req or 'shop' in req or 'market' in req) and name):
            if any(k in name for k in store_keywords):
                return True

        # Heuristic: if requesting water and name suggests water fountain/refill
        if (('water' in req or 'drinking_water' in req) and name):
            if any(k in name for k in ('water','fountain','refill')):
                return True

        return False

    for waypoint in waypoints:
        # Flexible filter by requested types/categories
        if not waypoint_matches_types(waypoint, waypoint_types):
            continue
            
        waypoint_coord = tuple(waypoint['coordinates'])  # [lat, lng]
        min_distance = float('inf')
        closest_segment_idx = -1
        
        # Check distance to each path segment
        for i in range(len(path) - 1):
            segment_start = tuple(path[i])      # [lat, lng]
            segment_end = tuple(path[i + 1])    # [lat, lng]
            
            distance = point_to_line_distance(waypoint_coord, segment_start, segment_end)
            
            if distance < min_distance:
                min_distance = distance
                closest_segment_idx = i
        
        # Include waypoint if within detour distance
        if min_distance <= max_detour:
            waypoint_with_distance = waypoint.copy()
            waypoint_with_distance['distance_to_path_m'] = round(min_distance, 1)
            waypoint_with_distance['closest_segment'] = closest_segment_idx
            nearby_waypoints.append(waypoint_with_distance)
    
    # Sort by distance to path
    nearby_waypoints.sort(key=lambda w: w['distance_to_path_m'])
    
    return nearby_waypoints


# Add this endpoint to app.py

@app.post("/waypoints/near_path")
async def waypoints_near_path(req: WaypointsNearPathRequest) -> Dict[str, Any]:
    """Find waypoints near the given path within the specified detour distance."""
    
    if not req.path or len(req.path) < 2:
        return {"error": "Path must contain at least 2 points"}
    
    try:
        normalized_path = _normalize_path_latlng(req.path)
        nearby_waypoints = find_waypoints_near_path(
            normalized_path, 
            req.max_detour_meters,
            req.waypoint_types
        )
        
        # Group by type for summary
        type_counts = {}
        for wp in nearby_waypoints:
            wp_type = wp.get('type', 'unknown')
            type_counts[wp_type] = type_counts.get(wp_type, 0) + 1
        
        return {
            "waypoints": nearby_waypoints,
            "total_found": len(nearby_waypoints),
            "type_counts": type_counts,
            "max_detour_meters": req.max_detour_meters,
            "types_requested": req.waypoint_types,
            "path_segments": len(normalized_path) - 1
        }
        
    except Exception as e:
        return {"error": f"Failed to find waypoints: {str(e)}"}


# Convenience GET variant so you can test in a browser without POST
@app.get("/waypoints/near_path")
async def waypoints_near_path_get(
    path: str,  # JSON-encoded [[lat,lng], ...] or [[lng,lat], ...]
    max_detour_meters: int = 100,
    waypoint_types: Optional[str] = "water,store",
) -> Dict[str, Any]:
    """GET helper for nearby waypoints. Provide `path` as a JSON-encoded array of coordinate pairs.

    Examples (URL-encoded):
      /waypoints/near_path?path=%5B%5B39.95%2C-75.16%5D%2C%5B39.96%2C-75.17%5D%5D&max_detour_meters=150&waypoint_types=water,store
    """
    try:
        coords = json.loads(path)
        if not isinstance(coords, list) or not coords or not isinstance(coords[0], list):
            return {"error": "path must be a JSON array of [lat,lng] (or [lng,lat]) pairs"}
    except Exception as e:
        return {"error": f"Invalid path JSON: {e}"}

    types_list: List[str] = []
    if waypoint_types:
        types_list = [t.strip() for t in waypoint_types.split(',') if t.strip()]

    try:
        normalized_path = _normalize_path_latlng(coords)
        nearby_waypoints = find_waypoints_near_path(
            normalized_path,
            max_detour_meters,
            types_list or ["water", "store"],
        )

        type_counts: Dict[str, int] = {}
        for wp in nearby_waypoints:
            wp_type = wp.get('type', 'unknown')
            type_counts[wp_type] = type_counts.get(wp_type, 0) + 1

        return {
            "waypoints": nearby_waypoints,
            "total_found": len(nearby_waypoints),
            "type_counts": type_counts,
            "max_detour_meters": max_detour_meters,
            "types_requested": types_list or ["water", "store"],
            "path_segments": len(normalized_path) - 1,
        }
    except Exception as e:
        return {"error": f"Failed to find waypoints: {e}"}


@app.get("/waypoints/all")
async def get_all_waypoints() -> Dict[str, Any]:
    """Get all available waypoints."""
    waypoints = load_waypoints_data()
    
    # Group by type for summary
    type_counts = {}
    for wp in waypoints:
        wp_type = wp.get('type', 'unknown')
        type_counts[wp_type] = type_counts.get(wp_type, 0) + 1
    
    return {
        "waypoints": waypoints,
        "total_count": len(waypoints),
        "type_counts": type_counts
    }


@app.get("/waypoints/types")
async def get_waypoint_types() -> Dict[str, Any]:
    """Get available waypoint types and their counts."""
    waypoints = load_waypoints_data()
    
    def derive_types(wp: Dict[str, Any]) -> List[str]:
        # Gather raw indicators
        typ = str(wp.get('type','')).lower()
        amenity = str(wp.get('amenity','')).lower()
        shop = str(wp.get('shop','')).lower()
        tags = wp.get('tags') or {}
        tag_amenity = str(tags.get('amenity','')).lower()
        tag_shop = str(tags.get('shop','')).lower()
        name = str(wp.get('name','')).lower()

        raw = {x for x in (typ, amenity, shop, tag_amenity, tag_shop) if x}

        # Category mappings
        store_like = {
            'convenience','supermarket','grocery','general','kiosk','marketplace','shop',
            'beverages','variety_store','chemist','pharmacy'
        }
        food_like = {
            'cafe','fast_food','restaurant','pub','bar','vending_machine','ice_cream'
        }
        water_like = {'drinking_water','water','water_point'}

        derived = set()
        if raw & store_like:
            derived.add('store')
        if raw & food_like:
            derived.add('food')
        if raw & water_like:
            derived.add('water')

        # Heuristics from name
        if name:
            if any(k in name for k in ('store','shop','market','mart','7-eleven','7 eleven','bodega')):
                derived.add('store')
            if any(k in name for k in ('cafe','coffee','burger','pizza','deli','grill','restaurant')):
                derived.add('food')
            if any(k in name for k in ('water','fountain','refill')):
                derived.add('water')

        # Expose raw values too
        return list(derived | raw)

    counts: Dict[str, int] = {}
    for wp in waypoints:
        for t in derive_types(wp):
            counts[t] = counts.get(t, 0) + 1
    
    return {
        "available_types": sorted(counts.keys()),
        "type_counts": counts,
        "total_waypoints": len(waypoints)
    }


# Add required imports at the top

# Update the lifespan function to also load waypoints
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the segmented graph and waypoints on startup."""
    global G
    # Load graph (existing code)
    enhanced_graph_path = os.path.join(os.path.dirname(__file__), "data", "graph_segments_with_shade.gpickle")
    original_graph_path = os.path.join(os.path.dirname(__file__), "data", "graph_segments.gpickle")
    
    try:
        if os.path.exists(enhanced_graph_path):
            with open(enhanced_graph_path, "rb") as f:
                G = pickle.load(f)
            print(f"Loaded enhanced graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            sample_edge = next(iter(G.edges(data=True)), None)
            if sample_edge and 'shade_fraction_9' in sample_edge[2]:
                print("✅ Shade data available for enhanced pathfinding")
            else:
                print("⚠️ No shade data found in enhanced graph")
        else:
            with open(original_graph_path, "rb") as f:
                G = pickle.load(f)
            print(f"Loaded original graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            print("⚠️ Enhanced graph not found - shade-aware pathfinding not available")
    except Exception as e:
        print(f"Failed to load graph: {e}")
        G = None
    
    # Load waypoints
    load_waypoints_data()
    
    yield

## Removed duplicate lifespan/app/CORS. Using startup event defined at top.


def find_nearest_node(lat: float, lon: float) -> Optional[Tuple[float, float]]:
    """Find the nearest graph node to the given lat,lon using geodesic distance."""
    if G is None:
        return None
    
    min_dist = float('inf')
    nearest_node = None
    
    for node in G.nodes():
        # Node format is (lon, lat)
        node_lon, node_lat = node
        dist = geod.inv(lon, lat, node_lon, node_lat)[2]  # geodesic distance in meters
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node


def lonlat_to_latlng(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """Convert list of (lon, lat) tuples to [[lat, lng]] format for frontend."""
    return [[lat, lon] for lon, lat in coords]


class WeightsRequest(BaseModel):
    prompt: str


class NearestNodeRequest(BaseModel):
    lat: float
    lng: float


class ShortestPathRequest(BaseModel):
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    add_water_stop: Optional[bool] = False


class ShadeAwarePathRequest(BaseModel):
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    time: Optional[int] = 9  # Hour 0-23, default 9am
    shade_penalty: Optional[float] = 1.0  # Penalty factor for shaded areas
    add_water_stop: Optional[bool] = False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/llm/weights")
async def llm_weights(req: WeightsRequest) -> Dict[str, Any]:
    """
    Convert plain English into a simple weights JSON.
    This is a placeholder — replace with a real LLM call or rule-based parser.
    """
    prompt = req.prompt.lower()
    weights = {
        "avoid_highways": False,
        "prefer_scenic": False,
        "max_elevation_gain": None,
    }

    if "no highway" in prompt or "avoid highway" in prompt:
        weights["avoid_highways"] = True
    if "scenic" in prompt or "scenery" in prompt:
        weights["prefer_scenic"] = True
    if "flat" in prompt:
        weights["max_elevation_gain"] = 50

    return {"prompt": req.prompt, "weights": weights}


@app.get("/route/fetch")
async def fetch_route_example() -> Dict[str, Any]:
    """Placeholder endpoint that would fetch OSM data using OSMnx and compute a route.
    For the demo we return a tiny fake GeoJSON-like object.
    """
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "demo"}, "geometry": {"type": "LineString", "coordinates": [[-75.1652,39.9526], [-75.16,39.955]]}}
        ]
    }


@app.post("/nearest_node")
async def nearest_node(req: NearestNodeRequest) -> Dict[str, Any]:
    """Find the nearest graph node to the given lat,lng coordinates."""
    if G is None:
        return {"error": "Graph not loaded"}
    
    nearest = find_nearest_node(req.lat, req.lng)
    if nearest is None:
        return {"error": "No nearest node found"}
    
    # Return in frontend format [lat, lng]
    lon, lat = nearest
    return {
        "nearest_node": [lat, lon],
        "node_id": nearest,
        "input": [req.lat, req.lng]
    }


@app.post("/shortest_path")
async def shortest_path(req: ShortestPathRequest) -> Dict[str, Any]:
    """Compute shortest path between start and end coordinates."""
    if G is None:
        return {"error": "Graph not loaded"}
    
    # Find nearest nodes for start and end points
    start_node = find_nearest_node(req.start_lat, req.start_lng)
    end_node = find_nearest_node(req.end_lat, req.end_lng)
    
    if start_node is None or end_node is None:
        return {"error": "Could not find nearest nodes"}
    
    try:
        # Compute initial direct path using NetworkX
        path_nodes = nx.shortest_path(G, start_node, end_node, weight='weight')
        
        # Convert path to coordinate list for frontend
        path_coords = lonlat_to_latlng(path_nodes)
        
        # Calculate total distance for direct path
        total_distance = nx.shortest_path_length(G, start_node, end_node, weight='weight')
        
        # Calculate shade statistics for the path (using 9am data as default)
        total_shade_length = 0
        total_path_length = 0
        shaded_segments = 0
        
        # Check if shade data is available
        sample_edge = next(iter(G.edges(data=True)), None)
        has_shade_data = sample_edge and 'shade_fraction_9' in sample_edge[2]
        
        if has_shade_data:
            for i in range(len(path_nodes) - 1):
                node1, node2 = path_nodes[i], path_nodes[i + 1]
                if G.has_edge(node1, node2):
                    edge_data = G[node1][node2]
                    total_path_length += edge_data.get('weight', 0)
                    
                    # Use 9am shade data as default
                    shade_length = edge_data.get('shade_length_9', 0)
                    total_shade_length += shade_length
                    
                    # Check if segment is shaded
                    is_shaded = edge_data.get('is_shaded_9', False)
                    if is_shaded:
                        shaded_segments += 1
        
        result = {
            "path": path_coords,
            "start_node": [start_node[1], start_node[0]],  # [lat, lng]
            "end_node": [end_node[1], end_node[0]],        # [lat, lng]
            "total_distance_m": total_distance,
            "num_segments": len(path_nodes) - 1,
            "shade_mode": "standard",
            "analysis_time": "9:00"  # Default time for standard routing
        }

        # If user requested water stop, pick a waypoint on/near the initial path and route via it
        if getattr(req, 'add_water_stop', False):
            try:
                # Build lat,lng polyline from direct path for selection
                path_latlng = path_coords
                candidates = find_waypoints_near_path(path_latlng, max_detour=100, waypoint_types=['water', 'store'])
                chosen = candidates[0] if candidates else None
                if chosen:
                    wp_lat = float(chosen.get('lat') or chosen.get('latitude') or chosen['coordinates'][0])
                    wp_lng = float(chosen.get('lon') or chosen.get('lng') or chosen.get('longitude') or chosen['coordinates'][1])
                    wp_node = find_nearest_node(wp_lat, wp_lng)
                    if wp_node is not None:
                        p1 = nx.shortest_path(G, start_node, wp_node, weight='weight')
                        p2 = nx.shortest_path(G, wp_node, end_node, weight='weight')
                        via_nodes = p1[:-1] + p2
                        # Graph nodes are stored as (lon, lat); lonlat_to_latlng expects that
                        via_coords = lonlat_to_latlng(via_nodes)
                        via_dist = nx.shortest_path_length(G, start_node, wp_node, weight='weight') + \
                                   nx.shortest_path_length(G, wp_node, end_node, weight='weight')

                        # Replace original result with via-stop result
                        result.update({
                            "path": via_coords,
                            "total_distance_m": via_dist,
                            "num_segments": len(via_nodes) - 1,
                            "routed_via_water_stop": True,
                            "waypoints": [{
                                "id": chosen.get('id'),
                                "type": chosen.get('type') or 'water',
                                "name": chosen.get('name') or 'Water Stop',
                                "lat": wp_lat,
                                "lon": wp_lng,
                                "coordinates": [wp_lat, wp_lng],
                                "amenity": chosen.get('amenity',''),
                                "shop": chosen.get('shop',''),
                                "opening_hours": chosen.get('opening_hours',''),
                                "website": chosen.get('website',''),
                                "phone": chosen.get('phone',''),
                            }]
                        })
            except Exception as e:
                print(f"add_water_stop selection failed (standard): {e}")
        
        # Add shade statistics if available
        if has_shade_data:
            shade_percentage = (total_shade_length / total_path_length * 100) if total_path_length > 0 else 0
            result.update({
                "shaded_segments": shaded_segments,
                "shade_percentage": round(shade_percentage, 1),
                "total_shade_length_m": round(total_shade_length, 1),
                "original_distance_m": total_distance,  # Same as total_distance for standard routing
                "shade_aware_distance_m": total_distance,  # Same as total_distance for standard routing
                "shade_penalty_applied": 1.0,  # No penalty applied
                "shade_penalty_added_m": 0.0   # No penalty added
            })
        
        return result
        
    except nx.NetworkXNoPath:
        return {"error": "No path found between the points"}
    except Exception as e:
        return {"error": f"Path computation failed: {str(e)}"}


def calculate_shade_aware_weight(edge_attrs: Dict[str, Any], shade_penalty: float, is_daylight: bool, hour: int = 9) -> float:
    """Calculate edge weight with shade penalty applied for the specified hour."""
    base_weight = edge_attrs.get('weight', 0)
    
    if not is_daylight:
        # Night time - no shade penalty
        return base_weight
    
    # Get shade length for the specified hour (fallback to 9 if hour not available)
    shade_length_key = f'shade_length_{hour}'
    shade_length = edge_attrs.get(shade_length_key, edge_attrs.get('shade_length_9', 0))
    
    # Apply penalty: base_weight + (shade_length × penalty_factor)
    return base_weight + ((base_weight - shade_length) * shade_penalty)
    # return base_weight + (shade_length * shade_penalty)


@app.post("/shortest_path_shade")
async def shortest_path_shade_aware(req: ShadeAwarePathRequest) -> Dict[str, Any]:
    """Compute shortest path with shade awareness for daylight hours."""
    if G is None:
        return {"error": "Graph not loaded"}
    
    # Check if it's night time (≤6am or ≥19pm)
    is_night = req.time <= 6 or req.time >= 19
    
    if is_night:
        # Use standard shortest path for night time
        standard_req = ShortestPathRequest(
            start_lat=req.start_lat,
            start_lng=req.start_lng,
            end_lat=req.end_lat,
            end_lng=req.end_lng
        )
        result = await shortest_path(standard_req)
        if isinstance(result, dict) and 'path' in result:
            result['shade_mode'] = 'night'
            result['shade_penalty_applied'] = False
        return result
    
    # Check if shade data is available for the requested hour
    sample_edge = next(iter(G.edges(data=True)), None)
    if not sample_edge:
        return {"error": "No edges available in graph"}
    
    # Check for hour-specific shade data, fallback to 9
    shade_fraction_key = f'shade_fraction_{req.time}'
    has_hour_data = shade_fraction_key in sample_edge[2]
    fallback_hour_data = 'shade_fraction_9' in sample_edge[2]
    
    if not has_hour_data and not fallback_hour_data:
        return {"error": f"Shade data not available for {req.time}:00 or fallback 9:00 - use /shortest_path instead"}
    
    if not has_hour_data:
        print(f"Warning: No shade data for {req.time}:00, using 9:00 data as fallback")
    
    # Find nearest nodes for start and end points
    start_node = find_nearest_node(req.start_lat, req.start_lng)
    end_node = find_nearest_node(req.end_lat, req.end_lng)
    
    if start_node is None or end_node is None:
        return {"error": "Could not find nearest nodes"}
    
    try:
        # Create a temporary graph with shade-aware weights
        temp_graph = G.copy()
        is_daylight = True  # We already checked for night time above
        
        # Update edge weights with shade penalty for the specified hour
        for node1, node2, edge_attrs in temp_graph.edges(data=True):
            new_weight = calculate_shade_aware_weight(edge_attrs, req.shade_penalty, is_daylight, req.time)
            edge_attrs['shade_aware_weight'] = new_weight
        
        # Compute initial shortest path using shade-aware weights
        path_nodes = nx.shortest_path(temp_graph, start_node, end_node, weight='shade_aware_weight')
        
        # Convert path to coordinate list for frontend
        path_coords = lonlat_to_latlng(path_nodes)
        
        # Calculate distances for direct path
        original_distance = nx.shortest_path_length(G, start_node, end_node, weight='weight')
        shade_aware_distance = nx.shortest_path_length(temp_graph, start_node, end_node, weight='shade_aware_weight')
        
        # Calculate shade statistics for the path using hour-specific data
        total_shade_length = 0
        total_path_length = 0
        shaded_segments = 0
        
        # Define hour-specific keys
        shade_length_key = f'shade_length_{req.time}'
        is_shaded_key = f'is_shaded_{req.time}'
        
        for i in range(len(path_nodes) - 1):
            node1, node2 = path_nodes[i], path_nodes[i + 1]
            if temp_graph.has_edge(node1, node2):
                edge_data = temp_graph[node1][node2]
                total_path_length += edge_data.get('weight', 0)
                
                # Try hour-specific data, fallback to 9
                shade_length = edge_data.get(shade_length_key, edge_data.get('shade_length_9', 0))
                total_shade_length += shade_length
                
                # Check if segment is shaded at this hour
                is_shaded = edge_data.get(is_shaded_key, edge_data.get('is_shaded_9', False))
                if is_shaded:
                    shaded_segments += 1
        
        shade_percentage = (total_shade_length / total_path_length * 100) if total_path_length > 0 else 0

        routed_via = False
        used_waypoints: List[Dict[str, Any]] = []
        # If requested, select a waypoint along the initial path and rebuild the path via it
        if getattr(req, 'add_water_stop', False):
            try:
                candidates = find_waypoints_near_path(path_coords, max_detour=100, waypoint_types=['water', 'store'])
                chosen = candidates[0] if candidates else None
                if chosen:
                    wp_lat = float(chosen.get('lat') or chosen.get('latitude') or chosen['coordinates'][0])
                    wp_lng = float(chosen.get('lon') or chosen.get('lng') or chosen.get('longitude') or chosen['coordinates'][1])
                    wp_node = find_nearest_node(wp_lat, wp_lng)
                    if wp_node is not None:
                        p1 = nx.shortest_path(temp_graph, start_node, wp_node, weight='shade_aware_weight')
                        p2 = nx.shortest_path(temp_graph, wp_node, end_node, weight='shade_aware_weight')
                        path_nodes = p1[:-1] + p2
                        path_coords = lonlat_to_latlng(path_nodes)
                        # Recompute distances and stats for via-stop path
                        shade_aware_distance = nx.shortest_path_length(temp_graph, start_node, wp_node, weight='shade_aware_weight') + \
                                              nx.shortest_path_length(temp_graph, wp_node, end_node, weight='shade_aware_weight')
                        # Recompute shade stats
                        total_shade_length = 0
                        total_path_length = 0
                        shaded_segments = 0
                        for i in range(len(path_nodes) - 1):
                            n1, n2 = path_nodes[i], path_nodes[i + 1]
                            if temp_graph.has_edge(n1, n2):
                                ed = temp_graph[n1][n2]
                                total_path_length += ed.get('weight', 0)
                                total_shade_length += ed.get(shade_length_key, ed.get('shade_length_9', 0))
                                if ed.get(is_shaded_key, ed.get('is_shaded_9', False)):
                                    shaded_segments += 1
                        shade_percentage = (total_shade_length / total_path_length * 100) if total_path_length > 0 else 0
                        routed_via = True
                        used_waypoints = [{
                            "id": chosen.get('id'),
                            "type": chosen.get('type') or 'water',
                            "name": chosen.get('name') or 'Water Stop',
                            "lat": wp_lat,
                            "lon": wp_lng,
                            "coordinates": [wp_lat, wp_lng],
                            "amenity": chosen.get('amenity',''),
                            "shop": chosen.get('shop',''),
                            "opening_hours": chosen.get('opening_hours',''),
                            "website": chosen.get('website',''),
                            "phone": chosen.get('phone',''),
                        }]
            except Exception as e:
                print(f"add_water_stop selection failed (shade): {e}")
        
        return {
            "path": path_coords,
            "start_node": [start_node[1], start_node[0]],  # [lat, lng]
            "end_node": [end_node[1], end_node[0]],        # [lat, lng]
            "original_distance_m": original_distance,
            "shade_aware_distance_m": shade_aware_distance,
            "shade_penalty_applied": req.shade_penalty,
            "analysis_time": f"{req.time}:00",
            "shade_mode": "daylight",
            "num_segments": len(path_nodes) - 1,
            "shaded_segments": shaded_segments,
            "shade_percentage": round(shade_percentage, 1),
            "total_shade_length_m": round(total_shade_length, 1),
            "shade_penalty_added_m": round(shade_aware_distance - original_distance, 1),
            "routed_via_water_stop": routed_via,
            "waypoints": used_waypoints,
        }
        
    except nx.NetworkXNoPath:
        return {"error": "No path found between the points"}
    except Exception as e:
        return {"error": f"Shade-aware path computation failed: {str(e)}"}


@app.get("/graph/edges")
async def get_graph_edges(limit: Optional[int] = None) -> Dict[str, Any]:
    """Export all graph edges with start/end coordinates for frontend analysis.
    
    Args:
        limit: Optional limit on number of edges to return (useful for testing)
    """
    if G is None:
        return {"error": "Graph not loaded"}
    
    edges = []
    
    try:
        # Iterate through all edges in the graph
        edge_iter = G.edges(data=True)
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            edge_iter = list(edge_iter)[:limit]
        
        for i, (node1, node2, edge_data) in enumerate(edge_iter):
            # node1 and node2 are tuples of (lon, lat)
            lon1, lat1 = node1
            lon2, lat2 = node2
            
            # Create edge object with frontend-compatible format
            edge = {
                "id": f"edge_{i}",
                "a": {"lat": lat1, "lng": lon1},
                "b": {"lat": lat2, "lng": lon2}
            }
            
            # Optionally include edge weight/distance if available
            if 'weight' in edge_data:
                edge["weight"] = edge_data['weight']
            
            edges.append(edge)
        
        total_edges = G.number_of_edges()
        
        return {
            "type": "graph_edges",
            "count": len(edges),
            "total_available": total_edges,
            "limited": limit is not None and limit < total_edges,
            "edges": edges
        }
        
    except Exception as e:
        return {"error": f"Failed to export edges: {str(e)}"}


@app.get("/graph/edges/download")
async def download_graph_edges(limit: Optional[int] = None):
    """Download graph edges as a JSON file.
    
    Args:
        limit: Optional limit on number of edges to download (useful for testing)
    """
    if G is None:
        return JSONResponse(
            content={"error": "Graph not loaded"}, 
            status_code=500
        )
    
    edges = []
    
    try:
        # Iterate through all edges in the graph
        edge_iter = G.edges(data=True)
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            edge_iter = list(edge_iter)[:limit]
        
        for i, (node1, node2, edge_data) in enumerate(edge_iter):
            # node1 and node2 are tuples of (lon, lat)
            lon1, lat1 = node1
            lon2, lat2 = node2
            
            # Create edge object with frontend-compatible format
            edge = {
                "id": f"edge_{i}",
                "a": {"lat": lat1, "lng": lon1},
                "b": {"lat": lat2, "lng": lon2}
            }
            
            # Include edge weight/distance if available
            if 'weight' in edge_data:
                edge["weight"] = edge_data['weight']
            
            edges.append(edge)
        
        total_edges = G.number_of_edges()
        filename = f"graph_edges{'_limited' if limit is not None and limit < total_edges else ''}.json"
        
        response_data = {
            "type": "graph_edges",
            "count": len(edges),
            "total_available": total_edges,
            "limited": limit is not None and limit < total_edges,
            "limit_applied": limit,
            "generated_at": G.graph.get('created_at', 'unknown') if hasattr(G, 'graph') else 'unknown',
            "edges": edges
        }
        
        # Return as downloadable JSON file
        return JSONResponse(
            content=response_data,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/json"
            }
        )
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to export edges: {str(e)}"}, 
            status_code=500
        )
