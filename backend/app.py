from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple, Optional

# helper that finds a shop waypoint and returns routed paths
from backend.find_shop_waypoint import route_via_nearby_shop, _nearest_node
import networkx as nx
import os
from pyproj import Geod
import pickle
from contextlib import asynccontextmanager

# Global graph variable
G: Optional[nx.Graph] = None
geod = Geod(ellps="WGS84")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the segmented graph on startup."""
    global G
    graph_path = os.path.join(os.path.dirname(__file__), "graph_segments.gpickle")
    try:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    except Exception as e:
        print(f"Failed to load graph: {e}")
        G = None
    yield


app = FastAPI(title="PennApps Demo Backend", lifespan=lifespan)

# Add CORS middleware
# Allow Vite dev server origins during development; adjust for production as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        # Compute shortest path using NetworkX
        path_nodes = nx.shortest_path(G, start_node, end_node, weight='weight')
        
        # Convert path to coordinate list for frontend
        path_coords = lonlat_to_latlng(path_nodes)
        
        # Calculate total distance
        total_distance = nx.shortest_path_length(G, start_node, end_node, weight='weight')
        
        return {
            "path": path_coords,
            "start_node": [start_node[1], start_node[0]],  # [lat, lng]
            "end_node": [end_node[1], end_node[0]],        # [lat, lng]
            "total_distance_m": total_distance,
            "num_segments": len(path_nodes) - 1
        }
        
    except nx.NetworkXNoPath:
        return {"error": "No path found between the points"}
    except Exception as e:
        return {"error": f"Path computation failed: {str(e)}"}


class RouteRequest(BaseModel):
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    shop_tags: Dict[str, int] = {}
    tag_match_rule: Optional[str] = 'any'
    max_dist: Optional[int] = 1000


@app.post("/route_via_shop")
def route_via_shop(req: RouteRequest):
    if not req.shop_tags:
        # If no shops are requested, just return the shortest path.
        # This handles the "include shop stop" being unchecked.
        if G is None:
            return {"error": "Graph not loaded"}
        
        start_node = _nearest_node(G, req.start_lng, req.start_lat)
        end_node = _nearest_node(G, req.end_lng, req.end_lat)

        if not start_node or not end_node:
            return {"error": "Could not find nearest nodes for start/end points"}

        try:
            base_path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight="length")
            base_path_coords = [ (G.nodes[n]['x'], G.nodes[n]['y']) for n in base_path_nodes]
            
            return {
                "base_path": lonlat_to_latlng(base_path_coords),
                "via_path": [],
                "shop_points": [],
                "shop_label": None,
                "note": "No shop requested, returning direct path.",
            }
        except nx.NetworkXNoPath:
            return {"error": "No path found between start and end points"}

    try:
        result = route_via_nearby_shop(
            start_lon=req.start_lng,
            start_lat=req.start_lat,
            end_lon=req.end_lng,
            end_lat=req.end_lat,
            shop_tags=req.shop_tags,
            tag_match_rule=req.tag_match_rule,
            G=G,
            max_search_dist_m=req.max_dist,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Internal server error: {str(e)}"}

    # Convert to lat/lng arrays for frontend
    def node_to_latlng(n: Tuple[float, float]) -> List[float]:
        return [n[1], n[0]]

    base_path = [node_to_latlng(n) for n in result.get("base_path", [])]
    via_path = [node_to_latlng(n) for n in result.get("route_u_shop_v", [])]

    shop_points = []
    if result.get("shop_points"):
        shop_points = [{"point": [p[1], p[0]], "label": label} for p, label in result["shop_points"]]

    return {
        "base_path": base_path,
        "via_path": via_path,
        "shop_points": shop_points,
        "note": result.get("note"),
    }
