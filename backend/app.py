from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import os
import pickle
from contextlib import asynccontextmanager

import networkx as nx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyproj import Geod

# -----------------------------
# Load segmented graph on start
# -----------------------------

G: Optional[nx.Graph] = None
geod = Geod(ellps="WGS84")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global G
    graph_path = os.path.join(os.path.dirname(__file__), "graph_segments.gpickle")
    try:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        print(f"[startup] Loaded graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    except Exception as e:
        print(f"[startup] Failed to load graph: {e}")
        G = None
    yield


app = FastAPI(title="PennApps Demo Backend", lifespan=lifespan)

# CORS (wide-open for dev; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models (existing + new)
# -----------------------------

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


class AccessiblePathRequest(BaseModel):
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    prefer_paved: bool = True
    prefer_smoothness: bool = True
    avoid_steps: bool = False
    avoid_uneven: bool = False


class BBoxRequest(BaseModel):
    north: float
    south: float
    east: float
    west: float


# -----------------------------
# Utilities (existing)
# -----------------------------

def find_nearest_node(lat: float, lon: float) -> Optional[Tuple[float, float]]:
    """Nearest node in geodesic distance. Graph nodes are (lon, lat)."""
    if G is None:
        return None
    min_dist = float("inf")
    nearest = None
    for node in G.nodes():
        node_lon, node_lat = node
        dist = geod.inv(lon, lat, node_lon, node_lat)[2]
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest


def lonlat_to_latlng(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """Convert (lon,lat) -> [lat,lng]"""
    return [[lat, lon] for lon, lat in coords]


# -----------------------------
# Enhanced accessibility enrichment
# -----------------------------

import aiohttp
import asyncio

# cache: osmid -> minimal tag dict we care about
WAY_TAG_CACHE: dict[int, dict] = {}

OVERPASS_API = "https://overpass-api.de/api/interpreter"

ROUGH_SURFACES = {
    "cobblestone", "sett", "unpaved", "gravel", "compacted", "fine_gravel",
    "grass", "ground", "dirt", "sand", "wood", "pebblestone", "cobblestone:flattened"
}

# Specific uneven terrain indicators
UNEVEN_SURFACES = {
    "cobblestone", "sett", "pebblestone", "rocks", "stone", "boulders", "scree"
}

SMOOTHNESS_ORDER = ["impassable", "very_bad", "bad", "intermediate", "good", "excellent", "perfect"]
SMOOTHNESS_SCORE = {v: i for i, v in enumerate(SMOOTHNESS_ORDER)}  # 0 worst

# Highway type defaults (when surface/smoothness tags are missing)
HIGHWAY_SURFACE_DEFAULTS = {
    "motorway": {"surface": "asphalt", "smoothness": "excellent"},
    "trunk": {"surface": "asphalt", "smoothness": "good"},
    "primary": {"surface": "asphalt", "smoothness": "good"},
    "secondary": {"surface": "asphalt", "smoothness": "good"},
    "tertiary": {"surface": "asphalt", "smoothness": "intermediate"},
    "residential": {"surface": "asphalt", "smoothness": "good"},
    "service": {"surface": "asphalt", "smoothness": "intermediate"},
    "unclassified": {"surface": "unknown", "smoothness": "intermediate"},
    "track": {"surface": "unpaved", "smoothness": "bad"},
    "path": {"surface": "unpaved", "smoothness": "bad"},
    "footway": {"surface": "paved", "smoothness": "good"},
    "cycleway": {"surface": "asphalt", "smoothness": "good"},
}


def _edge_osmids(d: dict) -> list[int]:
    osmid = d.get("osmid")
    if osmid is None:
        return []
    if isinstance(osmid, (list, tuple, set)):
        return [int(x) for x in osmid if x is not None]
    try:
        return [int(osmid)]
    except Exception:
        return []


def collect_edge_osmids_along_path(path_nodes: List[Tuple[float, float]]) -> List[int]:
    """Collect OSM way ids present on edges of a node path."""
    if G is None or not path_nodes or len(path_nodes) < 2:
        return []
    out: list[int] = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not G.has_edge(u, v):
            continue
        data = G.get_edge_data(u, v)
        # multi-edge dict or single dict
        if isinstance(data, dict) and any(isinstance(k, int) for k in data.keys()):
            first = data[min(data.keys())]
            out.extend(_edge_osmids(first))
        else:
            out.extend(_edge_osmids(data))
    # dedupe preserving order
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


async def _overpass_query(ql: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(OVERPASS_API, data={"data": ql}, timeout=60) as r:
            r.raise_for_status()
            return await r.json()


async def fetch_way_tags_from_overpass(osmids: list[int]) -> dict[int, dict]:
    """Fetch surface/smoothness/etc for osmid list, chunked, cached."""
    uncached = [i for i in osmids if i not in WAY_TAG_CACHE]
    if not uncached:
        return {}
    result: dict[int, dict] = {}
    chunks = [uncached[i:i + 150] for i in range(0, len(uncached), 150)]
    for chunk in chunks:
        id_list = ";".join(str(i) for i in chunk)
        ql = f"""
        [out:json][timeout:25];
        (
          way(id:{id_list});
        );
        out tags;
        """
        try:
            data = await _overpass_query(ql)
        except Exception:
            continue
        for el in data.get("elements", []):
            if el.get("type") == "way" and "id" in el:
                osmid = int(el["id"])
                tags = el.get("tags", {}) or {}
                keep = {
                    k: tags.get(k) for k in [
                        "surface", "smoothness", "sidewalk",
                        "kerb", "kerb:height", "step_count",
                        "incline", "footway", "tracktype"
                    ] if tags.get(k) is not None
                }
                result[osmid] = keep
                WAY_TAG_CACHE[osmid] = keep
    return result


def apply_tags_to_graph(tag_map: dict[int, dict]) -> None:
    """Write fetched tags back onto matching edges (non-destructive)."""
    if G is None or not tag_map:
        return
    for u, v, d in G.edges(data=True):
        osmids = _edge_osmids(d)
        if not osmids:
            continue
        merged = {}
        for i in osmids:
            if i in tag_map:
                merged.update(tag_map[i])
        if merged:
            for k, vtag in merged.items():
                if k not in d or d[k] in (None, "", "unknown"):
                    d[k] = vtag


async def ensure_attrs_for_path(path_nodes: List[Tuple[float, float]]) -> None:
    osmids = collect_edge_osmids_along_path(path_nodes)
    if not osmids:
        return
    fetched = await fetch_way_tags_from_overpass(osmids)
    apply_tags_to_graph(fetched)


async def ensure_attrs_for_bbox(north: float, south: float, east: float, west: float) -> int:
    """Optional: enrich all highway ways in a bbox."""
    ql = f"""
    [out:json][timeout:25];
    way["highway"]({south},{west},{north},{east});
    out ids tags;
    """
    try:
        data = await _overpass_query(ql)
    except Exception:
        return 0
    tag_map: dict[int, dict] = {}
    for el in data.get("elements", []):
        if el.get("type") == "way" and "id" in el:
            osmid = int(el["id"])
            tags = el.get("tags", {}) or {}
            keep = {
                k: tags.get(k) for k in [
                    "surface", "smoothness", "sidewalk",
                    "kerb", "kerb:height", "step_count",
                    "incline", "footway", "tracktype"
                ] if tags.get(k) is not None
            }
            tag_map[osmid] = keep
            WAY_TAG_CACHE.setdefault(osmid, keep)
    before = sum(1 for *_e, d in G.edges(data=True) if d.get("surface") or d.get("smoothness"))
    apply_tags_to_graph(tag_map)
    after = sum(1 for *_e, d in G.edges(data=True) if d.get("surface") or d.get("smoothness"))
    return max(0, after - before)


async def detect_area_context_from_overpass(lat: float, lng: float, radius_m: float = 500) -> dict:
    """Detect if we're in a historic district, old town, etc. that might have cobblestone."""
    
    ql = f"""
    [out:json][timeout:15];
    (
      way["historic"="yes"](around:{radius_m},{lat},{lng});
      way["tourism"="yes"](around:{radius_m},{lat},{lng});
      way["place"~"^(old_town|historic)$"](around:{radius_m},{lat},{lng});
      way["name"~"(?i)(old|historic|cobble|stone|brick)"](around:{radius_m},{lat},{lng});
      relation["historic"](around:{radius_m},{lat},{lng});
    );
    out geom;
    """
    
    try:
        data = await _overpass_query(ql)
        context = {
            "historic_area": False,
            "tourist_area": False,
            "old_town": False,
            "likely_cobblestone": False
        }
        
        for element in data.get("elements", []):
            tags = element.get("tags", {})
            
            if tags.get("historic"):
                context["historic_area"] = True
            if tags.get("tourism"):
                context["tourist_area"] = True
            if tags.get("place") in ["old_town", "historic"]:
                context["old_town"] = True
                
            # Check names for cobblestone indicators
            name = (tags.get("name") or "").lower()
            if any(word in name for word in ["cobble", "stone", "brick", "old", "historic"]):
                context["likely_cobblestone"] = True
                
        return context
    except:
        return {"historic_area": False, "tourist_area": False, "old_town": False, "likely_cobblestone": False}


async def enhance_route_with_area_context(path_nodes: List[Tuple[float, float]]) -> None:
    """Enhance route edges with area context for better surface prediction."""
    if not path_nodes:
        return
        
    # Sample a few points along the route to detect context
    sample_points = path_nodes[::max(1, len(path_nodes) // 5)]  # Sample ~5 points
    
    contexts = []
    for lon, lat in sample_points:
        context = await detect_area_context_from_overpass(lat, lon, 300)
        contexts.append(context)
    
    # Aggregate context - if any sample suggests cobblestone, flag the whole route
    route_context = {
        "historic_area": any(c["historic_area"] for c in contexts),
        "tourist_area": any(c["tourist_area"] for c in contexts),
        "old_town": any(c["old_town"] for c in contexts),
        "likely_cobblestone": any(c["likely_cobblestone"] for c in contexts)
    }
    
    # Apply context to graph edges along the route
    if G and (route_context["likely_cobblestone"] or route_context["historic_area"]):
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)
                if isinstance(edge_data, dict) and any(isinstance(k, int) for k in edge_data.keys()):
                    edge_data = edge_data[min(edge_data.keys())]
                
                # Only modify if no explicit surface tag exists
                if not edge_data.get("surface"):
                    highway = edge_data.get("highway", "").lower()
                    if highway in ["residential", "tertiary", "unclassified", "service"]:
                        if route_context["likely_cobblestone"]:
                            edge_data["_context_surface"] = "cobblestone"
                        elif route_context["historic_area"]:
                            edge_data["_context_surface"] = "paving_stones"


def get_effective_surface_tags(d: dict) -> dict:
    """Get surface/smoothness tags, using intelligent defaults when missing."""
    # Safe string conversion helper
    def safe_str_lower(value):
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value).lower()
        return str(value).lower()
    
    highway_type = safe_str_lower(d.get("highway"))
    
    # Start with actual tags
    surface = safe_str_lower(d.get("surface"))
    smoothness = safe_str_lower(d.get("smoothness"))
    
    # Check for context surface
    if not surface:
        surface = safe_str_lower(d.get("_context_surface"))
    
    # Apply defaults if tags are missing
    if not surface or not smoothness:
        defaults = HIGHWAY_SURFACE_DEFAULTS.get(highway_type, {})
        if not surface:
            surface = defaults.get("surface", "unknown")
        if not smoothness:
            smoothness = defaults.get("smoothness", "intermediate")
    
    # Additional heuristics for historical areas
    name = safe_str_lower(d.get("name"))
    if any(word in name for word in ["old", "historic", "cobble", "stone"]):
        if surface in ["unknown", "paved"]:
            surface = "cobblestone"
        if smoothness in ["unknown", "good"]:
            smoothness = "intermediate"
    
    # Adjust smoothness based on surface when not explicitly tagged
    if not d.get("smoothness"):
        if surface in ["cobblestone", "paving_stones"]:
            smoothness = "intermediate"
        elif surface in UNEVEN_SURFACES:
            smoothness = "bad"
    
    return {"surface": surface, "smoothness": smoothness}


def accessibility_weight(u: Tuple, v: Tuple, d: dict,
                         prefer_paved: bool = True,
                         avoid_steps: bool = False,
                         prefer_smoothness: bool = True,
                         avoid_uneven: bool = False) -> float:
    """Enhanced weight function with intelligent defaults for missing tags."""
    # Safe string conversion helper
    def safe_str_lower(value):
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value).lower()
        return str(value).lower()
    
    base = float(d.get("weight") or d.get("length") or 1.0)
    
    # Get effective surface tags (with defaults and context)
    tags = get_effective_surface_tags(d)
    surface = tags["surface"]
    smoothness = tags["smoothness"]
    
    # Other tags with safe string handling
    footway = safe_str_lower(d.get("footway"))
    step_ct = d.get("step_count")
    incline = safe_str_lower(d.get("incline"))
    sidewalk = safe_str_lower(d.get("sidewalk"))
    tracktype = safe_str_lower(d.get("tracktype"))
    highway_type = safe_str_lower(d.get("highway"))

    penalty = 0.0

    # Enhanced uneven terrain detection
    if avoid_uneven:
        uneven_penalty = 0.0
        
        # Surface-based unevenness
        if surface in UNEVEN_SURFACES:
            uneven_penalty += 1.2 * base  # Strong penalty for clearly uneven
        elif surface in ["unpaved", "gravel", "dirt", "grass"]:
            uneven_penalty += 0.4 * base  # Moderate penalty for potentially uneven
        
        # Smoothness-based unevenness
        if smoothness in ["very_bad", "bad", "impassable"]:
            uneven_penalty += 1.0 * base
        elif smoothness == "intermediate" and surface in ["cobblestone", "paving_stones"]:
            uneven_penalty += 0.6 * base  # Cobblestone penalty
        elif smoothness == "intermediate":
            uneven_penalty += 0.2 * base  # Small penalty for uncertain smoothness
        
        # Track quality
        if tracktype in ["grade4", "grade5"]:
            uneven_penalty += 0.8 * base
        elif tracktype in ["grade3"]:
            uneven_penalty += 0.3 * base
            
        # Highway type heuristics (when other info unavailable)
        if highway_type == "track" and surface == "unknown":
            uneven_penalty += 0.6 * base
        elif highway_type == "path" and surface == "unknown":
            uneven_penalty += 0.4 * base
            
        penalty += uneven_penalty

    # Existing logic (unchanged)
    if prefer_paved and surface in ROUGH_SURFACES:
        penalty += 0.5 * base

    if prefer_smoothness and smoothness:
        score = SMOOTHNESS_SCORE.get(smoothness)
        if score is not None:
            penalty += max(0, (3 - score)) * 0.15 * base

    if avoid_steps and (footway == "steps" or step_ct is not None):
        penalty += 5.0 * base

    if sidewalk in ("both", "left", "right", "yes"):
        penalty -= 0.05 * base

    if "%" in incline:
        try:
            pct = float(incline.replace("up", "").replace("down", "").replace("%", "").strip())
            if pct > 6:
                penalty += 0.2 * base
        except Exception:
            pass

    return max(0.2 * base, base + penalty)


# -----------------------------
# Endpoints (existing kept)
# -----------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/llm/weights")
async def llm_weights(req: WeightsRequest) -> Dict[str, Any]:
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
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "demo"},
             "geometry": {"type": "LineString",
                          "coordinates": [[-75.1652, 39.9526], [-75.16, 39.955]]}}
        ]
    }


@app.post("/nearest_node")
async def nearest_node(req: NearestNodeRequest) -> Dict[str, Any]:
    if G is None:
        return {"error": "Graph not loaded"}
    nearest = find_nearest_node(req.lat, req.lng)
    if nearest is None:
        return {"error": "No nearest node found"}
    lon, lat = nearest
    return {"nearest_node": [lat, lon], "node_id": nearest, "input": [req.lat, req.lng]}


@app.post("/shortest_path")
async def shortest_path(req: ShortestPathRequest) -> Dict[str, Any]:
    if G is None:
        return {"error": "Graph not loaded"}
    start_node = find_nearest_node(req.start_lat, req.start_lng)
    end_node = find_nearest_node(req.end_lat, req.end_lng)
    if start_node is None or end_node is None:
        return {"error": "Could not find nearest nodes"}
    try:
        path_nodes = nx.shortest_path(G, start_node, end_node, weight="weight")
        total_distance = nx.shortest_path_length(G, start_node, end_node, weight="weight")
    except nx.NetworkXNoPath:
        return {"error": "No path found between the points"}
    return {
        "path": lonlat_to_latlng(path_nodes),
        "start_node": [start_node[1], start_node[0]],
        "end_node": [end_node[1], end_node[0]],
        "total_distance_m": total_distance,
        "num_segments": len(path_nodes) - 1,
    }


# -----------------------------
# Enhanced accessibility endpoint
# -----------------------------

@app.post("/route/shortest_path_accessible")
async def shortest_path_accessible(req: AccessiblePathRequest) -> Dict[str, Any]:
    if G is None:
        return {"error": "Graph not loaded"}

    start_node = find_nearest_node(req.start_lat, req.start_lng)
    end_node = find_nearest_node(req.end_lat, req.end_lng)
    if start_node is None or end_node is None:
        return {"error": "Could not find nearest nodes"}

    # Get base path first
    try:
        base_path_nodes = nx.shortest_path(G, start_node, end_node, weight="weight")
    except nx.NetworkXNoPath:
        return {"error": "No path found between the points"}

    # Enhance with OSM surface tags AND area context
    await ensure_attrs_for_path(base_path_nodes)
    await enhance_route_with_area_context(base_path_nodes)

    # Use enhanced weight function with context awareness
    def w(u, v, d):
        return accessibility_weight(
            u, v, d,
            prefer_paved=req.prefer_paved,
            avoid_steps=req.avoid_steps,
            prefer_smoothness=req.prefer_smoothness,
            avoid_uneven=req.avoid_uneven,
        )

    try:
        path_nodes = nx.shortest_path(G, start_node, end_node, weight=w)
        access_cost = nx.shortest_path_length(G, start_node, end_node, weight=w)
        base_cost = nx.shortest_path_length(G, start_node, end_node, weight="weight")
    except nx.NetworkXNoPath:
        # fall back to base path; still return an access_cost estimate
        path_nodes = base_path_nodes
        access_cost = sum(
            w(u, v, (G.get_edge_data(u, v)[min(G.get_edge_data(u, v).keys())]
                     if isinstance(G.get_edge_data(u, v), dict) and any(isinstance(k, int) for k in G.get_edge_data(u, v).keys())
                     else G.get_edge_data(u, v)))
            for u, v in zip(path_nodes[:-1], path_nodes[1:])
        )
        base_cost = nx.shortest_path_length(G, start_node, end_node, weight="weight")

    return {
        "path": lonlat_to_latlng(path_nodes),
        "start_node": [start_node[1], start_node[0]],
        "end_node": [end_node[1], end_node[0]],
        "num_segments": len(path_nodes) - 1,
        "distance_m": base_cost,
        "access_cost": access_cost,
        "prefs": req.model_dump(),
    }


@app.post("/augment/bbox")
async def augment_bbox(req: BBoxRequest) -> Dict[str, Any]:
    if G is None:
        return {"error": "Graph not loaded"}
    applied = await ensure_attrs_for_bbox(req.north, req.south, req.east, req.west)
    return {"updated_edges": applied}