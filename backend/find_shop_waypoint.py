from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd

import networkx as nx
import osmnx as ox
from osmnx._errors import InsufficientResponseError
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from pyproj import Transformer, Geod
import matplotlib.pyplot as plt

from .build_graph_2 import build_graph_segments, DEFAULT_ROAD_TAGS


def _features_from_point_compat(center_point, tags, dist):
    """Compatibility wrapper for features_from_point."""
    try:
        # OSMnx v2.x signature
        return ox.features.features_from_point(center_point, tags, dist=dist)
    except AttributeError:
        # Fallback to OSMnx v1.x signature (geometries_from_point)
        return ox.geometries_from_point(center_point, tags, dist=dist)
    except InsufficientResponseError:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    except Exception as e:
        print(f"Error fetching features from point: {e}")
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")


def _features_from_bbox_compat(north, south, east, west, tags=None):
    """Compatibility wrapper supporting OSMnx v1.x and v2.x.
    - v2.x: ox.features.features_from_bbox
    - v1.x: ox.geometries_from_bbox
    Handles differing call signatures.
    """
    try:
        # OSMnx v2.x signature
        return ox.features.features_from_bbox((south, west, north, east), tags)
    except AttributeError:
        # Fallback to OSMnx v1.x signature
        return ox.geometries_from_bbox(north, south, east, west, tags)
    except InsufficientResponseError:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    except Exception as e:
        print(f"Error fetching features from bbox: {e}")
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")


# ---------------------------
# Helpers
# ---------------------------

def _nearest_node(G: nx.Graph, x: float, y: float) -> Optional[Tuple[float, float]]:
    """Return the nearest node (lon, lat tuple) in G to (x=lon, y=lat) without requiring OSMnx CRS.

    We compute geodesic distance using WGS84 directly over G's node coordinates.
    """
    if not G.nodes:
        return None
    geod = Geod(ellps="WGS84")
    best = None
    best_d = float("inf")
    for n_data in G.nodes(data=True):
        node_coords = (n_data[1]['x'], n_data[1]['y']) # (lon, lat)
        d = geod.inv(x, y, node_coords[0], node_coords[1])[2]
        if d < best_d:
            best_d = d
            best = node_coords
    return best


def _edge_linestring(u_node, v_node, d) -> LineString:
    """Get a LineString for an edge, using edge geometry when present."""
    geom = d.get("geometry")
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        # Taking the first line of a multiline - adjust if specific logic is needed
        if geom.geoms:
            return geom.geoms[0]
        # Fallback if MultiLineString is empty
        return LineString([(u_node[0], u_node[1]), (v_node[0], v_node[1])])
    # fallback: straight segment between node coordinates
    return LineString([(u_node[0], u_node[1]), (v_node[0], v_node[1])])


def _path_linestring(G: nx.Graph, path_nodes_ids: List[int]) -> LineString:
    """Merge the edge segments of a path into a single LineString."""
    if not path_nodes_ids:
        return LineString([])
    
    path_coords = []
    
    # Get node (lon, lat) tuples from node IDs
    nodes_lonlat = {node_id: (G.nodes[node_id]['x'], G.nodes[node_id]['y']) for node_id in path_nodes_ids}

    for u_id, v_id in zip(path_nodes_ids[:-1], path_nodes_ids[1:]):
        # Retrieve edge data, handling potential multiple edges between same nodes
        edges_data = G.get_edge_data(u_id, v_id)
        if not edges_data:
            continue # No edge found, skip

        # If it's a MultiGraph, get the first edge's data; otherwise, it's already an attribute dict
        if G.is_multigraph():
            edge_data = next(iter(edges_data.values()))
        else:
            edge_data = edges_data
        
        u_lonlat = nodes_lonlat[u_id]
        v_lonlat = nodes_lonlat[v_id]
        
        ls = _edge_linestring(u_lonlat, v_lonlat, edge_data)
        
        if ls.is_empty:
            continue

        if not path_coords:
            path_coords.extend(ls.coords)
        else:
            # Check if the last point of previous segment matches the first of current
            # This handles cases where _edge_linestring might return different start/end points than nodes_lonlat
            if path_coords[-1] == ls.coords[0]:
                path_coords.extend(ls.coords[1:])
            else:
                # If there's a mismatch, it might indicate a gap or issue in segments
                # For robustness, we can just add all coordinates or find closest
                # For now, append all and let LineString handle it
                path_coords.extend(ls.coords)

    if not path_coords:
        return LineString([])

    return LineString(path_coords)


def _project_to_local(points: List[Point]) -> Tuple[List[Point], Transformer]:
    """Project lon/lat Points to a local metric projection for distance calcs."""
    # Auto-select an appropriate UTM via EPSG:3857 as a safe metric fallback
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return [Point(*tf.transform(p.x, p.y)) for p in points], tf


def _project_linestring(ls: LineString, tf: Transformer) -> LineString:
    if ls.is_empty:
        return ls
    xs, ys = zip(*ls.coords)
    X, Y = tf.transform(xs, ys)
    return LineString(list(zip(X, Y)))


# ---------------------------
# Core functionality
# ---------------------------

def fetch_shops_from_point(
    center_point: Tuple[float, float],
    radius_m: int,
    tags: Optional[Dict[str, Iterable[str]]] = None,
):
    """Fetch OSM shop POIs as points within a radius of a center point."""
    # Use a specific set of tags for relevant shops, inspired by the user's notebook.
    if tags is None:
        tags = {
            "shop": ["supermarket", "convenience", "grocery", "general", "cafe", "bakery"],
            "amenity": ["fast_food", "cafe", "vending_machine", "drinking_water", "restaurant", "pub"],
        }

    print(f"Fetching POIs with tags: {tags} within {radius_m}m of {center_point}")

    # Try fetching; if it fails or returns nothing, return empty GeoDataFrame gracefully
    try:
        gdf = _features_from_point_compat(center_point, tags, dist=radius_m)
    except Exception as e:
        print(f"Error in fetch_shops_from_point: {e}")
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    if gdf.empty:
        print("No features found in radius for the given tags.")
        return gdf

    # Adopted from user's notebook: Normalize geometries to points
    pois_points = gdf.copy()

    # For Polygons/MultiPolygons, use centroid
    poly_mask = pois_points.geom_type.isin(["Polygon", "MultiPolygon"])
    pois_points.loc[poly_mask, 'geometry'] = pois_points.loc[poly_mask, 'geometry'].centroid

    # For LineString/MultiLineString, use midpoint
    line_mask = pois_points.geom_type.isin(["LineString", "MultiLineString"])
    midpoints = []
    for g in pois_points.loc[line_mask, 'geometry']:
        if g.geom_type == "LineString":
            midpoints.append(g.interpolate(0.5, normalized=True))
        elif g.geom_type == "MultiLineString" and g.geoms:
            longest_part = max(g.geoms, key=lambda p: p.length)
            midpoints.append(longest_part.interpolate(0.5, normalized=True))
        else:
            midpoints.append(g.centroid)
    pois_points.loc[line_mask, 'geometry'] = midpoints
    
    # Ensure we only have points now
    pois_points = pois_points[pois_points.geom_type == 'Point'].copy()
    
    if pois_points.empty:
        print("No point geometries remaining after processing.")
        return pois_points

    # Rename 'geometry' to 'point_geom' for consistency with downstream functions
    pois_points.rename(columns={'geometry': 'point_geom'}, inplace=True)
    pois_points.set_geometry('point_geom', inplace=True)

    # Keep a useful label
    name_series = pois_points.get("name")
    shop_series = pois_points.get("shop")
    amenity_series = pois_points.get("amenity")
    
    # Fill in labels robustly
    pois_points["label"] = ""
    if name_series is not None:
        pois_points["label"] = name_series.fillna("")
    if shop_series is not None:
        # Ensure we don't overwrite existing names
        pois_points["label"] = pois_points["label"].apply(lambda x: x if x else '').astype(str) + shop_series.apply(lambda x: f" ({x})" if pd.notna(x) else "")
    if amenity_series is not None:
        pois_points["label"] = pois_points["label"].apply(lambda x: x if x else '').astype(str) + amenity_series.apply(lambda x: f" ({x})" if pd.notna(x) else "")
    
    pois_points["label"] = pois_points["label"].str.strip().replace("", "poi")

    print(f"Successfully processed {len(pois_points)} features into points with labels.")
    return pois_points


def fetch_shops_within_bbox(
    north: float,
    south: float,
    east: float,
    west: float,
    tags: Optional[Dict[str, Iterable[str]]] = None,
):
    """Fetch OSM shop POIs as points within a bbox."""
    # Use a specific set of tags for relevant shops, inspired by the user's notebook.
    tags = {
        "shop": ["supermarket", "convenience", "grocery", "general"],
        "amenity": ["fast_food", "cafe", "vending_machine", "drinking_water"],
    }

    print(f"Fetching POIs with tags: {tags} in bbox N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")

    # Try fetching; if it fails or returns nothing, return empty GeoDataFrame gracefully
    try:
        gdf = _features_from_bbox_compat(north, south, east, west, tags)
    except Exception as e:
        print(f"Error in fetch_shops_within_bbox: {e}")
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    if gdf.empty:
        print("No features found in bounding box for the given tags.")
        return gdf

    # Adopted from user's notebook: Normalize geometries to points
    pois_points = gdf.copy()

    # For Polygons/MultiPolygons, use centroid
    poly_mask = pois_points.geom_type.isin(["Polygon", "MultiPolygon"])
    pois_points.loc[poly_mask, 'geometry'] = pois_points.loc[poly_mask, 'geometry'].centroid

    # For LineString/MultiLineString, use midpoint
    line_mask = pois_points.geom_type.isin(["LineString", "MultiLineString"])
    midpoints = []
    for g in pois_points.loc[line_mask, 'geometry']:
        if g.geom_type == "LineString":
            midpoints.append(g.interpolate(0.5, normalized=True))
        elif g.geom_type == "MultiLineString" and g.geoms:
            longest_part = max(g.geoms, key=lambda p: p.length)
            midpoints.append(longest_part.interpolate(0.5, normalized=True))
        else:
            midpoints.append(g.centroid)
    pois_points.loc[line_mask, 'geometry'] = midpoints
    
    # Ensure we only have points now
    pois_points = pois_points[pois_points.geom_type == 'Point'].copy()
    
    if pois_points.empty:
        print("No point geometries remaining after processing.")
        return pois_points

    # Rename 'geometry' to 'point_geom' for consistency with downstream functions
    pois_points.rename(columns={'geometry': 'point_geom'}, inplace=True)
    pois_points.set_geometry('point_geom', inplace=True)

    # Keep a useful label
    name_series = pois_points.get("name")
    shop_series = pois_points.get("shop")
    amenity_series = pois_points.get("amenity")
    
    # Fill in labels robustly
    pois_points["label"] = ""
    if name_series is not None:
        pois_points["label"] = name_series.fillna("")
    if shop_series is not None:
        # Ensure we don't overwrite existing names
        pois_points["label"] = pois_points["label"].apply(lambda x: x if x else '').astype(str) + shop_series.apply(lambda x: f" ({x})" if pd.notna(x) else "")
    if amenity_series is not None:
        pois_points["label"] = pois_points["label"].apply(lambda x: x if x else '').astype(str) + amenity_series.apply(lambda x: f" ({x})" if pd.notna(x) else "")
    
    pois_points["label"] = pois_points["label"].str.strip().replace("", "poi")

    print(f"Successfully processed {len(pois_points)} features into points with labels.")
    return pois_points


def find_shop_near_path(
    G: nx.Graph,
    path_node_ids: List[int], # Expecting node IDs now
    shops_gdf,
    max_dist_m: float = 100.0,
):
    """Return the nearest shop (row) to the path within max_dist_m; else None."""
    if shops_gdf is None or shops_gdf.empty or len(path_node_ids) < 2:
        print("find_shop_near_path: No shops_gdf, empty shops_gdf, or path too short.")
        return None

    path_ls = _path_linestring(G, path_node_ids)
    if path_ls.is_empty:
        print("find_shop_near_path: Path LineString is empty.")
        return None

    # Project to metric space for distance checks
    shop_points = [Point(pt.x, pt.y) for pt in shops_gdf["point_geom"].to_list()]
    
    if not shop_points:
        print("find_shop_near_path: No valid shop points to project.")
        return None
        
    proj_shop_pts, tf = _project_to_local(shop_points)
    proj_path = _project_linestring(path_ls, tf)

    # Compute distances and pick nearest
    dists = np.array([proj_path.distance(p) for p in proj_shop_pts])  # meters in EPSG:3857
    idx_min = int(np.argmin(dists))
    dmin = float(dists[idx_min])

    print(f"find_shop_near_path: Nearest shop is {dmin:.2f} meters from path.")

    if max_dist_m is not None and dmin > max_dist_m:
        print(f"find_shop_near_path: Nearest shop ({dmin:.2f}m) exceeds max_dist_m ({max_dist_m}m).")
        return None

    # Return the original row (in lon/lat)
    return shops_gdf.iloc[idx_min]


def route_via_nearby_shop(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    place: str = "Philadelphia, PA, USA",
    road_tags: Optional[Dict[str, Iterable[str]]] = None,
    shop_tags: Optional[Dict[str, Iterable[str]]] = None,
    max_shop_dist_m: float = 100.0,
    G: Optional[nx.MultiDiGraph] = None, # Expecting MultiDiGraph from build_graph_segments
):
    """Build segmented graph, find shortest path, snap a nearby shop, and re-route."""
    # Build/load the segmented graph over bbox/place. Allow passing a preloaded G to
    # avoid rebuilding the graph repeatedly (faster for API servers).
    if G is None:
        print("Building graph...")
        if bbox is None:
            G = build_graph_segments(place=place, tags=road_tags or DEFAULT_ROAD_TAGS)
            # derive bbox from graph nodes for shop query
            xs = [d["x"] for _, d in G.nodes(data=True)]
            ys = [d["y"] for _, d in G.nodes(data=True)]
            if not xs or not ys:
                raise ValueError("Graph built with no nodes; cannot derive bbox.")
            west, east = min(xs), max(xs)
            south, north = min(ys), max(ys)
        else:
            north, south, east, west = bbox
            G = build_graph_segments(bbox=bbox, tags=road_tags or DEFAULT_ROAD_TAGS)
        print(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    else:
        # If caller provided a graph G, derive a bbox from its nodes unless an explicit
        # bbox was provided.
        if bbox is None:
            xs = [d["x"] for _, d in G.nodes(data=True)]
            ys = [d["y"] for _, d in G.nodes(data=True)]
            if not xs or not ys:
                raise ValueError("Provided graph has no nodes; cannot derive bbox.")
            west, east = min(xs), max(xs)
            south, north = min(ys), max(ys)
        else:
            north, south, east, west = bbox
    
    print(f"Effective BBox for shop query: N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")

    # Snap start/end to graph nodes
    u_node_id = _nearest_node(G, start_lon, start_lat)
    v_node_id = _nearest_node(G, end_lon, end_lat)

    start_node_coords = (G.nodes[u_node_id]['x'], G.nodes[u_node_id]['y'])
    end_node_coords = (G.nodes[v_node_id]['x'], G.nodes[v_node_id]['y'])

    print(f"Start snapped to node {u_node_id} ({start_node_coords[0]:.4f}, {start_node_coords[1]:.4f})")
    print(f"End snapped to node {v_node_id} ({end_node_coords[0]:.4f}, {end_node_coords[1]:.4f})")


    # Base shortest path
    try:
        base_path_node_ids = nx.shortest_path(G, u_node_id, v_node_id, weight="weight")
        print(f"Base path found with {len(base_path_node_ids)} nodes.")
    except nx.NetworkXNoPath:
        print("No path found between start and end nodes.")
        return {
            "graph": G,
            "u": start_node_coords,
            "v": end_node_coords,
            "u_node_id": u_node_id,
            "v_node_id": v_node_id,
            "base_path": [],
            "shop_node_id": None,
            "shop_point": None,
            "shop_label": None,
            "route_uv": [],
            "route_u_shop_v": [],
            "note": "No path found between start and end.",
        }

    # Get shops using the new point-based strategy.
    
    # Calculate midpoint for the search
    center_lon = (start_lon + end_lon) / 2
    center_lat = (start_lat + end_lat) / 2
    center_point = (center_lat, center_lon) # OSMnx expects (lat, lon)

    shops = None
    print("Searching for shops using point-based search around route midpoint...")
    # Use a reasonable search radius, e.g., half the straight-line distance between start and end, but with min/max caps
    geod = Geod(ellps="WGS84")
    dist_m = geod.inv(start_lon, start_lat, end_lon, end_lat)[2]
    radius_m = int(min(max(dist_m / 2, 1000), 5000)) # Search between 1km and 5km radius

    shops = fetch_shops_from_point(
        center_point=center_point,
        radius_m=radius_m,
        tags=shop_tags,
    )
    
    if shops is None or shops.empty:
        print("No shops found via point-based search. Falling back to synthetic waypoint.")


    # --- Determine the waypoint to use ---
    chosen_shop_poi = find_shop_near_path(G, base_path_node_ids, shops, max_dist_m=max_shop_dist_m)
    
    shop_node_id_for_routing = None
    shop_point_for_display = None # This will hold the (lon, lat) of the chosen shop/waypoint
    shop_label_for_display = None
    note = None

    if chosen_shop_poi is not None:
        # A shop was found within max_dist_m
        shop_point_for_display = (chosen_shop_poi["point_geom"].x, chosen_shop_poi["point_geom"].y)
        shop_label_for_display = chosen_shop_poi.get("label", "shop")
        shop_node_id_for_routing = _nearest_node(G, shop_point_for_display[0], shop_point_for_display[1])
        print(f"Chosen shop: {shop_label_for_display} at ({shop_point_for_display[0]:.4f}, {shop_point_for_display[1]:.4f})")
        print(f"Snapped to node {shop_node_id_for_routing} ({(G.nodes[shop_node_id_for_routing]['x']):.4f}, {(G.nodes[shop_node_id_for_routing]['y']):.4f})")
        note = "Route via nearby shop."

    else:
        # No shop found within max_dist_m. Apply fallback logic.
        path_ls = _path_linestring(G, base_path_node_ids)
        if path_ls.is_empty:
            note = "Path geometry empty; cannot determine waypoint."
            # In this case, we won't have a shop point or path
            return {
                "graph": G, "u": start_node_coords, "v": end_node_coords,
                "u_node_id": u_node_id, "v_node_id": v_node_id,
                "base_path": base_path_node_ids, "shop_node_id": None,
                "shop_point": None, "shop_label": None,
                "route_uv": base_path_node_ids, "route_u_shop_v": base_path_node_ids,
                "note": note,
            }

        midpt = path_ls.interpolate(0.5, normalized=True)
        if shops is not None and not shops.empty:
            # Shops exist, but none are close enough. Pick the closest one to the route midpoint.
            print("No shops found within max_dist_m. Choosing closest shop to route midpoint.")
            chosen_fallback_shop = shops.iloc[int(shops["point_geom"].distance(midpt).argmin())]
            shop_point_for_display = (chosen_fallback_shop["point_geom"].x, chosen_fallback_shop["point_geom"].y)
            shop_label_for_display = chosen_fallback_shop.get("label", "shop")
            shop_node_id_for_routing = _nearest_node(G, shop_point_for_display[0], shop_point_for_display[1])
            note = "No shops within max_dist_m; chose closest shop to path midpoint."
            print(f"Fallback shop: {shop_label_for_display} at ({shop_point_for_display[0]:.4f}, {shop_point_for_display[1]:.4f})")
            print(f"Snapped to node {shop_node_id_for_routing} ({(G.nodes[shop_node_id_for_routing]['x']):.4f}, {(G.nodes[shop_node_id_for_routing]['y']):.4f})")
        else:
            # No POIs available at all — make a synthetic waypoint at the route midpoint node
            print("No POIs available at all. Inserting a synthetic waypoint on the route midpoint.")
            shop_node_id_for_routing = _nearest_node(G, float(midpt.x), float(midpt.y))
            shop_point_for_display = (G.nodes[shop_node_id_for_routing]['x'], G.nodes[shop_node_id_for_routing]['y'])
            shop_label_for_display = "synthetic_waypoint"
            note = "No POIs found; inserted a synthetic waypoint on the route."
            print(f"Synthetic waypoint at ({(G.nodes[shop_node_id_for_routing]['x']):.4f}, {(G.nodes[shop_node_id_for_routing]['y']):.4f})")

    # --- Compute the two-leg route via the determined waypoint ---
    path_u_shop_v_ids = base_path_node_ids # Default to base path if waypoint routing fails or no waypoint
    
    if shop_node_id_for_routing is not None:
        try:
            path_u_shop_ids = nx.shortest_path(G, u_node_id, shop_node_id_for_routing, weight="weight")
            path_shop_v_ids = nx.shortest_path(G, shop_node_id_for_routing, v_node_id, weight="weight")
            path_u_shop_v_ids = path_u_shop_ids[:-1] + path_shop_v_ids  # avoid duplicate meeting node
            print(f"Re-routed via waypoint: {len(path_u_shop_v_ids)} nodes.")
        except nx.NetworkXNoPath as e:
            print(f"Could not re-route via waypoint ({shop_label_for_display}): {e}. Using direct path.")
            path_u_shop_v_ids = base_path_node_ids # Fallback to direct path
            note = f"Could not re-route via {shop_label_for_display}. Using direct path. {note or ''}".strip()
    else:
        # If no shop_node_id_for_routing was ever determined (e.g., base path was empty)
        print("No valid waypoint for routing. Using direct path.")


    return {
        "graph": G,
        "u": start_node_coords, # Original start coords (lon, lat)
        "v": end_node_coords,   # Original end coords (lon, lat)
        "u_node_id": u_node_id, # Snapped start node ID
        "v_node_id": v_node_id, # Snapped end node ID
        "base_path": base_path_node_ids, # Node IDs for base path
        "shop_node_id": shop_node_id_for_routing, # Snapped shop node ID for display
        "shop_point": shop_point_for_display, # Original shop/waypoint coords (lon, lat) for display
        "shop_label": shop_label_for_display,
        "route_uv": base_path_node_ids, # Node IDs for direct route
        "route_u_shop_v": path_u_shop_v_ids, # Node IDs for via-waypoint route
        "note": note,
    }

# ---------------------------
# Plotting
# ---------------------------

def _plot_edges(ax, G: nx.Graph, path_node_ids: List[int], lw_factor=250.0, alpha=0.95, color: str = "blue", linestyle: Optional[str] = None):
    """Plot edges for a path using the provided color."""
    if not path_node_ids:
        return
    
    # Get node (lon, lat) tuples from node IDs
    nodes_lonlat = {node_id: (G.nodes[node_id]['x'], G.nodes[node_id]['y']) for node_id in path_node_ids}

    for u_id, v_id in zip(path_node_ids[:-1], path_node_ids[1:]):
        # Retrieve edge data, handling potential multiple edges between same nodes
        edges_data = G.get_edge_data(u_id, v_id)
        if not edges_data:
            continue # No edge found, skip
        
        # If it's a MultiGraph, get the first edge's data; otherwise, it's already an attribute dict
        if G.is_multigraph():
            edge_data = next(iter(edges_data.values()))
        else:
            edge_data = edges_data

        u_lonlat = nodes_lonlat[u_id]
        v_lonlat = nodes_lonlat[v_id]
        
        geom = edge_data.get("geometry")
        w = edge_data.get("weight", 1.0)
        lw = max(0.6, min(3.5, w / lw_factor))
        if isinstance(geom, LineString):
            xs, ys = geom.xy
            ax.plot(xs, ys, linewidth=lw, alpha=alpha, color=color, linestyle=linestyle if linestyle else 'solid')
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                xs, ys = part.xy
                ax.plot(xs, ys, linewidth=lw, alpha=alpha, color=color, linestyle=linestyle if linestyle else 'solid')
        else:
            ax.plot([u_lonlat[0], v_lonlat[0]], [u_lonlat[1], v_lonlat[1]], linewidth=lw, alpha=alpha, color=color, linestyle=linestyle if linestyle else 'solid')


def plot_route_with_waypoint(result, figsize=(10, 10)):
    """Visualize base route and via-shop route with three pointers."""
    G = result["graph"]
    u_lonlat = result["u"] # Original start coords
    v_lonlat = result["v"] # Original end coords
    u_node_id = result["u_node_id"] # Snapped start node ID
    v_node_id = result["v_node_id"] # Snapped end node ID
    shop_node_id = result["shop_node_id"] # Snapped shop node ID
    shop_point_lonlat = result["shop_point"] # Original shop coords
    shop_label = result["shop_label"]

    route_uv_ids = result["route_uv"]
    route_u_shop_v_ids = result["route_u_shop_v"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Draw the via-shop route (blue, prominent)
    _plot_edges(ax, G, route_u_shop_v_ids, lw_factor=220.0, alpha=0.95, color="blue")

    # Overlay the direct route (light gray, dashed)
    _plot_edges(ax, G, route_uv_ids, lw_factor=250.0, alpha=0.6, color="lightgray", linestyle="--")

    # Pointers: start, shop, end
    ax.scatter([G.nodes[u_node_id]['x']], [G.nodes[u_node_id]['y']], s=80, marker="o", color="green", edgecolor="black", zorder=5)      # start
    if shop_node_id is not None and shop_point_lonlat is not None:
        ax.scatter([G.nodes[shop_node_id]['x']], [G.nodes[shop_node_id]['y']], s=100, marker="^", color="red", edgecolor="black", zorder=5)  # shop node
        # Optionally, plot the actual shop point if it's different from the snapped node
        if (abs(shop_point_lonlat[0] - G.nodes[shop_node_id]['x']) > 1e-5 or 
            abs(shop_point_lonlat[1] - G.nodes[shop_node_id]['y']) > 1e-5):
            ax.scatter([shop_point_lonlat[0]], [shop_point_lonlat[1]], s=30, marker="*", color="magenta", zorder=4, label="Original Shop POI")
    ax.scatter([G.nodes[v_node_id]['x']], [G.nodes[v_node_id]['y']], s=80, marker="X", color="purple", edgecolor="black", zorder=5)      # end

    # Labels
    ax.text(G.nodes[u_node_id]['x'], G.nodes[u_node_id]['y'], " Start", fontsize=10, verticalalignment='bottom')
    if shop_node_id is not None:
        ax.text(G.nodes[shop_node_id]['x'], G.nodes[shop_node_id]['y'], f" {shop_label}", fontsize=10, verticalalignment='bottom')
    ax.text(G.nodes[v_node_id]['x'], G.nodes[v_node_id]['y'], " End", fontsize=10, verticalalignment='bottom')

    # Add a title and optional note
    title = "Route via Nearby Shop"
    if result["note"]:
        title += f"\n({result['note']})"
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.show()


# ---------------------------
# CLI
# ---------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Route via a shop near the shortest path.")
    p.add_argument("--start", nargs=2, type=float, metavar=("LON", "LAT"), required=True,
                   help="Starting coordinates (longitude latitude).")
    p.add_argument("--end", nargs=2, type=float, metavar=("LON", "LAT"), required=True,
                   help="Ending coordinates (longitude latitude).")
    p.add_argument("--north", type=float, default=None,
                   help="North boundary of the bounding box.")
    p.add_argument("--south", type=float, default=None,
                   help="South boundary of the bounding box.")
    p.add_argument("--east", type=float, default=None,
                   help="East boundary of the bounding box.")
    p.add_argument("--west", type=float, default=None,
                   help="West boundary of the bounding box.")
    p.add_argument("--place", type=str, default="Philadelphia, PA, USA",
                   help="Place name for graph generation if bbox is not provided.")
    p.add_argument("--shop-dist", type=float, default=100.0,
                   help="Max meters from path to consider a shop. (default: 100.0)")
    p.add_argument("--no-plot", action="store_true",
                   help="Do not display the route plot.")
    return p.parse_args()


def _maybe_bbox_from_args(args):
    if None not in (args.north, args.south, args.east, args.west):
        return (args.north, args.south, args.east, args.west)
    return None


def main():
    args = _parse_args()
    bbox = _maybe_bbox_from_args(args)
    
    try:
        res = route_via_nearby_shop(
            start_lon=args.start[0],
            start_lat=args.start[1],
            end_lon=args.end[0],
            end_lat=args.end[1],
            bbox=bbox,
            place=args.place,
            max_shop_dist_m=args.shop_dist,
        )
        if res["note"]:
            print("[find_shop_waypoint]", res["note"])
        
        if not args.no_plot:
            print("Generating plot...")
            plot_route_with_waypoint(res)
            print("Plot generated.")
        else:
            print("Plotting skipped as requested.")
            # If no plot, print the results in text form
            print("\n--- Route Summary ---")
            print(f"Start: ({res['u'][0]:.4f}, {res['u'][1]:.4f})")
            print(f"End: ({res['v'][0]:.4f}, {res['v'][1]:.4f})")
            print(f"Base path nodes: {len(res['base_path'])}")
            if res['shop_point']:
                print(f"Chosen Shop: {res['shop_label']} at ({res['shop_point'][0]:.4f}, {res['shop_point'][1]:.4f})")
                print(f"Route via shop nodes: {len(res['route_u_shop_v'])}")
            else:
                print("No shop waypoint chosen.")
            print(f"Note: {res['note'] if res['note'] else 'None'}")
            print("---------------------\n")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # For more detailed error info


if __name__ == "__main__":
    main()