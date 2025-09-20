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
    tag_match_rule: str = 'any',
):
    """Fetch OSM shop POIs as points within a radius of a center point."""
    # Use a specific set of tags for relevant shops, inspired by the user's notebook.
    if tags is None:
        tags = {
            "shop": ["supermarket", "convenience", "grocery", "general", "cafe"],
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

    # Filter by tag match rule ('any' or 'all')
    if tag_match_rule == 'all':
        # For 'all', a feature must have a value for each key in the tags dict
        for key, values in tags.items():
            if not values: continue # Skip empty tag lists
            # Drop rows where the key is missing or the value is not in the list of desired values
            gdf = gdf[gdf[key].isin(values)]
    
    if gdf.empty:
        print(f"No features remained after applying '{tag_match_rule}' rule.")
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
    start_node: Tuple[float, float],
    end_node: Tuple[float, float],
    shop_tags: Dict[str, int],
    tag_match_rule: str = 'any',
    max_search_dist_m: int = 2000,
) -> Dict:
    if not G or start_node is None or end_node is None:
        return {"error": "Invalid graph or start/end node."}

    try:
        base_path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight="length")
        base_path_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in base_path_nodes]
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        return {"error": f"Could not find a base path: {e}"}

    if not base_path_coords:
        return {"error": "Base path has no coordinates."}

    # Use the midpoint of the path to search for POIs
    mid_index = len(base_path_coords) // 2
    center_point_lon, center_point_lat = base_path_coords[mid_index]
    
    # Determine search radius based on path length, with a minimum and maximum
    base_path_ls = LineString(base_path_coords)
    # DEPRECATED: search_radius_m = max(200, min(int(base_path_ls.length * 0.1), 2000)) # 10% of path length, capped
    search_radius_m = max_search_dist_m

    all_found_pois = []
    
    tags_to_search = shop_tags
    if not tags_to_search:
        tags_to_search = {'convenience': 1} # Default to one convenience store if no tags are provided

    for tag, count in tags_to_search.items():
        if count <= 0:
            continue
        
        # Fetch POIs for the current tag, searching in both 'shop' and 'amenity' categories
        pois_gdf = fetch_shops_from_point(
            center_point=(center_point_lat, center_point_lon),
            radius_m=search_radius_m,
            tags={
                "shop": [tag],
                "amenity": [tag]
            },
            tag_match_rule='any' # POI can be a shop OR an amenity with the tag
        )

        if pois_gdf.empty:
            print(f"No POIs found for tag '{tag}'")
            continue

        # Find the nearest node in the graph for each POI
        pois_gdf['nearest_node'] = pois_gdf.apply(
            lambda row: _nearest_node(G, row.point_geom.x, row.point_geom.y), axis=1
        )
        pois_gdf = pois_gdf.dropna(subset=['nearest_node'])

        if pois_gdf.empty:
            print(f"No POIs for tag '{tag}' could be snapped to the graph.")
            continue

        # Calculate distance from the base path to each POI's nearest node
        # This is a simplification; a more accurate way would be to calculate path distance
        poi_nodes = [Point(G.nodes[n]['x'], G.nodes[n]['y']) for n in pois_gdf['nearest_node']]
        distances = [base_path_ls.distance(p) for p in poi_nodes]
        pois_gdf['distance_to_path'] = distances
        
        # Sort by distance and take the top 'count'
        pois_gdf = pois_gdf.sort_values(by='distance_to_path').head(count)
        all_found_pois.append(pois_gdf)

    if not all_found_pois:
        return {
            "note": "No shops found matching the criteria.",
            "base_path": base_path_coords,
            "route_u_shop_v": [],
            "shop_points": [],
        }

    final_pois_gdf = pd.concat(all_found_pois)
    
    # Ensure 'osmid' is a column for deduplication
    if 'osmid' not in final_pois_gdf.columns and 'osmid' in final_pois_gdf.index.names:
        final_pois_gdf = final_pois_gdf.reset_index()

    if 'osmid' in final_pois_gdf.columns:
        final_pois_gdf = final_pois_gdf.drop_duplicates(subset=['osmid'])
    else:
        # If no osmid, we cannot reliably deduplicate, so we proceed with what we have
        print("Warning: 'osmid' not found, cannot deduplicate POIs.")
    
    return {
        "base_path_coords": base_path_coords,
        "pois_gdf": final_pois_gdf,
    }


def route_via_nearby_shop(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    shop_tags: Dict[str, int],
    tag_match_rule: str,
    G: nx.Graph,
    max_search_dist_m: int = 2000,
) -> Dict:
    start_node = _nearest_node(G, start_lon, start_lat)
    end_node = _nearest_node(G, end_lon, end_lat)

    if not start_node or not end_node:
        return {"error": "Could not snap start or end points to the road network."}

    # Find candidate POIs
    path_info = find_shop_near_path(G, start_node, end_node, shop_tags, tag_match_rule, max_search_dist_m=max_search_dist_m)
    if "error" in path_info or "pois_gdf" not in path_info or path_info["pois_gdf"].empty:
        return path_info

    base_path_coords = path_info["base_path_coords"]
    pois_gdf = path_info["pois_gdf"]

    waypoints = list(pois_gdf['nearest_node'])
    
    # --- Waypoint Ordering: Insertion Heuristic ---
    # Project waypoints onto the base path and sort them by their order along the path.
    # This prevents the inefficient zig-zagging of a simple nearest-neighbor TSP approx.
    
    try:
        base_path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Fallback to greedy if base path fails for some reason (should be rare)
        base_path_nodes = []

    if base_path_nodes:
        # Create a mapping from each node in the base path to its position (index)
        node_to_pos = {node: i for i, node in enumerate(base_path_nodes)}
        
        # For each waypoint, find the closest node on the base path
        waypoint_projections = {}
        for wp in waypoints:
            min_dist = float('inf')
            closest_node_on_path = None
            # This is a simple projection; could be improved with path distance
            wp_geom = Point(G.nodes[wp]['x'], G.nodes[wp]['y'])
            for node_on_path in base_path_nodes:
                node_geom = Point(G.nodes[node_on_path]['x'], G.nodes[node_on_path]['y'])
                dist = wp_geom.distance(node_geom)
                if dist < min_dist:
                    min_dist = dist
                    closest_node_on_path = node_on_path
            
            if closest_node_on_path:
                waypoint_projections[wp] = node_to_pos[closest_node_on_path]

        # Sort waypoints based on their projected position on the base path
        ordered_waypoints = sorted(waypoints, key=lambda wp: waypoint_projections.get(wp, float('inf')))
    else:
        # Fallback to the old greedy method if no base path was found
        ordered_waypoints = []
        
    # --- Path Construction ---
    # Stitch together the path from start -> sorted_waypoints -> end
    
    path_segments = []
    current_node = start_node
    
    # Path from start to each ordered waypoint
    for waypoint in ordered_waypoints:
        try:
            segment = nx.shortest_path(G, source=current_node, target=waypoint, weight='length')
            path_segments.extend(segment[:-1]) # Avoid duplicating nodes
            current_node = waypoint
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print(f"Warning: Could not find path to waypoint {waypoint}. Skipping.")
            continue

    # Add the final leg from the last waypoint to the destination
    try:
        final_segment = nx.shortest_path(G, source=current_node, target=end_node, weight='length')
        path_segments.extend(final_segment)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"error": "Could not find a path from the last waypoint to the destination."}


    full_via_path_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path_segments]
    
    # Get coordinates and labels for the visited waypoints in order
    ordered_shop_points = []
    for wp_node in ordered_waypoints:
        poi_row = pois_gdf[pois_gdf['nearest_node'] == wp_node].iloc[0]
        point_coords = (poi_row.point_geom.x, poi_row.point_geom.y)
        label = poi_row.label
        ordered_shop_points.append((point_coords, label))

    return {
        "base_path": base_path_coords,
        "route_u_shop_v": full_via_path_coords,
        "shop_points": ordered_shop_points,
        "note": f"Found and routed through {len(ordered_waypoints)} waypoints.",
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
    shop_points_lonlat = result.get("shop_points", []) # List of original shop coords
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
    
    # Plot all waypoints
    for point in shop_points_lonlat:
        # Find the nearest node on the graph to this point for plotting the marker
        node_id = _nearest_node(G, point[0], point[1])
        if node_id:
            ax.scatter([G.nodes[node_id]['x']], [G.nodes[node_id]['y']], s=100, marker="^", color="red", edgecolor="black", zorder=5)
            ax.scatter([point[0]], [point[1]], s=30, marker="*", color="magenta", zorder=4, label="Original POI")

    ax.scatter([G.nodes[v_node_id]['x']], [G.nodes[v_node_id]['y']], s=80, marker="X", color="purple", edgecolor="black", zorder=5)      # end

    # Labels
    ax.text(G.nodes[u_node_id]['x'], G.nodes[u_node_id]['y'], " Start", fontsize=10, verticalalignment='bottom')
    # Labeling multiple waypoints can get crowded, so we'll skip it for now
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