"""
Build a NetworkX graph where each LineString segment becomes its own edge.

Compared to build_graph.py (one edge per full geometry), this creates finer
granularity by splitting each LineString (and each part of a MultiLineString)
into edges between consecutive coordinate pairs.

CLI:
    python -m backend.build_graph_2
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import os
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, MultiLineString, base
from pyproj import Geod
import matplotlib.pyplot as plt
import pickle


def _features_from_bbox_compat(north, south, east, west, tags=None):
    """Compatibility wrapper supporting OSMnx v1.x and v2.x.
    - v2.x: ox.features.features_from_bbox
    - v1.x: ox.geometries_from_bbox
    Handles differing call signatures and bbox tuple ordering.
    """
    # Try v2.x API first
    try:
        return ox.features.features_from_bbox(north, south, east, west, tags)
    except AttributeError:
        # features submodule not available (likely v1.x) → use geometries_from_bbox
        try:
            return ox.geometries_from_bbox(north, south, east, west, tags)
        except TypeError:
            # Some builds require tags as a named argument
            return ox.geometries_from_bbox(north, south, east, west, tags=tags)
    except TypeError:
        # Some OSMnx v2 builds expect a bbox tuple in (south, west, north, east)
        try:
            return ox.features.features_from_bbox((south, west, north, east), tags)
        except AttributeError:
            # v1.x fallback
            try:
                return ox.geometries_from_bbox(north, south, east, west, tags)
            except TypeError:
                return ox.geometries_from_bbox(north, south, east, west, tags=tags)

DEFAULT_ROAD_TAGS: Dict[str, Iterable[str]] = {
    "highway": [
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "service",
    ]
}


def _geodesic_length_m(ls: LineString, geod: Geod) -> float:
    if ls.is_empty:
        return 0.0
    lons, lats = zip(*list(ls.coords))
    return float(geod.line_length(lons, lats))


def _add_segment_edge(
    G: nx.Graph,
    a: Tuple[float, float],
    b: Tuple[float, float],
    geod: Geod,
    attrs: Optional[dict] = None,
) -> None:
    # Round to keep node identity stable and avoid tiny duplicates
    u = (round(a[0], 6), round(a[1], 6))
    v = (round(b[0], 6), round(b[1], 6))
    if u == v:
        return

    if u not in G:
        G.add_node(u, x=u[0], y=u[1])
    if v not in G:
        G.add_node(v, x=v[0], y=v[1])

    seg = LineString([u, v])
    length_m = _geodesic_length_m(seg, geod)
    edge_attrs = {"weight": length_m, "geometry": seg}
    if attrs:
        edge_attrs.update(attrs)
    G.add_edge(u, v, **edge_attrs)


def build_graph_segments(
    place: str = "Philadelphia",
    bbox: Optional[Tuple[float, float, float, float]] = (40.0, 39.9, -75.1, -75.2),
    tags: Optional[Dict[str, Iterable[str]]] = None,
) -> nx.Graph:
    """Build a graph with one edge per LineString segment.

    Args:
        place: Place name (used only if bbox is None to derive bounds).
        bbox: (north, south, east, west). Defaults to a small area in Philadelphia.
        tags: OSM tag filter. Defaults to a highway set matching build_graph.py.
    """
    print("[build_graph_2] Start building segmented graph...")
    if tags is None:
        tags = DEFAULT_ROAD_TAGS
    print(f"[build_graph_2] Using tags: {list(tags.keys())}")

    if bbox is None:
        print(f"[build_graph_2] bbox not provided. Geocoding '{place}'...")
        # OSMnx v2.x: use geocoder submodule
        city = ox.geocoder.geocode_to_gdf(place)
        minx, miny, maxx, maxy = city.total_bounds
        bbox = (maxy, miny, maxx, minx)
        print(
            f"[build_graph_2] Derived bbox: (north={bbox[0]}, south={bbox[1]}, east={bbox[2]}, west={bbox[3]})"
        )
    else:
        print(
            f"[build_graph_2] Using provided bbox: (north={bbox[0]}, south={bbox[1]}, east={bbox[2]}, west={bbox[3]})"
        )

    north, south, east, west = bbox
    print("[build_graph_2] Fetching OSM geometries...")
    # OSMnx v2.x: use features submodule (compat wrapper handles signature differences)
    gdf = _features_from_bbox_compat(north, south, east, west, tags)
    print(f"[build_graph_2] Fetched {len(gdf)} features")

    G = nx.Graph()
    geod = Geod(ellps="WGS84")
    seg_edges = 0
    skipped = 0

    for i, (_, row) in enumerate(gdf.iterrows(), start=1):
        geom: base.BaseGeometry = row.get("geometry")
        if geom is None or geom.is_empty:
            skipped += 1
            continue

        # Attributes to propagate
        attr = {}
        for k in ("highway", "name", "oneway", "maxspeed"):
            if k in row and row[k] is not None:
                attr[k] = row[k]

        def segmentize(ls: LineString):
            coords = list(ls.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                _add_segment_edge(G, a, b, geod, attrs=attr)
                nonlocal seg_edges
                seg_edges += 1

        if isinstance(geom, LineString):
            segmentize(geom)
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                if isinstance(part, LineString) and not part.is_empty:
                    segmentize(part)
        else:
            skipped += 1

        if i % 500 == 0:
            print(
                f"[build_graph_2] Processed {i} features... nodes={G.number_of_nodes()} edges={G.number_of_edges()}"
            )

    total_len_m = sum(d.get("weight", 0.0) for _, _, d in G.edges(data=True))
    print("[build_graph_2] Build complete.")
    print(
        f"[build_graph_2] Nodes={G.number_of_nodes()} Edges={G.number_of_edges()} SegEdges={seg_edges} TotalLen={total_len_m/1000:.2f} km"
    )
    return G


def save_graph_gpickle(G: nx.Graph, path: str) -> None:
    """Save a NetworkX graph to a gpickle file.

    Args:
        G: Graph to save.
        path: Output file path (e.g., "backend/graph.gpickle").
    """
    # gpickle.write_gpickle(G, path)
    with open(path, "wb") as f:
        pickle.dump(G, f)

def plot_graph_matplotlib(
    G: nx.Graph,
    figsize: Tuple[int, int] = (10, 10),
    edge_color: str = "steelblue",
    weight_scale: float = 200.0,
    alpha: float = 0.9,
    node_size: float = 3.0,
    node_color: str = "black",
    node_alpha: float = 0.7,
) -> None:
    """Plot a NetworkX graph using matplotlib.

    - Uses `geometry` on edges if present (preferred for curvy edges).
    - Falls back to straight segments using node coordinates `x` (lon), `y` (lat).
    - Scales line width by `weight/weight_scale` and clamps to [0.3, 3.0].
    - Plots nodes as small dots using stored `x` (lon) and `y` (lat) attributes.
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()

    has_geom = any("geometry" in d for _, _, d in G.edges(data=True))

    if has_geom:
        for u, v, d in G.edges(data=True):
            geom = d.get("geometry")
            w = d.get("weight", 1.0)
            lw = max(0.3, min(3.0, w / weight_scale))
            if isinstance(geom, LineString):
                xs, ys = geom.xy
                ax.plot(xs, ys, color=edge_color, linewidth=lw, alpha=alpha)
            elif isinstance(geom, MultiLineString):
                for part in geom.geoms:
                    xs, ys = part.xy
                    ax.plot(xs, ys, color=edge_color, linewidth=lw, alpha=alpha)
            else:
                # Fallback to straight line between nodes
                ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
                vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
                ax.plot([ux, vx], [uy, vy], color=edge_color, linewidth=lw, alpha=alpha)
    else:
        for u, v, d in G.edges(data=True):
            ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
            vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
            w = d.get("weight", 1.0)
            lw = max(0.3, min(3.0, w / weight_scale))
            ax.plot([ux, vx], [uy, vy], color=edge_color, linewidth=lw, alpha=alpha)

    # Plot nodes as dots
    try:
        xs = [data["x"] for _, data in G.nodes(data=True)]
        ys = [data["y"] for _, data in G.nodes(data=True)]
        ax.scatter(xs, ys, s=node_size, c=node_color, alpha=node_alpha, zorder=3)
    except KeyError:
        # If some nodes lack x/y, skip node plotting
        pass

    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    G = build_graph_segments()
    out_path = os.path.join(os.path.dirname(__file__), "graph_segments.gpickle")
    try:
        save_graph_gpickle(G, out_path)
        print(f"[build_graph_2] Saved segmented graph to: {out_path}")
    except Exception as e:
        print(f"[build_graph_2] Failed to save: {e}")
    plot_graph_matplotlib(G)