#!/usr/bin/env python3
"""
Tree Visualization Script

This script loads tree data from JSON and creates a visualization map.
Usage: python visualize_trees.py [json_file] [output_html]
"""

import json
import pandas as pd
import folium
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Downtown Philadelphia bounding box
PHILLY_BBOX = (39.97, 39.94, -75.15, -75.17)

def load_trees_from_json(json_file: str) -> pd.DataFrame:
    """Load tree data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    trees = []
    for tree in data['trees']:
        trees.append({
            'latitude': tree['latitude'],
            'longitude': tree['longitude'],
            'density': tree['density'],
            'source': 'loaded_from_json'
        })
    
    return pd.DataFrame(trees)

def create_tree_visualization(trees_df: pd.DataFrame, output_file: str = 'tree_visualization.html') -> None:
    """Create a visualization of the trees."""
    logger.info(f"Creating visualization for {len(trees_df)} trees...")
    
    # Create map centered on downtown Philadelphia
    center_lat = (PHILLY_BBOX[0] + PHILLY_BBOX[1]) / 2
    center_lon = (PHILLY_BBOX[2] + PHILLY_BBOX[3]) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles=None)
    
    # Add satellite and street map layers
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', 
        name='Satellite'
    ).add_to(m)
    folium.TileLayer(tiles='OpenStreetMap', name='Street Map').add_to(m)
    
    # Add bounding box rectangle
    folium.Rectangle(
        bounds=[[PHILLY_BBOX[1], PHILLY_BBOX[3]], [PHILLY_BBOX[0], PHILLY_BBOX[2]]], 
        color='red', 
        fill=False, 
        weight=3
    ).add_to(m)
    
    # Add trees with density-based coloring
    for idx, tree in trees_df.iterrows():
        density = tree['density']
        
        # Color-code by density
        if density > 0.8:
            color, size = 'darkred', 5
        elif density > 0.6:
            color, size = 'red', 4
        elif density > 0.4:
            color, size = 'orange', 3
        elif density > 0.3:
            color, size = 'blue', 2
        else:
            color, size = 'green', 2
        
        folium.CircleMarker(
            location=[tree['latitude'], tree['longitude']],
            radius=size,
            popup=f"Tree {idx}<br>Density: {density:.3f}<br>Grid: 5m",
            color=color,
            fill=True,
            fillOpacity=0.8
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    m.save(output_file)
    logger.info(f"Tree visualization saved to {output_file}")

def main():
    """Main function."""
    # Get input and output files from command line arguments
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'tree_positions.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'tree_visualization.html'
    
    try:
        # Load tree data
        trees_df = load_trees_from_json(json_file)
        logger.info(f"Loaded {len(trees_df)} trees from {json_file}")
        
        # Create visualization
        create_tree_visualization(trees_df, output_file)
        
        print(f"\n✓ Visualization complete!")
        print(f"  - Trees loaded: {len(trees_df)}")
        print(f"  - Output file: {output_file}")
        print(f"  - Open the HTML file in your browser to view the map")
        
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file}")
        print(f"Usage: python visualize_trees.py [json_file] [output_html]")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
