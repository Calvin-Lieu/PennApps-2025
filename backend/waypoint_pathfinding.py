"""
Waypoint Pathfinding Module

This module handles fetching and processing waypoints (water fountains, stores) 
from OpenStreetMap for the PennApps tree routing project.

Author: PennApps 2025 Team
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import json
from pyproj import Geod
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import folium
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
WATER_TAGS = {
    'amenity': ['drinking_water']
}

STORE_TAGS = {
    'shop': ['convenience', 'supermarket', 'grocery', 'beverages']
}

# Downtown Philadelphia bounding box (focused area)
PHILLY_BBOX = (39.97, 39.94, -75.15, -75.17)  # (north, south, east, west)

# Configure OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False


class WaypointFetcher:
    """Handles fetching waypoints from OpenStreetMap."""
    
    def __init__(self, bbox: Tuple[float, float, float, float] = PHILLY_BBOX):
        """
        Initialize the waypoint fetcher.
        
        Args:
            bbox: Bounding box as (north, south, east, west)
        """
        self.bbox = bbox
        self.north, self.south, self.east, self.west = bbox
        
    def fetch_water_fountains(self) -> pd.DataFrame:
        """Fetch water fountains from OpenStreetMap."""
        logger.info("Fetching water fountains...")
        try:
            water_gdf = ox.features_from_bbox(
                bbox=(self.west, self.south, self.east, self.north), 
                tags=WATER_TAGS
            )
            logger.info(f"Found {len(water_gdf)} water fountains")
            return water_gdf
        except Exception as e:
            logger.error(f"Error fetching water fountains: {e}")
            return pd.DataFrame()
    
    def fetch_stores(self) -> pd.DataFrame:
        """Fetch stores from OpenStreetMap."""
        logger.info("Fetching stores...")
        try:
            store_gdf = ox.features_from_bbox(
                bbox=(self.west, self.south, self.east, self.north), 
                tags=STORE_TAGS
            )
            logger.info(f"Found {len(store_gdf)} stores")
            return store_gdf
        except Exception as e:
            logger.error(f"Error fetching stores: {e}")
            return pd.DataFrame()


class WaypointProcessor:
    """Processes waypoint data into structured format."""
    
    @staticmethod
    def extract_waypoint_data(gdf: pd.DataFrame, waypoint_type: str) -> List[Dict[str, Any]]:
        """
        Extract waypoint data from GeoDataFrame into structured format.
        
        Args:
            gdf: GeoDataFrame containing waypoint data
            waypoint_type: Type of waypoint ('water' or 'store')
            
        Returns:
            List of dictionaries containing waypoint information
        """
        waypoints = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom and not geom.is_empty:
                if geom.geom_type == 'Point':
                    lon, lat = geom.x, geom.y
                    # Handle missing name column gracefully
                    name = (row.get('name', None) or 
                           row.get('amenity', None) or 
                           row.get('shop', None) or 
                           f'{waypoint_type}_{idx}')
                    
                    waypoint_data = {
                        'id': f"{waypoint_type}_{idx}",
                        'type': waypoint_type,
                        'name': str(name),
                        'coordinates': [lat, lon],  # [lat, lng] for frontend compatibility
                        'longitude': lon,
                        'latitude': lat,
                        'amenity': str(row.get('amenity', '')),
                        'shop': str(row.get('shop', '')),
                        'opening_hours': str(row.get('opening_hours', '')),
                        'website': str(row.get('website', '')),
                        'phone': str(row.get('phone', ''))
                    }
                    waypoints.append(waypoint_data)
                    
                elif geom.geom_type == 'MultiPoint':
                    for i, point in enumerate(geom.geoms):
                        lon, lat = point.x, point.y
                        name = (row.get('name', None) or 
                               row.get('amenity', None) or 
                               row.get('shop', None) or 
                               f'{waypoint_type}_{idx}_{i}')
                        
                        waypoint_data = {
                            'id': f"{waypoint_type}_{idx}_{i}",
                            'type': waypoint_type,
                            'name': str(name),
                            'coordinates': [lat, lon],
                            'longitude': lon,
                            'latitude': lat,
                            'amenity': str(row.get('amenity', '')),
                            'shop': str(row.get('shop', '')),
                            'opening_hours': str(row.get('opening_hours', '')),
                            'website': str(row.get('website', '')),
                            'phone': str(row.get('phone', ''))
                        }
                        waypoints.append(waypoint_data)
        return waypoints


class WaypointVisualizer:
    """Creates visualizations for waypoints."""
    
    def __init__(self, bbox: Tuple[float, float, float, float] = PHILLY_BBOX):
        """
        Initialize the visualizer.
        
        Args:
            bbox: Bounding box as (north, south, east, west)
        """
        self.bbox = bbox
        self.north, self.south, self.east, self.west = bbox
        
    def create_map(self, water_waypoints: List[Dict], store_waypoints: List[Dict], 
                   output_file: str = 'waypoints_map.html') -> folium.Map:
        """
        Create an interactive map showing all waypoints.
        
        Args:
            water_waypoints: List of water fountain waypoints
            store_waypoints: List of store waypoints
            output_file: Output file path for the map
            
        Returns:
            Folium map object
        """
        logger.info("Creating comprehensive map of all waypoints...")
        
        # Create map centered on Philadelphia
        center_lat = (self.north + self.south) / 2
        center_lon = (self.east + self.west) / 2
        
        philly_map = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=16,
            tiles='OpenStreetMap'
        )
        
        # Add bounding box rectangle to show search area
        bbox_coords = [
            [self.south, self.west],   # SW
            [self.north, self.west],   # NW  
            [self.north, self.east],   # NE
            [self.south, self.east],   # SE
            [self.south, self.west]    # Close rectangle
        ]
        folium.PolyLine(
            bbox_coords, 
            color='red', 
            weight=3, 
            opacity=0.8, 
            popup='Search Area'
        ).add_to(philly_map)
        
        # Add water fountains
        water_count = 0
        for wp in water_waypoints:
            lat, lon = wp['latitude'], wp['longitude']
            name = wp['name']
            amenity = wp['amenity']
            
            popup_text = f"""
            <b>Water Fountain</b><br>
            Name: {name}<br>
            Type: {amenity}<br>
            Coordinates: {lat:.6f}, {lon:.6f}
            """
            
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_text, max_width=200),
                icon=folium.Icon(color='blue', icon='tint', prefix='fa'),
                tooltip=f"Water: {name}"
            ).add_to(philly_map)
            water_count += 1
        
        # Add stores
        store_count = 0
        for wp in store_waypoints:
            lat, lon = wp['latitude'], wp['longitude']
            name = wp['name']
            shop_type = wp['shop']
            
            popup_text = f"""
            <b>Store</b><br>
            Name: {name}<br>
            Type: {shop_type}<br>
            Coordinates: {lat:.6f}, {lon:.6f}
            """
            
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_text, max_width=200),
                icon=folium.Icon(color='green', icon='shopping-cart', prefix='fa'),
                tooltip=f"Store: {name}"
            ).add_to(philly_map)
            store_count += 1
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
             bottom: 50px; left: 50px; width: 200px; height: 120px; 
             background-color: white; border:2px solid grey; z-index:9999; 
             font-size:14px; padding: 10px">
        <p><b>Waypoints Found</b></p>
        <p><i class="fa fa-tint" style="color:blue"></i> Water Fountains: {water_count}</p>
        <p><i class="fa fa-shopping-cart" style="color:green"></i> Stores: {store_count}</p>
        <p><b>Total: {water_count + store_count}</b></p>
        </div>
        '''
        philly_map.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        philly_map.save(output_file)
        logger.info(f"Map saved to {output_file}")
        logger.info(f"Map created with {water_count} water fountains and {store_count} stores")
        
        return philly_map


class WaypointManager:
    """Main class for managing waypoint operations."""
    
    def __init__(self, bbox: Tuple[float, float, float, float] = PHILLY_BBOX):
        """
        Initialize the waypoint manager.
        
        Args:
            bbox: Bounding box as (north, south, east, west)
        """
        self.bbox = bbox
        self.fetcher = WaypointFetcher(bbox)
        self.processor = WaypointProcessor()
        self.visualizer = WaypointVisualizer(bbox)
        
    def fetch_all_waypoints(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Fetch all waypoints (water fountains and stores).
        
        Returns:
            Tuple of (water_waypoints, store_waypoints)
        """
        # Fetch raw data
        water_gdf = self.fetcher.fetch_water_fountains()
        store_gdf = self.fetcher.fetch_stores()
        
        # Process data
        water_waypoints = self.processor.extract_waypoint_data(water_gdf, 'water')
        store_waypoints = self.processor.extract_waypoint_data(store_gdf, 'store')
        
        logger.info(f"Extracted {len(water_waypoints)} water waypoints")
        logger.info(f"Extracted {len(store_waypoints)} store waypoints")
        
        return water_waypoints, store_waypoints
    
    def save_waypoints(self, water_waypoints: List[Dict], store_waypoints: List[Dict], 
                      output_file: str = 'waypoints_data.json') -> None:
        """
        Save waypoints to JSON file.
        
        Args:
            water_waypoints: List of water fountain waypoints
            store_waypoints: List of store waypoints
            output_file: Output file path
        """
        all_waypoints = water_waypoints + store_waypoints
        
        waypoint_data = {
            'metadata': {
                'total_waypoints': len(all_waypoints),
                'water_fountains': len(water_waypoints),
                'stores': len(store_waypoints),
                'bounding_box': {
                    'north': self.bbox[0],
                    'south': self.bbox[1],
                    'east': self.bbox[2],
                    'west': self.bbox[3]
                },
                'area_km2': (self.bbox[0] - self.bbox[1]) * (self.bbox[2] - self.bbox[3]) * 111 * 111
            },
            'waypoints': all_waypoints
        }
        
        with open(output_file, 'w') as f:
            json.dump(waypoint_data, f, indent=2)
        
        logger.info(f"Waypoints saved to {output_file}")
    
    def create_visualization(self, water_waypoints: List[Dict], store_waypoints: List[Dict], 
                           output_file: str = 'waypoints_map.html') -> folium.Map:
        """
        Create and save waypoint visualization.
        
        Args:
            water_waypoints: List of water fountain waypoints
            store_waypoints: List of store waypoints
            output_file: Output file path for the map
            
        Returns:
            Folium map object
        """
        return self.visualizer.create_map(water_waypoints, store_waypoints, output_file)


def main():
    """Main function to run the waypoint pathfinding system."""
    logger.info("Starting waypoint pathfinding system...")
    
    # Initialize manager
    manager = WaypointManager()
    
    # Fetch all waypoints
    water_waypoints, store_waypoints = manager.fetch_all_waypoints()
    
    # Save data
    manager.save_waypoints(water_waypoints, store_waypoints)
    
    # Create visualization
    manager.create_visualization(water_waypoints, store_waypoints)
    
    logger.info("Waypoint pathfinding system completed successfully!")
    
    return water_waypoints, store_waypoints


if __name__ == "__main__":
    water_waypoints, store_waypoints = main()
