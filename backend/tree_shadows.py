"""
Tree Shadow Generator for PennApps Route Planning App

This module generates circular shadow polygons for trees based on density data.
Filters trees by density >= 0.2 and maps density [0.2, 1.0] to shadow radius [1m, 5m] linearly.
"""

import json
import math
import os
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TreeShadowGenerator:
    """
    Generates circular shadow polygons for trees based on density data.
    
    Features:
    - Loads tree data from tree_positions.json
    - Filters trees by density >= 0.2
    - Maps density [0.2, 1.0] to shadow radius [1m, 5m] linearly
    - Generates circular polygons with 16 points each
    - Returns GeoJSON FeatureCollection format
    """
    
    def __init__(self, tree_data_path: str):
        """
        Initialize the TreeShadowGenerator.
        
        Args:
            tree_data_path: Path to tree_positions.json file
        """
        self.tree_data_path = tree_data_path
        self.trees: List[Dict[str, Any]] = []
        self.filtered_trees: List[Dict[str, Any]] = []
        self.shadow_polygons: List[Dict[str, Any]] = []
        
        # Constants for coordinate conversion (approximation)
        self.METERS_PER_DEGREE_LAT = 111000  # ~111km per degree latitude
        self.MIN_DENSITY = 0.2
        self.MAX_DENSITY = 1.0
        self.MIN_RADIUS = 8.0  # meters - increased for better visibility
        self.MAX_RADIUS = 25.0  # meters - much larger for route impact
        self.POLYGON_POINTS = 24  # more points for smoother curves
        
        logger.info(f"TreeShadowGenerator initialized with data path: {tree_data_path}")
    
    def load_tree_data(self) -> None:
        """Load tree data from JSON file."""
        try:
            if not os.path.exists(self.tree_data_path):
                raise FileNotFoundError(f"Tree data file not found: {self.tree_data_path}")
            
            with open(self.tree_data_path, 'r') as f:
                data = json.load(f)
            
            self.trees = data.get('trees', [])
            total_trees = len(self.trees)
            
            logger.info(f"Loaded {total_trees} trees from {self.tree_data_path}")
            
            if total_trees == 0:
                logger.warning("No trees found in the data file")
            
        except Exception as e:
            logger.error(f"Error loading tree data: {e}")
            raise
    
    def filter_trees_by_density(self) -> None:
        """Filter trees by density >= 0.2."""
        if not self.trees:
            logger.warning("No trees loaded. Call load_tree_data() first.")
            return
        
        self.filtered_trees = [
            tree for tree in self.trees 
            if tree.get('density', 0) >= self.MIN_DENSITY
        ]
        
        original_count = len(self.trees)
        filtered_count = len(self.filtered_trees)
        
        logger.info(f"Filtered {original_count} trees to {filtered_count} trees with density >= {self.MIN_DENSITY}")
        logger.info(f"Filtering removed {original_count - filtered_count} trees ({((original_count - filtered_count) / original_count * 100):.1f}%)")
    
    def density_to_radius(self, density: float) -> float:
        """
        Map tree density [0.2, 1.0] to shadow radius [1m, 5m] linearly.
        
        Args:
            density: Tree density value
            
        Returns:
            Shadow radius in meters
        """
        # Clamp density to valid range
        density = max(self.MIN_DENSITY, min(self.MAX_DENSITY, density))
        
        # Linear mapping: radius = min_radius + (density - min_density) * (max_radius - min_radius) / (max_density - min_density)
        normalized_density = (density - self.MIN_DENSITY) / (self.MAX_DENSITY - self.MIN_DENSITY)
        radius = self.MIN_RADIUS + normalized_density * (self.MAX_RADIUS - self.MIN_RADIUS)
        
        return radius
    
    def meters_to_degrees(self, meters: float, latitude: float) -> tuple[float, float]:
        """
        Convert meters to degrees for latitude and longitude.
        
        Args:
            meters: Distance in meters
            latitude: Latitude for longitude conversion
            
        Returns:
            Tuple of (lat_degrees, lng_degrees)
        """
        # Latitude: 1 degree ≈ 111,000 meters (constant)
        lat_degrees = meters / self.METERS_PER_DEGREE_LAT
        
        # Longitude: varies by latitude, 1 degree ≈ 111,000 * cos(latitude) meters
        lng_degrees = meters / (self.METERS_PER_DEGREE_LAT * math.cos(math.radians(latitude)))
        
        return lat_degrees, lng_degrees
    
    def generate_organic_tree_canopy(self, center_lat: float, center_lng: float, radius_meters: float, tree_id: int = 0) -> List[List[float]]:
        """
        Generate an organic, tree canopy-like polygon with wavy edges.
        
        Args:
            center_lat: Center latitude
            center_lng: Center longitude
            radius_meters: Base radius in meters
            tree_id: Tree ID for consistent randomization
            
        Returns:
            List of [lng, lat] coordinate pairs forming an organic polygon
        """
        coordinates = []
        
        # Convert radius to degrees
        lat_degrees, lng_degrees = self.meters_to_degrees(radius_meters, center_lat)
        
        # Use tree_id as seed for consistent but varied shapes per tree
        import random
        random.seed(tree_id)
        
        # Generate multiple frequency components for organic variation
        # Primary wave (large lobes)
        primary_freq = random.uniform(3, 7)  # 3-7 major lobes
        primary_amplitude = random.uniform(0.2, 0.4)  # 20-40% radius variation
        
        # Secondary wave (smaller bumps)
        secondary_freq = random.uniform(8, 16)  # 8-16 smaller variations
        secondary_amplitude = random.uniform(0.1, 0.2)  # 10-20% radius variation
        
        # Tertiary wave (fine detail)
        tertiary_freq = random.uniform(20, 32)  # 20-32 fine details
        tertiary_amplitude = random.uniform(0.05, 0.1)  # 5-10% radius variation
        
        # Generate points around the organic shape
        for i in range(self.POLYGON_POINTS):
            angle = 2 * math.pi * i / self.POLYGON_POINTS
            
            # Calculate organic radius variation using multiple sine waves
            primary_variation = math.sin(primary_freq * angle) * primary_amplitude
            secondary_variation = math.sin(secondary_freq * angle) * secondary_amplitude
            tertiary_variation = math.sin(tertiary_freq * angle) * tertiary_amplitude
            
            # Combine variations for organic shape (always positive radius)
            radius_multiplier = 1.0 + primary_variation + secondary_variation + tertiary_variation
            radius_multiplier = max(0.3, radius_multiplier)  # Ensure minimum 30% of base radius
            
            # Apply organic radius to base coordinates
            organic_lat_degrees = lat_degrees * radius_multiplier
            organic_lng_degrees = lng_degrees * radius_multiplier
            
            # Calculate point coordinates
            point_lat = center_lat + organic_lat_degrees * math.sin(angle)
            point_lng = center_lng + organic_lng_degrees * math.cos(angle)
            
            # GeoJSON uses [longitude, latitude] format
            coordinates.append([point_lng, point_lat])
        
        # Close the polygon by repeating the first point
        coordinates.append(coordinates[0])
        
        return coordinates
    
    def generate_shadow_polygons(self) -> None:
        """Generate organic tree canopy shadow polygons for all filtered trees."""
        if not self.filtered_trees:
            logger.warning("No filtered trees available. Call filter_trees_by_density() first.")
            return
        
        self.shadow_polygons = []
        
        for tree in self.filtered_trees:
            try:
                tree_id = tree.get('id')
                latitude = tree.get('latitude')
                longitude = tree.get('longitude')
                density = tree.get('density', 0)
                
                # Validate required fields
                if any(x is None for x in [tree_id, latitude, longitude]):
                    logger.warning(f"Skipping tree with missing data: {tree}")
                    continue
                
                # Calculate shadow radius based on density
                radius = self.density_to_radius(density)
                
                # Generate organic tree canopy polygon
                polygon_coords = self.generate_organic_tree_canopy(latitude, longitude, radius, tree_id)
                
                # Create GeoJSON feature
                feature = {
                    "type": "Feature",
                    "properties": {
                        "id": tree_id,
                        "tree_id": tree_id,  # Include both for compatibility
                        "latitude": latitude,
                        "longitude": longitude,
                        "density": density,
                        "shadow_radius_m": round(radius, 2),
                        "type": "tree_shadow"
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon_coords]
                    }
                }
                
                self.shadow_polygons.append(feature)
                
            except Exception as e:
                logger.error(f"Error generating shadow for tree {tree.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Generated {len(self.shadow_polygons)} shadow polygons")
    
    def get_geojson_feature_collection(self) -> Dict[str, Any]:
        """
        Get shadow polygons as GeoJSON FeatureCollection.
        
        Returns:
            GeoJSON FeatureCollection with tree shadow polygons
        """
        if not self.shadow_polygons:
            logger.warning("No shadow polygons generated. Call generate_shadow_polygons() first.")
        
        return {
            "type": "FeatureCollection",
            "properties": {
                "generated_at": "runtime",
                "total_features": len(self.shadow_polygons),
                "description": "Tree shadow polygons for route planning",
                "density_filter": f">= {self.MIN_DENSITY}",
                "radius_mapping": f"density [{self.MIN_DENSITY}, {self.MAX_DENSITY}] -> radius [{self.MIN_RADIUS}m, {self.MAX_RADIUS}m]",
                "polygon_points": self.POLYGON_POINTS,
                "shape_type": "organic_tree_canopy"
            },
            "features": self.shadow_polygons
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the tree shadow generation.
        
        Returns:
            Dictionary with generation statistics
        """
        if not self.trees:
            return {"error": "No tree data loaded"}
        
        # Calculate density distribution
        density_values = [tree.get('density', 0) for tree in self.filtered_trees]
        radius_values = [self.density_to_radius(d) for d in density_values]
        
        stats = {
            "total_trees_loaded": len(self.trees),
            "trees_after_density_filter": len(self.filtered_trees),
            "shadow_polygons_generated": len(self.shadow_polygons),
            "filter_criteria": {
                "min_density": self.MIN_DENSITY,
                "trees_filtered_out": len(self.trees) - len(self.filtered_trees)
            },
            "radius_mapping": {
                "min_radius_m": self.MIN_RADIUS,
                "max_radius_m": self.MAX_RADIUS,
                "polygon_points": self.POLYGON_POINTS
            }
        }
        
        if density_values:
            stats["density_statistics"] = {
                "min": round(min(density_values), 3),
                "max": round(max(density_values), 3),
                "avg": round(sum(density_values) / len(density_values), 3)
            }
        
        if radius_values:
            stats["radius_statistics"] = {
                "min_m": round(min(radius_values), 2),
                "max_m": round(max(radius_values), 2),
                "avg_m": round(sum(radius_values) / len(radius_values), 2)
            }
        
        return stats
    
    def process_all(self) -> Dict[str, Any]:
        """
        Complete processing pipeline: load, filter, generate, and return GeoJSON.
        
        Returns:
            GeoJSON FeatureCollection with tree shadow polygons
        """
        try:
            logger.info("Starting complete tree shadow processing pipeline")
            
            # Step 1: Load tree data
            self.load_tree_data()
            
            # Step 2: Filter by density
            self.filter_trees_by_density()
            
            # Step 3: Generate shadow polygons
            self.generate_shadow_polygons()
            
            # Step 4: Return GeoJSON
            geojson = self.get_geojson_feature_collection()
            
            # Log statistics
            stats = self.get_statistics()
            logger.info(f"Processing complete: {stats}")
            
            return geojson
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise


# Global instance for the app
_tree_shadow_generator: Optional[TreeShadowGenerator] = None

def get_tree_shadow_generator(tree_data_path: str) -> TreeShadowGenerator:
    """
    Get or create the global TreeShadowGenerator instance.
    
    Args:
        tree_data_path: Path to tree_positions.json file
        
    Returns:
        TreeShadowGenerator instance
    """
    global _tree_shadow_generator
    
    if _tree_shadow_generator is None:
        _tree_shadow_generator = TreeShadowGenerator(tree_data_path)
        logger.info("Created new TreeShadowGenerator instance")
    
    return _tree_shadow_generator

def precompute_tree_shadows(tree_data_path: str) -> Dict[str, Any]:
    """
    Precompute tree shadows for server startup.
    
    Args:
        tree_data_path: Path to tree_positions.json file
        
    Returns:
        GeoJSON FeatureCollection with precomputed shadows
    """
    try:
        generator = get_tree_shadow_generator(tree_data_path)
        return generator.process_all()
    except Exception as e:
        logger.error(f"Failed to precompute tree shadows: {e}")
        raise
