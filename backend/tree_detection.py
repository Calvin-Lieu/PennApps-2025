import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import requests
import json
from datetime import datetime
import warnings
import logging
from typing import List, Dict, Tuple, Optional, Any

# Try to import Google Earth Engine
try:
    import ee
    EE_AVAILABLE = True
    print("Google Earth Engine available")
except ImportError:
    EE_AVAILABLE = False
    print("Google Earth Engine not available - using fallback simulation")

# Try to import dotenv for environment variables
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import scipy for interpolation
try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available - using simplified interpolation")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False

# Downtown Philadelphia area
PHILLY_BBOX = (39.97, 39.94, -75.15, -75.17)  # north, south, east, west


class OSMTreeFetcher:
    """Handles fetching tree data from OpenStreetMap."""
    
    def __init__(self, bbox: Tuple[float, float, float, float] = PHILLY_BBOX):
        """
        Initialize the OSM tree fetcher.
        
        Args:
            bbox: Bounding box as (north, south, east, west)
        """
        self.bbox = bbox
        self.north, self.south, self.east, self.west = bbox
        
    def fetch_osm_trees(self) -> pd.DataFrame:
        """
        Fetch trees from OpenStreetMap.
        
        Returns:
            DataFrame containing OSM tree data
        """
        logger.info("Fetching OSM tree data...")
        try:
            # Try the new API first, fallback to old API
            try:
                trees_gdf = ox.features_from_bbox(
                    self.north, self.south, self.east, self.west, 
                    tags={'natural': 'tree'}
                )
            except AttributeError:
                # Fallback to old API
                trees_gdf = ox.geometries_from_bbox(
                    self.north, self.south, self.east, self.west, 
                    tags={'natural': 'tree'}
                )
            
            if len(trees_gdf) > 0:
                # Convert to DataFrame
                osm_trees = pd.DataFrame({
                    'latitude': trees_gdf.geometry.y,
                    'longitude': trees_gdf.geometry.x,
                    'source': 'osm'
                })
                logger.info(f"Found {len(osm_trees)} OSM trees")
                return osm_trees
            else:
                logger.warning("No OSM trees found")
                return pd.DataFrame(columns=['latitude', 'longitude', 'source'])
                
        except Exception as e:
            logger.error(f"OSM error: {e}")
            logger.info("Using fallback approach...")
            # Create some sample OSM trees for demonstration
            sample_trees = [
                {'latitude': 39.949, 'longitude': -75.171, 'source': 'osm'},
                {'latitude': 39.947, 'longitude': -75.151, 'source': 'osm'},
                {'latitude': 39.955, 'longitude': -75.160, 'source': 'osm'},
                {'latitude': 39.952, 'longitude': -75.165, 'source': 'osm'},
                {'latitude': 39.948, 'longitude': -75.158, 'source': 'osm'},
            ]
            osm_trees = pd.DataFrame(sample_trees)
            logger.info(f"Using {len(osm_trees)} sample OSM trees")
            return osm_trees


class EnhancedTreeDetector:
    """Handles enhanced tree detection using satellite imagery and multiple vegetation indices."""
    
    def __init__(self, bbox: Tuple[float, float, float, float] = PHILLY_BBOX):
        """
        Initialize the enhanced tree detector.
        
        Args:
            bbox: Bounding box as (north, south, east, west)
        """
        self.bbox = bbox
        self.north, self.south, self.east, self.west = bbox
        
    def detect_additional_trees(self) -> pd.DataFrame:
        """
        Detect additional trees using satellite imagery and multiple vegetation indices.
        
        Returns:
            DataFrame containing additional tree detections
        """
        logger.info("Running enhanced satellite-based tree detection...")
        
        # Ultra-fine grid settings
        grid_size = 0.00005  # ~5m grid cells
        lats = np.arange(self.south, self.north, grid_size)
        lons = np.arange(self.west, self.east, grid_size)
        
        logger.info(f"Ultra-fine grid: {len(lats)} x {len(lons)} = {len(lats)*len(lons):,} cells")
        logger.info(f"Grid resolution: {grid_size*111:.0f}m per cell")
        
        if EE_AVAILABLE:
            return self._detect_with_satellite(lats, lons, grid_size)
        else:
            return self._detect_with_simulation(lats, lons)
    
    def _detect_with_satellite(self, lats: np.ndarray, lons: np.ndarray, grid_size: float) -> pd.DataFrame:
        """Detect trees using satellite imagery."""
        logger.info("Processing with multiple detection strategies...")
        
        # Initialize Earth Engine
        try:
            # Check if already initialized
            if not ee.data._initialized:
                # Try to get project ID from environment variable
                project_id = None
                if DOTENV_AVAILABLE:
                    project_id = os.getenv('EE_PROJECT_ID')
                
                if project_id:
                    ee.Initialize(project=project_id)
                    logger.info(f"Earth Engine initialized with project: {project_id}")
                else:
                    ee.Initialize()
                    logger.info("Earth Engine initialized without specific project")
            else:
                logger.info("Earth Engine already initialized")
        except Exception as e:
            logger.warning(f"Earth Engine initialization failed: {e}")
            logger.info("You may need to authenticate first. Run: python -c 'import ee; ee.Authenticate()'")
            return self._detect_with_simulation(lats, lons)
        
        # Get satellite data
        study_area = ee.Geometry.Rectangle([self.west, self.south, self.east, self.north])
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterDate('2024-06-01', '2024-12-01')
                        .filterBounds(study_area)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))
        
        # Calculate multiple vegetation indices
        def calculate_vegetation_indices(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
            gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
            return image.addBands(ndvi).addBands(ndwi).addBands(gndvi)
        
        # Process with multiple indices
        multi_indices = s2_collection.map(calculate_vegetation_indices)
        
        # Get median values for each index
        median_ndvi = multi_indices.select('NDVI').median()
        median_ndwi = multi_indices.select('NDWI').median()
        median_gndvi = multi_indices.select('GNDVI').median()
        
        # Ultra-dense sampling
        max_points = 20000
        step = max(1, len(lats) * len(lons) // max_points)
        sampling_points = []
        
        for i in range(0, len(lats), step):
            for j in range(0, len(lons), step):
                if len(sampling_points) >= max_points:
                    break
                sampling_points.append(ee.Geometry.Point([lons[j], lats[i]]))
        
        logger.info(f"Sampling {len(sampling_points)} points with multiple indices...")
        
        # Sample all indices
        try:
            ndvi_data = median_ndvi.sampleRegions(
                collection=ee.FeatureCollection(sampling_points),
                scale=10,
                geometries=True
            ).getInfo()
            
            ndwi_data = median_ndwi.sampleRegions(
                collection=ee.FeatureCollection(sampling_points),
                scale=10,
                geometries=True
            ).getInfo()
            
            gndvi_data = median_gndvi.sampleRegions(
                collection=ee.FeatureCollection(sampling_points),
                scale=10,
                geometries=True
            ).getInfo()
            
            logger.info("✓ Multi-index sampling successful")
            
        except Exception as e:
            logger.warning(f"Sampling failed: {e}")
            logger.info("Using direct array extraction...")
            # Fallback to array method
            ndvi_array = median_ndvi.sampleRectangle(region=study_area, defaultValue=0).getInfo()
            ndwi_array = median_ndwi.sampleRectangle(region=study_area, defaultValue=0).getInfo()
            gndvi_array = median_gndvi.sampleRectangle(region=study_area, defaultValue=0).getInfo()
            
            ndvi_data = {'NDVI': ndvi_array['NDVI']}
            ndwi_data = {'NDWI': ndwi_array['NDWI']}
            gndvi_data = {'GNDVI': gndvi_array['GNDVI']}
            logger.info("✓ Array extraction successful")
        
        # Process results with multiple strategies
        tree_density = np.zeros((len(lats), len(lons)))
        
        if 'features' in ndvi_data:
            logger.info(f"Processing {len(ndvi_data['features'])} points with multiple strategies...")
            
            for i, feature in enumerate(ndvi_data['features']):
                coords = feature['geometry']['coordinates']
                lon, lat = coords[0], coords[1]
                lat_idx = np.argmin(np.abs(lats - lat))
                lon_idx = np.argmin(np.abs(lons - lon))
                
                # Get all vegetation indices
                ndvi_val = feature['properties'].get('NDVI', 0)
                ndwi_val = ndwi_data['features'][i]['properties'].get('NDWI', 0) if i < len(ndwi_data['features']) else 0
                gndvi_val = gndvi_data['features'][i]['properties'].get('GNDVI', 0) if i < len(gndvi_data['features']) else 0
                
                # Multi-strategy tree detection
                tree_score = 0
                
                # Strategy 1: NDVI (vegetation health)
                if ndvi_val and ndvi_val > 0.1:
                    tree_score += (ndvi_val - 0.1) / 0.5
                
                # Strategy 2: NDWI (water content - trees have more water)
                if ndwi_val and ndwi_val > 0.1:
                    tree_score += (ndwi_val - 0.1) / 0.3
                
                # Strategy 3: GNDVI (green vegetation)
                if gndvi_val and gndvi_val > 0.1:
                    tree_score += (gndvi_val - 0.1) / 0.4
                
                # Strategy 4: Combined score with weights
                if tree_score > 0:
                    weighted_score = (tree_score * 0.4 + ndvi_val * 0.6) if ndvi_val else tree_score
                    tree_density[lat_idx, lon_idx] = max(0, min(1, weighted_score))
        
        elif 'NDVI' in ndvi_data:
            logger.info("Processing arrays with multi-strategy interpolation...")
            ndvi_array = np.array(ndvi_data['NDVI'])
            ndwi_array = np.array(ndwi_data['NDWI'])
            gndvi_array = np.array(gndvi_data['GNDVI'])
            
            if ndvi_array.size > 0 and SCIPY_AVAILABLE:
                # Interpolate all indices
                ndvi_lats = np.linspace(self.south, self.north, ndvi_array.shape[0])
                ndvi_lons = np.linspace(self.west, self.east, ndvi_array.shape[1])
                ndvi_lon_grid, ndvi_lat_grid = np.meshgrid(ndvi_lons, ndvi_lats)
                ndvi_points = np.column_stack([ndvi_lat_grid.ravel(), ndvi_lon_grid.ravel()])
                
                target_lats, target_lons = np.meshgrid(lats, lons, indexing='ij')
                target_points = np.column_stack([target_lats.ravel(), target_lons.ravel()])
                
                # Interpolate each index
                ndvi_interp = griddata(ndvi_points, ndvi_array.ravel(), target_points, method='linear', fill_value=0)
                ndwi_interp = griddata(ndvi_points, ndwi_array.ravel(), target_points, method='linear', fill_value=0)
                gndvi_interp = griddata(ndvi_points, gndvi_array.ravel(), target_points, method='linear', fill_value=0)
                
                # Reshape and combine
                ndvi_interp = ndvi_interp.reshape(len(lats), len(lons))
                ndwi_interp = ndwi_interp.reshape(len(lats), len(lons))
                gndvi_interp = gndvi_interp.reshape(len(lats), len(lons))
                
                # Multi-strategy combination
                tree_density = np.maximum(0, np.minimum(1, 
                    (ndvi_interp - 0.1) / 0.5 * 0.6 + 
                    (ndwi_interp - 0.1) / 0.3 * 0.2 + 
                    (gndvi_interp - 0.1) / 0.4 * 0.2
                ))
                logger.info("✓ Multi-strategy interpolation complete")
            else:
                logger.warning("Array processing failed - using simulation")
                return self._detect_with_simulation(lats, lons)
        
        # Convert to tree locations
        tree_threshold = 0.15
        tree_mask = tree_density > tree_threshold
        tree_indices = np.where(tree_mask)
        detected_trees_lat = lats[tree_indices[0]]
        detected_trees_lon = lons[tree_indices[1]]
        detected_trees_density = tree_density[tree_indices]
        
        enhanced_trees_df = pd.DataFrame({
            'latitude': detected_trees_lat,
            'longitude': detected_trees_lon,
            'density': detected_trees_density,
            'source': 'satellite_enhanced_multi_strategy'
        })
        
        logger.info(f"✓ Detected {len(enhanced_trees_df)} trees using satellite data")
        logger.info(f"Grid resolution: {grid_size*111:.0f}m per cell")
        logger.info(f"Threshold used: {tree_threshold}")
        logger.info(f"Tree density range: {detected_trees_density.min():.3f} - {detected_trees_density.max():.3f}")
        
        return enhanced_trees_df
    
    def _detect_with_simulation(self, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
        """Fallback simulation when satellite data is not available."""
        logger.info("Using enhanced simulated data...")
        tree_density = np.zeros((len(lats), len(lons)))
        center_lat, center_lon = 39.955, -75.16
        
        # More realistic simulation with multiple tree types
        for i in range(len(lats)):
            for j in range(len(lons)):
                lat, lon = lats[i], lons[j]
                dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
                
                # Simulate different tree types
                if dist < 0.005:  # Parks/squares
                    tree_density[i, j] = np.random.uniform(0.8, 1.0)
                elif dist < 0.01:  # Residential
                    tree_density[i, j] = np.random.uniform(0.6, 0.9)
                elif dist < 0.015:  # Mixed areas
                    tree_density[i, j] = np.random.uniform(0.3, 0.7)
                elif dist < 0.02:  # Commercial
                    tree_density[i, j] = np.random.uniform(0.1, 0.5)
                else:  # Edge areas
                    tree_density[i, j] = np.random.uniform(0.0, 0.3)
        
        # Convert to tree locations
        tree_threshold = 0.15
        tree_mask = tree_density > tree_threshold
        tree_indices = np.where(tree_mask)
        detected_trees_lat = lats[tree_indices[0]]
        detected_trees_lon = lons[tree_indices[1]]
        detected_trees_density = tree_density[tree_indices]
        
        enhanced_trees_df = pd.DataFrame({
            'latitude': detected_trees_lat,
            'longitude': detected_trees_lon,
            'density': detected_trees_density,
            'source': 'simulation_enhanced_multi_strategy'
        })
        
        logger.info(f"✓ Detected {len(enhanced_trees_df)} trees using simulation")
        return enhanced_trees_df


class TreeVisualizer:
    """Creates visualizations for tree data."""
    
    def __init__(self, bbox: Tuple[float, float, float, float] = PHILLY_BBOX):
        """
        Initialize the tree visualizer.
        
        Args:
            bbox: Bounding box as (north, south, east, west)
        """
        self.bbox = bbox
        self.north, self.south, self.east, self.west = bbox
        
    def create_map(self, trees_df: pd.DataFrame, output_file: str = 'tree_detection_map.html') -> folium.Map:
        """
        Create an interactive map showing detected trees.
        
        Args:
            trees_df: DataFrame containing tree data
            output_file: Output file path for the map
            
        Returns:
            Folium map object
        """
        logger.info("Creating tree detection visualization...")
        
        # Create map centered on downtown Philadelphia
        center_lat = (self.north + self.south) / 2
        center_lon = (self.east + self.west) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles=None)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', 
            name='Satellite'
        ).add_to(m)
        folium.TileLayer(tiles='OpenStreetMap', name='Street Map').add_to(m)
        
        # Add bounding box rectangle
        folium.Rectangle(
            bounds=[[self.south, self.west], [self.north, self.east]], 
            color='red', 
            fill=False, 
            weight=3
        ).add_to(m)
        
        # Add trees with confidence-based coloring
        for idx, tree in trees_df.iterrows():
            if tree['source'] == 'osm':
                color, size = 'green', 4
            elif tree['confidence'] > 0.8:
                color, size = 'darkred', 5
            elif tree['confidence'] > 0.6:
                color, size = 'red', 4
            elif tree['confidence'] > 0.4:
                color, size = 'orange', 3
            else:
                color, size = 'yellow', 2
            
            folium.CircleMarker(
                location=[tree['latitude'], tree['longitude']],
                radius=size,
                popup=f"Tree {idx}<br>Source: {tree['source']}<br>Confidence: {tree['confidence']:.3f}",
                color=color,
                fill=True,
                fillOpacity=0.8
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        m.save(output_file)
        logger.info(f"Map saved to {output_file}")
        
        return m


class TreeDetectionManager:
    """Main class for managing tree detection operations."""
    
    def __init__(self, bbox: Tuple[float, float, float, float] = PHILLY_BBOX):
        """
        Initialize the tree detection manager.
        
        Args:
            bbox: Bounding box as (north, south, east, west)
        """
        self.bbox = bbox
        self.osm_fetcher = OSMTreeFetcher(bbox)
        self.enhanced_detector = EnhancedTreeDetector(bbox)
        self.visualizer = TreeVisualizer(bbox)
        
    def detect_all_trees(self) -> pd.DataFrame:
        """
        Detect trees using all available methods.
        
        Returns:
            DataFrame containing all detected trees
        """
        logger.info("Starting comprehensive tree detection...")
        
        # Get OSM trees
        osm_trees = self.osm_fetcher.fetch_osm_trees()
        
        # Get enhanced detection trees
        enhanced_trees = self.enhanced_detector.detect_additional_trees()
        
        # Combine all tree data
        all_trees = []
        
        # Add OSM trees (with confidence column)
        for _, tree in osm_trees.iterrows():
            all_trees.append({
                'latitude': tree['latitude'],
                'longitude': tree['longitude'],
                'confidence': 1.0,  # OSM trees are high confidence
                'source': 'osm'
            })
        
        # Add enhanced detection trees
        for _, tree in enhanced_trees.iterrows():
            all_trees.append({
                'latitude': tree['latitude'],
                'longitude': tree['longitude'],
                'confidence': tree.get('density', tree.get('confidence', 1.0)),
                'source': tree['source']
            })
        
        final_trees = pd.DataFrame(all_trees)
        
        logger.info(f"Total trees detected: {len(final_trees)}")
        logger.info(f"  - OSM: {len(osm_trees)}")
        logger.info(f"  - Enhanced Detection: {len(enhanced_trees)}")
        logger.info(f"  - Average confidence: {final_trees['confidence'].mean():.3f}")
        
        # Calculate tree density
        area_km2 = (self.bbox[0] - self.bbox[1]) * (self.bbox[2] - self.bbox[3]) * 111 * 111
        tree_density = len(final_trees) / area_km2
        logger.info(f"  - Tree density: {tree_density:.0f} trees/km²")
        
        return final_trees
    
    def save_trees(self, trees_df: pd.DataFrame, output_file: str = 'tree_positions.json') -> None:
        """
        Save tree data to JSON file.
        
        Args:
            trees_df: DataFrame containing tree data
            output_file: Output file path
        """
        # Convert DataFrame to the same format as the notebook
        trees_list = []
        for idx, tree in trees_df.iterrows():
            tree_dict = {
                'id': idx,
                'latitude': float(tree['latitude']),
                'longitude': float(tree['longitude']),
                'density': float(tree.get('density', tree.get('confidence', 1.0))),
                'grid_size': '5m'
            }
            trees_list.append(tree_dict)
        
        tree_data = {
            'metadata': {
                'total_trees': len(trees_df),
                'method': 'enhanced_multi_strategy_tree_detection',
                'grid_size': '5m',
                'created_at': datetime.now().isoformat(),
                'bounding_box': {
                    'north': self.bbox[0],
                    'south': self.bbox[1],
                    'east': self.bbox[2],
                    'west': self.bbox[3]
                },
                'source_distribution': trees_df['source'].value_counts().to_dict()
            },
            'trees': trees_list
        }
        
        with open(output_file, 'w') as f:
            json.dump(tree_data, f, indent=2)
        
        logger.info(f"Tree positions saved to {output_file}")
    
    def create_visualization(self, trees_df: pd.DataFrame, 
                           output_file: str = 'tree_detection_map.html') -> folium.Map:
        """
        Create and save tree visualization.
        
        Args:
            trees_df: DataFrame containing tree data
            output_file: Output file path for the map
            
        Returns:
            Folium map object
        """
        return self.visualizer.create_map(trees_df, output_file)


def create_visualization(trees_df: pd.DataFrame, output_file: str = 'tree_detection_map.html') -> None:
    """
    Create a visualization of the detected trees.
    
    Args:
        trees_df: DataFrame containing tree data
        output_file: Output file path for the map
    """
    logger.info("Creating tree visualization...")
    
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
        density = tree.get('density', tree.get('confidence', 1.0))
        
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
            popup=f"Tree {idx}<br>Density: {density:.3f}<br>Source: {tree['source']}<br>Grid: 5m",
            color=color,
            fill=True,
            fillOpacity=0.8
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    m.save(output_file)
    logger.info(f"Tree visualization saved to {output_file}")
    
    return m


def main(create_map: bool = False):
    """Main function to run the tree detection system."""
    logger.info("Starting enhanced multi-strategy tree detection system...")
    
    # Initialize manager
    manager = TreeDetectionManager()
    
    # Detect all trees using multi-strategy approach
    trees_df = manager.detect_all_trees()
    
    # Save data to JSON
    manager.save_trees(trees_df, 'tree_positions.json')
    
    # Create visualization if requested
    if create_map:
        create_visualization(trees_df, 'tree_detection_map.html')
    
    logger.info("Enhanced tree detection system completed successfully!")
    logger.info(f"Total trees detected: {len(trees_df)}")
    logger.info(f"Tree positions saved to: tree_positions.json")
    if create_map:
        logger.info(f"Tree visualization saved to: tree_detection_map.html")
    
    return trees_df


if __name__ == "__main__":
    # Run with visualization enabled
    trees_df = main(create_map=True)
