"""
Combine graph_segments.gpickle with shade analysis data to create enhanced graph.

This script:
1. Loads the original graph from graph_segments.gpickle (or existing enhanced graph)
2. Loads shade analysis data from JSON file for a specific hour
3. Adds hour-specific shade attributes to each edge (shade_fraction_Xam/pm, shade_length_Xam/pm)
4. Preserves existing hour data and adds new hour data
5. Saves enhanced graph as graph_segments_with_shade.gpickle
"""

import os
import json
import pickle
import networkx as nx
from typing import Dict, Any, Optional
import argparse
from datetime import datetime


def format_hour_suffix(hour: int) -> str:
    """Convert hour (0-23) to simple number suffix like '9' or '15'."""
    return str(hour)


def load_graph(graph_path: str) -> nx.Graph:
    """Load NetworkX graph from pickle file."""
    print(f"Loading graph from {graph_path}...")
    
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph


def load_shade_analysis(json_path: str) -> Dict[str, Dict[str, Any]]:
    """Load shade analysis results from JSON file."""
    print(f"Loading shade analysis from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert edge list to dictionary for fast lookup
    shade_data = {}
    for edge_result in data['edges']:
        shade_data[edge_result['id']] = {
            'shade_fraction': edge_result['shadePct'],
            'shaded': edge_result['shaded'],
            'samples': edge_result['nSamples']
        }
    
    print(f"Loaded shade data for {len(shade_data)} edges")
    print(f"Analysis time: {data.get('analysisTime', 'unknown')}")
    print(f"Processing time: {data.get('processingTimeMs', 0)/1000:.1f}s")
    
    return shade_data


def enhance_graph_with_shade(graph: nx.Graph, shade_data: Dict[str, Dict[str, Any]], hour: int) -> nx.Graph:
    """Add hour-specific shade attributes to graph edges without overwriting existing hour data."""
    hour_suffix = format_hour_suffix(hour)
    print(f"Enhancing graph with shade data for {hour_suffix}...")
    
    enhanced_graph = graph.copy()
    edges_updated = 0
    edges_missing_shade = 0
    edges_already_have_hour_data = 0
    
    # Define hour-specific attribute names
    shade_fraction_attr = f'shade_fraction_{hour_suffix}'
    shade_length_attr = f'shade_length_{hour_suffix}'
    shade_samples_attr = f'shade_samples_{hour_suffix}'
    is_shaded_attr = f'is_shaded_{hour_suffix}'
    
    # Check if any edges already have this hour's data
    sample_edge = next(iter(enhanced_graph.edges(data=True)), None)
    if sample_edge and shade_fraction_attr in sample_edge[2]:
        print(f"⚠️  Warning: Graph already contains shade data for {hour_suffix}")
        response = input(f"Overwrite existing {hour_suffix} data? (y/N): ").strip().lower()
        if response != 'y':
            print(f"Aborted: Existing {hour_suffix} data preserved")
            return enhanced_graph
        print(f"Proceeding to overwrite {hour_suffix} data...")
    
    # Iterate through all edges and add shade data
    for i, (node1, node2, edge_attrs) in enumerate(enhanced_graph.edges(data=True)):
        edge_id = f"edge_{i}"
        
        if edge_id in shade_data:
            # Get shade data for this edge
            shade_info = shade_data[edge_id]
            shade_fraction = shade_info['shade_fraction']
            
            # Get edge weight (distance in meters)
            edge_weight = edge_attrs.get('weight', 0)
            
            # Calculate shade length (fraction of edge that's shaded × edge length)
            shade_length = shade_fraction * edge_weight
            
            # Add hour-specific shade attributes
            edge_attrs[shade_fraction_attr] = shade_fraction
            edge_attrs[shade_length_attr] = shade_length
            edge_attrs[shade_samples_attr] = shade_info['samples']
            edge_attrs[is_shaded_attr] = shade_info['shaded']
            
            edges_updated += 1
            
            if edges_updated % 1000 == 0:
                print(f"  Enhanced {edges_updated} edges...")
                
        else:
            # No shade data available - set defaults only if not already present
            if shade_fraction_attr not in edge_attrs:
                edge_attrs[shade_fraction_attr] = 0.0
                edge_attrs[shade_length_attr] = 0.0
                edge_attrs[shade_samples_attr] = 0
                edge_attrs[is_shaded_attr] = False
                edges_missing_shade += 1
            else:
                edges_already_have_hour_data += 1
    
    print(f"Enhanced {edges_updated} edges with {hour_suffix} shade data")
    if edges_missing_shade > 0:
        print(f"Warning: {edges_missing_shade} edges missing shade data (set to defaults)")
    if edges_already_have_hour_data > 0:
        print(f"Note: {edges_already_have_hour_data} edges already had {hour_suffix} data")
    
    return enhanced_graph
def update_graph_metadata(graph: nx.Graph, hour: int) -> None:
    """Update graph metadata to track hour-specific shade data."""
    hour_suffix = format_hour_suffix(hour)
    
    if not hasattr(graph, 'graph'):
        graph.graph = {}
    
    # Initialize shade metadata if not present
    if 'shade_analysis_hours' not in graph.graph:
        graph.graph['shade_analysis_hours'] = []
        graph.graph['shade_analysis_added'] = True
        graph.graph['shade_enhanced_at'] = datetime.now().isoformat()
    
    # Add this hour to the list if not already present
    if hour not in graph.graph['shade_analysis_hours']:
        graph.graph['shade_analysis_hours'].append(hour)
        graph.graph['shade_analysis_hours'].sort()  # Keep sorted
    
    # Update last enhancement time
    graph.graph['shade_last_enhanced_at'] = datetime.now().isoformat()
    graph.graph[f'shade_analysis_{hour_suffix}_added_at'] = datetime.now().isoformat()
    
    print(f"Updated metadata: Graph now has shade data for hours: {graph.graph['shade_analysis_hours']}")


def save_enhanced_graph(graph: nx.Graph, output_path: str) -> None:
    """Save enhanced graph to pickle file."""
    print(f"Saving enhanced graph to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"Saved enhanced graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")


def main():
    parser = argparse.ArgumentParser(description='Combine graph with hour-specific shade analysis data')
    parser.add_argument('--graph', '-g', required=True, help='Path to graph_segments.gpickle file (or existing enhanced graph)')
    parser.add_argument('--shade', '-s', required=True, help='Path to shade analysis JSON file')
    parser.add_argument('--hour', '-x', type=int, required=True, 
                       help='Hour (0-23) that the shade data represents')
    parser.add_argument('--output', '-o', default='data/graph_segments_with_shade.gpickle', 
                       help='Output path for enhanced graph')
    
    args = parser.parse_args()
    
    # Validate hour
    if not (0 <= args.hour <= 23):
        print(f"Error: Hour must be between 0-23, got {args.hour}")
        return 1
    
    # Verify input files exist
    if not os.path.exists(args.graph):
        print(f"Error: Graph file not found: {args.graph}")
        return 1
    
    if not os.path.exists(args.shade):
        print(f"Error: Shade analysis file not found: {args.shade}")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    hour_suffix = format_hour_suffix(args.hour)
    print(f"\n🕐 Processing shade data for {hour_suffix} ({args.hour}:00)")
    
    try:
        # Load input data
        graph = load_graph(args.graph)
        shade_data = load_shade_analysis(args.shade)
        
        # Show existing shade hours if any
        if hasattr(graph, 'graph') and 'shade_analysis_hours' in graph.graph:
            existing_hours = graph.graph['shade_analysis_hours']
            existing_suffixes = [format_hour_suffix(h) for h in existing_hours]
            print(f"Existing shade data hours: {existing_suffixes}")
        
        # Enhance graph with shade data for this specific hour
        enhanced_graph = enhance_graph_with_shade(graph, shade_data, args.hour)
        
        # Update metadata
        update_graph_metadata(enhanced_graph, args.hour)
        
        # Save enhanced graph
        save_enhanced_graph(enhanced_graph, args.output)
        
        print(f"\n✅ Graph enhancement complete for {hour_suffix}!")
        print(f"Enhanced graph saved to: {args.output}")
        
        # Print summary statistics for this hour
        total_edges = enhanced_graph.number_of_edges()
        shade_fraction_attr = f'shade_fraction_{hour_suffix}'
        is_shaded_attr = f'is_shaded_{hour_suffix}'
        
        shaded_edges = sum(1 for _, _, attrs in enhanced_graph.edges(data=True) 
                          if attrs.get(is_shaded_attr, False))
        avg_shade_fraction = sum(attrs.get(shade_fraction_attr, 0) 
                               for _, _, attrs in enhanced_graph.edges(data=True)) / total_edges
        
        print(f"\n📊 Shade Statistics for {hour_suffix}:")
        print(f"  Total edges: {total_edges}")
        print(f"  Shaded edges (≥50%): {shaded_edges} ({shaded_edges/total_edges*100:.1f}%)")
        print(f"  Average shade fraction: {avg_shade_fraction:.3f}")
        
        # Show all available hours
        if 'shade_analysis_hours' in enhanced_graph.graph:
            all_hours = enhanced_graph.graph['shade_analysis_hours']
            all_suffixes = [format_hour_suffix(h) for h in all_hours]
            print(f"\n🌅 Graph now contains shade data for: {all_suffixes}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())