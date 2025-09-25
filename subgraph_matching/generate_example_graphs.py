"""
Generate example graphs for subgraph matching demo.

Creates:
- G1.pkl: Target graph (synthetic, ~15 nodes)
- G2.pkl: Subgraph of G1 (sampled subgraph)
- G3.pkl: Non-subgraph (separate synthetic graph)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
import pickle
import numpy as np
from common import combined_syn

def generate_example_graphs():
    """Generate and save example graphs using synthetic data."""

    # Get the synthetic graph generator (same as training)
    generator = combined_syn.get_generator(np.arange(10, 20))  # Size range similar to training

    # Create target graph G1 (synthetic)
    G1 = generator.generate(size=15)
    print(f"G1 (target, synthetic): {len(G1)} nodes, {G1.number_of_edges()} edges")

    # Create G2 as subgraph of G1 (sample a smaller subgraph)
    # Use BFS sampling like in training
    start_node = list(G1.nodes())[0]
    visited = {start_node}
    queue = [start_node]
    while queue and len(visited) < 6:  # Aim for ~6 nodes
        current = queue.pop(0)
        neighbors = list(set(G1.neighbors(current)) - visited)
        for neighbor in neighbors[:2]:  # Limit branching
            if len(visited) >= 6:
                break
            visited.add(neighbor)
            queue.append(neighbor)
    G2 = G1.subgraph(visited).copy()
    print(f"G2 (subgraph of G1): {len(G2)} nodes, {G2.number_of_edges()} edges")

    # Create G3 as non-subgraph (separate synthetic graph)
    G3 = generator.generate(size=6)
    print(f"G3 (non-subgraph, synthetic): {len(G3)} nodes, {G3.number_of_edges()} edges")

    # Save graphs
    graphs_dir = os.path.dirname(__file__)
    pickle.dump(G1, open(os.path.join(graphs_dir, 'G1.pkl'), 'wb'))
    pickle.dump(G2, open(os.path.join(graphs_dir, 'G2.pkl'), 'wb'))
    pickle.dump(G3, open(os.path.join(graphs_dir, 'G3.pkl'), 'wb'))

    print("Graphs saved as G1.pkl, G2.pkl, G3.pkl in subgraph_matching folder")

if __name__ == "__main__":
    generate_example_graphs()