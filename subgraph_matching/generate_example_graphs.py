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
import argparse
from common import combined_syn
import random
from networkx.algorithms import isomorphism

# -------------------------------------------------------------
# Helper random graph generators (independent from training set)
# -------------------------------------------------------------
def gen_er(n, p=None, tries=10):
    if p is None:
        # default density similar to log2(n)/n
        p = min(0.9, max(0.01, np.log2(n)/n))
    for _ in range(tries):
        G = nx.gnp_random_graph(n, p)
        if nx.is_connected(G):
            return G
    return G  # may be disconnected as fallback

def gen_ws(n, k=None, p=0.1, tries=10):
    if k is None:
        k = max(2, min(n-1, int(np.log2(n))+1))
        if k % 2 == 1:
            k += 1  # Watts–Strogatz requires even k sometimes for some variants
    for _ in range(tries):
        try:
            G = nx.connected_watts_strogatz_graph(n, k, p)
            return G
        except Exception:
            continue
    # fallback (possibly disconnected)
    return nx.watts_strogatz_graph(n, k, p)

def gen_ba(n, m=None):
    if m is None:
        m = max(1, min(n-1, int(np.log2(n))+1))
    return nx.barabasi_albert_graph(n, m)

def gen_powerlaw_cluster(n, m=None, p=0.3):
    if m is None:
        m = max(1, min(n-1, int(np.log2(n))+1))
    return nx.powerlaw_cluster_graph(n, m, p)

def gen_tree(n):
    return nx.random_tree(n)

def gen_geometric(n, radius=None):
    if radius is None:
        radius = 0.4 * (np.log(n)/n + 0.1)
    return nx.random_geometric_graph(n, radius)

GEN_MAP = {
    'er': gen_er,
    'ws': gen_ws,
    'ba': gen_ba,
    'plc': gen_powerlaw_cluster,
    'tree': gen_tree,
    'geom': gen_geometric,
}

def bfs_sample_subgraph(G, sub_size, branching_limit=2):
    if sub_size >= len(G):
        return G.copy()
    start_node = random.choice(list(G.nodes()))
    visited = {start_node}
    queue = [start_node]
    while queue and len(visited) < sub_size:
        current = queue.pop(0)
        neighbors = list(set(G.neighbors(current)) - visited)
        random.shuffle(neighbors)
        for neighbor in neighbors[:branching_limit]:
            if len(visited) >= sub_size:
                break
            visited.add(neighbor)
            queue.append(neighbor)
    return G.subgraph(visited).copy()

def generate_example_graphs():
    """Generate and save example graphs.

    Modes:
      ensemble (default): Use training-style synthetic ensemble generator.
      er/ws/ba/plc/tree/geom: Use specified NetworkX random model.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ensemble'] + list(GEN_MAP.keys()), default='ensemble', help='Graph generation mode for G1 and G3.')
    parser.add_argument('--g1-size', type=int, default=5)
    parser.add_argument('--g2-size', type=int, default=3, help='Subgraph size (<= g1-size)')
    parser.add_argument('--g3-size', type=int, default=3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ensure-non-subgraph', action='store_true', default=True,
                        help='Actively enforce that G3 is NOT a subgraph of G1 (may modify G3).')
    parser.add_argument('--non-subgraph-attempts', type=int, default=25,
                        help='Max regeneration attempts for a non-subgraph G3 before structural tweak.')
    # Model-specific optional params
    parser.add_argument('--er-p', type=float, default=None)
    parser.add_argument('--ws-k', type=int, default=None)
    parser.add_argument('--ws-p', type=float, default=0.1)
    parser.add_argument('--ba-m', type=int, default=None)
    parser.add_argument('--plc-m', type=int, default=None)
    parser.add_argument('--plc-p', type=float, default=0.3)
    parser.add_argument('--geom-radius', type=float, default=None)
    args = parser.parse_args([] if '__file__' not in globals() else None)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # --- Generate G1 ---
    if args.mode == 'ensemble':
        generator = combined_syn.get_generator(np.arange(min(6, args.g1_size//2), max(args.g1_size+1, 8)))
        G1 = generator.generate(size=args.g1_size)
    else:
        gen_fn = GEN_MAP[args.mode]
        if args.mode == 'er':
            G1 = gen_fn(args.g1_size, p=args.er_p)
        elif args.mode == 'ws':
            G1 = gen_fn(args.g1_size, k=args.ws_k, p=args.ws_p)
        elif args.mode == 'ba':
            G1 = gen_fn(args.g1_size, m=args.ba_m)
        elif args.mode == 'plc':
            G1 = gen_fn(args.g1_size, m=args.plc_m, p=args.plc_p)
        elif args.mode == 'tree':
            G1 = gen_fn(args.g1_size)
        elif args.mode == 'geom':
            G1 = gen_fn(args.g1_size, radius=args.geom_radius)
    print(f"G1 ({args.mode}): {len(G1)} nodes, {G1.number_of_edges()} edges")

    # --- Generate G2 as subgraph of G1 ---
    g2_size = min(args.g2_size, len(G1))
    G2 = bfs_sample_subgraph(G1, g2_size)
    print(f"G2 (subgraph of G1): {len(G2)} nodes, {G2.number_of_edges()} edges")

    # --- Generate G3 independent random graph ---
    def gen_g3_once():
        if args.mode == 'ensemble':
            return generator.generate(size=args.g3_size)
        if args.mode == 'er':
            return gen_fn(args.g3_size, p=args.er_p)
        if args.mode == 'ws':
            return gen_fn(args.g3_size, k=args.ws_k, p=args.ws_p)
        if args.mode == 'ba':
            return gen_fn(args.g3_size, m=args.ba_m)
        if args.mode == 'plc':
            return gen_fn(args.g3_size, m=args.plc_m, p=args.plc_p)
        if args.mode == 'tree':
            return gen_fn(args.g3_size)
        if args.mode == 'geom':
            return gen_fn(args.g3_size, radius=args.geom_radius)
        raise ValueError("Unknown mode")

    def is_subgraph(G_large, G_small):
        # Return True if G_small is a subgraph of G_large
        GM = isomorphism.GraphMatcher(G_large, G_small)
        return GM.subgraph_is_isomorphic()

    G3 = gen_g3_once()
    attempts = 0
    if args.ensure_non_subgraph:
        while attempts < args.non_subgraph_attempts and is_subgraph(G1, G3):
            attempts += 1
            G3 = gen_g3_once()
        if is_subgraph(G1, G3):
            # Fallback: structurally modify G3 to break subgraph property.
            # Strategy: add a non-existing edge; if no non-edges remain, add a new node with an extra edge.
            non_edges = list(nx.non_edges(G3))
            if non_edges:
                u, v = random.choice(non_edges)
                G3.add_edge(u, v)
            else:
                new_id = max(G3.nodes) + 1 if len(G3.nodes) > 0 else 0
                G3.add_node(new_id)
                if len(G3.nodes) > 1:
                    G3.add_edge(new_id, random.choice([n for n in G3.nodes if n != new_id]))
            # One more sanity check (should now typically be False)
            if is_subgraph(G1, G3):
                print("Warning: Could not guarantee G3 is non-subgraph after modifications.")
            else:
                print("Applied structural tweak to ensure G3 is NOT a subgraph of G1.")
        else:
            if attempts > 0:
                print(f"Found non-subgraph G3 after {attempts} regeneration attempts.")
    print(f"G3 (independent {args.mode}, ensure_non_subgraph={args.ensure_non_subgraph}): {len(G3)} nodes, {G3.number_of_edges()} edges")

    # --- Save ---
    graphs_dir = os.path.dirname(__file__)
    with open(os.path.join(graphs_dir, 'G1.pkl'), 'wb') as f: pickle.dump(G1, f)
    with open(os.path.join(graphs_dir, 'G2.pkl'), 'wb') as f: pickle.dump(G2, f)
    with open(os.path.join(graphs_dir, 'G3.pkl'), 'wb') as f: pickle.dump(G3, f)
    print("Saved G1.pkl, G2.pkl, G3.pkl")
    print("(Reminder: these are raw NetworkX graphs; batching happens later in query)")

if __name__ == "__main__":
    generate_example_graphs()