import argparse
import csv
import time
import os
import json

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from subgraph_mining import decoder

from tqdm import tqdm
import matplotlib.pyplot as plt

from multiprocessing import Pool
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
from itertools import combinations

MAX_SEARCH_TIME = 60  
MAX_MATCHES_PER_QUERY = 10000  
DEFAULT_SAMPLE_ANCHORS = 1000  

def compute_graph_stats(G):
    """Compute graph statistics for filtering."""
    return {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'degree_seq': sorted([d for _, d in G.degree()], reverse=True),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
    }

def can_be_isomorphic(query_stats, target_stats):
    """Quick check if query could possibly be isomorphic to a subgraph of target."""
    if query_stats['n_nodes'] > target_stats['n_nodes']:
        return False
    if query_stats['n_edges'] > target_stats['n_edges']:
        return False
    if query_stats['avg_degree'] > target_stats['avg_degree']:
        return False
    query_degrees = query_stats['degree_seq']
    target_degrees = target_stats['degree_seq']
    return all(d <= target_degrees[i] for i, d in enumerate(query_degrees))

def arg_parse():
    parser = argparse.ArgumentParser(description='count graphlets in a graph')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--queries_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--count_method', type=str)
    parser.add_argument('--baseline', type=str)
    parser.add_argument('--node_anchored', action="store_true")
    parser.add_argument('--preserve_labels', action="store_true", help='Preserve Neo4j node and edge labels during counting')
    parser.add_argument('--max_query_size', type=int, default=20, help='Maximum query size to process')
    parser.add_argument('--sample_anchors', type=int, default=DEFAULT_SAMPLE_ANCHORS, help='Number of anchor nodes to sample for large graphs')
    parser.set_defaults(dataset="enzymes",
                       queries_path="results/out-patterns.p",
                       out_path="results/counts.json",
                       n_workers=4,
                       count_method="bin",
                       baseline="none",
                       preserve_labels=False)
    return parser.parse_args()

def gen_baseline_queries(queries, targets, method="mfinder",
    node_anchored=False):
    # use this to generate N size K queries
    #queries = [[0]*n for n in range(5, 21) for i in range(10)]
    if method == "mfinder":
        return utils.gen_baseline_queries_mfinder(queries, targets,
            node_anchored=node_anchored)
    elif method == "rand-esu":
        return utils.gen_baseline_queries_rand_esu(queries, targets,
            node_anchored=node_anchored)
    neighs = []
    for i, query in enumerate(queries):
        print(i)
        found = False
        if len(query) == 0:
            neighs.append(query)
            found = True
        while not found:
            if method == "radial":
                graph = random.choice(targets)
                node = random.choice(list(graph.nodes))
                neigh = list(nx.single_source_shortest_path_length(graph, node,
                    cutoff=3).keys())
                #neigh = random.sample(neigh, min(len(neigh), 15))
                neigh = graph.subgraph(neigh)
                neigh = neigh.subgraph(list(sorted(nx.connected_components(
                    neigh), key=len))[-1])
                neigh = nx.convert_node_labels_to_integers(neigh)
                print(i, len(neigh), len(query))
                if len(neigh) == len(query):
                    neighs.append(neigh)
                    found = True
            elif method == "tree":
                # https://academic.oup.com/bioinformatics/article/20/11/1746/300212
                graph = random.choice(targets)
                start_node = random.choice(list(graph.nodes))
                neigh = [start_node]
                frontier = list(set(graph.neighbors(start_node)) - set(neigh))
                while len(neigh) < len(query) and frontier:
                    new_node = random.choice(list(frontier))
                    assert new_node not in neigh
                    neigh.append(new_node)
                    frontier += list(graph.neighbors(new_node))
                    frontier = [x for x in frontier if x not in neigh]
                if len(neigh) == len(query):
                    neigh = graph.subgraph(neigh)
                    neigh = nx.convert_node_labels_to_integers(neigh)
                    neighs.append(neigh)
                    found = True
    return neighs

def count_graphlets_helper(inp):
    i, query, target, method, node_anchored, anchor_or_none, preserve_labels = inp
    
    # Remove self loops
    query = query.copy()
    query.remove_edges_from(nx.selfloop_edges(query))
    target = target.copy()
    target.remove_edges_from(nx.selfloop_edges(target))

    # Pre-compute query stats for method "freq"
    if method == "freq":
        ismags = nx.isomorphism.ISMAGS(query, query)
        n_symmetries = len(list(ismags.isomorphisms_iter(symmetry=False)))

    count = 0
    try:
        start_time = time.time()
        
        if method == "bin":
            if node_anchored:
                nx.set_node_attributes(target, 0, name="anchor")
                target.nodes[anchor_or_none]["anchor"] = 1
                
                if preserve_labels:
                    matcher = iso.GraphMatcher(target, query,
                        node_match=lambda n1, n2: (n1.get("anchor") == n2.get("anchor") and
                                                 n1.get("label") == n2.get("label")),
                        edge_match=lambda e1, e2: e1.get("type") == e2.get("type"))
                else:
                    matcher = iso.GraphMatcher(target, query,
                        node_match=iso.categorical_node_match(["anchor"], [0]))
                
                if time.time() - start_time > MAX_SEARCH_TIME:
                    return i, 0
                count = int(matcher.subgraph_is_isomorphic())
            else:
                if preserve_labels:
                    matcher = iso.GraphMatcher(target, query,
                        node_match=lambda n1, n2: n1.get("label") == n2.get("label"),
                        edge_match=lambda e1, e2: e1.get("type") == e2.get("type"))
                else:
                    matcher = iso.GraphMatcher(target, query)
                if time.time() - start_time > MAX_SEARCH_TIME:
                    return i, 0
                count = int(matcher.subgraph_is_isomorphic())
        elif method == "freq":
            if preserve_labels:
                matcher = iso.GraphMatcher(target, query,
                    node_match=lambda n1, n2: n1.get("label") == n2.get("label"),
                    edge_match=lambda e1, e2: e1.get("type") == e2.get("type"))
            else:
                matcher = iso.GraphMatcher(target, query)
            
            count = 0
            for _ in matcher.subgraph_isomorphisms_iter():
                if time.time() - start_time > MAX_SEARCH_TIME:
                    break
                count += 1
                if count >= MAX_MATCHES_PER_QUERY:
                    break
            count = count / n_symmetries
    except Exception as e:
        print(f"Error processing query {i}: {str(e)}")
        count = 0
        
    return i, count

def count_graphlets(queries, targets, n_workers=1, method="bin",
    node_anchored=False, min_count=0, preserve_labels=False, sample_anchors=DEFAULT_SAMPLE_ANCHORS):
    print(f"Processing {len(queries)} queries across {len(targets)} targets")
    
    # Pre-compute graph statistics
    target_stats = [compute_graph_stats(t) for t in targets]
    query_stats = [compute_graph_stats(q) for q in queries]
    
    n_matches = defaultdict(float)
    pool = Pool(processes=n_workers)
    
    # Generate work items with filtering
    inp = []
    for i, (query, q_stats) in enumerate(zip(queries, query_stats)):
        if query.number_of_nodes() > args.max_query_size:
            continue
            
        for target, t_stats in zip(targets, target_stats):
            if not can_be_isomorphic(q_stats, t_stats):
                continue
                
            if node_anchored:
                # Sample anchors for large graphs
                if target.number_of_nodes() > sample_anchors:
                    anchors = random.sample(list(target.nodes), sample_anchors)
                else:
                    anchors = list(target.nodes)
                    
                for anchor in anchors:
                    inp.append((i, query, target, method, node_anchored, anchor, preserve_labels))
            else:
                inp.append((i, query, target, method, node_anchored, None, preserve_labels))
    
    print(f"Generated {len(inp)} tasks after filtering")
    n_done = 0
    
    # Process in batches to manage memory
    batch_size = 1000
    for batch_start in range(0, len(inp), batch_size):
        batch = inp[batch_start:batch_start + batch_size]
        for i, n in pool.imap_unordered(count_graphlets_helper, batch):
            print(f"Processed {n_done}/{len(inp)} tasks, queries with matches: {len(n_matches)}",
                  end="\r")
            n_matches[i] += n
            n_done += 1
    
    print("\nDone counting")
    return [n_matches[i] for i in range(len(queries))]

def count_exact(queries, targets, args):
    """
    Alternative implementation that doesn't rely on orca.
    Uses networkx for graphlet counting instead.
    """
    import networkx as nx
    from collections import defaultdict
    
    def get_all_5_node_graphlets():
        """Generate all possible connected 5-node graphlets"""
        all_5_node_graphs = []
        for n_edges in range(4, 11):  # Min edges for connected graph to max possible edges
            for edges in combinations(combinations(range(5), 2), n_edges):
                G = nx.Graph()
                G.add_nodes_from(range(5))
                G.add_edges_from(edges)
                if nx.is_connected(G):
                    # Check if this graph is isomorphic to any we've already found
                    is_new = True
                    for existing in all_5_node_graphs:
                        if nx.is_isomorphic(G, existing):
                            is_new = False
                            break
                    if is_new:
                        all_5_node_graphs.append(G)
        return all_5_node_graphs

    n_matches_baseline = defaultdict(int)
    
    # Handle 5-node graphlets
    five_node_graphlets = get_all_5_node_graphlets()
    for target in targets:
        for graphlet in five_node_graphlets:
            count = 0
            if args.node_anchored:
                for node in target.nodes():
                    # Set anchor attribute
                    nx.set_node_attributes(target, 0, name="anchor")
                    target.nodes[node]["anchor"] = 1
                    
                    matcher = nx.isomorphism.GraphMatcher(
                        target, graphlet,
                        node_match=nx.isomorphism.categorical_node_match(["anchor"], [0])
                    )
                    if args.count_method == "bin":
                        count += int(matcher.subgraph_is_isomorphic())
                    else:
                        count += len(list(matcher.subgraph_isomorphisms_iter()))
            else:
                matcher = nx.isomorphism.GraphMatcher(target, graphlet)
                if args.count_method == "bin":
                    count += int(matcher.subgraph_is_isomorphic())
                else:
                    count += len(list(matcher.subgraph_isomorphisms_iter()))
            
            n_matches_baseline[len(graphlet)] += count

    # Get top 10 counts for 5-node graphlets
    counts5 = sorted([count for size, count in n_matches_baseline.items() 
                     if size == 5], reverse=True)[:10]
    
    # Handle 6-node graphlets using the provided atlas approach
    atlas_6 = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g) and len(g) == 6]
    queries_6 = []
    
    for g in atlas_6:
        for v in g.nodes:
            g_copy = g.copy()
            nx.set_node_attributes(g_copy, 0, name="anchor")
            g_copy.nodes[v]["anchor"] = 1
            is_dup = False
            for g2 in queries_6:
                if nx.is_isomorphic(g_copy, g2, 
                    node_match=lambda a, b: a["anchor"] == b["anchor"] if args.node_anchored else None):
                    is_dup = True
                    break
            if not is_dup:
                queries_6.append(g_copy)

    n_matches_6 = count_graphlets(queries_6, targets,
        n_workers=args.n_workers, method=args.count_method,
        node_anchored=args.node_anchored,
        min_count=10000)

    # Get top 20 counts for 6-node graphlets
    counts6 = sorted(n_matches_6, reverse=True)[:20]

    print("Average for size 5:", np.mean(np.log10(counts5)))
    print("Average for size 6:", np.mean(np.log10(counts6)))
    
    return counts5 + counts6

def load_neo4j_graph(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        graph = nx.Graph()
        
        # Add nodes with their attributes
        # data['nodes'] is a list of tuples: (node_id, attribute_dict)
        for node_id, node_attrs in data['nodes']:
            graph.add_node(node_id, **node_attrs)
            
        # Add edges with their attributes
        # data['edges'] is a list of tuples: (source, target, attribute_dict)
        for src, dst, edge_attrs in data['edges']:
            graph.add_edge(src, dst, **edge_attrs)
            
        return graph

if __name__ == "__main__":
    args = arg_parse()
    print("Using {} workers".format(args.n_workers))
    print("Baseline:", args.baseline)

    if args.dataset.endswith('.pkl'):
        print(f"Loading Neo4j graph from {args.dataset}")
        try:
            graph = load_neo4j_graph(args.dataset)
            print(f"Loaded Neo4j graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            dataset = [graph]
        except Exception as e:
            print(f"Error loading graph: {str(e)}")
            raise
    elif args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    elif args.dataset == 'cox2':
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
    elif args.dataset == 'reddit-binary':
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
    elif args.dataset == 'coil':
        dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
    elif args.dataset == 'ppi-pathways':
        graph = nx.Graph()
        with open("data/ppi-pathways.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                graph.add_edge(int(row[0]), int(row[1]))
        dataset = [graph]
    elif args.dataset in ['diseasome', 'usroads', 'mn-roads', 'infect']:
        fn = {"diseasome": "bio-diseasome.mtx",
            "usroads": "road-usroads.mtx",
            "mn-roads": "mn-roads.mtx",
            "infect": "infect-dublin.edges"}
        graph = nx.Graph()
        with open("data/{}".format(fn[args.dataset]), "r") as f:
            for line in f:
                if not line.strip(): continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        dataset = [graph]
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = decoder.make_plant_dataset(size)
    elif args.dataset == "analyze":
        with open("results/analyze.p", "rb") as f:
            cand_patterns, _ = pickle.load(f)
            queries = [q for score, q in cand_patterns[10]][:200]
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    targets = []
    for i in range(len(dataset)):
        graph = dataset[i]
        if not type(graph) == nx.Graph:
            graph = pyg_utils.to_networkx(dataset[i]).to_undirected()
        targets.append(graph)

    if args.dataset != "analyze":
        with open(args.queries_path, "rb") as f:
            queries = pickle.load(f)
            
    query_lens = [len(query) for query in queries]

    if args.baseline == "exact":
        n_matches_baseline = count_exact(queries, targets, args)
        n_matches = count_graphlets(queries[:len(n_matches_baseline)], targets,
            n_workers=args.n_workers, method=args.count_method,
            node_anchored=args.node_anchored, preserve_labels=args.preserve_labels)
    elif args.baseline == "none":
        n_matches = count_graphlets(queries, targets,
            n_workers=args.n_workers, method=args.count_method,
            node_anchored=args.node_anchored, preserve_labels=args.preserve_labels)
    else:
        baseline_queries = gen_baseline_queries(queries, targets,
            node_anchored=args.node_anchored, method=args.baseline)
        query_lens = [len(q) for q in baseline_queries]
        n_matches = count_graphlets(baseline_queries, targets,
            n_workers=args.n_workers, method=args.count_method,
            node_anchored=args.node_anchored, preserve_labels=args.preserve_labels)
            
    with open(args.out_path, "w") as f:
        json.dump((query_lens, n_matches, []), f)
