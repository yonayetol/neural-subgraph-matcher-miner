import argparse
import csv
from itertools import combinations
import time
import os

from deepsnap.batch import Batch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
from subgraph_mining.search_agents import GreedySearchAgent, MCTSSearchAgent

import matplotlib.pyplot as plt

import random
from scipy.io import mmread
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA

def prepare_nx_graph(g):
    """Prepare NetworkX graph by ensuring proper node and edge attributes"""
    cleaned_graph = nx.Graph()
    
    # Add nodes with id and label properties
    for node in g.nodes():
        node_attrs = {
            'id': g.nodes[node].get('id', str(node)),
            'label': g.nodes[node].get('label', 'default')
        }
        cleaned_graph.add_node(node, **node_attrs)
    
    # Add edges with source, target, and type properties
    for u, v in g.edges():
        edge_attrs = {
            'source': u,
            'target': v,
            'type': g.edges[u, v].get('type', 'default')
        }
        cleaned_graph.add_edge(u, v, **edge_attrs)
        
    return cleaned_graph

def batch_nx_graphs(graphs, anchors=None):
    """Create a batch from a list of NetworkX graphs"""
    # Clean and prepare graphs
    prepared_graphs = [prepare_nx_graph(g) for g in graphs]
    
    # Convert to DeepSnap format with specific attributes
    ds_graphs = []
    for i, g in enumerate(prepared_graphs):
        if anchors is not None:
            # Add anchor information if provided
            nx.set_node_attributes(g, {anchors[i]: True}, 'anchor')
        
        # Create DSGraph with specific edge and node attributes
        ds_graph = DSGraph(g, 
                          node_attrs=['id', 'label'],
                          edge_attrs=['source', 'target', 'type'])
        ds_graphs.append(ds_graph)
    
    return Batch.from_data_list(ds_graphs)

def make_plant_dataset(size):
    generator = combined_syn.get_generator([size])
    random.seed(3001)
    np.random.seed(14853)
    
    # Generate pattern with proper attributes
    pattern = generator.generate(size=10)
    for node in pattern.nodes():
        pattern.nodes[node]['id'] = str(node)
        pattern.nodes[node]['label'] = 'pattern_node'
    for u, v in pattern.edges():
        pattern.edges[u, v]['type'] = 'pattern_edge'
        
    pattern = prepare_nx_graph(pattern)
    nx.draw(pattern, with_labels=True)
    plt.savefig("plots/cluster/plant-pattern.png")
    plt.close()
    
    graphs = []
    for i in range(1000):
        graph = generator.generate()
        # Add attributes to the generated graph
        for node in graph.nodes():
            graph.nodes[node]['id'] = f"{i}_{node}"
            graph.nodes[node]['label'] = 'graph_node'
        for u, v in graph.edges():
            graph.edges[u, v]['type'] = 'graph_edge'
            
        n_old = len(graph)
        graph = nx.disjoint_union(graph, pattern)
        
        # Add connecting edges with attributes
        for j in range(1, 3):
            u = random.randint(0, n_old - 1)
            v = random.randint(n_old, len(graph) - 1)
            graph.add_edge(u, v, type='connecting_edge')
            
        graph = prepare_nx_graph(graph)
        graphs.append(graph)
    return graphs

def pattern_growth(dataset, task, args):
    # init model
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path,
        map_location=utils.get_device()))

    if task == "graph-labeled":
        dataset, labels = dataset

    # load data
    neighs_pyg, neighs = [], []
    print(len(dataset), "graphs")
    print("search strategy:", args.search_strategy)
    if task == "graph-labeled": print("using label 0")
    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0: continue
        if task == "graph-truncate" and i >= 1000: break
        if not type(graph) == nx.Graph:
            graph = pyg_utils.to_networkx(graph).to_undirected()
        graphs.append(graph)
    if args.use_whole_graphs:
        neighs = graphs
    else:
        anchors = []
        if args.sample_method == "radial":
            for i, graph in enumerate(graphs):
                print(i)
                for j, node in enumerate(graph.nodes):
                    if len(dataset) <= 10 and j % 100 == 0: print(i, j)
                    if args.use_whole_graphs:
                        neigh = graph.nodes
                    else:
                        neigh = list(nx.single_source_shortest_path_length(graph,
                            node, cutoff=args.radius).keys())
                        if args.subgraph_sample_size != 0:
                            neigh = random.sample(neigh, min(len(neigh),
                                args.subgraph_sample_size))
                    if len(neigh) > 1:
                        neigh = graph.subgraph(neigh)
                        if args.subgraph_sample_size != 0:
                            neigh = neigh.subgraph(max(
                                nx.connected_components(neigh), key=len))
                        neigh = nx.convert_node_labels_to_integers(neigh)
                        neigh.add_edge(0, 0)
                        neighs.append(neigh)
        elif args.sample_method == "tree":
            start_time = time.time()
            for j in tqdm(range(args.n_neighborhoods)):
                graph, neigh = utils.sample_neigh(graphs,
                    random.randint(args.min_neighborhood_size,
                        args.max_neighborhood_size))
                neigh = graph.subgraph(neigh)
                neigh = nx.convert_node_labels_to_integers(neigh)
                neigh.add_edge(0, 0)
                neighs.append(neigh)
                if args.node_anchored:
                    anchors.append(0)   # after converting labels, 0 will be anchor

    embs = []
    if len(neighs) % args.batch_size != 0:
        print("WARNING: number of graphs not multiple of batch size")
    for i in range(len(neighs) // args.batch_size):
        top = (i+1)*args.batch_size
        with torch.no_grad():
            try:
                batch = batch_nx_graphs(neighs[i*args.batch_size:top],
                    anchors=anchors[i*args.batch_size:top] if args.node_anchored else None)
                emb = model.emb_model(batch)
                emb = emb.to(torch.device("cpu"))
                embs.append(emb)
            except Exception as e:
                print(f"Warning: Batch processing failed for batch {i}: {str(e)}")
                continue

    if args.analyze:
        embs_np = torch.stack(embs).numpy()
        plt.scatter(embs_np[:,0], embs_np[:,1], label="node neighborhood")

    if args.search_strategy == "mcts":
        assert args.method_type == "order"
        agent = MCTSSearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, out_batch_size=args.out_batch_size)
    elif args.search_strategy == "greedy":
        agent = GreedySearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size)
    out_graphs = agent.run_search(args.n_trials)
    print(time.time() - start_time, "TOTAL TIME")
    x = int(time.time() - start_time)
    print(x // 60, "mins", x % 60, "secs")

    count_by_size = defaultdict(int)
    for pattern in out_graphs:
        plt.figure(figsize=(12, 8))
        
        labels = {
            node: f"ID: {pattern.nodes[node].get('id', 'N/A')}\n"
                  f"Label: {pattern.nodes[node].get('label', 'N/A')}"
            for node in pattern.nodes
        }
        
        pos = nx.spring_layout(pattern)
        
        if args.node_anchored:
            colors = ["red"] + ["blue"] * (len(pattern) - 1)
            nx.draw_networkx_nodes(pattern, pos, node_color=colors)
        else:
            nx.draw_networkx_nodes(pattern, pos, node_color='lightblue')
        
        nx.draw_networkx_edges(pattern, pos)
        nx.draw_networkx_labels(pattern, pos, labels, font_size=8)
        
        edge_labels = {
            (u, v): pattern.edges[u, v].get('type', 'N/A')
            for u, v in pattern.edges
        }
        nx.draw_networkx_edge_labels(pattern, pos, edge_labels, font_size=8)
        
        plt.title(f"Pattern Size: {len(pattern)}")
        plt.axis('off')
        
        output_path = "plots/cluster/{}-{}".format(len(pattern), count_by_size[len(pattern)])
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
        plt.close()
        
        count_by_size[len(pattern)] += 1

    if not os.path.exists("results"):
        os.makedirs("results")
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)

def main():
    if not os.path.exists("plots/cluster"):
        os.makedirs("plots/cluster")

    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    print("Using dataset {}".format(args.dataset))
    task = 'graph'  # Default task type
    
    if args.dataset.endswith('.pkl'):
        try:
            with open(args.dataset, 'rb') as f:
                graphs = pickle.load(f)
            if isinstance(graphs, list):
                dataset = graphs
            elif isinstance(graphs, nx.Graph):
                dataset = [graphs]
            else:
                raise ValueError(f"Unsupported data format in {args.dataset}")
            print(f"Loaded {len(dataset)} graphs from {args.dataset}")
        except Exception as e:
            print(f"Error loading pickle file: {str(e)}")
            return
    elif args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
    elif args.dataset == 'cox2':
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
        task = 'graph'
    elif args.dataset == 'reddit-binary':
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
        task = 'graph'
    elif args.dataset == 'dblp':
        dataset = TUDataset(root='/tmp/dblp', name='DBLP_v1')
        task = 'graph-truncate'
    elif args.dataset == 'coil':
        dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
        task = 'graph'
    elif args.dataset.startswith('roadnet-'):
        graph = nx.Graph()
        with open("data/{}.txt".format(args.dataset), "r") as f:
            for row in f:
                if not row.startswith("#"):
                    a, b = row.split("\t")
                    graph.add_edge(int(a), int(b))
        dataset = [graph]
        task = 'graph'
    elif args.dataset == "ppi":
        dataset = PPI(root="/tmp/PPI")
        task = 'graph'
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
        task = 'graph'
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = make_plant_dataset(size)
        task = 'graph'

    pattern_growth(dataset, task, args) 

if __name__ == '__main__':
    main()