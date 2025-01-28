import argparse
import csv
from itertools import combinations
import time
import os
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from deepsnap.batch import Batch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder

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
from concurrent.futures import ThreadPoolExecutor

class SearchAgent:
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20):
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.model = model
        self.dataset = dataset
        self.embs = embs
        self.node_anchored = node_anchored
        self.analyze = analyze
        self.model_type = model_type
        self.out_batch_size = out_batch_size
        
        # Pre-compute adjacency matrices for faster operations
        self.adj_matrices = [nx.to_scipy_sparse_array(g, format='csr') 
                           for g in self.dataset]
        # Pre-compute node degrees
        self.node_degrees = [np.array(g.degree)[:, 1] for g in self.dataset]
        
        # Batch size for embedding computation
        self.batch_size = 128

    def _batch_compute_embeddings(self, cand_neighs: List[nx.Graph], 
                                anchors: List[int] = None) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(cand_neighs), self.batch_size):
            batch = cand_neighs[i:i + self.batch_size]
            batch_anchors = anchors[i:i + self.batch_size] if anchors else None
            with torch.no_grad():
                embs = self.model.emb_model(utils.batch_nx_graphs(
                    batch, anchors=batch_anchors if self.node_anchored else None))
            all_embs.append(embs)
        return torch.cat(all_embs, dim=0)

    def _get_neighbors_fast(self, graph_idx: int, nodes: Set[int]) -> Set[int]:
        adj = self.adj_matrices[graph_idx]
        neighbor_matrix = adj[list(nodes), :]
        return set(neighbor_matrix.indices)

    def run_search(self, n_trials=1000): 
        self.cand_patterns = defaultdict(list)
        self.counts = defaultdict(lambda: defaultdict(list))
        self.n_trials = n_trials

        self.init_search()
        while not self.is_search_done():
            self.step()
        return self.finish_search()

    def init_search():
        raise NotImplementedError

    def step(self):
        """ Abstract method for executing a search step.
        Every step adds a new node to the subgraph pattern.
        Run_search calls step at least min_pattern_size times to generate a pattern of at least this
        size. To be inherited by concrete search strategy implementations.
        """
        raise NotImplementedError

class MCTSSearchAgent(SearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, c_uct=0.7):
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size)
        self.c_uct = c_uct
        assert not analyze

    def init_search(self):
        self.wl_hash_to_graphs = defaultdict(list)
        self.cum_action_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(float))
        self.visited_seed_nodes = set()
        self.max_size = self.min_pattern_size
        
        # Initialize cached data structures
        self.node_neighbors = defaultdict(set)
        self.pattern_cache = {}

    def is_search_done(self):
        return self.max_size == self.max_pattern_size + 1

    def has_min_reachable_nodes(self, graph_idx: int, start_node: int, n: int) -> bool:
        adj = self.adj_matrices[graph_idx]
        visited = {start_node}
        frontier = {start_node}
        
        for _ in range(n):
            if len(visited) >= n:
                return True
            new_frontier = set()
            for node in frontier:
                neighbors = set(adj[node].indices) - visited
                new_frontier.update(neighbors)
                visited.update(neighbors)
            frontier = new_frontier
            if not frontier:
                break
        return len(visited) >= n

    def step(self):
        ps = np.array([len(g) for g in self.dataset], dtype=np.float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        print("Size", self.max_size)
        print(len(self.visited_seed_nodes), "distinct seeds")
        
        def process_simulation(simulation_n):
            # Similar logic as before but with optimized operations
            best_graph_idx, best_start_node, best_score = None, None, -float("inf")
            for cand_graph_idx, cand_start_node in self.visited_seed_nodes:
                state = cand_graph_idx, cand_start_node
                my_visit_counts = sum(self.visit_counts[state].values())
                q_score = (sum(self.cum_action_values[state].values()) /
                    (my_visit_counts or 1))
                uct_score = self.c_uct * np.sqrt(np.log(simulation_n or 1) /
                    (my_visit_counts or 1))
                node_score = q_score + uct_score
                if node_score > best_score:
                    best_score = node_score
                    best_graph_idx = cand_graph_idx
                    best_start_node = cand_start_node

            if best_score >= self.c_uct * np.sqrt(np.log(simulation_n or 1)):
                graph_idx, start_node = best_graph_idx, best_start_node
                graph = self.dataset[graph_idx]
            else:
                found = False
                while not found:
                    graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
                    graph = self.dataset[graph_idx]
                    start_node = random.choice(list(graph.nodes))
                    if self.has_min_reachable_nodes(graph_idx, start_node,
                        self.min_pattern_size):
                        found = True
                self.visited_seed_nodes.add((graph_idx, start_node))

            # Use optimized neighbor computation
            neigh = [start_node]
            frontier = list(self._get_neighbors_fast(graph_idx, {start_node}))
            visited = {start_node}
            
            # Batch process candidate evaluations
            while frontier and len(neigh) < self.max_size:
                cand_neighs = []
                for cand_node in frontier:
                    cand_neigh = graph.subgraph(neigh + [cand_node])
                    cand_neighs.append(cand_neigh)

                # Batch compute embeddings
                cand_embs = self._batch_compute_embeddings(
                    cand_neighs,
                    anchors=[neigh[0]] * len(cand_neighs) if self.node_anchored else None
                )

                scores = torch.zeros(len(frontier), device=cand_embs.device)
                for emb_batch in self.embs:
                    emb_batch = emb_batch.to(cand_embs.device)
                    batch_preds = self.model.predict((emb_batch, cand_embs))
                    if self.model_type == "order":
                        batch_scores = -torch.sum(torch.argmax(
                            self.model.clf_model(batch_preds.unsqueeze(1)), axis=1
                        ), dim=0)
                    else:  # mlp
                        batch_scores = torch.sum(self.model(
                            emb_batch,
                            cand_embs.unsqueeze(0).expand(len(emb_batch), -1)
                        )[:, 0], dim=0)
                    scores += batch_scores

                best_idx = scores.argmax().item()
                best_node = frontier[best_idx]
                
                # Update frontier efficiently using pre-computed adjacency
                new_neighbors = self._get_neighbors_fast(graph_idx, {best_node})
                frontier = list((set(frontier) | new_neighbors) - visited - {best_node})
                visited.add(best_node)
                neigh.append(best_node)

                # Update pattern cache
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                
                pattern_hash = utils.wl_hash(neigh_g, node_anchored=self.node_anchored)
                self.wl_hash_to_graphs[pattern_hash].append(neigh_g)

            return neigh, graph_idx, scores[best_idx].item()

        # Process simulations in parallel batches
        n_workers = min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            simulation_range = range(self.n_trials //
                (self.max_pattern_size + 1 - self.min_pattern_size))
            results = list(executor.map(process_simulation, simulation_range))

        # Update state based on simulation results
        for neigh, graph_idx, score in results:
            graph = self.dataset[graph_idx]
            neigh_g = graph.subgraph(neigh).copy()
            pattern_hash = utils.wl_hash(neigh_g, node_anchored=self.node_anchored)
            
            # Update visit counts and values
            state = (graph_idx, neigh[0])
            self.cum_action_values[state][pattern_hash] += -np.log(score + 1) + 1
            self.visit_counts[state][pattern_hash] += 1

        self.max_size += 1

    def finish_search(self):
        counts = defaultdict(lambda: defaultdict(int))
        for _, v in self.visit_counts.items():
            for s2, count in v.items():
                counts[len(random.choice(self.wl_hash_to_graphs[s2]))][s2] += count

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size+1):
            for wl_hash, count in sorted(counts[pattern_size].items(),
                key=lambda x: x[1], reverse=True)[:self.out_batch_size]:
                cand_patterns_uniq.append(random.choice(
                    self.wl_hash_to_graphs[wl_hash]))
                print("- outputting", count, "motifs of size", pattern_size)
        return cand_patterns_uniq

class GreedySearchAgent(SearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, rank_method="counts",
        model_type="order", out_batch_size=20, n_beams=1):
        """Greedy implementation of the subgraph pattern search.
        Args:
            rank_method: 'counts', 'margin', or 'hybrid' for action ranking
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size)
        self.rank_method = rank_method
        self.n_beams = n_beams
        print("Rank Method:", rank_method)
        
        # Pre-compute adjacency matrices for faster operations
        self.adj_matrices = [nx.to_scipy_sparse_array(g, format='csr') 
                           for g in self.dataset]

    def _get_neighbors_fast(self, graph_idx, nodes):
        """Efficiently get neighbors using pre-computed sparse adjacency matrix."""
        adj = self.adj_matrices[graph_idx]
        neighbor_matrix = adj[list(nodes), :]
        return set(neighbor_matrix.indices)

    def init_search(self):
        ps = np.array([len(g) for g in self.dataset], dtype=np.float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        beams = []
        # Process trials in batches to reduce memory pressure
        for trial in range(self.n_trials):
            graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
            graph = self.dataset[graph_idx]
            start_node = random.choice(list(graph.nodes))
            neigh = [start_node]
            # Use optimized neighbor finding
            frontier = list(self._get_neighbors_fast(graph_idx, {start_node}))
            visited = set([start_node])
            beams.append([(0, neigh, frontier, visited, graph_idx)])
        self.beam_sets = beams
        self.analyze_embs = [] if self.analyze else None

    def is_search_done(self):
        return len(self.beam_sets) == 0

    def step(self):
        new_beam_sets = []
        print("seeds come from", len(set(b[0][-1] for b in self.beam_sets)),
            "distinct graphs")
        analyze_embs_cur = [] if self.analyze else None

        for beam_set in tqdm(self.beam_sets):
            new_beams = []
            for _, neigh, frontier, visited, graph_idx in beam_set:
                graph = self.dataset[graph_idx]
                if len(neigh) >= self.max_pattern_size or not frontier:
                    continue

                # Process candidates in batches
                batch_size = 128  # Adjust based on available memory
                for i in range(0, len(frontier), batch_size):
                    batch_frontier = frontier[i:i + batch_size]
                    
                    # Create subgraphs for batch
                    cand_neighs = [graph.subgraph(neigh + [node]) 
                                 for node in batch_frontier]
                    anchors = [neigh[0] for _ in batch_frontier] if self.node_anchored else None

                    # Compute embeddings for batch
                    with torch.no_grad():
                        cand_embs = self.model.emb_model(utils.batch_nx_graphs(
                            cand_neighs, anchors=anchors))

                    # Score computation
                    for node_idx, (cand_node, cand_emb) in enumerate(zip(batch_frontier, cand_embs)):
                        score = 0
                        n_embs = 0
                        
                        # Process embeddings in batches
                        for emb_batch in self.embs:
                            n_embs += len(emb_batch)
                            device = utils.get_device()
                            
                            if self.model_type == "order":
                                preds = self.model.predict((
                                    emb_batch.to(device),
                                    cand_emb
                                ))
                                batch_score = -torch.sum(torch.argmax(
                                    self.model.clf_model(preds.unsqueeze(1)), 
                                    axis=1
                                )).item()
                            else:  # mlp
                                batch_score = torch.sum(self.model(
                                    emb_batch.to(device),
                                    cand_emb.unsqueeze(0).expand(len(emb_batch), -1)
                                )[:,0]).item()
                                
                            score += batch_score

                        # Use optimized neighbor finding for new frontier
                        new_frontier = list(self._get_neighbors_fast(graph_idx, {cand_node}) - 
                                         visited - {cand_node})
                        
                        new_beams.append((
                            score,
                            neigh + [cand_node],
                            new_frontier,
                            visited | {cand_node},
                            graph_idx
                        ))

            # Keep top-k beams
            new_beams = list(sorted(new_beams, key=lambda x: x[0]))[:self.n_beams]
            
            # Update patterns
            for score, neigh, frontier, visited, graph_idx in new_beams[:1]:
                graph = self.dataset[graph_idx]
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                
                # Set anchor nodes
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                
                # Add to records
                self.cand_patterns[len(neigh_g)].append((score, neigh_g))
                
                if self.rank_method in ["counts", "hybrid"]:
                    pattern_hash = utils.wl_hash(neigh_g, node_anchored=self.node_anchored)
                    self.counts[len(neigh_g)][pattern_hash].append(neigh_g)
                
                if self.analyze and len(neigh) >= 3:
                    with torch.no_grad():
                        emb = self.model.emb_model(utils.batch_nx_graphs(
                            [neigh_g], 
                            anchors=[neigh[0]] if self.node_anchored else None
                        )).squeeze(0)
                        analyze_embs_cur.append(emb.detach().cpu().numpy())
                        
            if len(new_beams) > 0:
                new_beam_sets.append(new_beams)
                
        self.beam_sets = new_beam_sets
        if self.analyze:
            self.analyze_embs.append(analyze_embs_cur)

    def finish_search(self):
        if self.analyze:
            print("Saving analysis info in results/analyze.p")
            with open("results/analyze.p", "wb") as f:
                pickle.dump((self.cand_patterns, self.analyze_embs), f)
            
            xs, ys = [], []
            for embs_ls in self.analyze_embs:
                for emb in embs_ls:
                    xs.append(emb[0])
                    ys.append(emb[1])
                    
            print("Saving analysis plot in results/analyze.png")
            plt.scatter(xs, ys, color="red", label="motif")
            plt.legend()
            plt.savefig("plots/analyze.png")
            plt.close()

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):
            if self.rank_method == "hybrid":
                cur_rank_method = "margin" if len(max(
                    self.counts[pattern_size].values(), key=len)) < 3 else "counts"
            else:
                cur_rank_method = self.rank_method

            if cur_rank_method == "margin":
                wl_hashes = set()
                cands = self.cand_patterns[pattern_size]
                cand_patterns_uniq_size = []
                for pattern in sorted(cands, key=lambda x: x[0]):
                    wl_hash = utils.wl_hash(pattern[1], node_anchored=self.node_anchored)
                    if wl_hash not in wl_hashes:
                        wl_hashes.add(wl_hash)
                        cand_patterns_uniq_size.append(pattern[1])
                        if len(cand_patterns_uniq_size) >= self.out_batch_size:
                            break
                cand_patterns_uniq.extend(cand_patterns_uniq_size)
            elif cur_rank_method == "counts":
                for _, neighs in list(sorted(self.counts[pattern_size].items(),
                    key=lambda x: len(x[1]), reverse=True))[:self.out_batch_size]:
                    cand_patterns_uniq.append(random.choice(neighs))
            else:
                print("Unrecognized rank method")
                
        return cand_patterns_uniq
