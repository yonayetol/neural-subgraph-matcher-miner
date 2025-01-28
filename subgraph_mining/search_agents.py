import argparse
import csv
from itertools import combinations
import time
import os
import gc
from typing import List, Set, Dict, Tuple, Optional, Union
import psutil

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

from common import utils 

class MemoryManager:
    """Utility class to manage memory usage"""
    @staticmethod
    def check_memory_usage() -> float:
        return psutil.Process().memory_percent()

    @staticmethod
    def clear_memory(threshold: float = 0.8):
        if MemoryManager.check_memory_usage() > threshold:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class SearchAgent:
    """Base class for search strategies"""
    def __init__(self, min_pattern_size: int, max_pattern_size: int, model: nn.Module, 
                 dataset: List[nx.Graph], embs: torch.Tensor, node_anchored: bool = False, 
                 analyze: bool = False, model_type: str = "order", out_batch_size: int = 20, 
                 batch_size: int = 1000, memory_threshold: float = 0.8):
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.model = model
        self.dataset = dataset
        self.embs = embs
        self.node_anchored = node_anchored
        self.analyze = analyze
        self.model_type = model_type
        self.out_batch_size = out_batch_size
        self.batch_size = batch_size
        self.memory_threshold = memory_threshold

    @staticmethod
    def has_min_reachable_nodes(graph: nx.Graph, start_node: int, n: int, max_depth: int = None) -> bool:
        """Memory-efficient BFS implementation"""
        if max_depth is None:
            max_depth = n
        
        visited = {start_node}
        current_level = {start_node}
        
        for depth in range(max_depth):
            if len(visited) >= n:
                return True
                
            next_level = set()
            for node in current_level:
                neighbors = set(graph.neighbors(node)) - visited
                next_level.update(neighbors)
                if len(visited) + len(next_level) >= n:
                    return True
            
            visited.update(next_level)
            current_level = next_level
            
            if not current_level:
                break
        
        return len(visited) >= n

    def process_embeddings_batch(self, cand_emb: torch.Tensor, 
                               emb_batch: torch.Tensor) -> Tuple[float, int]:
        """Process embeddings in batches to manage memory"""
        score = 0
        n_embs = 0
        
        for i in range(0, len(emb_batch), self.batch_size):
            batch = emb_batch[i:i + self.batch_size].to(utils.get_device())
            score += torch.sum(self.model.predict((batch, cand_emb))).item()
            n_embs += len(batch)
            del batch
            MemoryManager.clear_memory(self.memory_threshold)
            
        return score, n_embs

class MCTSSearchAgent(SearchAgent):
    """Monte Carlo Tree Search implementation"""
    def __init__(self, *args, c_uct: float = 0.7, n_trials: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_uct = c_uct
        self.n_trials = n_trials

    def init_search(self):
        """Initialize search parameters"""
        self.wl_hash_to_graphs = defaultdict(list)
        self.cum_action_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(float))
        self.visited_seed_nodes = set()
        self.max_size = self.min_pattern_size

    def is_search_done(self) -> bool:
        """Check if search is complete"""
        return self.max_size == self.max_pattern_size + 1

    def _find_best_seed(self, simulation_n: int) -> Optional[Tuple[int, int]]:
        """Find the best seed node for the search"""
        best_graph_idx, best_start_node, best_score = None, None, -float("inf")
        
        for state in self.visited_seed_nodes:
            graph_idx, start_node = state
            my_visit_counts = sum(self.visit_counts[state].values())
            if my_visit_counts == 0:
                continue
                
            q_score = sum(self.cum_action_values[state].values()) / my_visit_counts
            uct_score = self.c_uct * np.sqrt(np.log(simulation_n or 1) / my_visit_counts)
            node_score = q_score + uct_score
            
            if node_score > best_score:
                best_score = node_score
                best_graph_idx = graph_idx
                best_start_node = start_node
                
        if best_score >= self.c_uct * np.sqrt(np.log(simulation_n or 1)):
            return best_graph_idx, best_start_node
        return None

    def _process_frontier(self, graph: nx.Graph, neigh: List[int], 
                         frontier: List[int], visited: Set[int], 
                         cur_state: Tuple[int, int]) -> Optional[Tuple[int, float, Tuple]]:
        """Process frontier nodes and find best next node"""
        cand_neighs, anchors = [], []
        
        for cand_node in frontier:
            cand_neigh = graph.subgraph(neigh + [cand_node])
            cand_neighs.append(cand_neigh)
            if self.node_anchored:
                anchors.append(neigh[0])

        cand_embs = self.model.emb_model(utils.batch_nx_graphs(
            cand_neighs, anchors=anchors if self.node_anchored else None))

        best_v_score, best_node_score, best_node = float("inf"), -float("inf"), None
        
        for cand_node, cand_emb in zip(frontier, cand_embs):
            score, n_embs = self.process_embeddings_batch(cand_emb, self.embs[0])
            
            v_score = -np.log(score/n_embs + 1) + 1
            neigh_g = graph.subgraph(neigh + [cand_node]).copy()
            neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
            
            for v in neigh_g.nodes:
                neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                
            next_state = utils.wl_hash(neigh_g, node_anchored=self.node_anchored)
            
            parent_visit_counts = sum(self.visit_counts[cur_state].values())
            my_visit_counts = sum(self.visit_counts[next_state].values())
            
            q_score = (sum(self.cum_action_values[next_state].values()) /
                (my_visit_counts or 1))
            uct_score = self.c_uct * np.sqrt(np.log(parent_visit_counts or 1) /
                (my_visit_counts or 1))
            node_score = q_score + uct_score
            
            if node_score > best_node_score:
                best_node_score = node_score
                best_v_score = v_score
                best_node = cand_node
                
        return best_node, best_v_score, next_state

    def step(self):
        """Execute one step of the search"""
        ps = np.array([len(g) for g in self.dataset], dtype=np.float32)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))
        
        print(f"Size {self.max_size}")
        print(f"{len(self.visited_seed_nodes)} distinct seeds")
        
        n_trials_per_size = self.n_trials // (self.max_pattern_size + 1 - self.min_pattern_size)
        
        for simulation_n in tqdm(range(n_trials_per_size)):
            MemoryManager.clear_memory(self.memory_threshold)
            
            best_state = self._find_best_seed(simulation_n)
            if best_state is None:
                found = False
                while not found:
                    graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
                    graph = self.dataset[graph_idx]
                    start_node = random.choice(list(graph.nodes))
                    if self.has_min_reachable_nodes(graph, start_node, self.min_pattern_size):
                        found = True
                self.visited_seed_nodes.add((graph_idx, start_node))
                best_state = (graph_idx, start_node)

            graph_idx, start_node = best_state
            graph = self.dataset[graph_idx]
            
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = {start_node}
            neigh_g = nx.Graph()
            neigh_g.add_node(start_node, anchor=1)
            cur_state = graph_idx, start_node
            state_list = [cur_state]
            
            while frontier and len(neigh) < self.max_size:
                result = self._process_frontier(graph, neigh, frontier, visited, cur_state)
                if result is None:
                    break
                    
                best_node, best_v_score, next_state = result
                frontier = list(((set(frontier) | set(graph.neighbors(best_node))) - visited) - {best_node})
                visited.add(best_node)
                neigh.append(best_node)
                
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                    
                cur_state = next_state
                state_list.append(cur_state)
                self.wl_hash_to_graphs[cur_state].append(neigh_g)
                
                for i in range(len(state_list) - 1):
                    self.cum_action_values[state_list[i]][state_list[i+1]] += best_v_score
                    self.visit_counts[state_list[i]][state_list[i+1]] += 1
                    
        self.max_size += 1

    def finish_search(self) -> List[nx.Graph]:
        """Complete the search and return results"""
        counts = defaultdict(lambda: defaultdict(int))
        for _, v in self.visit_counts.items():
            for s2, count in v.items():
                counts[len(random.choice(self.wl_hash_to_graphs[s2]))][s2] += count

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):
            size_patterns = sorted(counts[pattern_size].items(), key=lambda x: x[1], reverse=True)
            for wl_hash, count in size_patterns[:self.out_batch_size]:
                cand_patterns_uniq.append(random.choice(self.wl_hash_to_graphs[wl_hash]))
                print(f"- outputting {count} motifs of size {pattern_size}")
        
        return cand_patterns_uniq
    
class GreedySearchAgent(SearchAgent):
    """Memory-optimized greedy implementation of the subgraph pattern search."""
    
    def __init__(self, min_pattern_size: int, max_pattern_size: int, model: nn.Module, 
                 dataset: List[nx.Graph], embs: torch.Tensor, node_anchored: bool = False, 
                 analyze: bool = False, rank_method: str = "counts", model_type: str = "order", 
                 out_batch_size: int = 20, n_beams: int = 1, batch_size: int = 1000,
                 memory_threshold: float = 0.8):
        """Initialize the greedy search agent.
        
        Args:
            rank_method: Strategy for ranking next actions ('counts', 'margin', or 'hybrid')
            n_beams: Number of beam search paths to maintain
            Other parameters inherited from SearchAgent
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
                        embs, node_anchored, analyze, model_type, out_batch_size,
                        batch_size, memory_threshold)
        self.rank_method = rank_method
        self.n_beams = n_beams
        self.analyze_embs = []
        print(f"Rank Method: {rank_method}")

    def init_search(self):
        """Initialize search parameters and beam sets."""
        ps = np.array([len(g) for g in self.dataset], dtype=np.float32)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        beams = []
        for trial in range(self.n_trials):
            if trial % 100 == 0:  # Periodic memory cleanup
                MemoryManager.clear_memory(self.memory_threshold)
                
            graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
            graph = self.dataset[graph_idx]
            start_node = random.choice(list(graph.nodes))
            
            # Initialize beam with start node
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            beams.append([(0, neigh, frontier, visited, graph_idx)])
            
        self.beam_sets = beams
        self.cand_patterns = defaultdict(list)
        self.counts = defaultdict(lambda: defaultdict(list))

    def is_search_done(self) -> bool:
        """Check if search is complete."""
        return len(self.beam_sets) == 0

    def _evaluate_candidate(self, cand_emb: torch.Tensor, emb_batches: List[torch.Tensor]) -> float:
        """Evaluate a candidate node using batched processing."""
        score, n_embs = 0, 0
        for emb_batch in emb_batches:
            batch_score = 0
            # Process in smaller sub-batches
            for i in range(0, len(emb_batch), self.batch_size):
                sub_batch = emb_batch[i:i + self.batch_size].to(utils.get_device())
                if self.model_type == "order":
                    batch_score -= torch.sum(torch.argmax(
                        self.model.clf_model(self.model.predict((
                        sub_batch, cand_emb)).unsqueeze(1)), axis=1)).item()
                elif self.model_type == "mlp":
                    batch_score += torch.sum(self.model(
                        sub_batch,
                        cand_emb.unsqueeze(0).expand(len(sub_batch), -1)
                    )[:,0]).item()
                n_embs += len(sub_batch)
                del sub_batch
            score += batch_score
            MemoryManager.clear_memory(self.memory_threshold)
        return score, n_embs

    def _process_beam(self, beam: List[Tuple]) -> List[Tuple]:
        """Process a single beam to generate next states."""
        new_beams = []
        for _, neigh, frontier, visited, graph_idx in beam:
            graph = self.dataset[graph_idx]
            if len(neigh) >= self.max_pattern_size or not frontier:
                continue

            # Prepare candidates
            cand_neighs, anchors = [], []
            for cand_node in frontier:
                cand_neigh = graph.subgraph(neigh + [cand_node])
                cand_neighs.append(cand_neigh)
                if self.node_anchored:
                    anchors.append(neigh[0])

            # Process candidates in batches
            batch_size = min(self.batch_size, len(cand_neighs))
            for i in range(0, len(cand_neighs), batch_size):
                batch_neighs = cand_neighs[i:i + batch_size]
                batch_anchors = anchors[i:i + batch_size] if anchors else None
                
                cand_embs = self.model.emb_model(utils.batch_nx_graphs(
                    batch_neighs, anchors=batch_anchors if self.node_anchored else None))
                
                for j, (cand_node, cand_emb) in enumerate(zip(frontier[i:i + batch_size], cand_embs)):
                    score, _ = self._evaluate_candidate(cand_emb, [self.embs[0]])
                    
                    new_frontier = list(((set(frontier) |
                        set(graph.neighbors(cand_node))) - visited) -
                        set([cand_node]))
                    
                    new_beams.append((
                        score,
                        neigh + [cand_node],
                        new_frontier,
                        visited | set([cand_node]),
                        graph_idx
                    ))
                
                MemoryManager.clear_memory(self.memory_threshold)

        return sorted(new_beams, key=lambda x: x[0])[:self.n_beams]

    def step(self):
        """Execute one step of the greedy search."""
        new_beam_sets = []
        analyze_embs_cur = []
        
        print(f"Processing {len(self.beam_sets)} beam sets from",
              f"{len(set(b[0][-1] for b in self.beam_sets))} distinct graphs")
        
        for beam_idx, beam_set in enumerate(tqdm(self.beam_sets)):
            if beam_idx % 10 == 0:  # Periodic cleanup
                MemoryManager.clear_memory(self.memory_threshold)
                
            new_beams = self._process_beam(beam_set)
            
            # Record patterns
            for score, neigh, frontier, visited, graph_idx in new_beams[:1]:
                graph = self.dataset[graph_idx]
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                
                self.cand_patterns[len(neigh_g)].append((score, neigh_g))
                
                if self.rank_method in ["counts", "hybrid"]:
                    self.counts[len(neigh_g)][utils.wl_hash(neigh_g,
                        node_anchored=self.node_anchored)].append(neigh_g)
                
                if self.analyze and len(neigh) >= 3:
                    emb = self.model.emb_model(utils.batch_nx_graphs(
                        [neigh_g], anchors=[neigh[0]] if self.node_anchored
                        else None)).squeeze(0)
                    analyze_embs_cur.append(emb.detach().cpu().numpy())
            
            if len(new_beams) > 0:
                new_beam_sets.append(new_beams)
        
        self.beam_sets = new_beam_sets
        self.analyze_embs.append(analyze_embs_cur)

    def finish_search(self) -> List[nx.Graph]:
        """Complete the search and return results."""
        if self.analyze:
            self._save_analysis_results()

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):
            if self.rank_method == "hybrid":
                cur_rank_method = ("margin" if len(max(
                    self.counts[pattern_size].values(), key=len)) < 3 
                    else "counts")
            else:
                cur_rank_method = self.rank_method

            if cur_rank_method == "margin":
                patterns = self._get_margin_ranked_patterns(pattern_size)
            elif cur_rank_method == "counts":
                patterns = self._get_count_ranked_patterns(pattern_size)
            else:
                print("Unrecognized rank method")
                continue
                
            cand_patterns_uniq.extend(patterns)
            
        return cand_patterns_uniq

    def _save_analysis_results(self):
        """Save analysis results if analyze flag is True."""
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

    def _get_margin_ranked_patterns(self, pattern_size: int) -> List[nx.Graph]:
        """Get patterns ranked by margin score."""
        wl_hashes = set()
        patterns = []
        cands = self.cand_patterns[pattern_size]
        
        for pattern in sorted(cands, key=lambda x: x[0]):
            wl_hash = utils.wl_hash(pattern[1], node_anchored=self.node_anchored)
            if wl_hash not in wl_hashes:
                wl_hashes.add(wl_hash)
                patterns.append(pattern[1])
                if len(patterns) >= self.out_batch_size:
                    break
        return patterns

    def _get_count_ranked_patterns(self, pattern_size: int) -> List[nx.Graph]:
        """Get patterns ranked by frequency count."""
        patterns = []
        sorted_counts = sorted(
            self.counts[pattern_size].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for _, neighs in sorted_counts[:self.out_batch_size]:
            patterns.append(random.choice(neighs))
        return patterns