"""
Simple subgraph querying functionality for the neural subgraph matcher.
"""

import os
import torch
import networkx as nx
from subgraph_matching.alignment import gen_alignment_matrix
from subgraph_matching.train import build_model
from subgraph_matching.config import parse_encoder
from common import utils
import argparse

def query_subgraph(target_graph, query_graph, threshold=2.5): 
    """
    Check if query_graph is a subgraph of target_graph using the trained model.

    Args:
        target_graph: NetworkX graph (target)
        query_graph: NetworkX graph (query)
        threshold: Confidence threshold for matching (default: 0.9)

    Returns:
        bool: True if query is likely a subgraph of target, False otherwise
    """
    try:
        # Parse arguments for model building
        parser = argparse.ArgumentParser()
        utils.parse_optimizer(parser)
        parse_encoder(parser)
        args = parser.parse_args([])  # Use default arguments
        args.test = True

        # Build the model
        model = build_model(args)

        # Generate alignment matrix
        alignment_matrix = gen_alignment_matrix(model, query_graph, target_graph,
                                              method_type=args.method_type)

        # Check if there's any alignment above threshold
        max_alignment = alignment_matrix.max()
        model_decision = max_alignment <= threshold

        # For conformity checking: verify with exact subgraph isomorphism
        from networkx.algorithms import isomorphism
        GM = isomorphism.GraphMatcher(target_graph, query_graph)
        exact_result = GM.subgraph_is_isomorphic()

        # Log conformity (optional - can be removed for production)
        if model_decision != exact_result:
            print(f"Model decision ({model_decision}) differs from exact check ({exact_result}). "
                  f"Alignment score: {max_alignment:.4f}")

        # Return model's decision (neural-based)
        return model_decision

    except Exception as e:
        print(f"Error in subgraph querying: {e}")
        return False

# For backward compatibility
def is_subgraph(query, target):
    """Alias for query_subgraph"""
    return query_subgraph(target, query)