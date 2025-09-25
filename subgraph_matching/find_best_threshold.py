"""
Find the best threshold for subgraph matching by testing on the saved G1, G2, G3 graphs.

G1+G2 should be True (G2 is subgraph of G1), G1+G3 should be False (G3 is not subgraph of G1).
"""

import pickle
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from query_subgraph import query_subgraph

def find_best_threshold():
    """Test different thresholds on the saved graphs and find the best one."""

    # Load saved graphs
    graphs_dir = os.path.dirname(__file__)
    G1 = pickle.load(open(os.path.join(graphs_dir, 'G1.pkl'), 'rb'))
    G2 = pickle.load(open(os.path.join(graphs_dir, 'G2.pkl'), 'rb'))
    G3 = pickle.load(open(os.path.join(graphs_dir, 'G3.pkl'), 'rb'))

    print(f"G1: {len(G1)} nodes, {G1.number_of_edges()} edges")
    print(f"G2: {len(G2)} nodes, {G2.number_of_edges()} edges")
    print(f"G3: {len(G3)} nodes, {G3.number_of_edges()} edges")
    print()

    # Test thresholds from 0 to 10 in steps of 0.1
    thresholds = np.arange(0, 10.1, 0.1)

    max_threshold_with_at_least_1 = None
    best_thresh = 0,0
    for thresh in thresholds:
        # Test G1 + G2 (should be True)
        result_pos = query_subgraph(G1, G2, threshold=thresh)

        # Test G1 + G3 (should be False)
        result_neg = query_subgraph(G1, G3, threshold=thresh)

        if result_pos == True:  x = 1 
        else: x = 0 

        if result_neg == False: y = 1 
        else: y = 0  
        
        correct = x + y

        if correct > best_thresh[1]:
            best_thresh = (thresh, correct)
            
        print(f"Threshold {thresh:.3f}: \n\tG1+G2={result_pos} (expected True) \n\tG1+G3={result_neg} (expected False) \n\t- Correct: {correct}/2\n\t- the best we have got is {best_thresh}\n")

        if correct == 2: 
            print(f"we have got the answer ... {best_thresh}")
            return best_thresh[0]
        
    print(f"\nBest threshold: {best_thresh}")

    return best_thresh[0]

if __name__ == "__main__":
    find_best_threshold()