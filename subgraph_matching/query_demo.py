"""
Subgraph Querying Demo with Web Server

This module provides subgraph querying functionality with automatic web serving
of interactive visualizations on port 10000.
"""

import networkx as nx
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import webbrowser
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
import random
import pickle 
from find_best_threshold import find_best_threshold
from query_subgraph import query_subgraph

sys.path.insert(0, os.path.abspath(".."))

# Import visualization modules
try:
    from visualizer.visualizer import extract_graph_data, process_html_template
except ImportError:
    # Fallback import if running from different directory
    import visualizer.visualizer as viz
    extract_graph_data = viz.extract_graph_data
    process_html_template = viz.process_html_template

class SubgraphQueryDemo:
    """
    Demo class for subgraph querying with web visualization.
    """

    def __init__(self):
        # Set output directory relative to the script location (subgraph_matching folder)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_dir, "query_results")
        os.makedirs(self.output_dir, exist_ok=True)

    def create_example_graphs(self):
        """Create example target and query graphs."""

        # Check if saved graphs exist
        graphs_dir = os.path.dirname(__file__)
        g1_path = os.path.join(graphs_dir, 'G1.pkl') #main target graph
        g2_path = os.path.join(graphs_dir, 'G2.pkl') #subgraph of G1
        g3_path = os.path.join(graphs_dir, 'G3.pkl') #non-subgraph

        return pickle.load(open(g1_path, 'rb')), pickle.load(open(g2_path, 'rb'))

    def run_query_and_visualize(self):
        """
        Run subgraph query and create visualizations.

        Returns: tuple: (result, target_html_path, query_html_path, target_data, query_data, target_graph, query_graph)
        """
        target_graph, query_graph = self.create_example_graphs()

        print(f"Target graph: {len(target_graph)} nodes, {target_graph.number_of_edges()} edges")
        print(f"Query graph: {len(query_graph)} nodes, {query_graph.number_of_edges()} edges")

        # Perform subgraph query
        print("\nPerforming subgraph query...")
        result = query_subgraph(target_graph, query_graph, threshold=2.6)
        print(f"Query is {'a subgraph' if result else 'NOT a subgraph'} of target")
       
        # Create visualizations
        print("\nGenerating visualizations...")
        try:
            # Prepare graph data
            target_data = extract_graph_data(target_graph)
            query_data = extract_graph_data(query_graph)

            if result is None:
                target_data['metadata']['title'] = "Target Graph - Query Result Unknown (Dependencies Not Installed)"
                query_data['metadata']['title'] = "Query Graph - Result Unknown"
            else:
                target_data['metadata']['title'] = f"Target Graph - Query {'IS' if result else 'IS NOT'} a Subgraph"
                query_data['metadata']['title'] = f"Query Graph - {'Matches' if result else 'Does Not Match'}"

            # Generate HTML files
            template_path = os.path.join(os.path.dirname(__file__), "..", "visualizer", "template.html")

            target_html = process_html_template(
                target_data,
                template_path=template_path,
                output_filename="target_graph.html",
                output_dir=self.output_dir
            )

            query_html = process_html_template(
                query_data,
                template_path=template_path,
                output_filename="query_graph.html",
                output_dir=self.output_dir
            )

            print(f"Visualizations saved to:")
            print(f"  Target: {target_html}")
            print(f"  Query: {query_html}")

            return result, target_html, query_html, target_data, query_data, target_graph, query_graph

        except Exception as e:
            print(f"Visualization failed: {e}")
            return result, None, None, None, None, None, None

    def create_combined_visualization(self, target_data, query_data, result=None):
        """
        Create a combined HTML file with a single centered graph showing both datasets.

        Args:
            target_data: Target graph data
            query_data: Query graph data
            result: Query result (optional)

        Returns:
            str: Path to the combined HTML file
        """
        template_path = os.path.join(os.path.dirname(__file__), "..", "visualizer", "template.html")

        # Read the template
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()

        # Create combined graph data by merging both graphs
        combined_nodes = []
        combined_edges = []

        # Add target nodes with offset
        target_offset_x = -200
        target_offset_y = -100
        for node in target_data['nodes']:
            combined_node = node.copy()
            combined_node['x'] += target_offset_x
            combined_node['y'] += target_offset_y
            combined_node['label'] = f"Target: {node.get('label', 'node')}"
            combined_nodes.append(combined_node)

        # Add query nodes with offset
        query_offset_x = 200
        query_offset_y = 100
        for node in query_data['nodes']:
            combined_node = node.copy()
            combined_node['x'] += query_offset_x
            combined_node['y'] += query_offset_y
            combined_node['label'] = f"Query: {node.get('label', 'node')}"
            combined_nodes.append(combined_node)

        # Add edges from both graphs
        for edge in target_data['edges']:
            combined_edges.append(edge.copy())
        for edge in query_data['edges']:
            combined_edges.append(edge.copy())

        # Create combined metadata
        combined_metadata = {
            'title': f"Combined Graph - Target vs Query ({'Match' if result else 'No Match'})",
            'nodeCount': len(combined_nodes),
            'edgeCount': len(combined_edges),
            'isDirected': target_data['metadata'].get('isDirected', False),
            'density': len(combined_edges) / (len(combined_nodes) * (len(combined_nodes) - 1) / 2) if len(combined_nodes) > 1 else 0
        }

        # Create combined legend
        combined_legend = {
            'nodeTypes': [
                {'label': 'Target: gene', 'color': 'rgba(59, 130, 246, 0.7)'},
                {'label': 'Target: transcript', 'color': 'rgba(34, 197, 94, 0.7)'},
                {'label': 'Target: protein', 'color': 'rgba(245, 101, 101, 0.7)'},
                {'label': 'Query: gene', 'color': 'rgba(139, 69, 19, 0.7)'},
                {'label': 'Query: transcript', 'color': 'rgba(75, 0, 130, 0.7)'},
                {'label': 'Query: protein', 'color': 'rgba(255, 20, 147, 0.7)'}
            ],
            'edgeTypes': [
                {'label': 'Target: transcribed_to', 'color': 'rgba(34, 100, 94, 0.7)'},
                {'label': 'Target: translates_to', 'color': 'rgba(245, 190, 101, 0.7)'},
                {'label': 'Query: transcribed_to', 'color': 'rgba(0, 100, 0, 0.7)'},
                {'label': 'Query: translates_to', 'color': 'rgba(255, 69, 0, 0.7)'}
            ]
        }

        combined_data = {
            'metadata': combined_metadata,
            'nodes': combined_nodes,
            'edges': combined_edges,
            'legend': combined_legend
        }

        # Convert to JSON
        import json
        combined_data_json = json.dumps(combined_data, indent=2)

        # Extract the script content and replace GRAPH_DATA
        script_start = template_content.find('<script>')
        script_end = template_content.find('</script>', script_start) + len('</script>')
        script_content = template_content[script_start:script_end]

        # Replace the GRAPH_DATA definition
        old_graph_data_start = script_content.find('const GRAPH_DATA = {')
        old_graph_data_end = script_content.find('};', old_graph_data_start) + 2
        old_graph_data = script_content[old_graph_data_start:old_graph_data_end]

        new_graph_data = f'const GRAPH_DATA = {combined_data_json};'
        new_script_content = script_content.replace(old_graph_data, new_graph_data)

        # Replace the script in the template
        combined_html = template_content.replace(script_content, new_script_content)

        # Update the title
        combined_html = combined_html.replace(
            '<title>Interactive Graph Visualizer</title>',
            '<title>Combined Graph Visualization - Target vs Query</title>'
        )

        # Write combined HTML file
        combined_html_path = os.path.join(self.output_dir, "combined_graphs.html")
        with open(combined_html_path, 'w', encoding='utf-8') as f:
            f.write(combined_html)

        return combined_html_path

    def _kill_process_on_port(self, port):
        """
        Kill the process using the specified port on Windows.

        Args:
            port: Port number to free
        """
        import subprocess

        try:
            # Find the PID using the port
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            lines = result.stdout.split('\n')

            pid = None
            for line in lines:
                if f':{port} ' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        break

            if pid:
                print(f"Killing process {pid} using port {port}")
                subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True)
                # Wait a moment for the port to be freed
                import time
                time.sleep(1)
            else:
                print(f"Could not find process using port {port}")

        except Exception as e:
            print(f"Failed to kill process on port {port}: {e}")


def main():
    """Main demo function."""
    print("Neural Subgraph Matcher - Query Demo")
    print("=" * 50)

    demo = SubgraphQueryDemo()

    # Run query and create visualizations
    result, target_html, query_html, target_data, query_data, target_graph, query_graph = demo.run_query_and_visualize()

    if target_html and query_html and target_data and query_data:
        print("\nDemo completed successfully!")
        print(f"Results saved in: {demo.output_dir}")

        # Create combined visualization
        combined_html = demo.create_combined_visualization(target_data, query_data, result)

        print(f"Combined visualization saved to: {combined_html}")

        # Start web server on port 10000
        print("\nStarting web server on port 10000...")
        print(f"Access the visualization at: http://localhost:10000/combined_graphs.html")

        # Change to output directory to serve files
        original_cwd = os.getcwd()
        os.chdir(demo.output_dir)

        try:
            # Kill any existing process on port 10000
            demo._kill_process_on_port(10000)

            # Start HTTP server
            server = HTTPServer(('localhost', 10000), SimpleHTTPRequestHandler)
            print("Server started. Press Ctrl+C to stop.")

            # Run server in a separate thread to allow graceful shutdown
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down server...")
                server.shutdown()
                server.server_close()

        except Exception as e:
            print(f"Failed to start server: {e}")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    else:
        print("\nDemo completed with errors. Check the error messages above.")


if __name__ == "__main__":
    main()