"""
Subgraph Querying Demo with Web Server

This module provides subgraph querying functionality with automatic web serving
of interactive visualizations on port 3000.
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(".."))

# Import visualization modules
try:
    from visualizer.visualizer import extract_graph_data, process_html_template
except ImportError:
    # Fallback import if running from different directory
    import visualizer.visualizer as viz
    extract_graph_data = viz.extract_graph_data
    process_html_template = viz.process_html_template

# Query import will be done conditionally
QUERY_AVAILABLE = False
try:
    from query_subgraph import query_subgraph
    QUERY_AVAILABLE = True
except ImportError:
    print("Warning: Query functionality not available (dependencies not installed)")
    query_subgraph = None


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
        # Target graph: larger connected graph
        target = nx.gnp_random_graph(15, 0.2, seed=42)

        # Query graph: actual subgraph of target
        if len(target) >= 6:
            nodes = list(target.nodes())[:6]
            query = target.subgraph(nodes).copy()
        else:
            query = nx.gnp_random_graph(5, 0.3, seed=43)

        return target, query

    def run_query_and_visualize(self, target_graph=None, query_graph=None):
        """
        Run subgraph query and create visualizations.

        Args:
            target_graph: Target NetworkX graph (optional)
            query_graph: Query NetworkX graph (optional)

        Returns:
            tuple: (result, target_html_path, query_html_path, target_data, query_data, target_graph, query_graph)
        """
        if target_graph is None or query_graph is None:
            target_graph, query_graph = self.create_example_graphs()

        print(f"Target graph: {len(target_graph)} nodes, {target_graph.number_of_edges()} edges")
        print(f"Query graph: {len(query_graph)} nodes, {query_graph.number_of_edges()} edges")

        # Perform subgraph query
        print("\nPerforming subgraph query...")
        if QUERY_AVAILABLE and query_subgraph:
            try:
                result = query_subgraph(target_graph, query_graph)
                print(f"Query is {'a subgraph' if result else 'NOT a subgraph'} of target")
            except Exception as e:
                print(f"Query failed: {e}")
                result = None  # Unknown result
        else:
            print("Query functionality not available (dependencies not installed)")
            print("Proceeding with visualization only...")
            result = None  # Unknown result

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
        Create a combined HTML file with both graphs side by side.

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

        # Convert data to JSON
        import json
        target_data_json = json.dumps(target_data, indent=2)
        query_data_json = json.dumps(query_data, indent=2)

        # Create combined HTML with two canvases side by side
        combined_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://unpkg.com/jspdf@2.5.1/dist/jspdf.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/pdf-lib/dist/pdf-lib.min.js"></script>
    <title>Combined Graph Visualizations</title>
    <style>
        :root {{
            /* Color Palette - Light Theme */
            --bg-primary: #fafafa;
            --bg-secondary: rgba(255, 255, 255, 0.9);
            --border-light: rgba(0, 0, 0, 0.1);
            --text-primary: #374151;
            --text-secondary: #6b7280;

            /* Node Colors */
            --node-default: rgba(59, 130, 246, 0.7);
            --node-anchor: rgba(239, 68, 68, 0.8);
            --node-border: rgba(0, 0, 0, 0.3);

            /* Edge Colors */
            --edge-default: rgba(107, 114, 128, 0.6);
            --edge-hover: rgba(59, 130, 246, 0.8);

            /* UI Elements */
            --card-bg: rgba(255, 255, 255, 0.95);
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --card-border: rgba(0, 0, 0, 0.1);

            /* Grid Colors */
            --grid-minor: rgba(0, 0, 0, 0.03);
            --grid-major: rgba(0, 0, 0, 0.08);
            --grid-axis: rgba(59, 130, 246, 0.15);
            --grid-dots: rgba(0, 0, 0, 0.1);
            --grid-center: rgba(59, 130, 246, 0.2);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-primary);
            overflow: hidden;
            height: 100vh;
            width: 100vw;
            display: flex;
            flex-direction: column;
        }}

        .header {{
            text-align: center;
            padding: 20px;
            background: var(--card-bg);
            border-bottom: 1px solid var(--border-light);
            z-index: 1000;
        }}

        .header h1 {{
            font-size: 24px;
            color: var(--text-primary);
            margin-bottom: 8px;
        }}

        .header p {{
            font-size: 16px;
            color: var(--text-secondary);
        }}

        .graphs-container {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}

        .graph-section {{
            flex: 1;
            position: relative;
            border-right: 1px solid var(--border-light);
        }}

        .graph-section:last-child {{
            border-right: none;
        }}

        .canvas-container {{
            position: relative;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }}

        .graph-canvas {{
            position: absolute;
            top: 0;
            left: 0;
            cursor: grab;
            background-color: var(--bg-primary);
            background-image:
                radial-gradient(circle at 1px 1px, rgba(0,0,0,0.02) 1px, transparent 0);
            background-size: 20px 20px;
        }}

        .graph-canvas:active {{
            cursor: grabbing;
        }}

        /* UI card styling */
        .ui-card {{
            position: absolute;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            box-shadow: var(--card-shadow);
            backdrop-filter: blur(10px);
            z-index: 1000;
            padding: 12px;
            color: var(--text-primary);
            font-size: 14px;
        }}

        /* Position adjustments for side-by-side layout */
        .top-left {{
            top: 20px;
            left: 20px;
        }}

        .top-center {{
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
        }}

        .top-right {{
            top: 20px;
            right: 20px;
        }}

        .bottom-right-upper {{
            bottom: 180px;
            right: 20px;
        }}

        .bottom-right-lower {{
            bottom: 20px;
            right: 20px;
        }}

        /* Title bar styling */
        #title-bar {{
            text-align: center;
            font-weight: 600;
            font-size: 16px;
            min-width: 300px;
        }}

        #title-bar .subtitle {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        /* Zoom controls styling */
        #zoom-controls {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            align-items: center;
        }}

        #zoom-controls button {{
            width: 36px;
            height: 36px;
            border: 1px solid var(--border-light);
            border-radius: 6px;
            background: white;
            color: var(--text-primary);
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }}

        #zoom-controls button:hover {{
            background: var(--bg-primary);
            border-color: var(--text-secondary);
        }}

        #zoom-controls button:active {{
            transform: scale(0.95);
        }}

        /* Legend card styling */
        #legend-card {{
            min-width: 220px;
            max-width: 280px;
            max-height: 400px;
            overflow-y: auto;
        }}

        #legend-card h3 {{
            margin-bottom: 16px;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-light);
            padding-bottom: 8px;
        }}

        .legend-section {{
            margin-bottom: 16px;
        }}

        .legend-section:last-child {{
            margin-bottom: 0;
        }}

        .legend-section h4 {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
            font-size: 12px;
            padding: 2px 0;
        }}

        .legend-item:last-child {{
            margin-bottom: 0;
        }}

        .legend-color {{
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 1px solid var(--border-light);
            flex-shrink: 0;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }}

        .legend-shape {{
            width: 18px;
            height: 18px;
            border: 1px solid var(--border-light);
            flex-shrink: 0;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }}

        .legend-item span {{
            color: var(--text-primary);
            font-weight: 500;
            line-height: 1.2;
        }}

        /* Controls card styling */
        #controls-card {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-width: 200px;
        }}

        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .control-group label {{
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
        }}

        .toggle-switch {{
            position: relative;
            display: inline-block;
            width: 44px;
            height: 24px;
        }}

        .toggle-switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}

        .toggle-slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: 0.3s;
            border-radius: 24px;
        }}

        input:checked + .toggle-slider {{
            background-color: var(--node-default);
        }}

        input:checked + .toggle-slider:before {{
            transform: translateX(20px);
        }}

        .toggle-slider:before {{
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: 0.3s;
            border-radius: 50%;
        }}

        .export-buttons {{
            display: flex;
            gap: 8px;
        }}

        .export-buttons button {{
            flex: 1;
            padding: 8px 12px;
            border: 1px solid var(--border-light);
            border-radius: 4px;
            background: white;
            color: var(--text-primary);
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s ease;
        }}

        .export-buttons button:hover {{
            background: var(--bg-primary);
            border-color: var(--text-secondary);
        }}

        /* Context menu styling */
        #context-menu {{
            position: absolute;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 6px;
            box-shadow: var(--card-shadow);
            backdrop-filter: blur(10px);
            z-index: 2000;
            padding: 4px 0;
            min-width: 120px;
        }}

        #context-menu.hidden {{
            display: none;
        }}

        .context-menu-item {{
            padding: 8px 16px;
            cursor: pointer;
            font-size: 14px;
            color: var(--text-primary);
            transition: background-color 0.2s ease;
        }}

        .context-menu-item:hover {{
            background-color: var(--bg-primary);
        }}

        /* Isolation effects */
        .node-isolated {{
            filter: blur(2px);
            opacity: 0.3;
        }}

        .edge-isolated {{
            filter: blur(2px);
            opacity: 0.2;
        }}

        /* Responsive design */
        @media (max-width: 768px) {{
            .graphs-container {{
                flex-direction: column;
            }}
            .graph-section {{
                border-right: none;
                border-bottom: 1px solid var(--border-light);
            }}
            .graph-section:last-child {{
                border-bottom: none;
            }}
            .ui-card {{
                padding: 8px;
                font-size: 12px;
            }}
            .top-left, .top-right {{
                top: 10px;
            }}
            .top-left {{
                left: 10px;
            }}
            .top-right {{
                right: 10px;
            }}
            .bottom-right-upper {{
                bottom: 140px;
                right: 10px;
            }}
            .bottom-right-lower {{
                bottom: 10px;
                right: 10px;
            }}
            #title-bar {{
                min-width: 200px;
                font-size: 14px;
            }}
            #zoom-controls button {{
                width: 32px;
                height: 32px;
                font-size: 16px;
            }}
            #legend-card, #controls-card {{
                min-width: 160px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Neural Subgraph Matcher - Query Demo</h1>
        <p>Target Graph vs Query Graph Comparison</p>
    </div>

    <div class="graphs-container">
        <!-- Target Graph Section -->
        <div class="graph-section">
            <div id="title-bar-target" class="ui-card top-center">
                <div id="graph-title-target">{target_data['metadata']['title']}</div>
                <div class="subtitle" id="graph-stats-target">Loading...</div>
            </div>

            <div id="zoom-controls-target" class="ui-card top-right">
                <button id="zoom-in-target" title="Zoom In">+</button>
                <button id="zoom-out-target" title="Zoom Out">−</button>
                <button id="recenter-target" title="Recenter">⌂</button>
            </div>

            <div id="legend-card-target" class="ui-card bottom-right-upper">
                <h3>Legend</h3>
                <div id="legend-content-target">
                    <!-- Legend items will be populated by JavaScript -->
                </div>
            </div>

            <div id="controls-card-target" class="ui-card bottom-right-lower">
                <div class="control-group">
                    <label>Show Labels</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="label-toggle-target" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>

                <div class="control-group">
                    <label>Export</label>
                    <div class="export-buttons">
                        <button id="export-pdf-target">PDF</button>
                        <button id="export-png-target">PNG</button>
                    </div>
                </div>
            </div>

            <div id="context-menu-target" class="hidden">
                <div class="context-menu-item" id="isolate-node-target">Isolate</div>
                <div class="context-menu-item" id="copy-label-target">Copy Label</div>
                <div class="context-menu-item" id="cancel-isolate-target" style="display: none;">Cancel Isolate</div>
            </div>

            <div class="canvas-container">
                <canvas id="graph-canvas-target"></canvas>
            </div>
        </div>

        <!-- Query Graph Section -->
        <div class="graph-section">
            <div id="title-bar-query" class="ui-card top-center">
                <div id="graph-title-query">{query_data['metadata']['title']}</div>
                <div class="subtitle" id="graph-stats-query">Loading...</div>
            </div>

            <div id="zoom-controls-query" class="ui-card top-right">
                <button id="zoom-in-query" title="Zoom In">+</button>
                <button id="zoom-out-query" title="Zoom Out">−</button>
                <button id="recenter-query" title="Recenter">⌂</button>
            </div>

            <div id="legend-card-query" class="ui-card bottom-right-upper">
                <h3>Legend</h3>
                <div id="legend-content-query">
                    <!-- Legend items will be populated by JavaScript -->
                </div>
            </div>

            <div id="controls-card-query" class="ui-card bottom-right-lower">
                <div class="control-group">
                    <label>Show Labels</label>
                    <label class="toggle-switch">
                        <input type="checkbox" id="label-toggle-query" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>

                <div class="control-group">
                    <label>Export</label>
                    <div class="export-buttons">
                        <button id="export-pdf-query">PDF</button>
                        <button id="export-png-query">PNG</button>
                    </div>
                </div>
            </div>

            <div id="context-menu-query" class="hidden">
                <div class="context-menu-item" id="isolate-node-query">Isolate</div>
                <div class="context-menu-item" id="copy-label-query">Copy Label</div>
                <div class="context-menu-item" id="cancel-isolate-query" style="display: none;">Cancel Isolate</div>
            </div>

            <div class="canvas-container">
                <canvas id="graph-canvas-query"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Target graph data
        const TARGET_GRAPH_DATA = {target_data_json};

        // Query graph data
        const QUERY_GRAPH_DATA = {query_data_json};

        // Include the full JavaScript from the template
        {template_content.split('<script>')[1].split('</script>')[0]}
    </script>
</body>
</html>
"""

        # Convert data to JSON
        import json
        target_data_json = json.dumps(target_data, indent=2)
        query_data_json = json.dumps(query_data, indent=2)

        combined_html = combined_html.replace('{target_data_json}', target_data_json)
        combined_html = combined_html.replace('{query_data_json}', query_data_json)

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

        # Open in Chrome browser directly
        import subprocess
        import platform

        try:
            if platform.system() == 'Windows':
                subprocess.run(['start', 'chrome', combined_html], shell=True)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', '-a', 'Google Chrome', combined_html])
            else:  # Linux
                subprocess.run(['google-chrome', combined_html])
            print("Opened combined visualization in Chrome browser")
        except Exception as e:
            print(f"Could not open in Chrome automatically: {e}")
            print(f"Please open {combined_html} manually in your browser")
    else:
        print("\nDemo completed with errors. Check the error messages above.")


if __name__ == "__main__":
    main()