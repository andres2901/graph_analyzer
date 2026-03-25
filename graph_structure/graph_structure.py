#!/usr/bin/env python3
"""
Graph Structural Analyzing tool

This tool perform a structural analysis of an undirected graph
and their attribute-defined subgraphs. returning multiple metrics
related to composition, connectivity, assortativity, and centrality.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from graph_structure.graph_module import GraphObject
from graph_structure.graph_module import SubGraphObject
from graph_structure.version import __version__

# CONSTANTS

UW = "Unweighted"
W = "Weighted"
#exec(open("version.py").read())


def load_graph(edges_file: str, 
               node_file: str, 
               attribute: str
    ) -> list[pd.DataFrame, pd.DataFrame, str]:
    """Check and store the input files of the graph

    Args:
        gff_path: Path to the input GFF file.
        fasta_path: Path to the input fasta file.
        attribute: String representing the attribute for subgraphs.

    Returns:
        list [edges_df, nodes_df, graph_type] where:
            - edges_df: pandas dataframe of edges in a network.
            - nodes_df: pandas dataframe of nodes attributes of a network.
            - graph_type: String representing type of graph to analyze.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        pd.errors.ParserError: If either input file cannot be parsed.
        OSError: If any other I/O error occurs.
    """
    
    try:
        edges_df = pd.read_csv(
            edges_file, 
            sep='\t', 
            dtype={'Source': str, 'Target': str,'Weight': float}
        )
        
        if not set(["Source","Target"]).issubset(edges_df.columns.values):
            print(f"'Source' or 'Target' columns were not found in the edge file")
            sys.exit(1)
        else:
            if len(edges_df.columns) == 3:
                if "Weight" not in edges_df.columns.values:
                    print(f"'Weight' column has a bad name in the edge file")
                    sys.exit(1)
                else:
                    graph_type = W
            elif len(edges_df.columns) > 3:
                sys.exit(f"edge file have more than three colums")
            else:
                graph_type = UW
    except FileNotFoundError:
        sys.exit(f"Error: Data file '{edges_df}' not found.")
    except PermissionError:
        sys.exit(f"Error: No read permission for '{edges_df}'.")
    except pd.errors.ParserError as e:
        sys.exit(f"Error parsing data file '{edges_df}': {e}")
        
    try:
        nodes_df = pd.read_csv(
            node_file, 
            sep='\t', 
            dtype={'NodeID': str}
        )

        if "NodeID" not in nodes_df.columns.values:
            sys.exit(f"'NodeID' column is missing in node attributes file")
        else:
            if len(nodes_df.columns) == 1:
                sys.exit(f"nodes attribute file only have the NodeID column")
    except FileNotFoundError:
        sys.exit(f"Error: Data file '{node_file}' not found.")
    except PermissionError:
        sys.exit(f"Error: No read permission for '{node_file}'.")
    except pd.errors.ParserError as e:
        sys.exit(f"Error parsing data file '{node_file}': {e}")
        
    if attribute not in nodes_df.columns:
        sys.exit(f"Atribute '{attribute}' is not available in the given node atribute table")

    nodes_id_nodes = np.sort(nodes_df["NodeID"].to_numpy())
    nodes_id_edges = np.unique(np.array([edges_df["Source"].to_numpy(),edges_df["Target"].to_numpy()]))
    
    if not np.array_equal(nodes_id_nodes, nodes_id_edges):
        difference = np.setxor1d(nodes_id_nodes,nodes_id_edges)
        print(f"The following nodes differ between files:")
        print(f"{difference}")
        sys.exit(1)

    print(f"Input is a {graph_type} graph")

    return edges_df, nodes_df, graph_type


def process_graph(edges_file: str, 
                  node_file: str, 
                  attribute: str,
                  output_dir: str
    ) -> None:
    """Process the data and write the output.

    Args:
        gff_path: Path to the input GFF file.
        fasta_path: Path to the input fasta file.
        attribute: String representing the attribute for subgraphs.
        output_dir: Path to the output directory.

    Returns:
        None

    Raises:
        PermissionError: If the file cannot be written.
        OSError: If any other I/O error occurs.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    edges, nodes, graph_type = load_graph(edges_file, node_file, attribute)
    input_graph = GraphObject(edges, nodes, graph_type)
    assortativity = input_graph.assortativity(attribute)
    input_graph.distributions_statistic()
        
    main_stats = os.path.join(output_dir, "Main_graph_stats.txt")
    graph_stats = input_graph.stats()

    print(f"Writing results in {output_dir} directory")
    
    try:
        with open(main_stats, 'w') as f:
            f.write(f"Statistics of graph '{edges_file}':\n")
            f.write(f"  Graph type: {graph_stats[0]}\n")
            f.write(f"  Number of nodes: {graph_stats[1]}\n")
            f.write(f"  Number of edges: {graph_stats[2]}\n")
            f.write(f"  Density: {graph_stats[3]}\n")
            f.write(f"  Number of connected components: {graph_stats[4]}\n")
            f.write(f"  Transitivity: {graph_stats[5]}\n")
            f.write(f"  Assortativity of '{attribute}': {assortativity}\n")   
    except PermissionError:
        sys.exit(f"Error: No write permission for '{main_stats}'.")
    except OSError as e:
        sys.exit(f"Error writing file '{main_stats}': {e}")
    
    distributions = input_graph.distributions_statistic()
    
    if graph_type == W:
        distributions[0].to_csv(os.path.join(output_dir, "Node_characteristics.txt"),
                                sep='\t', index_label = "NodeID")
        distributions[1].to_csv(os.path.join(output_dir, "Node_stats.txt"),
                                sep='\t')
 
        distributions[2].to_csv(os.path.join(output_dir, "Edge_stats.txt"),
                                sep='\t')
    elif graph_type == UW:
        distributions[0].to_csv(os.path.join(output_dir, "Node_characteristics.txt"),
                                sep='\t')
        distributions[1].to_csv(os.path.join(output_dir, "Node_stats.txt"),
                                sep='\t', index_label = "NodeID")
    
    subgraph = SubGraphObject(input_graph, attribute)
    general_sub = subgraph.subgraphs()
    subgraph.calculate_metrics()
    subgraph_metric = subgraph.metrics()
    subgraph_distribution = subgraph.distributions()
    
    subgraph_dir = os.path.join(output_dir, "SubGraphs")
    if not os.path.exists(subgraph_dir):
            os.makedirs(subgraph_dir)
    
    for key in general_sub:
        graph = general_sub.get(key)
        metrics = subgraph_metric.get(key)
        distributions = subgraph_distribution.get(key)
        info_dir = os.path.join(subgraph_dir, key)

        if not os.path.exists(info_dir):
            os.makedirs(info_dir)

        edge_file = os.path.join(info_dir, f"{key}_edges_file.txt")

        nx.to_pandas_edgelist(graph).rename(columns = { 'source' : 'Source', 'target' : 'Target', 'weight' : 'Weight'}).to_csv(edge_file, sep='\t', index = False)

        node_file = os.path.join(info_dir, f"{key}_nodes_file.txt")

        pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index').to_csv(node_file, sep='\t', index_label = "NodeID")

        filename = os.path.join(info_dir, "Main_subgraph_stats.txt")
        try:
            with open(filename, 'w') as f:
                f.write(f"Statistics of subgraph '{key}':\n")
                f.write(f"  Graph type: {graph_stats[0]}\n")
                f.write(f"  Number of nodes: {metrics.get('node_number')}\n")
                f.write(f"  Number of edges: {metrics.get('edge_number')}\n")
                f.write(f"  Density: {metrics.get('density')}\n")
                f.write(f"  Number of connected components: {metrics.get('connect_components')}\n")
                f.write(f"  Transitivity: {metrics.get('transitivity')}\n")
        except PermissionError:
            sys.exit(f"Error: No write permission for '{filename}'.")
        except OSError as e:
            sys.exit(f"Error writing file '{filename}': {e}")

        if graph_type == W:
            distributions.get('nodes').to_csv(os.path.join(info_dir, "Node_characteristics.txt"),
                                sep='\t', index_label = "NodeID")
            distributions.get('nodes').describe().to_csv(os.path.join(info_dir, "Node_stats.txt"),
                                sep='\t')
            if metrics.get('edge_number') > 0:
                distributions.get('Weight')['Weight'].describe().to_csv(os.path.join(info_dir, "Edge_stats.txt"),
                                    sep='\t')
        elif graph_type == UW:
            distributions.get('nodes').to_csv(os.path.join(info_dir, "Node_characteristics.txt"),
                                sep='\t')
            distributions.get('nodes').describe().to_csv(os.path.join(info_dir, "Node_stats.txt"),
                                sep='\t')

    
def main() -> None:
    """Parse command-line arguments and launch the graph analyzer pipeline."""
    parser = argparse.ArgumentParser(prog = f"graph_structure", \
        description=f"Structural properties analysis of graph and attribute-base subgraphs v{__version__}")
    parser.add_argument('-e', '--edges-file', type=str, required=True, help="Input TSV file with edges.")
    parser.add_argument('-n', '--node-file', type=str, required=True, help="Input TSV file with node attributes.")
    parser.add_argument('-a', '--attribute', type=str, required=True, help="Name of the attribute for subgraphs")
    parser.add_argument('-o', '--output-dir', type=str, default='./', help="Directory for output files.")
    
    args = parser.parse_args()
    process_graph(args.edges_file, args.node_file, args.attribute, args.output_dir)
        

if __name__ == "__main__":
    main()
