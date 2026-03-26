# Graph Structure

Package focused on reading and analyzing graphs structures. It will perform a structural analysis on an undirected graph and their attribute-defined subgraphs. By providing an edge list and node attributes, it will automate the extraction of the following metrics of the graph and attribute-based subgraph:

- Number of nodes.
- Number of edges.
- Graph density.
- Number of connected components. 
- Transitivity.
- Node-level characteristics:
  - Node Degree Distribution.
  - Betweenness centrality distribution.
  - Closeness centrality distribution
- Edge-level characteristics (if weighted graph):
  - Edges weight distribution 
- Assortativity of the selected attribute

For the distributions, it also include descriptive statistics that summarize the central tendency, dispersion and shape of each property.

### Installation

`pip install graph-structure`  

Upgrade to a newer version using:  
`pip install graph-structure --upgrade`

### Usage

```
usage: graph_structure [-h] -e EDGES_FILE -n NODE_FILE -a ATTRIBUTE [-o OUTPUT_DIR] 

Structural properties analysis of graph and attribute-base subgraphs

options:
  -h, --help            show this help message and exit
  -e, --edges-file EDGES_FILE
                        Input TSV file with edges.
  -n, --node-file NODE_FILE
                        Input TSV file with node attributes.
  -a, --attribute ATTRIBUTE
                        Name of the attribute for subgraphs
  -o, --output-dir OUTPUT_DIR
                        Directory for output files.

```

## Input format

The input formats are the same as those accepted for [cytoscape](https://cytoscape.org/) to import a network and attributes from table.

### 1. Network Edges File
A two or three-column tab-delimited file. The header must include `Source` and `Target` (and optionally `Weight`).
```
Source	Target	Weight
node01	node03	85.71
node01	node04	90.91
node01	node06	95.24
node02	node05	90.91
node02	node09	70.42
node03	node04	67.65
node03	node06	95.0
node05	node09	94.12
node07  node08  100
```

### 2. Node Attribute File
A tab-delimited file with at least two columns. The first column **must** be named `NodeID`. Subsequent columns represent the attributes used for subgraph extraction.
```
NodeID  GenomeID  Country
node01  ID01  Sweden
node02  ID01  Sweden
node03	ID02  USA
node04	ID03  Norway
node05	ID03  Norway
node06  ID04  Sweden
node07  ID05  Spain
node08  ID05  Spain
node09  ID06  Denmark
```
