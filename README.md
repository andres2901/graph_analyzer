# Graph Analyzer

It performs a structural analysis on an undirected graph and their attribute-defined subgraphs. By providing an edge list and node attributes, you can automate the extraction of the following metrics of the graph and attribute-based subgraph

- Number of nodes.
- Number of edges.
- Graph density.
- Number of connected components. 
- Transitivity.
- Node Degree Distribution.
- Betweenness centrality distribution.
- Closeness centrality distribution
- Edges weight distribution (if weighted graph)
- Assortativity of the selected attribute

All generated statistics and processed data are organized and saved into a dedicated directory for easy post-processing and visualization.

## Input example

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
node10
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
node10  ID07  Japan
```
