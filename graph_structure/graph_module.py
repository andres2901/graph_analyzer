import networkx as nx
import pandas as pd
import numpy as np

# CONSTANTS

UW = "Unweighted"
W = "Weighted"
VALID_TYPES = {W, UW}


class GraphObject:
    """
    A class used to represent graphs and their related metrics.

    Attributes:
        edges: Pandas dataframe of edges.
        nodes: Pandas dataframe of node attributes.
        graph_type: Type of graph.
        graph: NetworkX graph.
        weight: weight distribution of the graph edges.
        node_number: Number of nodes in the graph.
        egde_number: NUmber of edges in the graph.
        density: density of the graph.
        connec_comp: Number of connected components.
        transitivity: Transitivity of the graph.
        degree: node degree distribution.
        betweenness: node betweenness centrality distribution.
        closeness: node closeness centrality distribution.
        node_distribution: Pandas dataframe of all nodes distribution.
        assortativity: assortativity of a given attribute.
    """

    def __init__(
        self,
        edges: pd.DataFrame,
        nodes: pd.DataFrame,
        graph_type: str
    ) -> None:
        """Initialize Graph Object that will store all the metrics.

        Args:
            edges: pandas dataframe of edges in a network.
            nodes: pandas dataframe of nodes attributes of a network.
            graph_type: string identifying the type of graph: weighted or unweighted
         
        Raises:
            ValueError: If the graph type is not valid.
        """

        if graph_type not in VALID_TYPES:
            raise ValueError(f"Unknown graph type: '{self.graph_type}'. "
                             f"Use '{W}' or '{UW}'.")

        self.graph_type = graph_type
        self.edges = edges
        self.nodes = nodes.set_index("NodeID").to_dict("index")

        if self.graph_type == W:
            graph = nx.Graph()
            self.weight = pd.to_numeric(self.edges["Weight"], errors='coerce')
            for _, row in self.edges.iterrows():
                graph.add_edge(row["Source"], row["Target"], weight=float(row["Weight"]))
        elif self.graph_type == UW:
            pairs = [tuple(x) for x in self.edges[["Source", "Target"]].to_numpy(dtype = str)]
            graph = nx.Graph(pairs)
        else:
            raise ValueError(f"Unknown graph type: '{self.graph_type}'. "
                             f"Use '{W}' or '{UW}'.")

        self.graph = graph
        nx.set_node_attributes(self.graph,self.nodes)
        self.node_number = self.graph.number_of_nodes()
        self.edge_number = self.graph.number_of_edges()
        self.density = nx.density(self.graph)
        self.connec_comp = nx.number_connected_components(self.graph)
        self.transitivity = nx.transitivity(self.graph)
        
        self.degree = {node:deg for (node, deg) in self.graph.degree()}
        self.betweenness = nx.betweenness_centrality(self.graph, normalized = False)
        self.closeness = nx.closeness_centrality(self.graph)
        self.node_distributions =  pd.DataFrame.from_records([self.degree, self.betweenness,self.closeness], index = ["Degree", "Betweeness", "Closeness"]).transpose()           
      
    def assortativity(self, attribute: str) -> float:
        """Identify assortativity of the value
        
        Args:
            attribute: node attribute to calculate the assortativity.
            
        Returns:
            float
        
        Raises:
            raise ValueError"""
            
        if attribute not in pd.DataFrame.from_dict(self.nodes, orient='index').columns:
            raise ValueError(f"Unknown attribute: '{attribute}'")
        
        self.assortativity = nx.attribute_assortativity_coefficient(self.graph, attribute)
        return self.assortativity
        
        
    def distributions_statistic(self) -> list[pd.DataFrame]:
        """Return basic statistics of distributions"""
        
        if self.graph_type == W:
            return self.node_distributions, self.node_distributions.describe().apply(lambda s: s.apply('{0:.4f}'.format)), self.weight.describe().apply(lambda x: format(x, '.4f'))
        elif self.graph_type == UW:
            return self.node_distributions, self.node_distributions.describe()
        
    def nodes_dict(self) -> dict[dict]:
        """Return the node dictionary"""
        
        return self.nodes
    
    def graph_type(self) -> str:
        """Return graph type"""
        
        return self.graph_type
    
    def stats(self) -> list[numeric]:
        """Return stats"""
        
        return self.graph_type, self.node_number, self.edge_number, self.density,\
               self.connec_comp, self.transitivity
    
    def base_graph(self) -> dict[dict]:
        """Return the node dictionary"""
        
        return self.graph


class SubGraphObject():
    """
    A class used to represent subgraphs and their related metrics.
    
    Attributes:
        attribute: Attribute used to defined subgraphs.
        nodes: pandas DataFrame of node attributes.
        unique_values: values of the attribute of interest.
        sub_graphs: subgraphs divided based on unique_values attribute.
        subgraph_metrics: dictionary with main metrics of each subgraph.
    """
    
    def __init__(
        self,
        main_graph: GraphObject,
        attribute: str
    ) -> None:
        """
        Initialize Sub Graph Object that will store all the metrics.
        """

        self.attribute = attribute
        self.graph_type = main_graph.graph_type
        self.nodes = pd.DataFrame.from_dict(main_graph.nodes_dict(), orient='index')
        if self.attribute not in self.nodes.columns:
            raise ValueError("Atribute '{self.attribute}' is not available in the given graph node directory")

        self.unique_values = self.nodes[self.attribute].unique()
        if len(self.unique_values) == 1:
            raise Exception("Attribute selected only have one value. No subgraph can be extracted")
        
        sub_graphs = dict()
        for value in self.unique_values:
            sub_graphs.update({value : main_graph.base_graph().subgraph(self.nodes[(self.nodes[attribute]== value)].index.to_list())})
        
        self.sub_graphs = sub_graphs
              
        
    def calculate_metrics(self) -> None:
        """Calculate metrics for the stuff"""
    
        self.subgraph_metrics = dict()
        self.subgraph_distributions = dict()
    
        for key in self.sub_graphs:
            sub_graph = self.sub_graphs.get(key)
            
            metrics = dict()
            metrics.update({'node_number' : sub_graph.number_of_nodes()})
            metrics.update({'edge_number' : sub_graph.number_of_edges()})
            metrics.update({'density' : nx.density(sub_graph)})
            metrics.update({'connect_components' : nx.number_connected_components(sub_graph)})
            metrics.update({'transitivity' : nx.transitivity(sub_graph)})
            
            self.subgraph_metrics.update({key : metrics})
            
            distributions = dict()
            degree = {node:deg for (node, deg) in sub_graph.degree()}
            betweenness = nx.betweenness_centrality(sub_graph, normalized = False)
            closeness = nx.closeness_centrality(sub_graph)
            node_distributions =  pd.DataFrame.from_records([degree, betweenness,closeness], index = ["Degree", "Betweeness", "Closeness"]).transpose() 
            distributions.update({'nodes' : node_distributions})
            if self.graph_type == W:
                distributions.update({'Weight' : nx.to_pandas_edgelist(sub_graph).rename(columns = { 'source' : 'Source', 'target' : 'Target', 'weight' : 'Weight'})})

            self.subgraph_distributions.update({key : distributions})


    def subgraphs(self) -> dict[Graph]:
        """Return dictionary of subgraphs"""
        
        return self.sub_graphs

    def metrics(self) -> dict[dict]:
        """Return dictionary of metrics"""

        return self.subgraph_metrics

    def distributions(self) -> list[objects]:
        """Return dictionary of distributions"""

        return self.subgraph_distributions

