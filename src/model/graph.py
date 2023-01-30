"""
This python file contains useful methods and classes for processing data
"""

import numpy as np 
import networkx as nx

class GraphOT: 
    """
    A class for graph objects that are used in GraphOT operations.
    """

    def __init__(self, graph: nx.Graph, prob_method="degree", cost_method="shortest_path", 
                info = ""): 
        """
        Initialize a GraphOT instance. 

        Parameters
        ----------
        graph : nx.Graph
            The networkx object representing the 

        prob_method : string
            The method to use for endowing the nodes with a probability 
            distibution. Acceptable strings are "uniform" and "degree". 
            If "uniform", then every node will receive the same, normalized
            value. If "degree", then every node will received a value 
            proportional to how many neighbors it has in the graph, normalized. 
        """
        self.graph = graph
        self.node_dist = self.compute_prob(prob_method)
        self.cost = self.compute_cost(cost_method)
        self.info = info
        
    def compute_prob(self, prob_method:str) -> np.ndarray: 
        """
        Compute the probability distribution according to outlined method

        Parameters
        ----------
        prob_method : string
            The method to use for endowing the nodes with a probability 
            distibution. Acceptable strings are "uniform" and "degree". 
            If "uniform", then every node will receive the same, normalized
            value. If "degree", then every node will received a value 
            proportional to how many neighbors it has in the graph, normalized. 

        Returns
        ----------
        node_dist : ndarray (n)
            Probability distribution corresponding to the nodes of the graph.
        """
        n = self.graph.number_of_nodes()
        node_dist = np.zeros(n)

        if prob_method == "uniform": 
            node_dist += 1
            node_dist /= np.sum(node_dist)
            return node_dist

        elif prob_method == "degree":
            # traverse through all connections  
            for edge in self.graph.edges: 
                src = edge[0]; dst = edge[1]
                # add weights if present, otherwise just add 1's 
                if nx.is_weighted(self.graph): 
                    node_dist[src] += self.graph[src][dst]['weight']
                    node_dist[dst] += self.graph[src][dst]['weight']
                else: 
                    node_dist[src] += 1
                    node_dist[dst] += 1
            # normalize
            node_dist /= np.sum(node_dist)
            return node_dist

        # if `prob_method` is not anticipated, return the uniform node distribution
        print("Warning: Non-identifiable probability method.")
        return self.compute_prob("uniform")

    def compute_cost(self, cost_method: str, check_metric=False) -> np.ndarray: 
        """
        Compute the relational cost matrix according to the outlined method

        Parameters
        ----------
        cost_method : string
            The method to use for computing relational distance between 
            the nodes in the graph. Acceptable strings are "adjacency" 
            and "shortest_path". 

        check_metric : boolean (optional)
            Ensures that the provided method results in a metric over the 
            graph space. If not, throw an exception. 

        Returns 
        ----------
        cost_matrix : ndarray (n, n)
            Relational cost matrix where entry (i, j) represents the
            distance between node (i) and node (j) in the graph.
        """
        
        if check_metric: 
            metrics = ["shortest_path"]
            assert cost_method in metrics, "Non-metric for relational cost matrix"

        if cost_method == "adjacency": 
            # extract the dense representation of the graph 
            return nx.adjacency_matrix(self.graph).toarray()

        elif cost_method == "shortest_path": 
            # floyd_warshall is a shortest path method that 
            # works on graphs with negative edges 
            return nx.floyd_warshall_numpy(self.graph)
        
        # if `cost_method` is not anticipated, return the shortest_path method result
        print("Warning: Non-identifiable cost method.")
        return self.compute_cost("shortest_path")

    def get_node_dist(self):
        """
        Returns the graph's node distribution
        """
        return np.copy(self.node_dist)

    def get_cost(self): 
        """
        Returns the graph's cost matrix
        """
        return np.copy(self.cost)

    def extract_info(self): 
        """
        Returns the graph's node distribution AND cost matrix
        """
        return np.copy(self.node_dist), np.copy(self.cost)

    # TODO: Need to add functionalities to save and load GraphOT objects
        