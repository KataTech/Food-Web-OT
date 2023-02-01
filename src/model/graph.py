"""
This python file contains useful methods and classes for processing data
"""

import numpy as np 
import networkx as nx
import pickle, os
from collections import defaultdict

# Visualization libraries primarily for statistics of GraphOT_Factory
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
        self.graph = nx.convert_node_labels_to_integers(graph, label_attribute="name")
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
        
class GraphOT_Factory: 
    """
    Generates and maintains a set of GraphOT objects. Contains
    neat operations such as compute the GW_Barycenter of a set of OT graphs
    """

    def __init__(self, name2graph: dict):
        """
        Initialzies an instance of GraphOT factory. 

        Parameters
        ----------
        name2graph : dict : str -> nx.Graph
            A mapping from graph names to the networkX objects
        
        """ 
        self.factory = name2graph
        self.ot_factory = self.make(name2graph)

    def make(self, name2graph):
        """
        Make a dictionary of GraphOT objects

        Parameters
        ----------
        name2graph : dict : str -> nx.Graph
            A mapping from graph names to the networkX objects
        """ 
        ot_factory = {}
        for name, nx_graph in name2graph.items(): 
            ot_factory[name] = GraphOT(nx_graph)
        return ot_factory
    
    def save(self, save_path: str): 
        """
        Save the current GraphOT_Factory

        Parameters
        ----------
        save_path : str
            The path for saving the current factory
            Path should end with ".pkl" for consistency
        """
        with open(save_path, "wb") as f: 
            pickle.dump(self, f)
        
    @staticmethod
    def load(load_path: str): 
        """
        Load a GraphOT object from the specified path.

        Parameters
        ----------
        load_path : str
            The path to a pickle file storing the GraphOT_Factory
        """
        with open(load_path, "rb") as f: 
            graphOT_factory = pickle.load(f)
        return graphOT_factory
        
    def summary(self, save=False, save_path="", save_title="summary"): 
        """
        Compute and output summary statistics of this graph factory

        Parameters
        ----------
        save : bool 
            Whether to save the computed instances of the 
            summary call
        save_path : str
            The path for saving the computed statistics and mappings 
        save_title : str 
            The title of the save file

        Returns 
        ----------
        info : dict
            A mapping containing all the computed variables of this function
        """
        # Initialize all the relevant variables 
        total_nodes = 0.0                           # the total number of nodes
        total_edges = 0.0                           # the total number of edges
        max_nodes = -float("inf")                   # the largest number of nodes 
        max_name = ""                               # the graph with the maximum nodes
        min_nodes = float("inf")                    # the smallest number of nodes
        min_name = ""                               # the graph with the minimal nodes
        graph_ct = 0.0                              # the total number of graphs
        graph_with_cycles = 0                       # the number of graphs with at least one cycle
        has_cycle = []                              # the name of graphs with cycle(s) 
        animal_freq = defaultdict(int)              # the mapping of animal to occurence over all graphs 
        edge_distribution = defaultdict(int)        # the number of times X edges populated a graph
        node_distribution = defaultdict(int)        # the number of times X nodes populated a graph
        tot_deg_distribution = defaultdict(int)     # the mapping of degree to the number of nodes with that degree
        graph2deg_dist = {}                         # the mapping of graph name to the degree distribution
        num_undirected = 0                          # the number of undirected graphs

        # Iterate through the networkx graphs 
        for name, nx_graph in self.factory.items(): 
            deg_distribution = defaultdict(int)
            # extract the number of nodes and edges in the current graph
            num_nodes = nx_graph.number_of_nodes()
            num_edges = nx_graph.number_of_edges()
            # compare to current min and max
            if max_nodes < num_nodes: 
                max_nodes = num_nodes
                max_name = name
            if min_nodes > num_nodes: 
                min_nodes = num_nodes
                min_name = name
            # keep track of the animal appearances and degrees
            for node in nx_graph.nodes: 
                animal_freq[node] += 1
                degree = nx_graph.degree[node]
                deg_distribution[degree] += 1
                tot_deg_distribution[degree] += 1
            # check for cycles in the graph
            if nx.is_directed(nx_graph) and len(list(nx.simple_cycles(nx_graph))) > 0: 
                graph_with_cycles += 1
                has_cycle.append(name)
            elif not nx.is_directed(nx_graph): 
                num_undirected += 1
            # keep track of total edges 
            total_nodes += num_nodes
            total_edges += num_edges 
            # increment the edge and node distributions
            node_distribution[num_nodes] += 1
            edge_distribution[num_edges] += 1
            # store the degree distribution of the current graph
            graph2deg_dist[name] = deg_distribution
            # increment the graph_ct
            graph_ct += 1


        # calculate average node and edges 
        average_node = total_nodes / float(graph_ct)
        average_edge = total_edges / float(graph_ct)

        # output 
        if num_undirected > 0: 
            print("WARNING: Cannot Detected Cycles in Undirected Graphs")
            print(f"Number of Undirected Graphs  : {num_undirected}\n\n")

        print(f"--------------- Summary ---------------")
        print(f"Total number of graphs       : {graph_ct}")
        print(f"Average number of nodes      : {average_node}")
        print(f"Average number of edges      : {average_edge}")
        print(f"Largest food web             : {max_name} with {max_nodes} animals")
        print(f"Smallest food web            : {min_name} with {min_nodes} animals")
        print(f"Number of graphs with cycles : {graph_with_cycles}")
        
        # set-up the 3-subplots for respective distributions
        _, ax = plt.subplots(3, 1, figsize=(8, 15))
        # plot node distribution
        ax[0].bar(node_distribution.keys(), node_distribution.values(), 5, color='g', alpha=0.8)
        ax[0].set_title("Node Distribution")
        # plot edge distribution
        ax[1].bar(edge_distribution.keys(), edge_distribution.values(), 5, color='r', alpha=0.8)
        ax[1].set_title("Edge Distribution")
        # plot degree distribution
        ax[2].bar(tot_deg_distribution.keys(), tot_deg_distribution.values(), 5, color='b', alpha=0.8)
        ax[2].set_title("Degree Distribution")

        # store all the computed variables into the info mapping 
        info = {}
        info["total_nodes"]     = total_nodes
        info["total_edges"]     = total_edges
        info["max_nodes"]       = max_nodes
        info["max_name"]        = max_name
        info["min_nodes"]       = min_nodes
        info["min_name"]        = min_name
        info["graph_ct"]        = graph_ct
        info["graph_wl_cycles"] = graph_with_cycles
        info["has_cycle"]       = has_cycle
        info["animal_freq"]     = animal_freq
        info["edge_dist"]       = edge_distribution
        info["node_dist"]       = node_distribution
        info["tot_deg_dist"]    = tot_deg_distribution
        info["graph2deg_dist"]  = graph2deg_dist
        info["num_undirected"]  = num_undirected

        # save the info if required
        if save: 
            file_path = os.path.join(save_path, save_title + ".pkl")
            with open(file_path, "wb") as f: 
                pickle.dump(info, f)
        
        return info

    #TODO: Implement a filter operation that takes into account 
    # some function by which to filter, some function by which to 
    # retrieve certain variable values from the GraphOT / NetworkX,
    # the type of objects to filter by, and outputs a new GraphOT_Factory
            
