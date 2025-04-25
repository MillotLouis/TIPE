import networkx as nx
import simpy
import random
import matplotlib.pyplot as plt
from Codes.Dijkstra import *


class NetworkSimulator:
    def __init__(self):
        self.env = simpy.Environment()
        self.G = nx.Graph()
        self.routing_algorithm = get_path_dijkstra
        self.table = {node:{node:(float('inf'),[]) for node in self.G} for node in self.G} 
        
    def add_node(self, node_id,x,y,bat):
        self.G.add_node((node_id,{"pos":[x,y],"bat":bat}))
        
    def add_link(self, src, dest, weight=1, capacity=1):
        self.G.add_edge(src, dest, weight=weight, capacity=capacity)
        
    def set_routing_algorithm(self, algorithm):
        """Permet de choisir l'algorithme de routage (ex: nx.bellman_ford_path)"""
        self.routing_algorithm = algorithm
        
    def send_packet(self, src, dest, payload):
        path = self.routing_algorithm(self.G, src, dest, weight='weight')
        return self.env.process(self._transmit_packet(path, payload))
    
    def get_distance(self,node1,node2):
        """Calcule la distance entre node1 et node2"""
        return ((self.G.nodes[node2]["pos"][0] - self.G.nodes[node1]["pos"][0])**2 + (self.G.nodes[node2]["pos"][1] - self.G.nodes[node1]["pos"][1])**2)**0.5
    
    def get_battery(self,node):
        """Renvoie la batterie de node"""
        return self.G[node]["bat"]
    
    def get_neighbors(self,node):
        """Renvoie un iterator sur les voisins de node"""
        return self.G.neighbors(node) 
    
    def _transmit_packet(self, path, payload):
        """Simule la transmission du paquet à travers le chemin"""