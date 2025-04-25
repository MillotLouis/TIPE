import networkx as nx
import simpy
import random
import matplotlib.pyplot as plt
from Codes.Dijkstra import *


class Node:
    def __init__(self,env,id,pos,initial_battery):
        self.env = env
        self.id = id
        self.pos = pos
        self.battery = initial_battery
        self.routing_table = {}
        self.seq_num = 0
        self.pending = {}

    def update_battery(self,cons):
        self.battery = max(0,self.battery - cons)
        return self.battery > 0 #Indique si un noeud est tombé en panne car plus de batterie
    


class Network:
    def __init__(self):
        self.env = simpy.Environment()
        self.G = nx.Graph()

    def add_node(self,id,pos,battery=100):
        new_node = Node(self.env,id,pos,battery)
        self.G.add_node(id,obj=new_node)

    def add_link(self, src, dest, weight):
        self.G.add_edge(src, dest, weight=weight)

    def remove_link(self,n1,n2):
        self.G.remove_edge(n1,n2)

    def update_weight(self,n1,n2,weight):
        self.G[n1][n2]["weight"] = weight
    
    def get_distance(self,node1,node2):
        """Calcule la distance entre node1 et node2"""
        return ((self.G.nodes[node2]["obj"].pos[0] - self.G.nodes[node1]["obj"].pos[0])**2 + (self.G.nodes[node2]["obj"].pos[1] - self.G.nodes[node1]["obj"].pos[1])**2)**0.5

    def get_battery(self,node):
        """Renvoie la batterie de node"""
        return self.G[node]["obj"].battery
    
    def get_neighbors(self,node):
        """Renvoie un iterator sur les voisins de node"""
        return self.G.neighbors(node)
    
