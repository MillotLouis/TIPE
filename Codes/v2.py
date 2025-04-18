import networkx as nx
import simpy
import random
import matplotlib.pyplot as plt


class NetworkSimulator:
    def __init__(self):
        self.env = simpy.Environment()
        self.G = nx.DiGraph()
        self.routing_algorithm = nx.dijkstra_path  # Algorithme par défaut
        
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
    
    def dist(self,node1,node2):
        return ((self.G.nodes[node2]["pos"][0] - self.G.nodes[node1]["pos"][0])**2 + (self.G.nodes[node2]["pos"][1] - self.G.nodes[node1]["pos"][1])**2)**0.5
    
    
    def _transmit_packet(self, path, payload):
        """Simule la transmission du paquet à travers le chemin"""

        

# Exemple d'utilisation
def main():


    nx.draw(net.G, with_labels=True, node_color='lightblue')
    plt.show()

if __name__ == "__main__":
    main()