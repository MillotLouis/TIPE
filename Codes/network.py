import networkx as nx
import simpy


class Node:
    def __init__(self,env,id,pos,initial_battery):
        self.env = env
        self.id = id
        self.pos = pos
        self.battery = initial_battery
        self.routing_table = {}
        self.seq_num = 0
        self.pending = {}

class Network:
    def __init__(self,conso,seuil,a,b):
        """
        conso : tuple (x,y) : x = pourcentage de batterie consomée à chaque transmission de rreq, y = ... à chaque transmission de message
        seuil : seuil en dessous duquel un noeud évite de transmettre des messages : on lui affecte poids très grand
        """
        self.env = simpy.Environment()
        self.G = nx.Graph()
        self.conso = conso
        self.seuil = seuil
        self.a = a
        self.b = b

    def add_node(self,id,pos,battery):
        """Ajoute un noeud au graphe"""
        new_node = Node(self.env,id,pos,battery)
        self.G.add_node(id,obj=new_node)
        new_node.routing_table = {}

    def add_link(self, n1, n2):
        """Ajoute une arrête entre n1 et n2 si il leur reste de la batterie"""
        if self.get_battery(n1) > 0 and self.get_battery(n2) > 0:
            weight = self.get_distance(n1,n2)
            self.G.add_edge(n1, n2, weight=weight)

    def remove_link(self,n1,n2):
        """Supprime le lien entre n1 et n2"""
        self.G.remove_edge(n1,n2)

    def update_weight(self,n1,n2,weight):
        """Modifie le poids de l'arrête entre n1 et n2"""
        self.G[n1][n2]["weight"] = weight

    def update_battery(self,node,percent):
        """Retire percent% de batterie à node"""
        self.battery = max(0,self.battery - percent)
        return self.battery > 0 #Indique si un noeud est tombé en panne car plus de batterie
    
    def get_distance(self,n1,n2):
        """Renvoie la distance entre node1 et node2"""
        return ((self.G.nodes[n2]["obj"].pos[0] - self.G.nodes[n1]["obj"].pos[0])**2 + (self.G.nodes[n2]["obj"].pos[1] - self.G.nodes[n1]["obj"].pos[1])**2)**0.5

    def get_battery(self,node):
        """Renvoie la batterie de node"""
        return self.G[node]["obj"].battery
    
    def get_neighbors(self,node):
        """Renvoie un iterator sur les voisins de node"""
        return self.G.neighbors(node)
    
    ## Protocole AODV ###

    def broadcast_rreq(self,src:Node,dest:Node):
        """Démarre une découverte de route de src à dest"""
        src.seq_num += 1
        rreq = {
            'src':src,
            'dest':dest,
            's_seq':src.seq_num
            's'
        }





    
