import networkx as nx
import simpy
import copy

from node import Node

class Message:
    def __init__(self,typ,src_id,src_seq,dest_seq,dest_id,weight,prev_hop,data=None):
        self.type = typ
        # self.data = data # Surement pas utile
        self.src_id = src_id
        self.src_seq = src_seq
        self.dest_seq = dest_seq
        self.dest_id = dest_id
        self.weight = weight
        self.prev_hop = prev_hop

# ♥ Passer uniquement les objets en argument de fonction pas les id #

class Network:
    def __init__(self,conso,seuil,coeff_dist,coeff_bat,coeff_conso):
        """
        conso : tuple (x,y) : x = pourcentage de batterie consomée à chaque transmission de rreq, y = ... à chaque transmission de message
        seuil : seuil en dessous duquel un noeud évite de transmettre des messages : on lui affecte poids très grand
        a,b : paramètres de pondération des arcs : weight = a*distance + b*(1/batterie)
        """
        self.env = simpy.Environment()
        self.G = nx.Graph()
        self.conso = conso
        self.seuil = seuil
        self.coeff_dist = coeff_dist
        self.coeff_bat = coeff_bat
        self.coeff_conso = coeff_conso
        
        self.messages_forwarded = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.rreq_sent = 0
        self.rrep_sent = 0
        self.battery_history = []

    def add_node(self,id,pos,max_dist,battery=100):
        """Ajoute un noeud au graphe"""
        new_node = Node(self.env,id,pos,battery,max_dist,self)
        self.G.add_node(id,obj=new_node)

    def add_link(self, n1, n2):
        """Ajoute une arrête entre n1 et n2 si il leur reste de la batterie"""
        if n1.alive and n2.alive:
            self.G.add_edge(n1.id, n2.id)

    def update_battery(self,node,type,dist):
        """Retire percent% de batterie à node"""
        cons = self.conso[0] if type == 'RRE' else self.conso[1]
        node.battery = max(0,node.battery - (self.coeff_cons*dist + cons))
        if node.battery <= 0:
            self.G.remove_edges_from(list(self.G.neighbors(node.id))) #Supprime les connexions avec ce noeud
            node.alive = False
        return node.alive
        
    def get_distance(self,n1,n2):
        """Renvoie la distance entre node1 et node2,
        on passe uniquement les objets en argument
        """
        return ((n2.pos[0] - n1.pos[0])**2 + (n2.pos[1] - n1.pos[1])**2)**0.5

    
    def calculate_weight(self,n1,n2):
        """Calcule le poids de l'arc n1 vers n2 en prenant en compte la distance à ce dernier et sa batterie"""
        bat = n2.battery
        dist = self.get_distance(n1,n2)
        return self.coeff_dist*dist + self.coeff_bat*(1/bat)

    def broadcast_rreq(self,node,rreq):
        for n in self.G.neighbors(node.id):
            neighbor = n["obj"]
            dist = self.get_distance(node,neighbor)
            if dist <= node.max_dist:
                if self.update_battery(node,"RRE",neighbor):
                    self.rreq_sent += 1
                    new_rreq = copy.deepcopy(rreq)
                    neighbor.pending.put(new_rreq)
                #sinon, ajouter compteur perdus peut être


    def unicast_rrep(self,node,rrep):
        next_hop = node.routing_table.get(rrep.src_id,(None,0,0))[0]
        if next_hop:
            dist = self.get_distance(node, self.G[next_hop]["obj"])
            rrep.prev_hop = node.id
            yield self.env.timeout(0.001) #délai à modifier peut-être
            if dist <= node.max_dist:
                #else rrep perdu à ajouter au compteur
                if self.update_battery(node,"RRE",dist):
                    self.rrep_sent += 1
                    self.G[next_hop]["obj"].pending.put(rrep)
