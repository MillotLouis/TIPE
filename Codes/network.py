import networkx as nx
import simpy

class Message:
    def __init__(self,typ,src_id,src_seq,dest_seq,dest_id,hop,prev_hop,data=None):
        self.type = typ
        self.data = data
        self.src_id = src_id
        self.src_seq = src_seq
        self.dest_seq = dest_seq
        self.dest_id = dest_id
        self.weight = weight
        self.prev_hop = prev_hop

class Node:
    def __init__(self,env,id,pos,initial_battery,max_dist,network):
        self.env = env
        self.id = id
        self.pos = pos
        self.battery = initial_battery
        self.routing_table = {} # {dest: (next_hop, seq_num, hops)}
        self.seq_num = 0
        self.pending = simpy.Store(env)
        self.max_dist = max_dist
        self.alive = True
        self.network = network
        env.process(self.process_messages())

    def process_messages(self):
        while True:
            if self.alive:
                msg = yield self.pending.get() #bloque le process jusqu'à recevoir un item de la queue
                if msg.type == "RREQ":
                    self.handle_rreq(msg)
                elif msg.type == "RREP":
                    self.handle_rrep(msg)

    def init_rreq(self,dest_id):
        self.seq_num += 1
        rreq = Message(
            typ="RREQ",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_id=dest_id,
            dest_seq=self.routing_table.get(dest_id, (None, 0, 0))[2], #dernier numéro de séquence connu pour la source
            hop=0,
            prev_hop=self.id
        )
        self.network.broadcast_rreq(self.id,rreq)

    def handle_rreq(self, rreq:Message):
        self.update_route(rreq.src_id,rreq.prev_hop,rreq.src_seq,rreq.weight)## On met à jour l'entrée de la table de routage à destination du noeud source (reverse path) afin de pouvoir renvoyer le rrep plus tard
        
        if self.id == rreq.dest_id:
            self.send_rrep

        current_seq = self.routing_table.get(rreq['dest_id'], (None, None, 0))[2]

    def handle_rrep(self,rrep):
        pass

    def send_rrep(self,rrep):
        pass

    def update_route(self,dest,next_hop,seq_num,weight):
        current = self.routing_table.get(dest, (None, -1, float('inf')))

        if (seq_num > current[1]) or (seq_num == current[1] and weight < current[2]):
            self.routing_table[dest] = (next_hop, seq_num, weight) #On met à jour l'entrée dans la table de routage si le numéro de séquence est supérieur à celui connu OU si il est égal mais la route a un poids préférable        

class Network:
    def __init__(self,conso,seuil,a,b):
        """
        conso : tuple (x,y) : x = pourcentage de batterie consomée à chaque transmission de rreq, y = ... à chaque transmission de message
        seuil : seuil en dessous duquel un noeud évite de transmettre des messages : on lui affecte poids très grand
        a,b : paramètres de pondération des arcs : weight = a*distance + b*(1/batterie)
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
        if n1.alive and n2.alive:
            self.G.add_edge(n1, n2)


    def update_battery(self,node,type):
        """Retire percent% de batterie à node"""
        cons = self.conso[0] if type == 'RRE' else self.conso[1]
        self.battery = max(0,self.battery - cons)
        
        if node.battery <= 0:
            self.G.remove_edges_from(list(self.get_neighbors(node))) #Supprime les connexions avec ce noeud
        
        return self.battery > 0 #Indique si un noeud est tombé en panne car plus de batterie
    
    def get_distance(self,n1,n2):
        """Renvoie la distance entre node1 et node2"""
        return ((self.G.nodes[n2.id]["obj"].pos[0] - self.G.nodes[n1.id]["obj"].pos[0])**2 + (self.G.nodes[n2.id]["obj"].pos[1] - self.G.nodes[n1.id]["obj"].pos[1])**2)**0.5

    def get_battery(self,node):
        """Renvoie la batterie de node"""
        return self.G[node]["obj"].battery
    
    def get_neighbors(self,node):
        """Renvoie un iterator sur les voisins de node"""
        return self.G.neighbors(node.id)
    
    def calculate_weight(self,n1,n2)
        """Calcule le poids de l'arc n1 vers n2 en prenant en compte la distance à ce dernier et sa batterie"""
        bat = self.get_battery(n2)
        dist = self.get_distance(n1,n2)
        return self.a*dist + self.b*(1/bat)
