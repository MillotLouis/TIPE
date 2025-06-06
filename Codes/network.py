import networkx as nx
import simpy
import copy

class Message:
    def __init__(self,typ,src_id,src_seq,dest_seq,dest_id,weight,prev_hop,data=None):
        self.type = typ
        self.data = data
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

    def add_node(self,id,pos,battery):
        """Ajoute un noeud au graphe"""
        new_node = Node(self.env,id,pos,battery)
        self.G.add_node(id,obj=new_node)
        new_node.routing_table = {}

    def add_link(self, n1, n2):
        """Ajoute une arrête entre n1 et n2 si il leur reste de la batterie"""
        if n1.alive and n2.alive:
            self.G.add_edge(n1, n2)


    def update_battery(self,node,type,dist):
        """Retire percent% de batterie à node"""
        cons = self.conso[0] if type == 'RRE' else self.conso[1]
        node.battery = max(0,node.battery - (self.network.coeff_cons*dist + cons))
        if node.battery <= 0:
            self.G.remove_edges_from(list(self.get_neighbors(node))) #Supprime les connexions avec ce noeud
            node.alive = False
        return node.alive
        
    def get_distance(self,n1,n2):
        """Renvoie la distance entre node1 et node2,
        on passe uniquement les objets en argument
        """
        return ((n2.pos[0] - n1.pos[0])**2 + (n2.pos[1] - n1.pos[1])**2)**0.5


    def get_neighbors(self,node):
        """Renvoie un iterator sur les voisins de node"""
        return self.G.neighbors(node.id)
    
    def calculate_weight(self,n1,n2):
        """Calcule le poids de l'arc n1 vers n2 en prenant en compte la distance à ce dernier et sa batterie"""
        bat = n2.battery
        dist = self.get_distance(n1,n2)
        return self.coeff_dist*dist + self.coeff_bat*(1/bat)

    def broadcast_rreq(self,node,rreq):
        for n in self.get_neighbors(node):
            neighbor = n["obj"]
            dist = self.get_distance(node,neighbor)
            if self.get_distance <= node.max_dist:
                if self.update_battery(node,"RRE",neighbor):
                    new_rreq = copy.deepcopy(rreq)
                    neighbor.pending.put(new_rreq)


    def unicast_rrep(self,node,rrep):
        next_hop = self.routing_table.get(rrep.src_id,(None,0,0))[0]
        self.update_battery(node,"RRE",next_hop)
        if next_hop:
            #else paquet perdu à ajouter
            self.env.timeout(0.001)
            self.G[next_hop]["obj"].pending.put(rrep)

class Node:
    def __init__(self,env,id,pos,initial_battery,max_dist,network:Network):
        self.env = env
        self.id = id
        self.pos = pos
        self.battery = initial_battery
        self.routing_table = {} # {dest: (next_hop, seq_num, weight)} #ajouter TTL quand implémente envoi messages
        self.seq_num = 0
        self.pending = simpy.Store(env)
        self.max_dist = max_dist
        self.alive = True
        self.network = network
        self.seen = set() #évite les boucles de routage infinies
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
        self.seq_num += 1 #indispensable selon RFC 3561
        for n in self.network.get_neighbors:
            self.seen.add(self.id,self.seq_num,n["obj"].id) #sert à ce que les voisins du noeud de départ ne lui renvoient pas le RREQ directement
        rreq = Message(
            typ="RREQ",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_id=dest_id,
            dest_seq=self.routing_table.get(dest_id, (None, 0, 0))[2], #dernier numéro de séquence connu pour la source
            prev_hop=self.id,
            weight= 0
        )
        self.network.broadcast_rreq(self.id,rreq)

    def handle_rreq(self, rreq:Message):
        rreq.weight += self.network.calculate_weight(rreq.prev_hop,self.id) #On ajoute le poids de l'arc qui va du noeud précedent à nous
        
        self.update_route(rreq.src_id,rreq.prev_hop,rreq.src_seq,rreq.weight)## On met à jour l'entrée de la table de routage à destination du noeud source (reverse path) afin de pouvoir renvoyer le rrep plus tard
        
        if self.id == rreq.dest_id:
            best = self.collect_rreps()
            self.send_rrep(best)
            return
        
        if (rreq.src_id,rreq.src_seq,rreq.prev_hop) in self.seen or self.battery<self.network.seuil:
            return #Si on a déjà vu cette RREQ de la part du même voisin on ne la forward pas, on peut vérifier cela avec le seq_num de la source car il est incrémenté pour chaque RREQ
                   # Ou si le noeud est en dessous du seuil de batterie 


        ## Sinon on forward le RREQ
        self.seen.add(rreq.src_seq)
        rreq.prev_hop = self.id
        self.network.broadcast_rreq(self,rreq)

    def handle_rrep(self,rrep:Message):
        rrep.weight += self.network.calculate_weight(rrep.prev_hop,self.id) #On ajoute le poids de l'arc qui va du noeud précedent à nous
        
        self.update_route(rrep.src_id,rrep.prev_hop,rrep.src_seq,rrep.weight) #On met à jour la route vers le noeud des destination original

        if self.id != rrep.dest_id:
            self.network.unicast_rrep(self,rrep)
            
    def send_rrep(self,rreq):
        self.seq_num += 1
        rrep = Message(
            typ="RREP",
            src_id=rreq.dest_id,    # Original destination
            src_seq= self.seq_num,  # Destination's sequence number
            dest_seq=0,             # Not used in RREP
            dest_id=rreq.src_id,    # Original source
            weight=0,
            prev_hop=self.id
        )
        self.network.unicast_rrep(self,rrep)

    def update_route(self,dest,next_hop,seq_num,weight):
        current = self.routing_table.get(dest, (None, -1, float('inf')))

        if (seq_num > current[1]) or (seq_num == current[1] and weight < current[2]):
            self.routing_table[dest] = (next_hop, seq_num, weight) #On met à jour l'entrée dans la table de routage si le numéro de séquence est supérieur à celui connu OU si il est égal mais la route a un poids préférable        
