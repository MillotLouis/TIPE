import simpy

from network import Network,Message

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
        self.pending_rreqs = {} #permet de collecter les rreqs pour comparer le poids de celles venant d'une même source
        
        self.env.process(self.process_messages())

    def process_messages(self):
        while True:
            if self.alive:
                msg = yield self.pending.get() #bloque le process jusqu'à recevoir un item de la queue
                if msg.type == "RREQ":
                    self.handle_rreq(msg)
                elif msg.type == "RREP":
                    self.handle_rrep(msg)
                elif msg.type == "DATA":
                    self.handle_data(msg)

    def init_rreq(self,dest_id):
        self.seq_num += 1 #indispensable selon RFC 3561
        for n in self.network.G.neighbors(self.id):
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
        seen_key = (rreq.src_id,rreq.src_seq,rreq.prev_hop)
        if seen_key in self.seen or self.battery<self.network.seuil:
            return #Si on a déjà vu cette RREQ de la part du même voisin on ne la forward pas, on peut vérifier cela avec le seq_num de la source car il est incrémenté pour chaque RREQ
                   # Ou si le noeud est en dessous du seuil de batterie 
        
        ## Sinon on forward le RREQ
        rreq.weight += self.network.calculate_weight(self.network.G[rreq.prev_hop]["obj"],self) #On ajoute le poids de l'arc qui va du noeud précedent à nous
        
        self.update_route(
            dest=rreq.src_id,
            next_hop=rreq.prev_hop,
            seq_num=rreq.src_seq,
            weight=rreq.weight)     # On met à jour l'entrée de la table de routage à destination du noeud source (reverse path) afin de pouvoir renvoyer le rrep plus tard
        
        if self.id == rreq.dest_id:
            key = (rreq.src_id, rreq.src_seq)
            if key not in self.pending_rreqs:
                self.pending_rreqs[key] = []
                self.env.process(self.collect_rreps(key)) #on lance le processus de collecte des rreqs identiques
            self.pending_rreqs[key].append(rreq)    
            return
        
        self.seen.add(seen_key)
        rreq.prev_hop = self.id
        self.network.broadcast_rreq(self,rreq)

    def handle_rrep(self,rrep:Message):
        rrep.weight += self.network.calculate_weight(self.network.G[rrep.prev_hop]["obj"],self) #On ajoute le poids de l'arc qui va du noeud précedent à nous
        
        self.update_route(
            dest=rrep.src_id,
            next_hop=rrep.prev_hop,
            seq_num=rrep.src_seq,
            weight=rrep.weight) #On met à jour la route vers le noeud des destination original

        if self.id != rrep.dest_id:
            self.network.unicast_rrep(self,rrep)
            
    def send_rrep(self,rreq):
        self.seq_num += 1
    
        self.update_route(
                dest = rreq.src_id,
                next_hop=rreq.prev_hop,
                seq_num = rreq.src_seq,
                weight= rreq.weight
            )
        
        rrep = Message(
            typ="RREP",
            src_id=rreq.dest_id,    # destination originale
            src_seq= self.seq_num,  # numéro de séquence du noeud de destination orinal, sert à remplie la table de routage de la source originale
            dest_seq= -1,           # pas utilisé dans les RREP 
            dest_id=rreq.src_id,    # source originale
            weight=0,
            prev_hop=self.id
        )
        self.network.unicast_rrep(self,rrep)

    def update_route(self,dest,next_hop,seq_num,weight):
        current = self.routing_table.get(dest, (None, -1, float('inf')))
        
        if (seq_num > current[1]) or (seq_num == current[1] and weight < current[2]):
            self.routing_table[dest] = (next_hop, seq_num, weight) #On met à jour l'entrée dans la table de routage si le numéro de séquence est supérieur à celui connu OU si il est égal mais la route a un poids préférable        

    def collect_rreps(self,key):
        yield self.env.timeout(0.2)

        if key in self.pending_rreqs:
            rreqs = self.pending_rreqs.pop(key)
            best_rreq = min(rreqs, key= lambda r:r.weight)

        self.send_rrep(best_rreq)
        
        del self.pending_rreqs[key]

        return best_rreq
        

    def handle_data(self,data:Message):
        if data.dest_id == self.id:
            self.network.messages_received += 1
        else:
            next_hop = self.routing_table.get(data.dest_id,(None, 0, 0))[2]
            if next_hop:
                pass