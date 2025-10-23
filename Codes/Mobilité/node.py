import random
from collections import defaultdict
import simpy
import networkx as nx


class Node:
    def __init__(self, env, id, pos, initial_battery, max_dist, network, reg_aodv):
        self.env = env                             
        """environnement simpy"""
        
        self.id = id                               
        """id du noeud"""
        
        self.pos = pos                             
        """ position (x,y) du noeud """
        
        self.battery = initial_battery             
        """ batterie du noeud """
        
        self.initial_battery = initial_battery     
        """ Garde en mémoire la batterie initiale pour des calculs """
        
        self.routing_table = {}                    
        """ dest : {next_hop,seq_num,weight,expiry} """
        
        self.seq_num = 0                           
        """ numéro de séquence du noeud """
        
        self.pending = simpy.Store(env)            
        """ requêtes en attente """
        
        self.max_dist = max_dist                   
        """distance max d'émission / portée du noeud"""
        
        self.alive = True                          
        """ True si le noeud est vivant False si il est mort """
        
        self.network = network                     
        """ Classe network dont fait partie le noeud """
        
        self.seen = set() if reg_aodv else {}      
        """ (rreq.src_id, rreq.src_seq) : (compteur,meilleur poids) """
        
        self.collected_rreqs = {}                  
        """ Liste des rreqs collectés une fois que le premier est arrivé à la dest """
        
        self.to_be_sent = defaultdict(list)        
        """ Dictionnaire contenant la liste des messages à envoyer à chaque noeud """ 
        #default_dict permet de créer une liste vide comme valeur si une clé n'existe pas mais est accedé dans le dico
        
        self.reg_aodv = reg_aodv                   
        """ True si on utilise le AODV classique et False si on utilise le mien """
        
        self.MAX_DUPLICATES = 3                    
        """ On s'autorise 3 RREQs max par (src_id,src_seq)  """
        
        self.WEIGHT_SEUIL = 1.3                    
        """ Seuil à partir duquel on considère avoir vu une vraie amélioration """ 
        
        self.data_seq = 0
        """identique au numéro de séquence mais pour les messages"""

        self.env.process(self.process_messages())

    def process_messages(self):
        while self.alive:
            msg = yield self.pending.get() #On bloque le process jusqu'à avoir un nouveau message

            processing_delay = random.uniform(0.001, 0.005)
            yield self.env.timeout(processing_delay) # on bloque pour simuler un délai de processing
            
            
            if msg.type == "RREQ":
                self.handle_rreq(msg)
            elif msg.type == "RREP":
                self.handle_rrep(msg)
            elif msg.type == "DATA":
                self.handle_data(msg)

    def init_rreq(self, dest_id):
        from network import Message
        self.seq_num += 1 # IMPORTANT ! 
        self.network.rreq_sent += 1

        rreq = Message(
            typ="RREQ",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_id=dest_id,
            dest_seq=self.routing_table.get(dest_id, (None, 0, 0, 0))[1],
            prev_hop=self.id,
            weight=0
        )
        
        self.env.process(self.network.broadcast_rreq(self, rreq))

    def handle_rreq(self, rreq):
        prev_node = self.network.G.nodes[rreq.prev_hop]["obj"]
        weight = self.network.calculate_weight(prev_node, self) #inclus la pénalité si batterie en dessous du seuil
        rreq.weight += weight
        
        if rreq.src_id == self.id:
            return #on discard si on a déjà vu : évite les **boucles** ♥
                   #éviter que les RREQs soient renvoyés à la source       
        
        
        if self.reg_aodv:
            """reg_aodv = True"""
            if (rreq.src_id,rreq.src_seq) in self.seen:
                return
            self.seen.add((rreq.src_id,rreq.src_seq))

            if self.id == rreq.dest_id:
                self.send_rrep(rreq)
                return
        
        else:
            """reg_aodv = False"""
            #vérification pour éviter boucles
            seen_key = (rreq.src_id, rreq.src_seq)
            count,min_weight = self.seen.get(seen_key, (0,float('inf')))
            if count > self.MAX_DUPLICATES or rreq.weight * self.WEIGHT_SEUIL >= min_weight:
                    return
            else:
                self.seen[seen_key] = (count + 1,rreq.weight) 
            
            # Collecte des rreps si on est la destination
            if self.id == rreq.dest_id: #Si on est la destination du RREQ
                key = (rreq.src_id, rreq.src_seq)
                if key not in self.collected_rreqs:
                    self.collected_rreqs[key] = []
                    self.env.process(self.collect_rreps(key)) # on commence la collecte des RREPs
                self.collected_rreqs[key].append(rreq)
                return
            
        
        
        self.update_route(
            dest=rreq.src_id,
            next_hop=rreq.prev_hop,
            seq_num=rreq.src_seq,
            weight=rreq.weight
        )
        
        
        rreq.prev_hop = self.id
        self.env.process(self.network.broadcast_rreq(self, rreq))


    def handle_rrep(self, rrep):
        """Marche pareil si reg_aodv ou pas"""
        prev_node = self.network.G.nodes[rrep.prev_hop]["obj"]
        weight = self.network.calculate_weight(prev_node, self) #inclus la penalité si en dessous du seuil
        rrep.weight += weight
        
        self.update_route(
            dest=rrep.src_id,
            next_hop=rrep.prev_hop,
            seq_num=rrep.src_seq,
            weight=rrep.weight
        )

        if self.id == rrep.dest_id:
            if rrep.src_id in self.to_be_sent: #si on a déjà des messages en attente on les envoie
                for msg in self.to_be_sent[rrep.src_id]:
                    self.env.process(self.network.forward_data(self, msg))
                    self.network.messages_sent += 1
                    self.network.log_data_send(self.id, msg.src_seq, self.env.now)
                del self.to_be_sent[rrep.src_id]
        else:
            rrep.prev_hop = self.id 
            self.env.process(self.network.unicast_rrep(self, rrep))

    def send_rrep(self, rreq):
        from network import Message
        """Marche pareil si reg_aodv ou pas"""
        #appelé quand on est la destination d'un RREQ
        self.seq_num += 1
        self.network.rrep_sent += 1 

        self.update_route(
            dest=rreq.src_id,
            next_hop=rreq.prev_hop,
            seq_num=rreq.src_seq,
            weight=rreq.weight
        )
        
        rrep = Message(
            typ="RREP",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_id=rreq.src_id,
            dest_seq=-1,
            weight=0,
            prev_hop=self.id
        )

        self.env.process(self.network.unicast_rrep(self, rrep))

    def update_route(self, dest, next_hop, seq_num, weight):
        current = self.routing_table.get(dest, (None, -1, float('inf'), 0))
        
        if not self.reg_aodv:    
            # dynamic_ttl = max(1, self.network.ttl * (self.battery/100000))
            
            if (seq_num > current[1]) or (seq_num == current[1] and weight < current[2]):
                self.routing_table[dest] = (next_hop, seq_num, weight, self.env.now + self.network.ttl)

        else:
            if (seq_num > current[1]) or (seq_num == current[1] and weight < current[2]): #si la route est plus fraiche ou aussi fraiche avec un poids moindre
                self.routing_table[dest] = (next_hop, seq_num, weight, self.env.now + self.network.ttl)


    def collect_rreps(self, key):
        """Appelé uniquement si reg_aodv = False"""
        yield self.env.timeout(0.2)
        # On attend pour que tous les RREQs arrivent à la dest et soient stockés dans self.collected_rreqs[key]
        
        if key in self.collected_rreqs:
            rreqs = self.collected_rreqs.get(key)
            best_rreq = min(rreqs, key=lambda r: r.weight)
            self.send_rrep(best_rreq) #on envoie le meilleur

    def handle_data(self, data):
        """Marche pareil si reg_aodv ou pas"""
        if data.dest_id == self.id:
            self.network.messages_received += 1
            self.network.log_data_recv(data.src_id, data.src_seq, self.env.now)

        else:
            self.env.process(self.network.forward_data(self, data))

    def send_data(self, dest_id):
        from network import Message
        """Marche pareil si reg_aodv ou pas"""
        self.data_seq += 1
        msg = Message(
            typ="DATA",
            src_id=self.id,
            src_seq=self.data_seq,
            dest_id=dest_id,
            dest_seq=-1,
            weight=-1,
            prev_hop=self.id,
        )
        self.network.messages_initiated += 1
        self.network.log_data_init(self.id, self.data_seq, self.env.now)

        if dest_id in self.routing_table and self.routing_table.get(msg.dest_id, (None, 0, 0, 0))[3] >= self.env.now:
            #si la route existe et est toujours valide ie si la date d'expiration n'est pas encore dépassée
            self.network.messages_sent += 1
            self.env.process(self.network.forward_data(self, msg))
            self.network.log_data_send(self.id, msg.src_seq, self.env.now)
        else:
            #si route inexistante ou plus valide
            self.to_be_sent[dest_id].append(msg)
            self.init_rreq(dest_id)
