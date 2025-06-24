import random
import matplotlib.pyplot as plt
from collections import defaultdict
import simpy
import copy
import networkx as nx
import numpy as np

class Message:
    def __init__(self, typ, src_id, src_seq, dest_seq, dest_id, weight, prev_hop):
        self.type = typ
        self.src_id = src_id
        self.src_seq = src_seq
        self.dest_seq = dest_seq
        self.dest_id = dest_id
        self.weight = weight
        self.prev_hop = prev_hop
        

class Node:
    def __init__(self, env, id, pos, initial_battery, max_dist, network, reg_aodv):
        self.env = env
        self.id = id
        self.pos = pos
        self.battery = initial_battery
        self.initial_battery = initial_battery  # Store initial battery for metrics
        self.routing_table = {} # dest : {next_hop,seq_num,weight,expiry}
        self.seq_num = 0
        self.pending = simpy.Store(env)
        self.max_dist = max_dist
        self.alive = True
        self.network = network
        self.seen = set() if reg_aodv else {} # (rreq.src_id, rreq.src_seq) : (compteur,meilleur poids)
        self.pending_rreqs = {}
        self.to_be_sent = defaultdict(list)
        self.reg_aodv = reg_aodv #True si on utilise le AODV classique et False si on utilise le mien
        self.MAX_DUPLICATES = 3 #on s'autorise 3 RREQs par (src_id,src_seq) 
        self.WEIGHT_SEUIL = 1.3
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
                if key not in self.pending_rreqs:
                    self.pending_rreqs[key] = []
                    self.env.process(self.collect_rreps(key)) # on commence la collecte des RREPs
                self.pending_rreqs[key].append(rreq)
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
                del self.to_be_sent[rrep.src_id]
        else:
            rrep.prev_hop = self.id 
            self.env.process(self.network.unicast_rrep(self, rrep))

    def send_rrep(self, rreq):
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
            # Calculate dynamic TTL based on battery level
            dynamic_ttl = max(1, self.network.ttl * (self.battery/100000))
            
            if (seq_num > current[1]) or (seq_num == current[1] and weight < current[2]):
                self.routing_table[dest] = (next_hop, seq_num, weight, self.env.now + dynamic_ttl)

        else:
            if (seq_num > current[1]) or (seq_num == current[1] and weight < current[2]): #si la route est plus fraiche ou aussi fraiche avec un poids moindre
                self.routing_table[dest] = (next_hop, seq_num, weight, self.env.now + self.network.ttl)


    def collect_rreps(self, key):
        """Appelé uniquement si reg_aodv = False"""
        yield self.env.timeout(0.2)
        # On attend pour que tous les RREQs arrivent à la dest et soient stockés dans self.pending_rreqs[key]
        
        if key in self.pending_rreqs:
            rreqs = self.pending_rreqs.get(key)
            best_rreq = min(rreqs, key=lambda r: r.weight)
            self.send_rrep(best_rreq) #on envoie le meilleur

    def handle_data(self, data):
        """Marche pareil si reg_aodv ou pas"""
        if data.dest_id == self.id:
            self.network.messages_received += 1
        else:
            self.env.process(self.network.forward_data(self, data))

    def send_data(self, dest_id):
        """Marche pareil si reg_aodv ou pas"""
        msg = Message(
            typ="DATA",
            src_id=self.id,
            src_seq=-1,
            dest_id=dest_id,
            dest_seq=-1,
            weight=-1,
            prev_hop=self.id,
        )
        self.network.messages_initiated += 1

        if dest_id in self.routing_table and self.routing_table.get(msg.dest_id, (None, 0, 0, 0))[3] > self.env.now:
            #si la route existe et est toujours valide ie si la date d'expiration n'est pas encore dépassée
            self.network.messages_sent += 1
            self.env.process(self.network.forward_data(self, msg))
        else:
            #si route inexistante ou plus valide
            self.to_be_sent[dest_id].append(msg)
            self.init_rreq(dest_id)

class Network:
    def __init__(self, conso, seuil, coeff_dist, coeff_bat, coeff_conso, nb_nodes, ttl, reg_aodv):
        self.env = simpy.Environment()
        self.G = nx.Graph()
        self.conso = conso
        self.seuil = seuil
        self.coeff_dist = coeff_dist
        self.coeff_bat = coeff_bat
        self.coeff_conso = coeff_conso
        self.ttl = ttl
        self.stop = False
        self.reg_aodv = reg_aodv
        
        self.messages_forwarded = 0
        self.messages_initiated = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.rreq_sent = 0
        self.rreq_forwarded = 0
        self.rrep_sent = 0
        self.energy_consumed = 0
        self.nb_nodes = nb_nodes
        self.dead_nodes = 0
        self.seuiled = 0
        
        # NEW METRICS
        self.first_node_death_time = None
        self.ten_percent_death_time = None
        self.network_partition_time = None
        self.death_times = []  
        

    def add_node(self, id, pos, max_dist, reg_aodv, battery=100):
        """Marche pareil si reg_aodv ou pas"""
        new_node = Node(self.env, id, pos, battery, max_dist, self,reg_aodv)
        self.G.add_node(id, obj=new_node)

    def update_battery(self, node, msg_type, dist):
        """Marche pareil si reg_aodv ou pas"""
        cons = self.conso[0] if (msg_type == "RREQ" or msg_type == "RREP") else self.conso[1]
        energy_cost = self.coeff_conso * dist + cons
        node.battery = max(0, node.battery - energy_cost)
        self.energy_consumed += energy_cost
        
        if node.battery <= 0 and node.alive:
            self.env.process(self._kill_node(node))
        
        return node.battery > 0

    def _kill_node(self, node):
        """Enhanced to track death metrics"""
        yield self.env.timeout(0)
        
        self.G.remove_edges_from(list(self.G.edges(node.id)))
        node.alive = False
        self.dead_nodes += 1
        
        self.death_times.append(self.env.now)
        
        if self.first_node_death_time is None:
            self.first_node_death_time = self.env.now
            print(f"First node death at time {self.env.now:.2f}")
        
        if self.ten_percent_death_time is None and self.dead_nodes >= self.nb_nodes * 0.1:
            self.ten_percent_death_time = self.env.now
            print(f"10% nodes dead at time {self.env.now:.2f}")
        
        if self.network_partition_time is None:
            if self._is_network_partitioned():
                self.network_partition_time = self.env.now
                print(f"Network partitioned at time {self.env.now:.2f}")
        
        if self.dead_nodes >= self.nb_nodes / 2:
            self.stop = True

    def _is_network_partitioned(self):
        """Check if network is partitioned (disconnected)"""
        alive_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['obj'].alive]
        if len(alive_nodes) <= 1:
            return True
        
        alive_subgraph = self.G.subgraph(alive_nodes).copy()
        
        edges_to_remove = []
        for edge in alive_subgraph.edges():
            n1 = self.G.nodes[edge[0]]['obj']
            n2 = self.G.nodes[edge[1]]['obj']
            if self.get_distance(n1, n2) > n1.max_dist:
                edges_to_remove.append(edge)
        
        alive_subgraph.remove_edges_from(edges_to_remove)
        
        return not nx.is_connected(alive_subgraph) if len(alive_subgraph.nodes()) > 0 else True

    def get_distance(self, n1, n2):
        """Marche pareil si reg_aodv ou pas"""
        return ((n2.pos[0] - n1.pos[0])**2 + (n2.pos[1] - n1.pos[1])**2)**0.5

    def calculate_weight(self, n1, n2):
        if self.reg_aodv:
            return 1
        
        bat = max(n2.battery, 0.1)  
        dist = self.get_distance(n1, n2)
        
        dist_norm = dist / n1.max_dist
        bat_norm = 1 - (bat / 100000)
        
        weight = (self.coeff_dist * dist_norm) + (self.coeff_bat * bat_norm)

        if bat < self.seuil:
            self.seuiled += 1
            ecart = (self.seuil - bat) / self.seuil
            penalty = min(1.0, 0.5 * ecart)  # limite la pénalité à 1
            weight += penalty

        return weight

    def get_energy_stats(self):
        """Calculate average remaining energy and standard deviation"""
        alive_nodes = [self.G.nodes[n]['obj'] for n in self.G.nodes() if self.G.nodes[n]['obj'].alive]
        
        if not alive_nodes:
            return 0, 0
        
        remaining_energies = [node.battery for node in alive_nodes]
        avg_energy = np.mean(remaining_energies)
        std_energy = np.std(remaining_energies)
        
        return avg_energy, std_energy

    def broadcast_rreq(self, node, rreq):
        """Marche pareil si reg_aodv ou pas"""
        neighbors = list(self.G.neighbors(node.id))
        
        valid_neighbors = []
        for neighbor_id in neighbors:
            neighbor = self.G.nodes[neighbor_id]["obj"]
            if not neighbor.alive:
                continue
                
            dist = self.get_distance(node, neighbor)
            if dist <= node.max_dist:
                valid_neighbors.append((neighbor, dist))

        if valid_neighbors != []:
            max_dist = max(dist for _, dist in valid_neighbors) if not self.reg_aodv else node.max_dist
            if not self.update_battery(node, "RREQ", max_dist): return #on consomme la batterie une seule fois pour un broadcast

            for neighbor,dist in valid_neighbors:
                if not neighbor.alive:
                    continue #si il est mort on passe
                
                yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05)) #on ajoute un "jitter" aléatoire avant chaque transmission pour
                                                                                      #modéliser la réalité et éviter les problèmes de simulation : 
                                                                                      #tous les evenements sont planifiés à la même date => elle avance pas dans le temps
                new_rreq = copy.deepcopy(rreq)  #deepcopy pour avoir des objets différents sinon chaque noeud va modifier le même RREQ
                neighbor.pending.put(new_rreq)
                self.rreq_forwarded += 1

        
    def unicast_rrep(self, node, rrep):
        """Marche pareil si reg_aodv ou pas"""
        next_hop = node.routing_table.get(rrep.dest_id, (None, 0, 0, 0))[0]
        
        if next_hop is None:
            return 
            
        next_node = self.G.nodes[next_hop]["obj"]
        
        if not next_node.alive:
            return
            
        dist = self.get_distance(node, next_node)
        
        if dist <= node.max_dist:
            if self.update_battery(node, "RREP", dist): 
                yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))  #délai basé sur la distance, facteur arbitraire : 1ms / unité de distance
                next_node.pending.put(rrep)

    def forward_data(self, node, data):
        """Marche pareil si reg_aodv ou pas"""
        next_hop = node.routing_table.get(data.dest_id, (None, 0, 0, 0))[0]
        data.prev_hop = node.id
        
        if next_hop is None:
            return
            
        next_node = self.G.nodes[next_hop]["obj"]
        if not next_node.alive:
            return
            
        dist = self.get_distance(node, next_node)
        if dist <= node.max_dist:
            if self.update_battery(node, "DATA", dist):
                self.messages_forwarded += 1
                
                yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))
                next_node.pending.put(data)