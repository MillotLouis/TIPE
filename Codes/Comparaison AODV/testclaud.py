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
        self.death_times = []  # Track when each node dies
        

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
        
        # Remove edges and mark as dead
        self.G.remove_edges_from(list(self.G.edges(node.id)))
        node.alive = False
        self.dead_nodes += 1
        
        # Track death times for new metrics
        self.death_times.append(self.env.now)
        
        # First Node Death (FND)
        if self.first_node_death_time is None:
            self.first_node_death_time = self.env.now
            print(f"First node death at time {self.env.now:.2f}")
        
        # 10% Node Death
        if self.ten_percent_death_time is None and self.dead_nodes >= self.nb_nodes * 0.1:
            self.ten_percent_death_time = self.env.now
            print(f"10% nodes dead at time {self.env.now:.2f}")
        
        # Check for network partition
        if self.network_partition_time is None:
            if self._is_network_partitioned():
                self.network_partition_time = self.env.now
                print(f"Network partitioned at time {self.env.now:.2f}")
        
        # Original stopping condition
        if self.dead_nodes >= self.nb_nodes / 2:
            self.stop = True

    def _is_network_partitioned(self):
        """Check if network is partitioned (disconnected)"""
        alive_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['obj'].alive]
        if len(alive_nodes) <= 1:
            return True
        
        # Create subgraph with only alive nodes and valid connections
        alive_subgraph = self.G.subgraph(alive_nodes).copy()
        
        # Remove edges that are too long (beyond max_dist)
        edges_to_remove = []
        for edge in alive_subgraph.edges():
            n1 = self.G.nodes[edge[0]]['obj']
            n2 = self.G.nodes[edge[1]]['obj']
            if self.get_distance(n1, n2) > n1.max_dist:
                edges_to_remove.append(edge)
        
        alive_subgraph.remove_edges_from(edges_to_remove)
        
        # Check if network is connected
        return not nx.is_connected(alive_subgraph) if len(alive_subgraph.nodes()) > 0 else True

    def get_distance(self, n1, n2):
        """Marche pareil si reg_aodv ou pas"""
        return ((n2.pos[0] - n1.pos[0])**2 + (n2.pos[1] - n1.pos[1])**2)**0.5

    def calculate_weight(self, n1, n2):
        if self.reg_aodv:
            return 1
        
        bat = max(n2.battery, 0.1)  # Prevent division by zero
        dist = self.get_distance(n1, n2)
        
        # Normalized weights (0-1 range)
        dist_norm = dist / n1.max_dist
        bat_norm = 1 - (bat / 100000)  # Inverted battery (0=full, 1=empty)
        
        weight = (self.coeff_dist * dist_norm) + (self.coeff_bat * bat_norm)

        if bat < self.seuil:
            # Linear penalty instead of quadratic, with maximum cap
            self.seuiled += 1
            ecart = (self.seuil - bat) / self.seuil
            penalty = min(1.0, 0.5 * ecart)  # Cap penalty at 1.0
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

class Simulation:
    def __init__(self, nb_nodes, area_size, max_dist,conso,seuil,coeff_dist,coeff_bat,coeff_conso,ttl,reg_aodv, node_positions = None):
        #création des attributs
        self.nb_nodes = nb_nodes
        self.area_size = area_size
        self.max_dist = max_dist
        self.reg_aodv = reg_aodv
        self.energy_history = []
        self.dead_nodes_history = []
        self.time_points = []
        self.avg_energy_history = []  # NEW
        self.std_energy_history = []  # NEW

        #création du réseau
        self.net = Network(
            conso=conso,
            seuil=seuil,
            coeff_dist=coeff_dist,
            coeff_bat=coeff_bat,
            coeff_conso=coeff_conso,
            nb_nodes=nb_nodes,
            ttl=ttl,
            reg_aodv = reg_aodv
        )

        #création des noeuds
        self.node_positions = node_positions or {}
        for i in range(nb_nodes):
            if i in self.node_positions:
                pos = self.node_positions[i]
            else:
                pos = (random.uniform(0, self.area_size), random.uniform(0, self.area_size))
                self.node_positions[i] = pos
            
            self.net.add_node(i, pos, max_dist, battery=100000,reg_aodv=self.reg_aodv) #100 joules cf obsidian
        
        #création des noeuds
        self._create_links()
        

    def _create_links(self):
        for i in range(self.nb_nodes):
            for j in range(i + 1, self.nb_nodes):
                self.net.G.add_edge(i, j)
        # crée un réseau complet : toutes les connexions possibles sont crées mais pas utilisées car la distance est verifiée dans les fct de transmission
    
    def _random_communication(self):
        print("Starting random communication...")
        while not self.net.stop:
            src_id = random.randint(0, self.nb_nodes-1)
            dest_id = random.randint(0, self.nb_nodes-1)
            while dest_id == src_id:
                dest_id = random.randint(0, self.nb_nodes-1)
            #on choisit deux noeuds différents

            src_node = self.net.G.nodes[src_id]['obj']
            if src_node.alive:
                src_node.send_data(dest_id) # on lance le tranfert de données
            
            yield self.net.env.timeout(0.1) #petit délai pour pas flood

    def _monitor(self):
        while not self.net.stop:
            self.time_points.append(self.net.env.now)
            self.energy_history.append(self.net.energy_consumed)
            self.dead_nodes_history.append(self.net.dead_nodes)
            
            # NEW: Track energy statistics
            avg_energy, std_energy = self.net.get_energy_stats()
            self.avg_energy_history.append(avg_energy)
            self.std_energy_history.append(std_energy)
            
            yield self.net.env.timeout(0.25)  # ce qui donne à peu près tous les 2 messages envoyés, pas déconnant

    def get_metrics(self):
        return {
            "dead_nodes": self.net.dead_nodes,
            "energy": self.net.energy_consumed,
            "msg_recv": self.net.messages_received,
            "msg_sent": self.net.messages_sent,
            "rreq_sent": self.net.rreq_sent,
            "duration": self.net.env.now,
            "rrep_sent" : self.net.rrep_sent,
            "messages_forwarded" : self.net.messages_forwarded,
            "messages_initiated":self.net.messages_initiated,
            "rreq_forwarded":self.net.rreq_forwarded,
            "seuiled":self.net.seuiled,
            # NEW METRICS
            "first_node_death": self.net.first_node_death_time,
            "ten_percent_death": self.net.ten_percent_death_time,
            "network_partition": self.net.network_partition_time,
            "final_avg_energy": self.avg_energy_history[-1] if self.avg_energy_history else 0,
            "final_std_energy": self.std_energy_history[-1] if self.std_energy_history else 0,
        }
    
    def run(self):
        print("===== STARTING SIMULATION =====")
        self.net.env.process(self._random_communication())
        self.net.env.process(self._monitor())
        
        # while not self.net.stop:
        while self.net.env.now <= 3000:
            self.net.env.step()
        
        print("\n=== SIMULATION COMPLETE ===")
        # self.print_results()
        self.plot_results()

    def print_results(self):
        print("\n=== SIMULATION RESULTS ===")
        print(f"Durée: {self.net.env.now:.2f} unités de temps")
        print(f"Noeuds morts: {self.net.dead_nodes}/{self.nb_nodes}")
        print(f"Énergie consommée: {self.net.energy_consumed:.2f}")
        print(f"Messages envoyés: {self.net.messages_sent}")
        print(f"Messages transmis: {self.net.messages_forwarded}")
        print(f"Messages reçus: {self.net.messages_received}")
        print(f"RREQ envoyés: {self.net.rreq_sent}")
        print(f"RREQ transmis: {self.net.rreq_forwarded}")
        print(f"RREP envoyés: {self.net.rrep_sent}")
        print(f"Seuiled: {self.net.seuiled}")
        
        # NEW METRICS OUTPUT
        print(f"\n=== NEW METRICS ===")
        print(f"First Node Death (FND): {self.net.first_node_death_time:.2f}" if self.net.first_node_death_time else "First Node Death: Not reached")
        print(f"10% Node Death: {self.net.ten_percent_death_time:.2f}" if self.net.ten_percent_death_time else "10% Node Death: Not reached")
        print(f"Network Partition: {self.net.network_partition_time:.2f}" if self.net.network_partition_time else "Network Partition: Not reached")
        
        if self.avg_energy_history:
            print(f"Final Average Remaining Energy: {self.avg_energy_history[-1]:.2f}")
            print(f"Final Energy Std Deviation: {self.std_energy_history[-1]:.2f}")

    def plot_results(self):
        if not self.time_points:
            print("No data to plot")
            return
            
        plt.figure(figsize=(20, 12))
        
        # Energy and Dead Nodes
        plt.subplot(2, 3, 1)
        plt.plot(self.time_points, self.energy_history, 'b-')
        plt.xlabel('Temps')
        plt.ylabel('Énergie Consommée')
        plt.title('Consommation énergétique au cours du temps')
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.time_points, self.dead_nodes_history, 'r-')
        plt.xlabel('Temps')
        plt.ylabel('Noeuds morts')
        plt.title('Mort des noeuds au cours du temps')
        
        # Add vertical lines for key metrics
        if self.net.first_node_death_time:
            plt.axvline(x=self.net.first_node_death_time, color='orange', linestyle='--', label='FND')
        if self.net.ten_percent_death_time:
            plt.axvline(x=self.net.ten_percent_death_time, color='red', linestyle='--', label='10% Death')
        if self.net.network_partition_time:
            plt.axvline(x=self.net.network_partition_time, color='purple', linestyle='--', label='Partition')
        plt.legend()
        plt.grid(True)
        
        # NEW: Average Energy Over Time
        plt.subplot(2, 3, 3)
        plt.plot(self.time_points, self.avg_energy_history, 'g-', label='Avg Energy')
        plt.fill_between(self.time_points, 
                        [avg - std for avg, std in zip(self.avg_energy_history, self.std_energy_history)],
                        [avg + std for avg, std in zip(self.avg_energy_history, self.std_energy_history)],
                        alpha=0.3, color='g', label='±1 Std Dev')
        plt.xlabel('Temps')
        plt.ylabel('Énergie Restante')
        plt.title('Énergie moyenne restante ± écart-type')
        plt.legend()
        plt.grid(True)
        
        # Message Types
        plt.subplot(2, 3, 4)
        msg_types = ['Messages initiés','envoyés', 'transmis', 'reçus', 'RREQs envoyés', 'RREPs envoyés']
        counts = [
            self.net.messages_initiated,
            self.net.messages_sent,
            self.net.messages_forwarded,
            self.net.messages_received,
            self.net.rreq_sent,
            self.net.rrep_sent
        ]
        plt.bar(msg_types, counts, color=['cyan','blue', 'green', 'red', 'purple', 'orange'])
        plt.xlabel('Type')

def run_comparison_simulations(num_runs=10):
    """Run multiple simulations to compare regular AODV vs modified AODV"""
    
    # Simulation parameters
    params = {
        'nb_nodes': 40,
        'area_size': 800,
        'max_dist': 250,
        'conso': (1, 20),
        'seuil': 750,
        'coeff_dist': 0.6,
        'coeff_bat': 0.2,
        'coeff_conso': 0.005,
        'ttl': 100
    }
    
    # Store results for both algorithms
    regular_aodv_results = []
    modified_aodv_results = []
    
    print(f"Running {num_runs} simulations for each algorithm...")
    
    for run in range(num_runs):
        print(f"\n=== RUN {run + 1}/{num_runs} ===")
        
        # Generate base node positions for fair comparison
        base_positions = {}
        for i in range(params['nb_nodes']):
            base_positions[i] = (
                random.uniform(0, params['area_size']), 
                random.uniform(0, params['area_size'])
            )
        
        # Run Regular AODV
        print("Running Regular AODV...")
        sim_regular = Simulation(
            node_positions=base_positions,
            reg_aodv=True,
            **params
        )
        sim_regular.run()
        regular_aodv_results.append(sim_regular.get_metrics())
        
        # Run Modified AODV with same positions
        print("Running Modified AODV...")
        sim_modified = Simulation(
            node_positions=base_positions,
            reg_aodv=False,
            **params
        )
        sim_modified.run()
        modified_aodv_results.append(sim_modified.get_metrics())
    
    # Calculate and display averaged results
    print_averaged_results(regular_aodv_results, modified_aodv_results, num_runs)

def calculate_average_metrics(results):
    """Calculate average metrics from multiple simulation runs"""
    if not results:
        return {}
    
    # Get all metric keys from first result
    metric_keys = results[0].keys()
    averaged = {}
    
    for key in metric_keys:
        # Handle None values (for metrics that might not be reached)
        values = [r[key] for r in results if r[key] is not None]
        if values:
            averaged[key] = sum(values) / len(values)
            averaged[f"{key}_count"] = len(values)  # How many runs reached this metric
        else:
            averaged[key] = None
            averaged[f"{key}_count"] = 0
    
    return averaged

def print_averaged_results(regular_results, modified_results, num_runs):
    """Print comparison of averaged results"""
    
    regular_avg = calculate_average_metrics(regular_results)
    modified_avg = calculate_average_metrics(modified_results)
    
    print(f"\n" + "="*60)
    print(f"AVERAGED RESULTS OVER {num_runs} RUNS")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Regular AODV':<15} {'Modified AODV':<15} {'Improvement':<12}")
    print("-" * 70)
    
    # Main performance metrics
    metrics_to_compare = [
        ('duration', 'Duration', 'lower_better'),
        ('dead_nodes', 'Dead Nodes', 'lower_better'),
        ('energy', 'Energy Consumed', 'lower_better'),
        ('msg_recv', 'Messages Received', 'higher_better'),
        ('msg_sent', 'Messages Sent', 'context'),
        ('rreq_sent', 'RREQ Sent', 'lower_better'),
        ('rrep_sent', 'RREP Sent', 'context'),
        ('messages_forwarded', 'Messages Forwarded', 'context'),
        ('seuiled', 'Seuiled Events', 'lower_better')
    ]
    
    for metric_key, display_name, preference in metrics_to_compare:
        reg_val = regular_avg.get(metric_key, 0)
        mod_val = modified_avg.get(metric_key, 0)
        
        if reg_val != 0:
            if preference == 'lower_better':
                improvement = ((reg_val - mod_val) / reg_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            elif preference == 'higher_better':
                improvement = ((mod_val - reg_val) / reg_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:  # context
                improvement_str = "N/A"
        else:
            improvement_str = "N/A"
        
        print(f"{display_name:<25} {reg_val:<15.2f} {mod_val:<15.2f} {improvement_str:<12}")
    
    # Special metrics (may not always be reached)
    print(f"\n{'Special Metrics':<25} {'Regular AODV':<15} {'Modified AODV':<15} {'Runs Reached':<12}")
    print("-" * 70)
    
    special_metrics = [
        ('first_node_death', 'First Node Death'),
        ('ten_percent_death', '10% Node Death'),
        ('network_partition', 'Network Partition'),
        ('final_avg_energy', 'Final Avg Energy'),
        ('final_std_energy', 'Final Std Energy')
    ]
    
    for metric_key, display_name in special_metrics:
        reg_val = regular_avg.get(metric_key)
        mod_val = modified_avg.get(metric_key)
        reg_count = regular_avg.get(f"{metric_key}_count", 0)
        mod_count = modified_avg.get(f"{metric_key}_count", 0)
        
        reg_str = f"{reg_val:.2f}" if reg_val is not None else "N/A"
        mod_str = f"{mod_val:.2f}" if mod_val is not None else "N/A"
        count_str = f"{reg_count}/{mod_count}"
        
        print(f"{display_name:<25} {reg_str:<15} {mod_str:<15} {count_str:<12}")
    
    # Calculate and display delivery ratio
    reg_delivery_ratio = (regular_avg['msg_recv'] / regular_avg['messages_initiated']) * 100 if regular_avg['messages_initiated'] > 0 else 0
    mod_delivery_ratio = (modified_avg['msg_recv'] / modified_avg['messages_initiated']) * 100 if modified_avg['messages_initiated'] > 0 else 0
    
    print(f"\n{'Delivery Ratio':<25} {reg_delivery_ratio:<15.1f}% {mod_delivery_ratio:<15.1f}% {'':<12}")
    
    print("\n" + "="*60)

def calculate_optimal_max_dist(nb_nodes, area_size, connectivity_factor=1.2):
    """
    Calculate optimal max_dist based on network density to ensure connectivity
    connectivity_factor > 1 ensures some redundancy in connections
    """
    # Area per node
    area_per_node = (area_size ** 2) / nb_nodes
    
    # Radius for circular area per node
    radius_per_node = (area_per_node / 3.14159) ** 0.5
    
    # Max distance should be larger to ensure connectivity
    optimal_max_dist = radius_per_node * connectivity_factor * 2
    
    # Ensure it's not larger than area_size (unrealistic)
    return min(optimal_max_dist, area_size * 0.7)

def run_density_analysis(density_configs=None, num_runs=3):
    """
    Run simulations across different network densities
    density_configs: list of (nb_nodes, max_dist) tuples, if None uses calculated values
    """
    
    if density_configs is None:
        # Define density configurations: (nb_nodes, area_size)
        area_size = 800
        node_counts = [10, 15, 20, 25, 30, 35, 40]
        
        density_configs = []
        for nb_nodes in node_counts:
            max_dist = calculate_optimal_max_dist(nb_nodes, area_size)
            density_configs.append((nb_nodes, max_dist))
    
    # Base simulation parameters
    base_params = {
        'area_size': 800,
        'conso': (1, 20),
        'seuil': 750,
        'coeff_dist': 0.6,
        'coeff_bat': 0.2,
        'coeff_conso': 0.005,
        'ttl': 100
    }
    
    # Store results for analysis
    density_results = {
        'configs': [],
        'regular_aodv': [],
        'modified_aodv': []
    }
    
    print(f"Running density analysis with {num_runs} runs per configuration...")
    print(f"{'Nodes':<8} {'Max Dist':<12} {'Density':<12} {'Avg Degree':<12}")
    print("-" * 50)
    
    for nb_nodes, max_dist in density_configs:
        # Calculate network density metrics
        network_density = nb_nodes / (base_params['area_size'] ** 2) * 1000000  # nodes per km²
        avg_theoretical_degree = (nb_nodes - 1) * (3.14159 * (max_dist ** 2)) / (base_params['area_size'] ** 2)
        
        print(f"{nb_nodes:<8} {max_dist:<12.0f} {network_density:<12.2f} {avg_theoretical_degree:<12.1f}")
        
        # Store configuration
        config_info = {
            'nb_nodes': nb_nodes,
            'max_dist': max_dist,
            'density': network_density,
            'avg_degree': avg_theoretical_degree
        }
        density_results['configs'].append(config_info)
        
        # Run simulations for this density
        regular_results = []
        modified_results = []
        
        for run in range(num_runs):
            print(f"  Running density {nb_nodes} nodes, run {run + 1}/{num_runs}...")
            
            # Generate base positions
            base_positions = {}
            for i in range(nb_nodes):
                base_positions[i] = (
                    random.uniform(0, base_params['area_size']), 
                    random.uniform(0, base_params['area_size'])
                )
            
            # Simulation parameters for this density
            sim_params = {
                **base_params,
                'nb_nodes': nb_nodes,
                'max_dist': max_dist,
                'node_positions': base_positions
            }
            
            # Regular AODV
            sim_regular = Simulation(reg_aodv=True, **sim_params)
            sim_regular.run()
            regular_results.append(sim_regular.get_metrics())
            
            # Modified AODV
            sim_modified = Simulation(reg_aodv=False, **sim_params)
            sim_modified.run()
            modified_results.append(sim_modified.get_metrics())
        
        # Calculate averages for this density
        regular_avg = calculate_average_metrics(regular_results)
        modified_avg = calculate_average_metrics(modified_results)
        
        density_results['regular_aodv'].append(regular_avg)
        density_results['modified_aodv'].append(modified_avg)
    
    # Print density analysis results
    print_density_analysis(density_results)
    
    return density_results

def print_density_analysis(results):
    """Print comprehensive density analysis results"""
    
    print(f"\n" + "="*80)
    print("DENSITY ANALYSIS RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Nodes':<6} {'Density':<8} {'Regular AODV':<45} {'Modified AODV':<45}")
    print(f"{'Count':<6} {'(n/km²)':<8} {'Duration':<8} {'Dead':<6} {'Energy':<8} {'Del%':<8} {'RREQ':<8} {'Duration':<8} {'Dead':<6} {'Energy':<8} {'Del%':<8} {'RREQ':<8}")
    print("-" * 110)
    
    for i, config in enumerate(results['configs']):
        reg = results['regular_aodv'][i]
        mod = results['modified_aodv'][i]
        
        # Calculate delivery ratios
        reg_delivery = (reg['msg_recv'] / reg['messages_initiated']) * 100 if reg['messages_initiated'] > 0 else 0
        mod_delivery = (mod['msg_recv'] / mod['messages_initiated']) * 100 if mod['messages_initiated'] > 0 else 0
        
        print(f"{config['nb_nodes']:<6} {config['density']:<8.1f} "
              f"{reg['duration']:<8.1f} {reg['dead_nodes']:<6.0f} {reg['energy']:<8.0f} {reg_delivery:<8.1f} {reg['rreq_sent']:<8.0f} "
              f"{mod['duration']:<8.1f} {mod['dead_nodes']:<6.0f} {mod['energy']:<8.0f} {mod_delivery:<8.1f} {mod['rreq_sent']:<8.0f}")
    
    # Improvement analysis
    print(f"\n" + "="*80)
    print("IMPROVEMENT ANALYSIS (Modified vs Regular)")
    print("="*80)
    print(f"{'Nodes':<6} {'Duration':<10} {'Energy':<10} {'Dead Nodes':<12} {'Delivery':<10} {'RREQ Sent':<12}")
    print("-" * 65)
    
    for i, config in enumerate(results['configs']):
        reg = results['regular_aodv'][i]
        mod = results['modified_aodv'][i]
        
        # Calculate improvements (positive = better for modified)
        duration_imp = ((reg['duration'] - mod['duration']) / reg['duration']) * 100 if reg['duration'] > 0 else 0
        energy_imp = ((reg['energy'] - mod['energy']) / reg['energy']) * 100 if reg['energy'] > 0 else 0
        dead_imp = ((reg['dead_nodes'] - mod['dead_nodes']) / max(reg['dead_nodes'], 1)) * 100
        
        reg_delivery = (reg['msg_recv'] / reg['messages_initiated']) * 100 if reg['messages_initiated'] > 0 else 0
        mod_delivery = (mod['msg_recv'] / mod['messages_initiated']) * 100 if mod['messages_initiated'] > 0 else 0
        delivery_imp = mod_delivery - reg_delivery
        
        rreq_imp = ((reg['rreq_sent'] - mod['rreq_sent']) / reg['rreq_sent']) * 100 if reg['rreq_sent'] > 0 else 0
        
        print(f"{config['nb_nodes']:<6} {duration_imp:<+9.1f}% {energy_imp:<+9.1f}% {dead_imp:<+11.1f}% {delivery_imp:<+9.1f}% {rreq_imp:<+11.1f}%")
    
    # Network lifetime metrics
    print(f"\n" + "="*80)
    print("NETWORK LIFETIME METRICS")
    print("="*80)
    print(f"{'Nodes':<6} {'Regular AODV':<35} {'Modified AODV':<35}")
    print(f"{'Count':<6} {'FND':<8} {'10%Death':<10} {'Partition':<10} {'FND':<8} {'10%Death':<10} {'Partition':<10}")
    print("-" * 80)
    
    for i, config in enumerate(results['configs']):
        reg = results['regular_aodv'][i]
        mod = results['modified_aodv'][i]
        
        reg_fnd = f"{reg['first_node_death']:.1f}" if reg['first_node_death'] else "N/A"
        reg_10p = f"{reg['ten_percent_death']:.1f}" if reg['ten_percent_death'] else "N/A"
        reg_part = f"{reg['network_partition']:.1f}" if reg['network_partition'] else "N/A"
        
        mod_fnd = f"{mod['first_node_death']:.1f}" if mod['first_node_death'] else "N/A"
        mod_10p = f"{mod['ten_percent_death']:.1f}" if mod['ten_percent_death'] else "N/A"
        mod_part = f"{mod['network_partition']:.1f}" if mod['network_partition'] else "N/A"
        
        print(f"{config['nb_nodes']:<6} {reg_fnd:<8} {reg_10p:<10} {reg_part:<10} {mod_fnd:<8} {mod_10p:<10} {mod_part:<10}")


def plot_node_scaling_analysis(node_counts, area_size=800, max_dist=400, nb_runs=3):
    """
    Plots performance improvements of modified AODV vs regular AODV as node count varies.
    
    Parameters:
    - node_counts: list of node counts to test
    - area_size: fixed area size for all simulations
    - max_dist: fixed maximum transmission distance
    - nb_runs: number of runs per configuration for averaging
    """
    
    # Add safe run method to Simulation class
    add_safe_run_method()
    
    # Storage for results
    dead_nodes_improvement_list = []
    first_death_improvement_list = []
    ten_percent_death_improvement_list = []
    delivery_rate_improvement_list = []
    
    # Fixed parameters
    base_params = {
        'area_size': area_size,
        'max_dist': max_dist,
        'conso': (1, 20),
        'seuil': 750,
        'coeff_dist': 0.6,
        'coeff_bat': 0.2,
        'coeff_conso': 0.005,
        'ttl': 100
    }
    
    print("Starting node scaling analysis...")
    
    for nb_nodes in node_counts:
        print(f"\nTesting with {nb_nodes} nodes...")
        
        # Storage for this node count
        mod_results = []
        reg_results = []
        
        for run in range(nb_runs):
            print(f"  Run {run + 1}/{nb_runs}")
            
            # Create base simulation to get consistent node positions
            base_sim = Simulation(nb_nodes=nb_nodes, **base_params, reg_aodv=False)
            node_positions = base_sim.node_positions.copy()
            
            # Run modified AODV
            try:
                mod_sim = Simulation(
                    nb_nodes=nb_nodes,
                    node_positions=node_positions,
                    reg_aodv=False,
                    **base_params
                )
                mod_sim.run_safe()
                mod_metrics = mod_sim.get_metrics()
                
                # Calculate additional metrics for modified AODV
                mod_first_death = get_first_death_time(mod_sim)
                mod_ten_percent_death = get_ten_percent_death_time(mod_sim, nb_nodes)
                mod_delivery_rate = mod_metrics['msg_recv'] / max(mod_metrics['messages_initiated'], 1)
                
                mod_results.append({
                    'dead_nodes': mod_metrics['dead_nodes'],
                    'first_death': mod_first_death,
                    'ten_percent_death': mod_ten_percent_death,
                    'delivery_rate': mod_delivery_rate
                })
            except Exception as e:
                print(f"    Error in modified AODV run {run}: {e}")
                # Use default values if simulation fails
                mod_results.append({
                    'dead_nodes': nb_nodes,  # Assume all nodes died
                    'first_death': 0,
                    'ten_percent_death': 0,
                    'delivery_rate': 0
                })
            
            # Run regular AODV
            try:
                reg_sim = Simulation(
                    nb_nodes=nb_nodes,
                    node_positions=node_positions,
                    reg_aodv=True,
                    **base_params
                )
                reg_sim.run_safe()
                reg_metrics = reg_sim.get_metrics()
                
                # Calculate additional metrics for regular AODV
                reg_first_death = get_first_death_time(reg_sim)
                reg_ten_percent_death = get_ten_percent_death_time(reg_sim, nb_nodes)
                reg_delivery_rate = reg_metrics['msg_recv'] / max(reg_metrics['messages_initiated'], 1)
                
                reg_results.append({
                    'dead_nodes': reg_metrics['dead_nodes'],
                    'first_death': reg_first_death,
                    'ten_percent_death': reg_ten_percent_death,
                    'delivery_rate': reg_delivery_rate
                })
            except Exception as e:
                print(f"    Error in regular AODV run {run}: {e}")
                # Use default values if simulation fails
                reg_results.append({
                    'dead_nodes': nb_nodes,  # Assume all nodes died
                    'first_death': 0,
                    'ten_percent_death': 0,
                    'delivery_rate': 0
                })
        
        # Calculate averages
        mod_avg = {metric: np.mean([r[metric] for r in mod_results]) for metric in mod_results[0].keys()}
        reg_avg = {metric: np.mean([r[metric] for r in reg_results]) for metric in reg_results[0].keys()}
        
        # Calculate improvements (positive = better for modified AODV)
        dead_improvement = (reg_avg['dead_nodes'] - mod_avg['dead_nodes']) / max(reg_avg['dead_nodes'], 1) * 100
        first_death_improvement = (mod_avg['first_death'] - reg_avg['first_death']) / max(reg_avg['first_death'], 1) * 100
        ten_percent_improvement = (mod_avg['ten_percent_death'] - reg_avg['ten_percent_death']) / max(reg_avg['ten_percent_death'], 1) * 100
        delivery_improvement = (mod_avg['delivery_rate'] - reg_avg['delivery_rate']) * 100  # Percentage points
        
        dead_nodes_improvement_list.append(dead_improvement)
        first_death_improvement_list.append(first_death_improvement)
        ten_percent_death_improvement_list.append(ten_percent_improvement)
        delivery_rate_improvement_list.append(delivery_improvement)
        
        print(f"  Results: Dead nodes: {dead_improvement:.1f}%, "
              f"First death: {first_death_improvement:.1f}%, "
              f"10% death: {ten_percent_improvement:.1f}%, "
              f"Delivery: {delivery_improvement:.1f}pp")
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Dead nodes improvement
    plt.subplot(2, 2, 1)
    plt.plot(node_counts, dead_nodes_improvement_list, 'ro-', linewidth=2, markersize=6)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Dead Nodes Improvement (%)')
    plt.title('Reduction in Dead Nodes vs Node Count')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 2: First death time improvement
    plt.subplot(2, 2, 2)
    plt.plot(node_counts, first_death_improvement_list, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Number of Nodes')
    plt.ylabel('First Death Time Improvement (%)')
    plt.title('Delay in First Node Death vs Node Count')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 3: 10% death time improvement
    plt.subplot(2, 2, 3)
    plt.plot(node_counts, ten_percent_death_improvement_list, 'go-', linewidth=2, markersize=6)
    plt.xlabel('Number of Nodes')
    plt.ylabel('10% Death Time Improvement (%)')
    plt.title('Delay in 10% Node Death vs Node Count')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 4: Delivery rate improvement
    plt.subplot(2, 2, 4)
    plt.plot(node_counts, delivery_rate_improvement_list, 'mo-', linewidth=2, markersize=6)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Delivery Rate Improvement (pp)')
    plt.title('Message Delivery Rate vs Node Count')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('aodv_node_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n=== SCALING ANALYSIS SUMMARY ===")
    print(f"Node counts tested: {node_counts}")
    print(f"Average improvements:")
    print(f"  Dead nodes reduction: {np.mean(dead_nodes_improvement_list):.1f}%")
    print(f"  First death delay: {np.mean(first_death_improvement_list):.1f}%")
    print(f"  10% death delay: {np.mean(ten_percent_death_improvement_list):.1f}%")
    print(f"  Delivery rate: {np.mean(delivery_rate_improvement_list):.1f} percentage points")


def get_first_death_time(simulation):
    """Extract the time of first node death from simulation history"""
    if not simulation.dead_nodes_history or not simulation.time_points:
        return simulation.net.env.now
    
    for i, dead_count in enumerate(simulation.dead_nodes_history):
        if dead_count > 0:
            return simulation.time_points[i]
    return simulation.net.env.now  # No deaths occurred


def get_ten_percent_death_time(simulation, total_nodes):
    """Extract the time when 10% of nodes died"""
    if not simulation.dead_nodes_history or not simulation.time_points:
        return simulation.net.env.now
        
    threshold = max(1, int(0.1 * total_nodes))
    for i, dead_count in enumerate(simulation.dead_nodes_history):
        if dead_count >= threshold:
            return simulation.time_points[i]
    return simulation.net.env.now  # Threshold not reached


def add_safe_run_method():
    """Add a safe run method to the Simulation class"""
    def run_safe(self):
        print("===== STARTING SIMULATION =====")
        self.net.env.process(self._random_communication())
        self.net.env.process(self._monitor())
        
        max_time = 300
        try:
            while not self.net.stop and self.net.env.now < max_time:
                if not self.net.env._queue:  # Check if event queue is empty
                    print(f"No more events at time {self.net.env.now:.2f}")
                    break
                self.net.env.step()
        except Exception as e:
            print(f"Simulation stopped due to: {e} at time {self.net.env.now:.2f}")
        
        print(f"Simulation ended at time {self.net.env.now:.2f}")
    
    Simulation.run_safe = run_safe


if __name__ == "__main__":
    print("Starting AODV Density Analysis Study")
    
    # You can choose which analysis to run:
    
    # Option 1: Basic comparison with fixed parameters
    # run_comparison_simulations(num_runs=3)
    
    # Option 2: Density analysis (recommended)
    # run_density_analysis(num_runs=3)
    
    # Option 3: Custom density configurations
    # custom_configs = [(25, 250), (30, 200), (40, 150), (50, 100)]
    # run_density_analysis(density_configs=custom_configs, num_runs=3)

    node_counts = [10, 15, 20, 25, 30, 35, 40]
    plot_node_scaling_analysis(node_counts, nb_runs=2)  # Use 2 runs for faster testing