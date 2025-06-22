import random
import matplotlib.pyplot as plt
from collections import defaultdict
import simpy
import copy
import networkx as nx

class Message:
    def __init__(self, typ, src_id, src_seq, dest_seq, dest_id, weight, prev_hop):
        self.type = typ
        self.src_id = src_id
        self.src_seq = src_seq
        self.dest_seq = dest_seq
        self.dest_id = dest_id
        self.weight = weight
        self.prev_hop = prev_hop
        
    def __repr__(self):
        return (f"Message({self.type}, src={self.src_id}, dest={self.dest_id}, "
                f"prev_hop={self.prev_hop}, weight={self.weight:.2f})")

class Node:
    def __init__(self, env, id, pos, initial_battery, max_dist, network):
        self.env = env
        self.id = id
        self.pos = pos
        self.battery = initial_battery
        self.routing_table = {} # dest : {next_hop,seq_num,weight,expiry}
        self.seq_num = 0
        self.pending = simpy.Store(env)
        self.max_dist = max_dist
        self.alive = True
        self.network = network
        self.seen = set()
        self.pending_rreqs = {}
        self.to_be_sent = defaultdict(list)
        self.env.process(self.process_messages())

    def process_messages(self):
        while self.alive:
            msg = yield self.pending.get() #On bloque le process jusqu'à avoir un nouveau message

            processing_delay = random.uniform(0.001, 0.005)
            yield self.env.timeout(processing_delay) # on bloque pour simuler un délai de processing
            
            # print(f"[{self.env.now:.4f}] Node {self.id} received: {msg}")
            
            if msg.type == "RREQ":
                self.handle_rreq(msg)
            elif msg.type == "RREP":
                self.handle_rrep(msg)
            elif msg.type == "DATA":
                self.handle_data(msg)

    def init_rreq(self, dest_id):
        # print(f"[{self.env.now:.4f}] Node {self.id} INIT_RREQ to {dest_id}")
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
        # print(f"[{self.env.now:.4f}] Node {self.id} HANDLE_RREQ from {rreq.src_id} via {rreq.prev_hop}")
        
        seen_key = (rreq.src_id, rreq.src_seq, rreq.prev_hop)
        if seen_key in self.seen or rreq.src_id == self.id:
            return #on discard si on a déjà vu : évite les **boucles** ♥
                   #éviter que les RREQs soient renvoyés à la source       
        
        prev_node = self.network.G.nodes[rreq.prev_hop]["obj"]
        weight = self.network.calculate_weight(prev_node, self) #inclus la pénalité si batterie en dessous du seuil
        rreq.weight += weight
        
        self.update_route(
            dest=rreq.src_id,
            next_hop=rreq.prev_hop,
            seq_num=rreq.src_seq,
            weight=rreq.weight
        )
        
        if self.id == rreq.dest_id: #Si on est la destination du RREQ
            key = (rreq.src_id, rreq.src_seq)
            if key not in self.pending_rreqs:
                self.pending_rreqs[key] = []
                self.env.process(self.collect_rreps(key)) # on commence la collecte des RREPs
            self.pending_rreqs[key].append(rreq)
            return
        
        self.seen.add(seen_key)
        rreq.prev_hop = self.id
        self.env.process(self.network.broadcast_rreq(self, rreq))


    def handle_rrep(self, rrep):
        prev_node = self.network.G.nodes[rrep.prev_hop]["obj"]
        weight_add = self.network.calculate_weight(prev_node, self) #inclus la penalité si en dessous du seuil
        rrep.weight += weight_add
        
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
        
        if (seq_num > current[1]) or (seq_num == current[1] and weight < current[2]): #si la route est plus fraiche ou aussi fraiche avec un poids moindre
            self.routing_table[dest] = (next_hop, seq_num, weight, self.env.now + self.network.ttl)

    def collect_rreps(self, key):
        yield self.env.timeout(1)
        # On attend pour que tous les RREQs arrivent à la dest et soient stockés dans self.pending_rreqs[key]
        if key in self.pending_rreqs:
            rreqs = self.pending_rreqs.pop(key)
            best_rreq = min(rreqs, key=lambda r: r.weight)
            self.send_rrep(best_rreq) #on envoie le meilleur

    def handle_data(self, data):
        if data.dest_id == self.id:
            self.network.messages_received += 1
        else:
            self.env.process(self.network.forward_data(self, data))

    def send_data(self, dest_id):
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
    def __init__(self, conso, seuil, coeff_dist, coeff_bat, coeff_conso, nb_nodes, ttl):
        self.env = simpy.Environment()
        self.G = nx.Graph()
        self.conso = conso
        self.seuil = seuil
        self.coeff_dist = coeff_dist
        self.coeff_bat = coeff_bat
        self.coeff_conso = coeff_conso
        self.ttl = ttl
        self.stop = False
        
        self.messages_forwarded = 0
        self.messages_initiated = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.rreq_sent = 0
        self.rrep_sent = 0
        self.energy_consumed = 0
        self.nb_nodes = nb_nodes
        self.dead_nodes = 0
        self.seuiled = 0
        

    def add_node(self, id, pos, max_dist, battery=100):
        new_node = Node(self.env, id, pos, battery, max_dist, self)
        self.G.add_node(id, obj=new_node)

    def update_battery(self, node, msg_type, dist):
        cons = self.conso[0] if msg_type == "RREQ" else self.conso[1]
        energy_cost = self.coeff_conso * dist + cons
        node.battery = max(0, node.battery - energy_cost)
        self.energy_consumed += energy_cost
        
        if node.battery <= 0 and node.alive:
            self.env.process(self._kill_node(node))
        
        return node.battery > 0

    def _kill_node(self, node):
        yield self.env.timeout(0) #attend la fin du step de simulation pour pas supprimer un noeud quand on est en train de parcourir une liste le contenant
        self.G.remove_edges_from(list(self.G.edges(node.id)))
        node.alive = False
        self.dead_nodes += 1
        if self.dead_nodes >= self.nb_nodes / 2:
            self.stop = True #la moitié des noeuds sont morts on inqique qu'il faut arrêter la simulation au prochain check

    def get_distance(self, n1, n2):
        return ((n2.pos[0] - n1.pos[0])**2 + (n2.pos[1] - n1.pos[1])**2)**0.5

    def calculate_weight(self, n1, n2):
        bat = n2.battery if n2.battery > 0 else 0.001 # éviter division par 0
        dist = self.get_distance(n1, n2)
        weight = self.coeff_dist * dist + self.coeff_bat * (1 / bat)

        if n2.battery < self.seuil:
            self.seuiled += 1
            ecart = self.seuil - n2.battery
            penalite = 1000 * (1 + ecart/self.seuil) #augmentation exponentielle de la penalite en dessous du seuil
            weight += penalite

        return weight

    def broadcast_rreq(self, node, rreq):
        neighbors = list(self.G.neighbors(node.id))
        
        for neighbor_id in neighbors:
            neighbor = self.G.nodes[neighbor_id]["obj"]
            if not neighbor.alive:
                continue #si il est mort on passe
                
            dist = self.get_distance(node, neighbor)
            if dist <= node.max_dist:
                if self.update_battery(node, "RREQ", dist): #consomation + vérif de la batterie
                    yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05)) #on ajoute un "jitter" aléatoire avant chaque transmission pour
                                                                                      #modéliser la réalité et éviter les problèmes de simulation : 
                                                                                      #tous les evenements sont planifiés à la même date => elle avance pas dans le temps
                    new_rreq = copy.deepcopy(rreq)  #deepcopy pour avoir des objets différents sinon chaque noeud va modifier le même RREQ
                    neighbor.pending.put(new_rreq)

        
    def unicast_rrep(self, node, rrep):
        next_hop = node.routing_table.get(rrep.dest_id, (None, 0, 0, 0))[0]
        
        if next_hop is None:
            return 
            
        next_node = self.G.nodes[next_hop]["obj"]
        
        if not next_node.alive:
            return
            
        dist = self.get_distance(node, next_node)
        
        if dist <= node.max_dist:
            if self.update_battery(node, "RREQ", dist): 
                yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))  #délai basé sur la distance, facteur arbitraire : 1ms / unité de distance
                next_node.pending.put(rrep)

    def forward_data(self, node, data):
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
    def __init__(self, nb_nodes, area_size, max_dist, duration,conso,seuil,coeff_dist,coeff_bat,coeff_conso,ttl):
        self.nb_nodes = nb_nodes
        self.area_size = area_size
        self.max_dist = max_dist
        self.duration = duration
        
        self.net = Network(
            conso=conso,
            seuil=seuil,
            coeff_dist=coeff_dist,
            coeff_bat=coeff_bat,
            coeff_conso=coeff_conso,
            nb_nodes=nb_nodes,
            ttl=ttl
        )

        self.node_positions = {}
        for i in range(nb_nodes):
            pos = (random.uniform(0, area_size), random.uniform(0, area_size))
            self.node_positions[i] = pos
            self.net.add_node(i, pos, max_dist, battery=random.uniform(90,100))  
        
        self._create_links()
        
        self.energy_history = []
        self.dead_nodes_history = []
        self.messages_history = defaultdict(list)
        self.time_points = []
        print("===== SIMULATION READY =====")

    def _create_links(self):
        print("Creating random links...")
        nodes = list(self.net.G.nodes(data='obj'))
        node_ids = [n[0] for n in nodes]
        nb_nodes = len(node_ids)   
        min_degree = 2

        # Start with no edges
        self.net.G.remove_edges_from(list(self.net.G.edges()))

        # Ensure each node has at least min_degree neighbors
        for n1_id, n1 in nodes:
            possible_neighbors = [nid for nid in node_ids if nid != n1_id]
            random.shuffle(possible_neighbors)
            neighbors = set(self.net.G.neighbors(n1_id))
            needed = max(0, min_degree - len(neighbors))
            added = 0
            for n2_id in possible_neighbors:
                if n2_id not in neighbors and n2_id != n1_id:
                    self.net.G.add_edge(n1_id, n2_id)
                    neighbors.add(n2_id)
                    added += 1
                    if len(neighbors) >= min_degree:
                        break
            if added < needed:
                print(f"  Warning: Node {n1_id} could not reach min degree {min_degree}")

        # Optionally, add more random edges for extra connectivity
        extra_edges = random.randint(nb_nodes, nb_nodes * 2)
        for _ in range(extra_edges):
            n1_id, n2_id = random.sample(node_ids, 2)
            if not self.net.G.has_edge(n1_id, n2_id):
                self.net.G.add_edge(n1_id, n2_id)

        print(f"Created {self.net.G.number_of_edges()} links")

    def _random_communication(self):
        print("Starting random communication...")
        while self.net.env.now < self.duration and not self.net.stop:
            src_id = random.randint(0, self.nb_nodes-1)
            dest_id = random.randint(0, self.nb_nodes-1)
            while dest_id == src_id:
                dest_id = random.randint(0, self.nb_nodes-1)
                
            src_node = self.net.G.nodes[src_id]['obj']
            if src_node.alive and src_node.battery > self.net.seuil:
                print(f"[{self.net.env.now:.4f}] RANDOM COMM: {src_id} -> {dest_id}")
                src_node.send_data(dest_id)
            else:
                print(f"[{self.net.env.now:.4f}] Skip comm: node {src_id} battery {src_node.battery:.2f}")
            
            delay = random.expovariate(0.5)  # More frequent for debugging
            yield self.net.env.timeout(delay)

    def _monitor(self):
        print("Starting monitor...")
        while not self.net.stop and self.net.env.now < self.duration:
            self.time_points.append(self.net.env.now)
            self.energy_history.append(self.net.energy_consumed)
            self.dead_nodes_history.append(self.net.dead_nodes)
            self.messages_history['sent'].append(self.net.messages_sent)
            self.messages_history['forwarded'].append(self.net.messages_forwarded)
            self.messages_history['received'].append(self.net.messages_received)
            self.messages_history['rreq'].append(self.net.rreq_sent)
            self.messages_history['rrep'].append(self.net.rrep_sent)
            self.messages_history['initiated'].append(self.net.messages_initiated)
            
            print(f"[{self.net.env.now:.2f}] MONITOR: "
                    f"Energy={self.net.energy_consumed:.2f}, "
                    f"Dead={self.net.dead_nodes}, "
                    f"Sent={self.net.messages_sent}, "
                    f"RREQ={self.net.rreq_sent}")
            
            yield self.net.env.timeout(1.0)  # Update every 1 time unit

    def run(self):
        print("===== STARTING SIMULATION =====")
        self.net.env.process(self._random_communication())
        self.net.env.process(self._monitor())
        
        while not self.net.stop and self.net.env.now < self.duration:
            self.net.env.step()
            # print(f"\n--- STEP {self.net.env.now:.4f} ---")
        
        print("\n=== SIMULATION COMPLETE ===")
        self.print_results()
        self.plot_results()

    def print_results(self):
        print("\n=== SIMULATION RESULTS ===")
        print(f"Duration: {self.net.env.now:.2f} time units")
        print(f"Dead nodes: {self.net.dead_nodes}/{self.nb_nodes}")
        print(f"Energy consumed: {self.net.energy_consumed:.2f}")
        print(f"Messages sent: {self.net.messages_sent}")
        print(f"Messages forwarded: {self.net.messages_forwarded}")
        print(f"Messages received: {self.net.messages_received}")
        print(f"RREQ sent: {self.net.rreq_sent}")
        print(f"RREP sent: {self.net.rrep_sent}")
        print(f"Seuiled: {self.net.seuiled}")
        
        # Print final node status
        print("\nNode Status:")
        for i in range(self.nb_nodes):
            node = self.net.G.nodes[i]['obj']
            status = "ALIVE" if node.alive else "DEAD"
            print(f"Node {i}: {status}, Battery: {node.battery:.2f}, Position: {node.pos}")

    def plot_results(self):
        if not self.time_points:
            print("No data to plot")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Energy and Dead Nodes
        plt.subplot(2, 2, 1)
        plt.plot(self.time_points, self.energy_history, 'b-', label='Energy Consumed')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy Consumption Over Time')
        plt.grid(True)
        
        ax2 = plt.gca().twinx()
        ax2.plot(self.time_points, self.dead_nodes_history, 'r-', label='Dead Nodes')
        ax2.set_ylabel('Dead Nodes')
        plt.title('Energy and Node Mortality')
        plt.legend()
        
        # Message Types
        plt.subplot(2, 2, 2)
        msg_types = ['initiated','sent', 'forwarded', 'received', 'rreq', 'rrep']
        counts = [
            self.net.messages_initiated,
            self.net.messages_sent,
            self.net.messages_forwarded,
            self.net.messages_received,
            self.net.rreq_sent,
            self.net.rrep_sent
        ]
        plt.bar(msg_types, counts, color=['cyan','blue', 'green', 'red', 'purple', 'orange'])
        plt.xlabel('Message Type')
        plt.ylabel('Count')
        plt.title('Message Statistics')
        
        # Network Topology
        plt.subplot(2, 2, 3)
        for i, pos in self.node_positions.items():
            node_obj = self.net.G.nodes[i]['obj']
            color = 'green' if node_obj.alive else 'red'
            plt.plot(pos[0], pos[1], marker='o', markersize=10, color=color)
            plt.text(pos[0], pos[1], str(i), fontsize=9, ha='center', va='center')
        
        # Only plot links if nodes are within max_dist
        for edge in self.net.G.edges():
            n1 = self.net.G.nodes[edge[0]]['obj']
            n2 = self.net.G.nodes[edge[1]]['obj']
            dist = self.net.get_distance(n1, n2)
            if n1.alive and n2.alive and dist <= self.max_dist:
                plt.plot([n1.pos[0], n2.pos[0]], [n1.pos[1], n2.pos[1]], 'b-', alpha=0.3)
        
        plt.xlim(0, self.area_size)
        plt.ylim(0, self.area_size)
        plt.title('Network Topology (Green=Alive, Red=Dead)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        
        # Message Flow Over Time
        plt.subplot(2, 2, 4)
        for msg_type, color in zip(['sent', 'received', 'rreq', 'rrep'], 
                                ['blue', 'green', 'red', 'purple']):
            if msg_type in self.messages_history:
                plt.plot(self.time_points, self.messages_history[msg_type], color, label=msg_type)
        plt.xlabel('Time')
        plt.ylabel('Message Count')
        plt.title('Message Flow Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('aodv_debug_results.png')
        plt.show()
        print("Saved results to aodv_debug_results.png")

if __name__ == "__main__":
    print("===== STARTING DEBUG SIMULATION =====")
    sim = Simulation(
        nb_nodes=50,    # Small network for debugging
        area_size=100,
        max_dist=50,
        duration=100
    )
    sim.run()
    
#https://chat.deepseek.com/a/chat/s/e9f44a34-4df3-4d4d-b3d3-07ec7f5eb11e