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
        self.seen = {} # (rreq.src_id, rreq.src_seq, rreq.prev_hop) : meilleur poids
        self.pending_rreqs = {}
        self.to_be_sent = defaultdict(list)
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
        
        if rreq.src_id == self.id:
            return #on discard si on a déjà vu : évite les **boucles** ♥
                   #éviter que les RREQs soient renvoyés à la source       
        
        seen_key = (rreq.src_id, rreq.src_seq, rreq.prev_hop)
        if seen_key in self.seen:
            if rreq.weight >= self.seen[seen_key]:
                return
            else:
                self.seen[seen_key] = rreq.weight

        else:
            self.seen[seen_key] = rreq.weight

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
        yield self.env.timeout(2)
        # On attend pour que tous les RREQs arrivent à la dest et soient stockés dans self.pending_rreqs[key]
        if key in self.pending_rreqs:
            rreqs = self.pending_rreqs.get(key)
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
        self.rreq_forwarded = 0
        self.rrep_sent = 0
        self.energy_consumed = 0
        self.nb_nodes = nb_nodes
        self.dead_nodes = 0
        self.seuiled = 0
        

    def add_node(self, id, pos, max_dist, battery=100):
        new_node = Node(self.env, id, pos, battery, max_dist, self)
        self.G.add_node(id, obj=new_node)

    def update_battery(self, node, msg_type, dist):
        cons = self.conso[0] if (msg_type == "RREQ" or msg_type == "RREP") else self.conso[1]
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
        if self.dead_nodes >= self.nb_nodes / 4:
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
        
        valid_neighbors = []
        for neighbor_id in neighbors:
            neighbor = self.G.nodes[neighbor_id]["obj"]
            if not neighbor.alive:
                continue
                
            dist = self.get_distance(node, neighbor)
            if dist <= node.max_dist:
                valid_neighbors.append((neighbor, dist))

        if valid_neighbors != []:
            max_dist = max(dist for _, dist in valid_neighbors)
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
    def __init__(self, nb_nodes, area_size, max_dist,conso,seuil,coeff_dist,coeff_bat,coeff_conso,ttl):
        self.nb_nodes = nb_nodes
        self.area_size = area_size
        self.max_dist = max_dist
        
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
            self.net.add_node(i, pos, max_dist, battery=10000) #100 joules cf
        
        self._create_links()
        
        self.energy_history = []
        self.dead_nodes_history = []
        self.time_points = []

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
                src_node.send_data(dest_id) # on lance le tranfer de données
            
            yield self.net.env.timeout(0.1) #petit délai pour pas flood

    def _monitor(self):
        while not self.net.stop:
            self.time_points.append(self.net.env.now)
            self.energy_history.append(self.net.energy_consumed)
            self.dead_nodes_history.append(self.net.dead_nodes)
            
            yield self.net.env.timeout(0.25)  # ce qui donne à peu près tous les 2 messages envoyés, pas déconnant

    def run(self):
        print("===== STARTING SIMULATION =====")
        self.net.env.process(self._random_communication())
        self.net.env.process(self._monitor())
        
        while not self.net.stop:
            self.net.env.step()
            # print(f"\n--- STEP {self.net.env.now:.4f} ---")
        
        print("\n=== SIMULATION COMPLETE ===")
        self.print_results()
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
        
        # # Print final node status
        # print("\nNode Status:")
        # for i in range(self.nb_nodes):
        #     node = self.net.G.nodes[i]['obj']
        #     status = "ALIVE" if node.alive else "DEAD"
        #     print(f"Node {i}: {status}, Battery: {node.battery:.2f}, Position: {node.pos}")
        #inutile et déjà dans le plt

    def plot_results(self):
        if not self.time_points:
            print("No data to plot")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Energy and Dead Nodes
        plt.subplot(2, 2, 1)
        plt.plot(self.time_points, self.energy_history, 'b-')
        plt.xlabel('Temps')
        plt.ylabel('Énergie')
        plt.title('Consommation énergétique au cours du temps')
        plt.grid(True)
        
        plt.subplot(2,2,2)
        plt.plot(self.time_points, self.dead_nodes_history, 'r-')
        plt.ylabel('Noeuds morts')
        plt.title('Mort des noeuds au cours du temps')
        plt.grid(True)
        
        # Message Types
        plt.subplot(2, 2, 3)
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
        plt.ylabel('Nombre')
        plt.title('Statistiques de messages')

        
        plt.subplot(2, 2, 4)
        #affichage du réseau
        for i, pos in self.node_positions.items():
            node_obj = self.net.G.nodes[i]['obj']
            color = 'green' if node_obj.alive else 'red'
            plt.plot(pos[0], pos[1], marker='o', markersize=10, color=color)
            plt.text(pos[0], pos[1], str(i), fontsize=9, ha='center', va='center')
        
        for edge in self.net.G.edges():
            n1 = self.net.G.nodes[edge[0]]['obj']
            n2 = self.net.G.nodes[edge[1]]['obj']
            dist = self.net.get_distance(n1, n2)
            if n1.alive and n2.alive and dist <= self.max_dist: #on affiche que les vraies connections (dist<=max_dist) entre deux noeuds vivants
                plt.plot([n1.pos[0], n2.pos[0]], [n1.pos[1], n2.pos[1]], 'b-', alpha=0.3)
        
        plt.xlim(0, self.area_size)
        plt.ylim(0, self.area_size)
        plt.title('Réseau (Vert=Actif, Rouge=Mort)')
        plt.xlabel('Position en X')
        plt.ylabel('Position en Y')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('aodv_debug_results.png')
        # plt.get_current_fig_manager().full_screen_toggle() #plein écran mais marche pas
        plt.show()
        print("Saved results to aodv_debug_results.png")

if __name__ == "__main__":
    print("starting")
    #cf obsidian pour valeurs
    sim = Simulation(
        nb_nodes=25,
        area_size=800,
        max_dist=250,
        conso=(0.01,0.2),
        seuil = 5,
        coeff_dist= 0.25,
        coeff_bat= 1,
        coeff_conso= 0.01,
        ttl = 5
    )
    sim.run()

#https://chat.deepseek.com/a/chat/s/e9f44a34-4df3-4d4d-b3d3-07ec7f5eb11e