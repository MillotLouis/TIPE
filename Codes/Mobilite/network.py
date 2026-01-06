import random
import simpy
import copy
import numpy as np

from node import Node

class Message:
    def __init__(self, typ, src_id, src_seq, dest_seq, dest_id, weight, prev_hop):
        self.type = typ             
        """ type de message : requête donc "RREQ" ou "RREP" ou bien data : "DATA" """
        
        self.src_id = src_id        
        """ identifiant de l'émetteur du message """
        
        self.src_seq = src_seq      
        """ numéro de séquence de la source au moment de l'émission """
        
        self.dest_seq = dest_seq    
        """  dernier numéro de séquence connu (par la source) du destinataire au moment de l'émission """
        # ne sert pas pour l'instant, utile pour réponses intermédiaires
        
        self.dest_id = dest_id      
        """ identifiant du destinataire"""
        
        self.weight = weight        
        """ poids de la route empruntée par ce message, sera augmenté au fil des propagations """
        
        self.prev_hop = prev_hop    
        """ dernier noeud par lequel le message a été forwardé """

class Network:
    def __init__(self, conso, seuil, coeff_dist_weight, coeff_bat_weight, coeff_dist_bat, nb_nodes, ttl, reg_aodv):
        self.env = simpy.Environment()                   
        """ Environnement simpy """
        
        self.G = {}                             
        """ Graphe du réseau représenté par un dictionnaire node_id : node_obj """
        
        self.conso = conso                               
        """ consomation pour la transmition de requêtes / données : (req,donnée) """
        
        self.seuil = seuil                               
        """ seuil à partir duquel on applique la pénalité sur le poids des routes """
        
        self.coeff_dist_weight = coeff_dist_weight       
        """ coefficient de pondération : poids calculé avec dist_normalisée * coeff_dist_weight + ... """
        
        self.coeff_bat_weight = coeff_bat_weight         
        """ coefficient de pondération : poids calculé avec ... + batt_normalisée * coeff_bat_weight  """
        
        self.coeff_dist_bat = coeff_dist_bat             
        """ coefficient de pondération : consommation calculée avec ... + coeff_dist_bat * dist """
        
        self.ttl = ttl                                   
        """ ttl des routes """
        
        self.stop = False                                
        """ passé à True quand on veut que la simulation s'arrête """
        
        self.reg_aodv = reg_aodv                         
        """ True si on utilise AODV et false sinon """
        
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
        """ Nombre de routes pénalisées car en dessous du seuil """
        
        self.first_node_death_time = None
        self.ten_percent_death_time = None
        self.fifty_percent_death_time = None
        self.death_times = []                            
        """ Liste des dates auxquelles des noeuds sont morts  """
        
        # Pour calculer le delivery ratio
        self.data_log = {}              
        """ (src_id, data_seq) -> {'t_init': float, 't_send': float|None, 't_recv': float|None} """
        
        self.data_init_times = []       
        """ [(t_init, key)] """
        
        self.data_send_times = []       
        """ [(t_send, key)] """
        
    def add_node(self, id, pos, max_dist, reg_aodv, battery=100):
        """
        Ajoute un noeud au réseau
        Marche pareil si reg_aodv ou pas
        """
        new_node = Node(self.env, id, pos, battery, max_dist, self,reg_aodv)
        self.G[id] = new_node

    def update_battery(self, node, msg_type, dist):
        """
        Met à jour la batterie et tue le noeud si il n'en a plus 
        Marche pareil si reg_aodv ou pas
        """
        cons = self.conso[0] if (msg_type[:2] == "RR") else self.conso[1]
        energy_cost = self.coeff_dist_bat * dist + cons
        node.battery = max(0, node.battery - energy_cost)
        self.energy_consumed += energy_cost
        
        if node.battery == 0 and node.alive:
            self.env.process(self._kill_node(node))
        
        return node.battery > 0

    def _kill_node(self, node):
        """
        Tue un noeud, comptabilise cette mort dans les métriques et enlève toutes les connexions le concernant
        """
        yield self.env.timeout(0)
        
        node.alive = False
        self.dead_nodes += 1
        
        self.death_times.append(self.env.now)
        
        if self.first_node_death_time is None:
            self.first_node_death_time = self.env.now
            print(f"First node death at time {self.env.now:.2f}")
        
        if self.ten_percent_death_time is None and self.dead_nodes >= self.nb_nodes * 0.1:
            self.ten_percent_death_time = self.env.now
            print(f"10% nodes dead at time {self.env.now:.2f}")

        if self.fifty_percent_death_time is None and self.dead_nodes >= self.nb_nodes * 0.5:
            self.fifty_percent_death_time = self.env.now
            print(f"50% nodes dead at time {self.env.now:.2f}")
            self.stop = True
        

    def get_distance(self, n1, n2):
        """Marche pareil si reg_aodv ou pas"""
        return ((n2.pos[0] - n1.pos[0])**2 + (n2.pos[1] - n1.pos[1])**2)**0.5

    def calculate_weight(self, n1, n2):
        """
        Calcule le poids d'un saut
        Dépend de reg_aodv
        """
        if self.reg_aodv:
            return 1
        
        bat = max(n2.battery, 0.1)  
        dist = self.get_distance(n1, n2)
        
        dist_norm = dist / n1.max_dist
        bat_norm = 1 - (bat / n1.initial_battery)
        
        weight = (self.coeff_dist_weight * dist_norm) + (self.coeff_bat_weight * bat_norm)

        if bat < self.seuil:
            self.seuiled += 1
            ecart = (self.seuil - bat) / self.seuil
            penalty = min(1.0, 0.5 * ecart)  # limite la pénalité à 1
            weight += penalty

        return weight

    def get_energy_stats(self):
        """
        Calcule la batterie restante moyenne dans le réseau
        Calcule l'écart type sur ↑
        """
        alive_nodes = [node for node in self.G.values() if node.alive]
        
        if not alive_nodes:
            return 0, 0
        
        remaining_energies = [node.battery for node in alive_nodes]
        avg_energy = np.mean(remaining_energies)
        std_energy = np.std(remaining_energies)
        
        return avg_energy, std_energy

    def broadcast_rreq(self, node, rreq):
        """
        Broadcast une RREQ à tous les noeuds à portée de node
        Marche pareil si reg_aodv ou pas
        """
        # if valid_neighbors != []:
        #     max_dist = max(dist for _, dist in valid_neighbors) if not self.reg_aodv else node.max_dist
        #     if not self.update_battery(node, "RREQ", max_dist): return #on consomme la batterie une seule fois pour un broadcast

        for neighbor in self.G.values():
            if not neighbor.alive or self.get_distance(node, neighbor) > node.max_dist:
                continue

            yield self.env.timeout(random.uniform(0.01, 0.05)) #on ajoute un "jitter" aléatoire avant chaque transmission pour
                                                                #modéliser la réalité et éviter les problèmes de simulation : 
                                                                #tous les evenements sont planifiés à la même date => elle avance pas dans le temps
            new_rreq = copy.deepcopy(rreq)  #deepcopy pour avoir des objets différents sinon chaque noeud va modifier le même RREQ
            neighbor.pending.put(new_rreq)
            self.rreq_forwarded += 1

        
    def unicast_rrep(self, node, rrep):
        """
        Permet de renvoyer la RREP à la source
        Marche pareil si reg_aodv ou pas
        """
        next_hop = node.routing_table.get(rrep.dest_id, (None, 0, 0, 0))[0]
        
        if next_hop is None:
            return 
            
        next_node = self.G[next_hop]
        
        if not next_node.alive:
            return
            
        dist = self.get_distance(node, next_node)
        
        if dist <= node.max_dist:
            if self.update_battery(node, "RREP", dist if not self.reg_aodv else node.max_dist): 
                yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))  #délai basé sur la distance, facteur arbitraire : 1ms / unité de distance
                next_node.pending.put(rrep)

    def forward_data(self, node, data):
        """
        Transmet des paquets de type DATA selon la route stockée dans les noeuds
        Marche pareil si reg_aodv ou pas
        """
        next_hop = node.routing_table.get(data.dest_id, (None, 0, 0, 0))[0]
        data.prev_hop = node.id
        
        if next_hop is None:
            return
            
        next_node = self.G[next_hop]
        if not next_node.alive:
            return
            
        dist = self.get_distance(node, next_node)
        if dist <= node.max_dist:
            if self.update_battery(node, "DATA", dist if not self.reg_aodv else node.max_dist):
                self.messages_forwarded += 1
                
                yield self.env.timeout(dist * 0.001 + random.uniform(0.01, 0.05))
                next_node.pending.put(data)

    def log_data_init(self, src_id, data_seq, t_init):
        key = (src_id, data_seq)
        self.data_log[key] = {'t_init': t_init, 't_send': None, 't_recv': None}
        self.data_init_times.append((t_init, key))

    def log_data_send(self, src_id, data_seq, t_send):
        key = (src_id, data_seq)
        e = self.data_log.get(key)
        if e and e['t_send'] is None:
            e['t_send'] = t_send
            self.data_send_times.append((t_send, key))

    def log_data_recv(self, src_id, data_seq, t_recv):
        key = (src_id, data_seq)
        e = self.data_log.get(key)
        if e and e['t_recv'] is None:
            e['t_recv'] = t_recv




