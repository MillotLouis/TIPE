# import networkx as nx
# import simpy
# import copy

# import node

# class Message:
#     def __init__(self,typ,src_id,src_seq,dest_seq,dest_id,weight,prev_hop,data=None):
#         self.type = typ
#         # self.data = data # Surement pas utile
#         self.src_id = src_id
#         self.src_seq = src_seq
#         self.dest_seq = dest_seq
#         self.dest_id = dest_id
#         self.weight = weight
#         self.prev_hop = prev_hop

# # ♥ Passer uniquement les objets en argument de fonction pas les id #

# class Network:
#     def __init__(self,conso,seuil,coeff_dist,coeff_bat,coeff_conso,nb_nodes):
#         """
#         conso : tuple (x,y) : x = pourcentage de batterie consomée à chaque transmission de rreq, y = ... à chaque transmission de message
#         seuil : seuil en dessous duquel un noeud évite de transmettre des messages : on lui affecte poids très grand
#         a,b : paramètres de pondération des arcs : weight = a*distance + b*(1/batterie)
#         """
#         self.env = simpy.Environment()
#         self.G = nx.Graph()
#         self.conso = conso
#         self.seuil = seuil
#         self.coeff_dist = coeff_dist
#         self.coeff_bat = coeff_bat
#         self.coeff_conso = coeff_conso
#         self.stop = False #pour arrêter simulation
        
#         self.messages_forwarded = 0
#         self.messages_sent = 0
#         self.messages_received = 0
#         self.rreq_sent = 0
#         self.rrep_sent = 0
#         self.energy_consumed = 0
#         self.nb_nodes = nb_nodes
#         self.dead_nodes = 0
#         self.data_sent = 0

#     def add_node(self,id,pos,max_dist,battery=100):
#         """Ajoute un noeud au graphe"""
#         new_node = node.Node(self.env,id,pos,battery,max_dist,self)
#         self.G.add_node(id,obj=new_node)

#     def add_link(self, n1, n2):
#         """Ajoute une arrête entre n1 et n2 si il leur reste de la batterie"""
#         if n1.alive and n2.alive:
#             self.G.add_edge(n1.id, n2.id)

#     def update_battery(self,node,type,dist):
#         """Retire percent% de batterie à node"""
#         cons = self.conso[0] if type == 'RRE' else self.conso[1]
#         energy_cost = self.coeff_cons*dist + cons
#         node.battery = max(0,node.battery - energy_cost)
#         self.energy_consumed += energy_cost
        
#         if node.battery <= 0 and node.alive:
#             self.env.process(self._kill_node(node))

#         return node.alive
        
#     def _kill_node(self,node):
#         yield self.env.timeout(0) #permet d'attendre la fin de l'étape simpy pour ne pas supprimer un voisin si il est dans une liste en train d'être parcourue
#         self.G.remove_edges_from(list(self.G.edges(node.id)))
#         node.alive = False
#         self.dead_nodes += 1

#         if self.dead_nodes > self.nb_nodes / 2:
#             self.stop = True
    
#     def get_distance(self,n1,n2):
#         """Renvoie la distance entre node1 et node2,
#         on passe uniquement les objets en argument
#         """
#         return ((n2.pos[0] - n1.pos[0])**2 + (n2.pos[1] - n1.pos[1])**2)**0.5

    
#     def calculate_weight(self,n1,n2):
#         """Calcule le poids de l'arc n1 vers n2 en prenant en compte la distance à ce dernier et sa batterie"""
#         bat = n2.battery
#         dist = self.get_distance(n1,n2)
#         return self.coeff_dist*dist + self.coeff_bat*(1/bat)

#     def broadcast_rreq(self,node,rreq):
#         for n in self.G.neighbors(node.id):
#             neighbor = n["obj"]
#             dist = self.get_distance(node,neighbor)
#             if dist <= node.max_dist:
#                 if self.update_battery(node,"RRE",neighbor):
#                     self.rreq_sent += 1
#                     new_rreq = copy.deepcopy(rreq) #modifié, ne surtout pas set prev_hop à node.id
#                     neighbor.pending.put(new_rreq)
#                 #sinon, ajouter compteur perdus peut être


#     def unicast_rrep(self,node,rrep):
#         next_hop = node.routing_table.get(rrep.src_id,(None,0,0))[0]
#         if next_hop:
#             dist = self.get_distance(node, self.G[next_hop]["obj"])
#             rrep.prev_hop = node.id
#             if dist <= node.max_dist:
#                 #else rrep perdu à ajouter au compteur
#                 if self.update_battery(node,"RRE",dist):
#                     yield self.env.   (0.001) #délai à modifier peut-être
#                     self.rrep_sent += 1
#                     self.G[next_hop]["obj"].pending.put(rrep)

#     def forward_data(self,node:'node.Node',data):
#         next_hop = node.routing_table.get(data.dest_id,(None, 0, 0))[2]
#         if next_hop:
#             next_node = self.G[next_hop]["obj"]
#             dist = self.get_distance(node,next_node)
#             if dist <= node.max_dist:
#                 if self.update_battery(node,"DATA",dist):
#                     yield self.env.timeout(0.001) #délai à modifier peut-être
#                     next_node.pending.put(data)


# Attention, copier version dans aodv_test_script

