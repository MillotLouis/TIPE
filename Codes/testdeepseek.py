import networkx as nx
import simpy

class Message:
    def __init__(self, typ, src_id, src_seq, dest_seq, dest_id, hop, prev_hop, data=None):
        self.type = typ
        self.src_id = src_id        # Originator of the message
        self.src_seq = src_seq      # Originator's sequence number
        self.dest_seq = dest_seq    # Destination sequence number (for RREQ)
        self.dest_id = dest_id      # Final destination
        self.hop = hop              # Hop count
        self.prev_hop = prev_hop    # Immediate sender
        self.data = data

class Node:
    def __init__(self, env, id, pos, initial_battery, max_dist, network):
        self.env = env
        self.id = id
        self.pos = pos
        self.battery = initial_battery
        self.routing_table = {}  # {dest: (next_hop, seq_num, hops)}
        self.seq_num = 0
        self.pending = simpy.Store(env)
        self.max_dist = max_dist
        self.alive = True
        self.network = network
        env.process(self.process_messages())

    def process_messages(self):
        while True:
            msg = yield self.pending.get()
            if self.alive:
                if msg.type == "RREQ":
                    self.handle_rreq(msg)
                elif msg.type == "RREP":
                    self.handle_rrep(msg)

    def init_rreq(self, dest_id):
        """Initiate Route Request (RFC 3561 Section 5.1)"""
        self.seq_num += 1
        rreq = Message(
            typ="RREQ",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_seq=self.routing_table.get(dest_id, (None, 0, 0))[1],  # Last known dest seq
            dest_id=dest_id,
            hop=0,
            prev_hop=self.id
        )
        self.network.broadcast_rreq(self.id, rreq)

    def handle_rreq(self, rreq):
        """Process Route Request (RFC 3561 Section 5.3)"""
        # Update reverse path to source (Section 5.3.2)
        self.update_route(rreq.src_id, rreq.prev_hop, rreq.src_seq, rreq.hop)
        # Check if we're the destination or have fresh route (Section 5.3.3)
        if self.id == rreq.dest_id:
            self.send_rrep(rreq, is_destination=True)
        else:
            entry = self.routing_table.get(rreq.dest_id, (None, 0, float('inf')))
            if entry[1] >= rreq.dest_seq:  # Existing route is fresher
                self.send_rrep(rreq, entry[1], entry[2])
                return
                
            # Forward RREQ (Section 5.3.4)
            rreq.hop += 1
            rreq.prev_hop = self.id
            self.network.forward_rreq(self.id, rreq)

    def send_rrep(self, rreq, dest_seq=None, hops=None, is_destination=False):
        """Send Route Reply (RFC 3561 Section 5.4)"""
        if is_destination:
            self.seq_num += 1
            dest_seq = self.seq_num
            hops = 0
        else:
            hops += 1  # Add our hop to the existing route
            
        rrep = Message(
            typ="RREP",
            src_id=rreq.dest_id,    # Original destination
            src_seq=dest_seq,       # Destination's sequence number
            dest_seq=0,             # Not used in RREP
            dest_id=rreq.src_id,    # Original source
            hop=hops,
            prev_hop=self.id
        )
        
        # Use reverse path to source
        next_hop = self.routing_table.get(rreq.src_id, (None, 0, 0))[0]
        if next_hop:
            self.network.unicast_rrep(self.id, next_hop, rrep)

    def handle_rrep(self, rrep):
        """Process Route Reply (RFC 3561 Section 5.5)"""
        # Update forward path to destination (Section 5.5.2)
        self.update_route(rrep.src_id, rrep.prev_hop, rrep.src_seq, rrep.hop)
        
        # Forward towards source if not the originator
        if self.id != rrep.dest_id:
            rrep.hop += 1
            next_hop = self.routing_table.get(rrep.dest_id, (None, 0, 0))[0]
            if next_hop:
                self.network.unicast_rrep(self.id, next_hop, rrep)

    def update_route(self, dest, next_hop, seq_num, hops):
        """Update routing table following RFC 3561 rules"""
        current = self.routing_table.get(dest, (None, -1, float('inf')))
        
        # Update if: newer seq OR same seq with shorter path
        if (seq_num > current[1]) or (seq_num == current[1] and hops < current[2]):
            self.routing_table[dest] = (next_hop, seq_num, hops)

class Network:
    def __init__(self, conso, seuil, a, b):
        self.env = simpy.Environment()
        self.G = nx.Graph()
        self.conso = conso  # (RREQ_cons, DATA_cons)
        self.seuil = seuil  # Battery threshold
        self.a = a          # Distance coefficient
        self.b = b          # Battery coefficient

    def add_node(self, id, pos, battery):
        node = Node(self.env, id, pos, battery, max_dist=100, network=self)
        self.G.add_node(id, obj=node)

    def add_link(self, n1, n2):
        """Add bidirectional link with dynamic weights"""
        node1 = self.G.nodes[n1]["obj"]
        node2 = self.G.nodes[n2]["obj"]
        
        if node1.alive and node2.alive:
            weight = self.calculate_weight(n1, n2)
            self.G.add_edge(n1, n2, weight=weight)
            
            # Add reverse direction with different weight
            reverse_weight = self.calculate_weight(n2, n1)
            self.G.add_edge(n2, n1, weight=reverse_weight)

    def update_battery(self, node_id, msg_type):
        """Update battery and handle node death (RFC 3561 Section 7)"""
        node = self.G.nodes[node_id]["obj"]
        cons = self.conso[0] if msg_type == 'RRE' else self.conso[1]
        node.battery = max(0, node.battery - cons)
        
        if node.battery <= self.seuil:
            node.alive = False
            # Remove all associated edges
            edges = list(self.G.edges(node_id))
            self.G.remove_edges_from(edges)
            return False
        return True

    def broadcast_rreq(self, sender_id, rreq):
        """Flood RREQ to all neighbors (RFC 3561 Section 5.1)"""
        if self.update_battery(sender_id, 'RRE'):
            for neighbor in self.G.neighbors(sender_id):
                self.env.process(self.transmit(sender_id, neighbor, rreq))

    def unicast_rrep(self, sender_id, receiver_id, rrep):
        """Send RREP along reverse path (RFC 3561 Section 5.4)"""
        if self.update_battery(sender_id, 'RRE'):
            self.env.process(self.transmit(sender_id, receiver_id, rrep))

    def transmit(self, sender, receiver, packet):
        """Simulate wireless transmission with delay"""
        yield self.env.timeout(0.001)  # 1ms transmission delay
        receiver_node = self.G.nodes[receiver]["obj"]
        if receiver_node.alive:
            yield receiver_node.pending.put(packet)

    def calculate_weight(self, n1, n2):
        """Calculate edge weight according to a*distance + b*(1/battery)"""
        node2 = self.G.nodes[n2]["obj"]
        dist = self.get_distance(n1, n2)
        return self.a * dist + self.b * (1 / node2.battery)

    def get_distance(self, n1, n2):
        """Euclidean distance between nodes"""
        pos1 = self.G.nodes[n1]["obj"].pos
        pos2 = self.G.nodes[n2]["obj"].pos
        return ((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)**0.5