import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import simpy
import random
import copy
from collections import defaultdict
import time

# Your existing classes with critical fixes
class Message:
    def __init__(self, typ, src_id, src_seq, dest_seq, dest_id, weight, prev_hop, data=None):
        self.type = typ
        self.data = data
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
        self.initial_battery = initial_battery
        self.routing_table = {}
        self.seq_num = 0
        self.pending = simpy.Store(env)
        self.max_dist = max_dist
        self.alive = True
        self.network = network
        self.seen = set()
        self.pending_rreqs = {}
        
        # Metrics counters
        self.messages_sent = 0
        self.messages_received = 0
        self.rreq_sent = 0
        self.rrep_sent = 0
        self.data_sent = 0
        self.data_received = 0
        self.battery_history = [initial_battery]
        self.time_history = [0]
        
        env.process(self.process_messages())

    def process_messages(self):
        while True:
            try:
                if self.alive:
                    # Remove timeout to avoid dropping messages
                    msg = yield self.pending.get()
                    if hasattr(msg, 'type'):
                        self.messages_received += 1
                        print(f"Node {self.id} processing {msg.type} message")
                        
                        if msg.type == "RREQ":
                            self.handle_rreq(msg)
                        elif msg.type == "RREP":
                            self.handle_rrep(msg)
                        elif msg.type == "DATA":
                            self.handle_data(msg)
                else:
                    break
            except Exception as e:
                print(f"Error in node {self.id} message processing: {e}")
                break

    def init_rreq(self, dest_id):
        """Initialize route request - Fixed to properly handle broadcasting"""
        self.seq_num += 1
        print(f"Node {self.id} initiating RREQ for destination {dest_id}")
        
        rreq = Message(
            typ="RREQ",
            src_id=self.id,
            src_seq=self.seq_num,
            dest_id=dest_id,
            dest_seq=self.routing_table.get(dest_id, (None, 0, 0))[1],
            prev_hop=self.id,
            weight=0
        )
        self.rreq_sent += 1
        self.network.broadcast_rreq(self, rreq) 

    def handle_rreq(self, rreq):
        """Handle received RREQ - Fixed duplicate detection and routing"""
        seen_key = (rreq.src_id, rreq.src_seq)
        if seen_key in self.seen or self.battery < self.network.seuil:
            print(f"Node {self.id} ignoring duplicate RREQ from {rreq.src_id}")
            return
        
        self.seen.add(seen_key)
        print(f"Node {self.id} handling RREQ from {rreq.src_id} to {rreq.dest_id}")
        
        # Calculate weight from previous hop to current node
        if rreq.prev_hop != rreq.src_id:  # Not the original source
            prev_node = self.network.G.nodes[rreq.prev_hop]["obj"]
            rreq.weight += self.network.calculate_weight(prev_node, self)
        
        # CRITICAL FIX: Always update route to source for reverse path
        self.update_route(
            dest=rreq.src_id,
            next_hop=rreq.prev_hop,
            seq_num=rreq.src_seq,
            weight=rreq.weight
        )
        
        # Check if this node is the destination
        if self.id == rreq.dest_id:
            print(f"Node {self.id} is destination for RREQ from {rreq.src_id}")
            # Send RREP immediately instead of collecting
            self.send_rrep(rreq)
            return
        
        # Forward RREQ if not destination
        rreq.prev_hop = self.id
        print(f"Node {self.id} forwarding RREQ from {rreq.src_id} to {rreq.dest_id}")
        self.network.broadcast_rreq(self, rreq)

    def handle_rrep(self, rrep):
        """Handle received RREP - Fixed routing"""
        print(f"Node {self.id} received RREP from {rrep.src_id} to {rrep.dest_id}")
        
        # Calculate weight if not from original source
        if rrep.prev_hop != rrep.src_id:
            prev_node = self.network.G.nodes[rrep.prev_hop]["obj"]
            rrep.weight += self.network.calculate_weight(prev_node, self)
        
        # Update route to RREP source (the original destination)
        self.update_route(
            dest=rrep.src_id,
            next_hop=rrep.prev_hop,
            seq_num=rrep.src_seq,
            weight=rrep.weight
        )

        # Forward RREP if not the final destination
        if self.id != rrep.dest_id:
            print(f"Node {self.id} forwarding RREP from {rrep.src_id}")
            # CRITICAL FIX: Use routing table to find next hop towards destination
            if rrep.dest_id in self.routing_table:
                next_hop_id = self.routing_table[rrep.dest_id][0]
                if next_hop_id and next_hop_id in self.network.G.nodes:
                    next_hop = self.network.G.nodes[next_hop_id]["obj"]
                    if next_hop.alive:
                        rrep.prev_hop = self.id
                        self.env.process(self.network.unicast_message(self, rrep, next_hop_id))
            else:
                print(f"Node {self.id}: No route to forward RREP to {rrep.dest_id}")
        else:
            print(f"RREP from {rrep.src_id} reached final destination {self.id}")

    def handle_data(self, data_msg):
        """Handle received data messages"""
        self.data_received += 1
        print(f"Node {self.id} received data message from {data_msg.src_id} to {data_msg.dest_id}")
        
        if self.id == data_msg.dest_id:
            # Message reached destination
            self.network.successful_deliveries += 1
            print(f"✓ Data delivery successful: {data_msg.src_id} -> {self.id}")
        else:
            # Forward the message
            print(f"Node {self.id} forwarding data from {data_msg.src_id} to {data_msg.dest_id}")
            self.env.process(self.forward_data(data_msg))

    def forward_data(self, data_msg):
        """Forward data message to next hop"""
        if data_msg.dest_id in self.routing_table:
            next_hop_id = self.routing_table[data_msg.dest_id][0]
            if next_hop_id:
                yield from self.network.unicast_message(self, data_msg, next_hop_id)
            else:
                print(f"Node {self.id}: Invalid next hop for {data_msg.dest_id}")
                self.network.failed_deliveries += 1
        else:
            print(f"Node {self.id}: No route to forward data to {data_msg.dest_id}")
            self.network.failed_deliveries += 1

    def send_data(self, dest_id, data):
        """Send data to destination - Fixed timing"""
        print(f"Node {self.id} wants to send data to {dest_id}")
        
        if dest_id not in self.routing_table:
            print(f"Node {self.id}: No route to {dest_id}, initiating route discovery")
            self.init_rreq(dest_id)
            # Wait longer for route discovery
            yield self.env.timeout(5.0)
        
        if dest_id in self.routing_table:
            print(f"Node {self.id}: Route found to {dest_id}, sending data")
            print(f"Node {self.id}: Route to {dest_id} via {self.routing_table[dest_id]}")
            
            self.seq_num += 1
            data_msg = Message(
                typ="DATA",
                src_id=self.id,
                src_seq=self.seq_num,
                dest_id=dest_id,
                dest_seq=0,
                weight=0,
                prev_hop=self.id,
                data=data
            )
            self.data_sent += 1
            next_hop_id = self.routing_table[dest_id][0]
            if next_hop_id:
                yield from self.network.unicast_message(self, data_msg, next_hop_id)
        else:
            print(f"Node {self.id}: Still no route found to {dest_id} after discovery")
            self.network.failed_deliveries += 1

    def send_rrep(self, rreq):
        """Send RREP back to source - Fixed routing"""
        self.seq_num += 1
        print(f"Node {self.id} sending RREP to {rreq.src_id}")
        
        rrep = Message(
            typ="RREP",
            src_id=self.id,  # This node is the source of RREP
            src_seq=self.seq_num,
            dest_seq=-1,
            dest_id=rreq.src_id,  # Send back to original RREQ source
            weight=0,
            prev_hop=self.id
        )
        self.rrep_sent += 1
        
        # Use the route established by RREQ
        if rreq.src_id in self.routing_table:
            next_hop_id = self.routing_table[rreq.src_id][0]
            print(f"Node {self.id}: Sending RREP to {rreq.src_id} via {next_hop_id}")
            self.env.process(self.network.unicast_message(self, rrep, next_hop_id))
        else:
            print(f"Node {self.id}: No reverse route to {rreq.src_id}")

    def update_route(self, dest, next_hop, seq_num, weight):
        """Update routing table with better route"""
        current = self.routing_table.get(dest, (None, -1, float('inf')))
        
        if (seq_num > current[1]) or (seq_num == current[1] and weight < current[2]):
            old_route = self.routing_table.get(dest, "No route")
            self.routing_table[dest] = (next_hop, seq_num, weight)
            print(f"Node {self.id}: Updated route to {dest}: {old_route} -> {(next_hop, seq_num, weight)}")

    def update_battery_history(self):
        """Update battery history for visualization"""
        self.battery_history.append(self.battery)
        self.time_history.append(self.env.now)

class Network:
    def __init__(self, conso, seuil, coeff_dist, coeff_bat, coeff_conso):
        self.env = simpy.Environment()
        self.G = nx.Graph()
        self.conso = conso
        self.seuil = seuil
        self.coeff_dist = coeff_dist
        self.coeff_bat = coeff_bat
        self.coeff_conso = coeff_conso
        
        # Metrics
        self.total_messages = 0
        self.successful_deliveries = 0
        self.failed_deliveries = 0
        self.total_energy_consumed = 0
        self.dead_nodes = 0

    def add_node(self, id, pos, max_dist, battery=100):
        new_node = Node(self.env, id, pos, battery, max_dist, self)
        self.G.add_node(id, obj=new_node)
        return new_node

    def add_link(self, n1, n2):
        if n1.alive and n2.alive:
            self.G.add_edge(n1.id, n2.id)

    def update_battery(self, node, msg_type, dist):
        """Update node battery based on message transmission"""
        cons = self.conso[0] if msg_type == 'RREQ' else self.conso[1]
        energy_cost = self.coeff_conso * dist + cons
        node.battery = max(0, node.battery - energy_cost)
        self.total_energy_consumed += energy_cost
        
        if node.battery <= 0 and node.alive:
            self.env.process(self._kill_node(node))
        
        node.update_battery_history()
        return node.alive

    def get_distance(self, n1, n2):
        return ((n2.pos[0] - n1.pos[0])**2 + (n2.pos[1] - n1.pos[1])**2)**0.5

    def calculate_weight(self, n1, n2):
        bat = max(n2.battery, 0.1)
        dist = self.get_distance(n1, n2)
        return self.coeff_dist * dist + self.coeff_bat * (1/bat)

    def broadcast_rreq(self, node, rreq):
        """Broadcast RREQ to all neighbors"""
        neighbor_ids = list(self.G.neighbors(node.id))
        print(f"Node {node.id} broadcasting RREQ to {len(neighbor_ids)} neighbors")
        
        for neighbor_id in neighbor_ids:
            if neighbor_id in self.G.nodes:
                neighbor = self.G.nodes[neighbor_id]["obj"]
                if neighbor.alive and neighbor.id != rreq.prev_hop:  # Don't send back to sender
                    dist = self.get_distance(node, neighbor)
                    if dist <= node.max_dist:
                        if self.update_battery(node, "RREQ", dist):
                            new_rreq = copy.deepcopy(rreq)
                            new_rreq.prev_hop = node.id  # Set correct previous hop
                            neighbor.pending.put(new_rreq)
                            self.total_messages += 1
                            print(f"  -> Sent RREQ to node {neighbor_id}")

    def _kill_node(self, node):
        """Safely kill a node by removing its edges"""
        yield self.env.timeout(0)
        if node.alive:
            print(f"Node {node.id} died (battery depleted)")
            edges_to_remove = list(self.G.edges(node.id))
            self.G.remove_edges_from(edges_to_remove)
            node.alive = False
            self.dead_nodes += 1

    def unicast_message(self, node, msg, next_hop_id):
        """Generic unicast for any message type"""
        if next_hop_id in self.G.nodes:
            next_hop = self.G.nodes[next_hop_id]["obj"]
            if next_hop.alive:
                dist = self.get_distance(node, next_hop)
                if dist <= node.max_dist:
                    msg_type = "RREQ" if msg.type == "RREQ" else "OTHER"
                    if self.update_battery(node, msg_type, dist):
                        msg.prev_hop = node.id
                        next_hop.pending.put(msg)
                        self.total_messages += 1
                        print(f"  -> Sent {msg.type} from {node.id} to {next_hop_id}")
                        yield self.env.timeout(0.01)  # Small delay for realism
                    else:
                        print(f"Node {node.id} died while sending {msg.type}")
                        self.failed_deliveries += 1
                else:
                    print(f"Node {next_hop_id} out of range from {node.id}")
                    self.failed_deliveries += 1
            else:
                print(f"Next hop node {next_hop_id} is dead")
                self.failed_deliveries += 1
        else:
            print(f"Next hop node {next_hop_id} doesn't exist")
            self.failed_deliveries += 1

    def create_random_topology(self, num_nodes, area_size, max_dist, min_battery=50, max_battery=100):
        """Create a random network topology"""
        nodes = []
        for i in range(num_nodes):
            pos = (random.uniform(0, area_size), random.uniform(0, area_size))
            battery = random.uniform(min_battery, max_battery)
            node = self.add_node(i, pos, max_dist, battery)
            nodes.append(node)
        
        # Create edges based on distance
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                dist = self.get_distance(node1, node2)
                if dist <= min(node1.max_dist, node2.max_dist):
                    self.add_link(node1, node2)
        
        return nodes

class AODVTester:
    def __init__(self, network):
        self.network = network
        self.metrics_history = defaultdict(list)
        self.time_points = []

    def run_communication_test(self, source_id, dest_id, num_messages=5, interval=2.0):
        """Test communication between two nodes"""
        def communication_process():
            source = self.network.G.nodes[source_id]["obj"]
            for i in range(num_messages):
                try:
                    if source.alive and dest_id in self.network.G.nodes:
                        dest_node = self.network.G.nodes[dest_id]["obj"]
                        if dest_node.alive:
                            data = f"Message {i+1} from {source_id} to {dest_id}"
                            yield from source.send_data(dest_id, data)
                            print(f"Scheduled message {i+1} from {source_id} to {dest_id}")
                        else:
                            print(f"Destination node {dest_id} is dead")
                            break
                    else:
                        print(f"Source node {source_id} is dead or dest {dest_id} doesn't exist")
                        break
                    yield self.network.env.timeout(interval)
                except Exception as e:
                    print(f"Error in communication {source_id}->{dest_id}: {e}")
                    break
        
        return self.network.env.process(communication_process())

    def collect_metrics(self):
        """Collect network metrics at current time"""
        alive_nodes = sum(1 for node_id in self.network.G.nodes() 
                         if self.network.G.nodes[node_id]["obj"].alive)
        
        total_battery = sum(self.network.G.nodes[node_id]["obj"].battery 
                           for node_id in self.network.G.nodes())
        
        avg_battery = total_battery / len(self.network.G.nodes()) if len(self.network.G.nodes()) > 0 else 0
        
        metrics = {
            'time': self.network.env.now,
            'alive_nodes': alive_nodes,
            'dead_nodes': self.network.dead_nodes,
            'total_messages': self.network.total_messages,
            'successful_deliveries': self.network.successful_deliveries,
            'failed_deliveries': self.network.failed_deliveries,
            'avg_battery': avg_battery,
            'total_energy_consumed': self.network.total_energy_consumed
        }
        
        for key, value in metrics.items():
            if key != 'time':
                self.metrics_history[key].append(value)
        
        self.time_points.append(metrics['time'])
        return metrics

    def run_simulation(self, duration, metric_interval=1.0):
        """Run simulation and collect metrics"""
        def metric_collector():
            while self.network.env.now < duration:
                try:
                    self.collect_metrics()
                    yield self.network.env.timeout(metric_interval)
                except Exception as e:
                    print(f"Error in metric collection: {e}")
                    break
        
        self.network.env.process(metric_collector())
        
        def watchdog():
            last_time = 0
            stuck_count = 0
            while self.network.env.now < duration:
                yield self.network.env.timeout(1.0)
                if self.network.env.now == last_time:
                    stuck_count += 1
                    if stuck_count > 5:
                        print(f"Simulation appears stuck at time {self.network.env.now}. Terminating.")
                        break
                else:
                    stuck_count = 0
                    last_time = self.network.env.now
                    if int(self.network.env.now) % 5 == 0:
                        print(f"Simulation progress: {self.network.env.now:.1f}/{duration}")
        
        self.network.env.process(watchdog())
        
        try:
            self.network.env.run(until=duration)
        except Exception as e:
            print(f"Simulation error: {e}")
        
        self.collect_metrics()

    def visualize_network(self, title="Network Topology"):
        """Visualization with routing table info"""
        plt.figure(figsize=(15, 10))
        
        # Create subplot for network topology
        ax1 = plt.subplot(2, 2, 1)
        pos_dict = {}
        colors = []
        sizes = []
        
        for node_id in self.network.G.nodes():
            node = self.network.G.nodes[node_id]["obj"]
            pos_dict[node_id] = node.pos
            if node.alive:
                colors.append(node.battery / node.initial_battery)
                sizes.append(100 + node.battery)
            else:
                colors.append(0)
                sizes.append(50)
        
        if pos_dict:
            nodes = nx.draw_networkx_nodes(self.network.G, pos_dict, node_color=colors, 
                                         node_size=sizes, cmap=plt.cm.RdYlGn, ax=ax1)
            nx.draw_networkx_edges(self.network.G, pos_dict, ax=ax1, alpha=0.5)
            nx.draw_networkx_labels(self.network.G, pos_dict, font_size=8, ax=ax1)
            
            if nodes:
                plt.colorbar(nodes, ax=ax1, label='Battery Level (normalized)')
        
        ax1.set_title(f"{title} - Node Status")
        ax1.set_aspect('equal')
        
        # Plot metrics if available
        if self.time_points:
            # Alive vs Dead nodes  
            plt.subplot(2, 2, 2)
            plt.plot(self.time_points, self.metrics_history['alive_nodes'], 'g-', label='Alive', linewidth=2)
            plt.plot(self.time_points, self.metrics_history['dead_nodes'], 'r-', label='Dead', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Number of Nodes')
            plt.title('Node Status Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Battery levels
            plt.subplot(2, 2, 3)
            plt.plot(self.time_points, self.metrics_history['avg_battery'], 'b-', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Average Battery Level')
            plt.title('Average Battery Over Time')
            plt.grid(True, alpha=0.3)
            
            # Message statistics
            plt.subplot(2, 2, 4)
            plt.plot(self.time_points, self.metrics_history['successful_deliveries'], 'g-', 
                    label='Successful', linewidth=2)
            plt.plot(self.time_points, self.metrics_history['failed_deliveries'], 'r-', 
                    label='Failed', linewidth=2)
            plt.plot(self.time_points, self.metrics_history['total_messages'], 'b--', 
                    label='Total Sent', alpha=0.7)
            plt.xlabel('Time')
            plt.ylabel('Number of Messages')
            plt.title('Message Delivery Statistics')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def print_final_stats(self):
        """Print final simulation statistics with routing info"""
        print("\n" + "="*50)
        print("FINAL SIMULATION STATISTICS")
        print("="*50)
        
        total_nodes = len(self.network.G.nodes())
        alive_nodes = sum(1 for node_id in self.network.G.nodes() 
                         if self.network.G.nodes[node_id]["obj"].alive)
        
        print(f"Total Nodes: {total_nodes}")
        print(f"Alive Nodes: {alive_nodes}")
        print(f"Dead Nodes: {self.network.dead_nodes}")
        print(f"Survival Rate: {alive_nodes/total_nodes*100:.1f}%")
        print(f"Total Messages Sent: {self.network.total_messages}")
        print(f"Successful Deliveries: {self.network.successful_deliveries}")
        print(f"Failed Deliveries: {self.network.failed_deliveries}")
        
        # Calculate delivery success rate
        total_data_attempts = sum(self.network.G.nodes[node_id]["obj"].data_sent 
                                for node_id in self.network.G.nodes())
        if total_data_attempts > 0:
            delivery_rate = self.network.successful_deliveries / total_data_attempts * 100
            print(f"Data Delivery Success Rate: {delivery_rate:.1f}%")
        
        print(f"Total Energy Consumed: {self.network.total_energy_consumed:.2f}")
        print(f"Average Energy per Node: {self.network.total_energy_consumed/total_nodes:.2f}")
        
        # Print routing table info
        print(f"\nRouting Table Summary:")
        for node_id in self.network.G.nodes():
            node = self.network.G.nodes[node_id]["obj"]
            if node.alive:
                routes = len(node.routing_table)
                print(f"Node {node_id}: {routes} routes - {dict(node.routing_table)}")


def main():
    """Main test function with debugging"""
    print("Testing Fixed AODV Protocol")
    print("="*40)
    
    # Network parameters
    conso = (20, 10)
    seuil = 5
    coeff_dist = 0.5
    coeff_bat = 20.0
    coeff_conso = 0.05
    
    # Create network
    network = Network(conso, seuil, coeff_dist, coeff_bat, coeff_conso)
    tester = AODVTester(network)
    
    # Create smaller, more connected topology for testing
    print("Creating test network topology...")
    num_nodes = 20
    area_size = 50
    max_dist = 30
    nodes = network.create_random_topology(num_nodes, area_size, max_dist, 
                                          min_battery=80, max_battery=100)
    
    print(f"Created network with {len(nodes)} nodes")
    print(f"Network has {network.G.number_of_edges()} edges")
    
    # Check connectivity
    if nx.is_connected(network.G):
        print("✓ Network is connected")
    else:
        print("⚠ Network is not fully connected")
        components = list(nx.connected_components(network.G))
        for i, comp in enumerate(components):
            print(f"Component {i+1}: nodes {sorted(comp)}")
    
    # Schedule simple test communications
    print("Scheduling test communications...")
    
    # Test 1: Simple communication
    tester.run_communication_test(0, min(num_nodes-1, 4), 1, 8.0)
    
    # Run simulation
    simulation_duration = 1500.0
    print(f"Running simulation for {simulation_duration} time units...")
    
    start_time = time.time()
    tester.run_simulation(simulation_duration, metric_interval=1.0)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Print results and visualize
    tester.print_final_stats()
    tester.visualize_network("Fixed AODV Test Results")

if __name__ == "__main__":
    main()