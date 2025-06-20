import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import network  # Your network.py module
import node     # Your node.py module

class Simulation:
    def __init__(self, num_nodes=50, area_size=100, max_dist=25, duration=100):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.max_dist = max_dist
        self.duration = duration
        self.conso = (0.5, 1.0)  # (RREQ consumption, DATA consumption)
        self.seuil = 10           # Battery threshold
        self.coeff_dist = 0.7     # Distance coefficient
        self.coeff_bat = 0.3      # Battery coefficient
        self.coeff_conso = 0.01   # Distance consumption coefficient
        
        # Initialize network
        self.net = network.Network(
            conso=self.conso,
            seuil=self.seuil,
            coeff_dist=self.coeff_dist,
            coeff_bat=self.coeff_bat,
            coeff_conso=self.coeff_conso,
            nb_nodes=num_nodes
        )
        
        # Create nodes with random positions
        self.node_positions = {}
        for i in range(num_nodes):
            pos = (random.uniform(0, area_size), random.uniform(0, area_size))
            self.node_positions[i] = pos
            self.net.add_node(i, pos, max_dist, battery=100)
        
        # Create links based on proximity
        self._create_links()
        
        # Statistics
        self.energy_history = []
        self.dead_nodes_history = []
        self.messages_history = defaultdict(list)
        
    def _create_links(self):
        """Create links between nodes within communication range"""
        nodes = list(self.net.G.nodes(data='obj'))
        for i in range(len(nodes)):
            n1_id, n1 = nodes[i]
            for j in range(i+1, len(nodes)):
                n2_id, n2 = nodes[j]
                dist = self.net.get_distance(n1, n2)
                if dist <= self.max_dist:
                    self.net.G.add_edge(n1_id, n2_id)
    
    def _random_communication(self):
        """Generate random communication events between nodes"""
        while self.net.env.now < self.duration and not self.net.stop:
            # Random source and destination
            src_id = random.randint(0, self.num_nodes-1)
            dest_id = random.randint(0, self.num_nodes-1)
            while dest_id == src_id:
                dest_id = random.randint(0, self.num_nodes-1)
                
            # Get node objects
            src_node = self.net.G.nodes[src_id]['obj']
            dest_node = self.net.G.nodes[dest_id]['obj']
            
            # Only send if source is alive and has battery
            if src_node.alive and src_node.battery > self.seuil:
                src_node.send_data(dest_id)
            
            # Wait random time before next communication
            yield self.net.env.timeout(random.expovariate(0.1))
    
    def _monitor(self):
        """Track simulation metrics over time"""
        while not self.net.stop and self.net.env.now < self.duration:
            self.energy_history.append(self.net.energy_consumed)
            self.dead_nodes_history.append(self.net.dead_nodes)
            self.messages_history['sent'].append(self.net.messages_sent)
            self.messages_history['forwarded'].append(self.net.messages_forwarded)
            self.messages_history['received'].append(self.net.messages_received)
            self.messages_history['rreq'].append(self.net.rreq_sent)
            self.messages_history['rrep'].append(self.net.rrep_sent)
            yield self.net.env.timeout(0.1)
    
    def run(self):
        """Run the simulation"""
        self.net.env.process(self._random_communication())
        self.net.env.process(self._monitor())
        
        # Run until half nodes dead or time expires
        while not self.net.stop and self.net.env.now < self.duration:
            self.net.env.step()
        
        print("\n=== Simulation Complete ===")
        print(f"Duration: {self.net.env.now:.2f} time units")
        print(f"Dead nodes: {self.net.dead_nodes}/{self.num_nodes}")
        print(f"Energy consumed: {self.net.energy_consumed:.2f}")
        print(f"Messages sent: {self.net.messages_sent}")
        print(f"Messages received: {self.net.messages_received}")
        print(f"RREQ sent: {self.net.rreq_sent}")
        print(f"RREP sent: {self.net.rrep_sent}")
        
        self.plot_results()
    
    def plot_results(self):
        """Visualize simulation metrics"""
        time_points = np.linspace(0, self.net.env.now, len(self.energy_history))
        
        plt.figure(figsize=(15, 10))
        
        # Energy and Dead Nodes
        plt.subplot(2, 2, 1)
        plt.plot(time_points, self.energy_history, 'b-', label='Energy Consumed')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy Consumption Over Time')
        plt.grid(True)
        
        ax2 = plt.gca().twinx()
        ax2.plot(time_points, self.dead_nodes_history, 'r-', label='Dead Nodes')
        ax2.set_ylabel('Dead Nodes')
        plt.title('Energy and Node Mortality')
        plt.legend()
        
        # Message Types
        plt.subplot(2, 2, 2)
        msg_types = ['sent', 'forwarded', 'received', 'rreq', 'rrep']
        counts = [
            self.net.messages_sent,
            self.net.messages_forwarded,
            self.net.messages_received,
            self.net.rreq_sent,
            self.net.rrep_sent
        ]
        plt.bar(msg_types, counts, color=['blue', 'green', 'red', 'purple', 'orange'])
        plt.xlabel('Message Type')
        plt.ylabel('Count')
        plt.title('Message Statistics')
        
        # Network Topology
        plt.subplot(2, 2, 3)
        for i, pos in self.node_positions.items():
            if self.net.G.nodes[i]['obj'].alive:
                plt.plot(pos[0], pos[1], 'go', markersize=8)
            else:
                plt.plot(pos[0], pos[1], 'ro', markersize=8)
            plt.text(pos[0], pos[1], str(i), fontsize=8)
        
        # Draw edges
        for edge in self.net.G.edges():
            n1 = self.net.G.nodes[edge[0]]['obj']
            n2 = self.net.G.nodes[edge[1]]['obj']
            if n1.alive and n2.alive:
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
            plt.plot(time_points, self.messages_history[msg_type], color, label=msg_type)
        plt.xlabel('Time')
        plt.ylabel('Message Count')
        plt.title('Message Flow Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('aodv_simulation_results.png')
        plt.show()

if __name__ == "__main__":
    sim = Simulation(
        num_nodes=30,
        area_size=80,
        max_dist=20,
        duration=200
    )
    sim.run()