import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

class NetworkEnv:
    """
    Simulates a communication network graph where edges have 
    dynamic costs (latency + packet loss).
    """
    def __init__(self, num_nodes=20, edge_prob=0.2, mode='static'):
        # Create a random Erdős-Rényi graph
        self.graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
        self.mode = mode
        self.time_step = 0
        
        # Initialize edges with weights (latency + loss)
        for u, v in self.graph.edges():
            # Base weight between 1 and 10
            self.graph[u][v]['base_weight'] = random.uniform(1, 10)
            self.graph[u][v]['loss'] = random.uniform(0, 0.1) # 0-10% loss

    def get_edge_cost(self, u, v):
        """Calculates the current cost of an edge."""
        base = self.graph[u][v]['base_weight']
        loss = self.graph[u][v]['loss']
        
        if self.mode == 'static':
            return base + loss
        else:
            # DYNAMIC MODE: Weights fluctuate using a sine wave + noise
            # This simulates periodic network congestion
            fluctuation = np.sin(self.time_step * 0.5) + 1.0 
            noise = random.uniform(0, 2)
            return (base * fluctuation) + (loss * 10) + noise

    def step(self, u, v):
        """Moves the packet from u to v and returns cost."""
        self.time_step += 1
        cost = self. Ast_edge_cost(u, v) if hasattr(self, 'Ast_edge_cost') else self.get_edge_cost(u, v)
        return v, cost

class QLearningAgent:
    def __init__(self, nodes, adj_list, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.nodes = nodes
        self.adj_list = adj_list  # Dictionary of {node: [neighbors]}
        self.alpha = alpha        # Learning rate
        self.gamma = gamma        # Discount factor
        self.epsilon = epsilon    # Exploration rate
        
        # Q-Table: Rows = Current Node, Columns = Next Node
        self.q_table = np.zeros((len(nodes), len(nodes)))

    def choose_action(self, u):
        """Epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            # Explore: pick a random neighbor
            return random.choice(self.adj_list[u])
        else:
            # Exploit: pick neighbor with highest Q-value
            neighbors = self.adj_list[u]
            qs = self.q_table[u, neighbors]
            return neighbors[np.argmax(qs)]

    def update(self, u, v, reward, next_node_neighbors):
        """Bellman Equation Update."""
        # Reward is negative cost (since we want to minimize cost)
        max_future_q = np.max(self.q_lan_table[v, next_node_neighbors]) if next_node_neighbors else 0
        
        # Standard Q-learning update rule
        self.q_table[u, v] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[u, v])

def run_simulation(mode='static', episodes=100):
    env = NetworkEnv(num_nodes=25, edge_prob=0.25, mode=mode)
    nodes = list(env.graph.nodes())
    adj_list = {n: list(env.graph.neighbors(n)) for n in nodes}
    
    # Initialize Agent
    agent = QLearningAgent(nodes, adj_list)
    
    dijkstra_costs = []
    rl_costs = []
    convergence_history = []

    source = 0
    target = len(nodes) - 1

    for ep in range(episodes):
        # --- 1. Dijkstra's Path Cost (The Baseline) ---
        # We must update weights for Dijkstra every time to reflect dynamic changes
        temp_graph = env.graph.copy()
        for u, v in temp_graph.edges():
            temp_graph[u][v]['current_cost'] = env.get_edge_cost(u, v)
        
        try:
            path_d = nx.dijkstra_path(temp_graph, source, target, weight='current_cost')
            cost_d = sum(temp_graph[path_d[i]][path_d[i+1]]['current_cost'] for i in range(len(path_d)-1))
            dijkstra_costs.append(cost_d)
        except nx.NetworkXNoPath:
            dijkstra_costs.append(float('inf'))

        # --- 2. Q-Learning Agent Path Cost ---
        current_node = source
        path_rl = [source]
        total_rl_cost = 0
        steps = 0
        
        while current_node != target and steps < 50: # Prevent infinite loops in bad policies
            neighbors = adj_list[current_node]
            next_node = agent.choose_action(current_node)
            
            cost = env.get_edge_cost(current_node, next_node)
            reward = -cost  # Minimizing cost is maximizing negative cost
            
            # Update Q-Table
            agent.update(current_node, next_node, reward, adj_list[next_node])
            
            total_rl_cost += cost
            current_node = next_node
            path_rl.append(current_node)
            steps += 1
            
        rl_costs.append(total_rl_cost)
        convergence_history.append(np.mean(rl_costs[-20:])) # Moving average

    return dijkstra_costs, rl_costs, convergence_history

# ==========================================
# EXECUTION AND VISUALIZATION
# ==========================================

print("Running Static Simulation...")
d_static, r_static, conv_static = run_imulation(mode='static', episodes=200)

print("Running Dynamic Simulation (The Graduate Twist)...")
d_dynamic, r_dynamic, conv_dynamic = run_imulation(mode='dynamic', episodes=200)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Static Comparison
axes[0].plot(d_static, label='Dijkstra (Optimal)', color='blue', linestyle='--')
axes[0].plot(r_static, label='Q-Learning (Learned)', color='red', alpha=0.7)
axes[0].set_title("Static Environment: Dijkstra vs Q-Learning")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Path Cost")
axes[0].legend()

# Plot 2: Dynamic Comparison
axes[1].plot(d_dynamic, label='Dijkstra (Reactive)', color='blue', linestyle='--')
axes[1].plot(r_dynamic, label='Q-Learning (Adaptive)', color='red', alpha=0.7)
axes[1].set_title("Dynamic Environment: Dijkstra vs Q-Learning")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Path Cost")
axes[1].legend()

plt.tight_layout()
plt.show()
