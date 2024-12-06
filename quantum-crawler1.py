import requests
from bs4 import BeautifulSoup
from collections import deque
import re
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
# For the simple perceptron
import torch
import torch.nn as nn
import torch.optim as optim

#####################################
# 1. Web Crawling and Graph Building #
#####################################

def get_links(url):
    """Retrieve links from the given URL, return as a list of absolute URLs."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        base = re.match(r'^https?://[^/]+', url)
        base_url = base.group(0) if base else url
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('http'):
                full_link = href
            else:
                full_link = base_url + '/' + href.lstrip('/')
            links.append(full_link)
        return links
    except Exception:
        return []

def build_graph(start_url, max_nodes=50):
    """Perform a breadth-first crawl from start_url and build a graph up to max_nodes."""
    visited = set()
    queue = deque([start_url])
    visited.add(start_url)

    nodes = [start_url]  # store in a list
    edges = []

    while queue and len(nodes) < max_nodes:
        current = queue.popleft()
        neighbors = get_links(current)
        for n in neighbors[:5]:
            if n not in visited and len(nodes) < max_nodes:
                visited.add(n)
                queue.append(n)
                nodes.append(n)
            if n in visited:
                edges.append((current, n))

    N = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    adjacency = np.zeros((N, N))
    for (u, v) in edges:
        if u in node_index and v in node_index:
            adjacency[node_index[u], node_index[v]] = 1

    # Normalize rows to get a stochastic matrix
    for i in range(N):
        row_sum = np.sum(adjacency[i])
        if row_sum > 0:
            adjacency[i] = adjacency[i] / row_sum

    return nodes, adjacency

###################################
# 2. Quantum Simulation of Motion #
###################################

def quantum_walk_step(adjacency, num_steps=2):
    n_qubits = len(adjacency)
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Initialize your circuit here, for example:
    qc.h(range(n_qubits))  # Example initialization

    # Apply operations for quantum walk based on your adjacency matrix
    for _ in range(num_steps):
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if adjacency[i][j]:
                    qc.cx(i, j)  # Apply controlled-X gate if there's an edge

    # Save the statevector
    qc.save_statevector()

    # Setup the simulator
    simulator = AerSimulator(method='statevector')
    
    # Execute the circuit using the new 'run' method
    job = simulator.run(qc)
    result = job.result()

    # Retrieve the statevector
    state = result.get_statevector()

    return state

    
#################################
# 3. Feed into a Simple Perceptron
#################################

class SimplePerceptron(nn.Module):
    def __init__(self, input_size, hidden_size=16, output_size=2):
        super(SimplePerceptron, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    start_url = "https://www.tsubasakato.com"
    nodes, adjacency = build_graph(start_url, max_nodes=50)
    print("Crawled nodes:", len(nodes))

    # Perform pseudo quantum walk
    probs = quantum_walk_step(adjacency, num_steps=2)
    print("Quantum probabilities over nodes:", probs)

    # Feed into a perceptron
    input_size = len(probs)
    model = SimplePerceptron(input_size=input_size)

    x = torch.tensor(probs, dtype=torch.float32).unsqueeze(0)
    output = model(x)
    print("Perceptron output:", output.detach().numpy())

    # Example training step
    target = torch.tensor([1], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Training step done. Loss:", loss.item())
