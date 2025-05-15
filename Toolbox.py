import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import networkx as nx
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import numpy as np


## MAIN CLASS
class KirchhoffNet(nn.Module):
    def __init__(self, graph: nx.DiGraph, theta: float = 1.0):
        super(KirchhoffNet, self).__init__()
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.theta = theta

        self.edges = list(graph.edges(data=True))
        self.theta_sd_1 = nn.Parameter(torch.randn(len(self.edges)) * 0.1)
        self.theta_sd_2 = nn.Parameter(torch.randn(len(self.edges)) * 0.1)
    
    #EQ 7
    def g(self, vs, vd, theta_sd_1, theta_sd_2):
        return torch.relu(theta_sd_1 * (vs - vd) + theta_sd_2)

    #EQ 10
    def forward(self, t, v):
        dv_dt = torch.zeros_like(v)

        for node_idx in range(self.num_nodes):
            incoming = 0
            outgoing = 0
            for edge_idx, (src, dst, data) in enumerate(self.edges):
                theta1 = self.theta_sd_1[edge_idx]
                theta2 = self.theta_sd_2[edge_idx]

                conductance = 1.0 / data['resistance'] if data['resistance'] != 0 else 0.0

                if dst == node_idx:
                    incoming += conductance * self.g(v[src], v[dst], theta1, theta2)
                if src == node_idx:
                    outgoing += conductance * self.g(v[src], v[dst], theta1, theta2)

            dv_dt[node_idx] = (incoming - outgoing) / self.theta

        return dv_dt

## CONDUCTANCE MATRIX
def get_conductance_matrix(graph: nx.DiGraph) -> np.ndarray:
    n = graph.number_of_nodes()
    G = np.zeros((n, n))

    for u, v, data in graph.edges(data=True):
        conductance = 1.0 / data['resistance'] if data['resistance'] != 0 else 0.0
        G[u][v] -= conductance
        G[v][u] -= conductance
        G[u][u] += conductance
        G[v][v] += conductance

    return G

## NETLIST PARSER ## NETWORK X IMPLEMENTATION
def parse_netlist(filepath: str):
    graph = nx.DiGraph()
    voltages = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            parts = line.split()
            if parts[0].startswith('R') and len(parts) == 4:
                _, node1, node2, resistance = parts
                n1, n2 = int(node1), int(node2)
                r = float(resistance)
                graph.add_node(n1)
                graph.add_node(n2)
                graph.add_edge(n1, n2, resistance=r)
                graph.add_edge(n2, n1, resistance=r)  # Add reverse edge
            elif parts[0].startswith('V') and len(parts) == 4:
                _, node0, node1, value = parts
                voltages[int(node1)] = float(value)
    return graph, voltages

## SIMULATION
def simulate_kirchhoffnet(graph: nx.DiGraph, initial_voltages: torch.Tensor, T=10.0, steps=100):
    model = KirchhoffNet(graph)
    t = torch.linspace(0, T, steps)

    with torch.no_grad():
        voltages = odeint(model, initial_voltages, t, method='dopri5')

    plt.figure(figsize=(10, 6))
    for i in range(graph.number_of_nodes()):
        plt.plot(t, voltages[:, i].numpy(), label=f'v{i}(t)')

    plt.title('KirchhoffNet Node Voltages Over Time')
    plt.xlabel('Time (t)')
    plt.ylabel('Voltage')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Initial voltages: {initial_voltages}")
    print(f"Final voltages: {voltages[-1]}")

    G_matrix = get_conductance_matrix(graph)
    print("Conductance Matrix (Kirchhoff Matrix):")
    print(G_matrix)

## GUI GUI GUI
def launch_gui():
    root = tk.Tk()
    root.withdraw()

    use_netlist = messagebox.askyesno("Use Netlist", "Do you want to load circuit from a netlist file?")

    if use_netlist:
        filepath = filedialog.askopenfilename(title="Select Netlist File", filetypes=[("Text files", "*.txt")])
        if not filepath:
            messagebox.showerror("Error", "No file selected.")
            return
        G, voltage_dict = parse_netlist(filepath)
        num_nodes = G.number_of_nodes()
        initial_voltages = torch.zeros(num_nodes)
        for i in range(num_nodes):
            initial_voltages[i] = voltage_dict.get(i, 0.0)
    else:
        num_nodes = simpledialog.askinteger("KirchhoffNet Setup", "Enter number of nodes:")
        if num_nodes is None or num_nodes < 1:
            messagebox.showerror("Error", "Invalid number of nodes.")
            return
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))

        while True:
            edge_str = simpledialog.askstring("Add Edge", "Enter edge as 'src dst' e.g. 0 1 (leave empty to finish):")
            if not edge_str:
                break
            try:
                src, dst = map(int, edge_str.strip().split())
                if 0 <= src < num_nodes and 0 <= dst < num_nodes and src != dst:
                    resistance = simpledialog.askfloat("Edge Resistance", f"Resistance for edge {src}->{dst} (Ohms):")
                    if resistance is None:
                        messagebox.showerror("Error", "Input cancelled.")
                        return
                    G.add_edge(src, dst, resistance=resistance)
                else:
                    messagebox.showerror("Error", f"Invalid edge: {src} -> {dst}")
            except Exception as e:
                messagebox.showerror("Error", f"Please enter valid integers (e.g., '1 3'). Error: {str(e)}")

        if G.number_of_edges() == 0:
            messagebox.showerror("Error", "No edges entered.")
            return

        initial_voltages = []
        for i in range(G.number_of_nodes()):
            voltage = simpledialog.askfloat("Initial Voltage", f"Voltage at node {i}:")
            if voltage is None:
                messagebox.showerror("Error", "Input cancelled.")
                return
            initial_voltages.append(voltage)
        initial_voltages = torch.tensor(initial_voltages)

    simulate_kirchhoffnet(G, initial_voltages)

## MAIN
if __name__ == "__main__":
    launch_gui()
