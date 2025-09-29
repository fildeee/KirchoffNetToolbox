import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import networkx as nx
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import numpy as np

from spice_adapter_diodes import build_spice_netlist_diodes
from spice_runner import run_spice_netlist

# NEW: DC adapter + DC runner (for .op with diode models)
from spice_adapter_dc import build_spice_netlist_diodes_dc
from spice_runner import run_spice_netlist_dc


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

## --- NEW: SPICE DC (diodes) helper ---
def run_spice_dc_from_graph(G, initial_voltages, parent):
    # Optional: user-provided model card and/or include library
    model_card = simpledialog.askstring(
        "Optional .model card",
        "Paste a .model line for the diode (or leave blank for default):\n"
        "e.g. .model DDEF D(IS=1e-14 N=1.9 RS=0.01)",
        parent=parent
    )
    includes = []
    if messagebox.askyesno("Include library?", "Do you want to .include a model library file?", parent=parent):
        p = filedialog.askopenfilename(title="Pick model library (.lib/.cir)", parent=parent)
        if p: includes.append(p)

    nodes = sorted(G.nodes())
    if 0 not in nodes:
        nodes = [0] + nodes
    outputs = [n for n in nodes if n != 0]

    try:
        init_list = initial_voltages.tolist()
    except Exception:
        init_list = list(initial_voltages)

    # Map any non-zero initial voltage to a DC source; fallback to first non-ground node
    inputs_dc = {i: float(v) for i, v in enumerate(init_list) if i != 0 and abs(float(v)) > 1e-12}
    if not inputs_dc and outputs:
        inputs_dc = {outputs[0]: 0.8}

    netlist = build_spice_netlist_diodes_dc(
        nodes=nodes,
        edges=list(G.edges()),            # diode: anode=src, cathode=dst
        inputs_dc=inputs_dc,
        outputs=outputs,
        add_leaks=True, r_leak=1e9,
        include_files=includes,
        model_cards=[model_card] if model_card else [],
        title="Toolbox -> SPICE DC (diodes)"
    )
    vals = run_spice_netlist_dc(
        netlist,
        ngspice_path=r"C:\ProgramData\chocolatey\bin\ngspice.exe"
    )
    msg = "\n".join([f"{k} = {v:.6f} V" for k, v in sorted(vals.items())])
    messagebox.showinfo("DC Operating Point", msg or "No voltages parsed.", parent=parent)

## GUI GUI GUI
def launch_gui():
    root = tk.Tk()
    # Make dialogs reliably visible
    root.title("KirchhoffNet Toolbox")
    root.geometry("360x120")
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))

    use_netlist = messagebox.askyesno("Use Netlist", "Do you want to load circuit from a netlist file?", parent=root)

    if use_netlist:
        filepath = filedialog.askopenfilename(
            title="Select Netlist File",
            filetypes=[("Text files", "*.txt;*.cir;*.sp"), ("All files", "*.*")],
            parent=root
        )
        if not filepath:
            messagebox.showerror("Error", "No file selected.", parent=root)
            root.destroy()
            return
        G, voltage_dict = parse_netlist(filepath)
        num_nodes = G.number_of_nodes()
        initial_voltages = torch.zeros(num_nodes)
        for i in range(num_nodes):
            initial_voltages[i] = voltage_dict.get(i, 0.0)
    else:
        num_nodes = simpledialog.askinteger("KirchhoffNet Setup", "Enter number of nodes:", parent=root)
        if num_nodes is None or num_nodes < 1:
            messagebox.showerror("Error", "Invalid number of nodes.", parent=root)
            root.destroy()
            return
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))

        while True:
            edge_str = simpledialog.askstring(
                "Add Edge",
                "Enter edge as 'src dst' e.g. 0 1 (leave empty to finish):",
                parent=root
            )
            if not edge_str:
                break
            try:
                src, dst = map(int, edge_str.strip().split())
                if 0 <= src < num_nodes and 0 <= dst < num_nodes and src != dst:
                    resistance = simpledialog.askfloat(
                        "Edge Resistance",
                        f"Resistance for edge {src}->{dst} (Ohms):",
                        parent=root
                    )
                    if resistance is None:
                        messagebox.showerror("Error", "Input cancelled.", parent=root)
                        root.destroy()
                        return
                    G.add_edge(src, dst, resistance=resistance)
                else:
                    messagebox.showerror("Error", f"Invalid edge: {src} -> {dst}", parent=root)
            except Exception as e:
                messagebox.showerror("Error", f"Please enter valid integers (e.g., '1 3'). Error: {str(e)}", parent=root)

        if G.number_of_edges() == 0:
            messagebox.showerror("Error", "No edges entered.", parent=root)
            root.destroy()
            return

        initial_voltages = []
        for i in range(G.number_of_nodes()):
            voltage = simpledialog.askfloat("Initial Voltage", f"Voltage at node {i}:", parent=root)
            if voltage is None:
                messagebox.showerror("Error", "Input cancelled.", parent=root)
                root.destroy()
                return
            initial_voltages.append(voltage)
        initial_voltages = torch.tensor(initial_voltages)

    # Tom's request: offer DC .op with diode models
    if messagebox.askyesno(
        "Run SPICE DC (diodes)?",
        "Yes = DC .op with diode models\nNo = run your Python ODE",
        parent=root
    ):
        run_spice_dc_from_graph(G, initial_voltages, parent=root)
        root.destroy()
        return

    # Fall back to your existing Python ODE simulation
    simulate_kirchhoffnet(G, initial_voltages)
    root.destroy()

## Hook (transient SPICE; kept for future use)
def run_spice_for_graph(G, *,
                        inputs_dc=None,
                        outputs=None,
                        C0=1e-9, T=2e-3, dt=1e-6,
                        ngspice_path=r"C:\ProgramData\chocolatey\bin\ngspice.exe"):
    """
    G: your NetworkX graph. Node 0 is ground (if missing, we'll add it).
    inputs_dc: dict {node: volts} to drive the circuit (e.g. {1: 0.8})
    outputs: list of node ids to record (default: all non-ground nodes)
    """
    import networkx as nx

    nodes = sorted(G.nodes())
    if 0 not in nodes:
        nodes = [0] + nodes  # enforce a ground node

    # IMPORTANT: For diode simulation, edges should be **directed** as you intend current to flow.
    edges = list(G.edges())

    if outputs is None:
        outputs = [n for n in nodes if n != 0]

    if inputs_dc is None:
        # default: drive the first non-ground node at 0.8 V if nothing is provided
        nz = [n for n in nodes if n != 0]
        inputs_dc = {nz[0]: 0.8} if nz else {}

    netlist = build_spice_netlist_diodes(
        nodes=nodes,
        edges=edges,
        inputs_dc=inputs_dc,
        outputs=outputs,
        C0=C0, T=T, dt=dt,
        title="Toolbox -> SPICE (diodes)"  # ASCII-only to avoid encoding issues
    )

    t, signals, folder = run_spice_netlist(
        netlist,
        ngspice_path=ngspice_path,
        vector_names=[f"v({n})" for n in outputs],
    )
    return t, signals, folder


## MAIN
if __name__ == "__main__":
    launch_gui()
