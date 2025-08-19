# spice_adapter_diodes.py
from typing import Dict, Iterable, List, Tuple, Optional

def build_spice_netlist_diodes(
    nodes: Iterable[int],
    edges: List[Tuple[int, int]],
    *,
    diode_params_per_edge: Optional[Dict[Tuple[int,int], Dict[str, float]]] = None,
    bidirectional_edges: Optional[List[Tuple[int,int]]] = None,
    inputs_dc: Optional[Dict[int, float]] = None,  # node -> volts
    outputs: Optional[List[int]] = None,
    C0: float = 1e-9,
    T: float = 1e-3,
    dt: float = 1e-6,
    title: str = "KKF / Diode demo"
) -> str:
    """
    nodes: includes 0 (ground)
    edges: list of directed edges (s,d) to realize as diodes anode=s, cathode=d
    diode_params_per_edge: optional per-edge SPICE diode params (IS,N,RS,CJO,M,TT)
    bidirectional_edges: edges to realize as two antiparallel diodes (flow both ways)
    inputs_dc: constant DC sources at nodes (e.g., {1:0.7})
    outputs: which node voltages to export; default = all non-ground nodes
    """
    nodes = sorted(set(nodes))
    if 0 not in nodes:
        nodes = [0] + nodes
    ng_nodes = [n for n in nodes if n != 0]
    outputs = outputs or ng_nodes
    inputs_dc = inputs_dc or {}
    diode_params_per_edge = diode_params_per_edge or {}
    bidirectional_edges = set(bidirectional_edges or [])

    lines = []
    lines.append(f"* {title}")
    lines.append(".options TEMP=27")
    lines.append(f".param C0={C0:.12g}")
    lines.append("")

    # Default diode model if a specific edge doesn't override
    # Feel free to tweak these if you need stronger/weaker conduction
    lines.append(".model DDEF D(IS=1e-14 N=1.9 RS=10m CJO=2e-12 M=0.33 TT=5e-9)")
    lines.append("")

    # Per-node caps for stability / KKF-like dynamics
    for n in ng_nodes:
        lines.append(f"C_{n} {n} 0 {{C0}}")

    lines.append("")

    # Inputs as DC sources (cleaner than only .ic for diode circuits)
    for n, v in sorted(inputs_dc.items()):
        if n == 0:
            continue
        lines.append(f"V_{n} {n} 0 {v}")

    lines.append("")

    # Emit diode elements
    # If an edge has custom params, create a dedicated .model for it
    model_count = 0
    edge_to_model = {}
    for (s, d) in edges:
        pars = diode_params_per_edge.get((s, d))
        if pars:
            model_count += 1
            mname = f"DMOD{model_count}"
            # Build the .model line with provided fields (fallbacks if missing)
            IS = pars.get("IS", 1e-14); N = pars.get("N", 1.9); RS = pars.get("RS", 0.01)
            CJO = pars.get("CJO", 2e-12); M = pars.get("M", 0.33); TT = pars.get("TT", 5e-9)
            lines.append(f".model {mname} D(IS={IS} N={N} RS={RS} CJO={CJO} M={M} TT={TT})")
            edge_to_model[(s, d)] = mname
    if model_count:
        lines.append("")

    # Place diodes. Forward-only by default; antiparallel if requested.
    didx = 0
    for (s, d) in edges:
        mname = edge_to_model.get((s, d), "DDEF")
        didx += 1
        lines.append(f"D{didx} {s} {d} {mname}")
        if (s, d) in bidirectional_edges:
            didx += 1
            # opposite orientation for reverse flow
            lines.append(f"D{didx} {d} {s} {mname}")

    lines.append("")
    # Control: transient + CSV dump
    probes = " ".join([f"v({n})" for n in outputs])
    lines.append(".control")
    lines.append("set filetype=ascii")
    lines.append(f"tran {dt} {T} uic")
    lines.append(f"wrdata results.csv time {probes}")
    lines.append("quit")
    lines.append(".endc")
    lines.append(".end")

    return "\n".join(lines)
