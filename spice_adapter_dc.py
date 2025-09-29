# spice_adapter_dc.py
from typing import Dict, Iterable, List, Tuple, Optional

def build_spice_netlist_diodes_dc(
    nodes: Iterable[int],
    edges: List[Tuple[int, int]],
    *,
    inputs_dc: Optional[Dict[int, float]] = None,        # node -> volts
    outputs: Optional[List[int]] = None,                 # nodes to report
    add_leaks: bool = True,
    r_leak: float = 1e9,                                 # 1 GΩ leaks prevent floating nodes
    include_files: Optional[List[str]] = None,           # e.g. ["C:/models/diodes.lib"]
    model_cards: Optional[List[str]] = None,             # e.g. [".model DDEF D(IS=1e-14 N=1.9 RS=0.01)"]
    model_per_edge: Optional[Dict[Tuple[int,int], str]] = None,  # {(u,v): "DFAST"}
    default_model: str = "DDEF",
    default_model_card: str = ".model DDEF D(IS=1e-14 N=1.9 RS=0.01)",
    title: str = "KKF DC (diodes, .op only)"
) -> str:
    nodes = sorted(set(nodes))
    if 0 not in nodes:
        nodes = [0] + nodes
    ng_nodes = [n for n in nodes if n != 0]
    outputs = outputs or ng_nodes
    inputs_dc = inputs_dc or {}
    include_files = include_files or []
    model_cards = model_cards or []
    model_per_edge = model_per_edge or {}

    L: List[str] = []
    L.append(f"* {title}")
    # optional external libraries
    for p in include_files:
        L.append(f'.include "{p}"')
    # user-supplied model cards
    for card in model_cards:
        L.append(card)
    # ensure a default diode model exists
    if not any(c.lower().startswith(".model") and c.split()[1] == default_model for c in model_cards):
        L.append(default_model_card)
    L.append("")

    # DC sources
    for n, v in sorted(inputs_dc.items()):
        if n != 0:
            L.append(f"V{n} {n} 0 {v}")

    # optional mega-ohm leaks so .op has no floating nodes
    if add_leaks:
        for n in ng_nodes:
            L.append(f"RLEAK_{n} {n} 0 {r_leak:g}")

    L.append("")
    # diode instances: anode = src, cathode = dst
    didx = 0
    for (s, d) in edges:
        didx += 1
        mname = model_per_edge.get((s, d), default_model)
        L.append(f"D{didx} {s} {d} {mname}")

    L.append("")
    # DC operating point and print node voltages
    L.append(".control")
    L.append("set noaskquit")
    L.append("op")
    L.append("echo --- DC OP RESULTS ---")
    L.append("print " + " ".join([f"v({n})" for n in outputs]))
    L.append("quit")
    L.append(".endc")
    L.append(".end")
    return "\n".join(L)
