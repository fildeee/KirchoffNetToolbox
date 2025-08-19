# demo_diodes.py
# Minimal diode-based ngspice demo: build netlist -> run ngspice (batch) -> print & plot results.

from spice_adapter_diodes import build_spice_netlist_diodes
from spice_runner import run_spice_netlist

# >>> UPDATE THIS PATH ONLY IF YOUR ngspice.exe LIVES SOMEWHERE ELSE <<<
NGSPICE_PATH = r"C:\ProgramData\chocolatey\bin\ngspice.exe"


def main():
    # Nodes (0 is ground)
    nodes = [0, 1, 2, 3]

    # Directed edges: each becomes a diode anode=s, cathode=d (one-way conduction s -> d)
    edges = [(1, 2), (2, 3)]

    # Optional: tweak diode on (2,3) to be a little "stronger"
    diode_params = {
        (2, 3): {"IS": 5e-14, "N": 1.7, "RS": 0.005, "CJO": 1.5e-12, "M": 0.33, "TT": 3e-9}
    }

    # Input: constant DC drive at node 1 (acts like an input neuron voltage)
    inputs_dc = {1: 0.8}  # 0.8 V

    # Which node voltages to record
    outputs = [1, 2, 3]

    # Build the SPICE netlist (adds a small capacitor to ground at each non-ground node)
    netlist = build_spice_netlist_diodes(
        nodes=nodes,
        edges=edges,
        diode_params_per_edge=diode_params,
        bidirectional_edges=[],      # e.g. [(1,2)] if you want current to flow both ways
        inputs_dc=inputs_dc,
        outputs=outputs,
        C0=1e-9,                     # per-node capacitor to ground (stability / dynamics)
        T=2e-3,                      # total simulation time (2 ms)
        dt=1e-6,                     # timestep (1 µs)
        title="KKF Diode Network (minimal)"
    )

    # Run ngspice in batch mode and parse results
    t, signals, folder = run_spice_netlist(
        netlist,
        ngspice_path=NGSPICE_PATH,                      # <-- your ngspice.exe
        vector_names=[f"v({n})" for n in outputs],      # nice labels in the result dict
    )

    # Console summary Tom can read quickly
    print(f"Results folder: {folder}   (contains ckt.cir, results.csv, stdout.log)")
    for name in [f"v({n})" for n in outputs]:
        y = signals[name]
        print(f"{name}: start={y[0]:.4f} V   end={y[-1]:.4f} V")

    # Optional: quick plot (if matplotlib is installed)
    try:
        import matplotlib.pyplot as plt
        for name in [f"v({n})" for n in outputs]:
            plt.plot(t, signals[name], label=name)
        plt.xlabel("time (s)")
        plt.ylabel("voltage (V)")
        plt.title("Diode network transient")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
