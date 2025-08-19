# spice_runner.py
import os
import re
import sys
import tempfile
import subprocess
from typing import Dict, Tuple, List, Optional

import numpy as np

def _is_float_token(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _parse_wrdata_table(path: str, vector_names: Optional[List[str]] = None):
    """
    Parse ngspice 'wrdata' output which is whitespace-separated 'time/value' pairs per vector.
    Example row for 3 vectors: [time,time] [time,v(1)] [time,v(2)]  -> 6 numbers.
    Returns: time (np.ndarray), signals (dict name->np.ndarray)
    """
    rows: List[List[float]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip comment / header lines
            if line.lstrip().startswith("*") or not _is_float_token(line.split()[0]):
                continue
            parts = re.split(r"\s+", line)
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                # If first token was an index (integer), try dropping it
                if _is_float_token(parts[0]) and float(parts[0]).is_integer():
                    try:
                        rows.append([float(x) for x in parts[1:]])
                    except Exception:
                        raise RuntimeError(f"Unexpected numeric format in line:\n{line}")
                else:
                    raise RuntimeError(f"Unexpected non-numeric line in wrdata:\n{line}")

    if not rows:
        raise RuntimeError("wrdata file is empty or contains no numeric rows.")

    arr = np.asarray(rows, dtype=float)  # shape: (N, C)
    C = arr.shape[1]

    # If odd number of columns, first may be an index column; drop it.
    if C % 2 == 1:
        if np.allclose(arr[:, 0], np.arange(arr.shape[0])):
            arr = arr[:, 1:]
            C -= 1
        else:
            raise RuntimeError(f"Unexpected wrdata column count {C}. Cannot pair time/value columns.")

    # Now we expect pairs: [time,time] [time,val1] [time,val2] ...
    pairs = C // 2
    if pairs < 2:
        # At minimum, we expect the [time,time] pair plus one data pair
        raise RuntimeError(f"Not enough time/value pairs in wrdata (columns={C}).")

    time = arr[:, 0]  # master time

    signals: Dict[str, np.ndarray] = {}
    # For j=1..pairs-1, pick the 'value' column at 2*j+1
    for j in range(1, pairs):
        val_col = 2 * j + 1
        name = (
            vector_names[j - 1]
            if vector_names and (j - 1) < len(vector_names)
            else f"signal_{j}"
        )
        signals[name] = arr[:, val_col]

    return time, signals

def run_spice_netlist(
    netlist_text: str,
    ngspice_path: Optional[str] = None,
    vector_names: Optional[List[str]] = None,
) -> Tuple[List[float], Dict[str, List[float]], str]:
    """
    Run ngspice in batch mode on the given netlist text.
    - ngspice_path: optional full path to ngspice or ngspice_con.exe
    - vector_names: names for the data vectors passed to 'wrdata' after 'time'
                    e.g., if wrdata has 'time v(1) v(2) v(3)', pass ['v(1)','v(2)','v(3)']
    Returns: time list, dict of signal->list, and the working folder path where results.csv lives.
    """
    if ngspice_path is None:
        # Prefer console build if present on Windows
        ngspice_path = "ngspice.exe" if sys.platform.startswith("win") else "ngspice"
        # Some installations expose ngspice_con.exe
        alt = "ngspice_con.exe"
        try:
            subprocess.run([ngspice_path, "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            ngspice_path = alt

    workdir = tempfile.mkdtemp(prefix="kkf_spice_")
    ckt_path = os.path.join(workdir, "ckt.cir")
    csv_path = os.path.join(workdir, "results.csv")

    with open(ckt_path, "w", newline="\n") as f:
        f.write(netlist_text)

    proc = subprocess.run(
        [ngspice_path, "-b", ckt_path],
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ngspice failed.\nSTDOUT:\n{}\nSTDERR:\n{}".format(proc.stdout, proc.stderr)
        )

    if not os.path.exists(csv_path):
        raise RuntimeError("ngspice ran but 'results.csv' was not created. Check .control/wrdata.")

    t, sigs = _parse_wrdata_table(csv_path, vector_names=vector_names)
    # Convert to plain lists for easy printing/plotting
    t_list = t.tolist()
    sigs_list = {k: v.tolist() for k, v in sigs.items()}
    return t_list, sigs_list, workdir
