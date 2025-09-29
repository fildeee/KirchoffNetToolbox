# spice_runner.py
import os
import re
import sys
import shutil
import tempfile
import subprocess
from typing import Dict, Tuple, List, Optional

import numpy as np


# ---------------------------
# Helpers
# ---------------------------
def _is_float_token(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _pick_ngspice(ngspice_path: Optional[str]) -> str:
    """
    Choose an ngspice executable sensibly across platforms.
    """
    if ngspice_path:
        return ngspice_path

    if sys.platform.startswith("win"):
        # Prefer console build if present; otherwise fall back.
        for cand in ("ngspice_con.exe", "ngspice.exe"):
            p = shutil.which(cand)
            if p:
                return p
        return "ngspice.exe"
    else:
        return shutil.which("ngspice") or "ngspice"


def _win_hide_startup():
    """
    Return (startupinfo, creationflags) to hide spawned windows on Windows.
    No-op on other OSes.
    """
    if not sys.platform.startswith("win"):
        return None, 0
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return si, subprocess.CREATE_NO_WINDOW


# ---------------------------
# wrdata (.tran) CSV parser
# ---------------------------
def _parse_wrdata_table(path: str, vector_names: Optional[List[str]] = None):
    """
    Parse ngspice 'wrdata' output which is whitespace-separated 'time/value' pairs per vector.
    Example row for 3 vectors: [time,time] [time,v(1)] [time,v(2)]  -> 6 numbers.
    Returns: time (np.ndarray), signals (dict name->np.ndarray)
    """
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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


# ---------------------------
# Transient runner (.tran + wrdata)
# ---------------------------
def run_spice_netlist(
    netlist_text: str,
    ngspice_path: Optional[str] = None,
    vector_names: Optional[List[str]] = None,
) -> Tuple[List[float], Dict[str, List[float]], str]:
    """
    Run ngspice in batch mode on the given netlist text (transient).
    - ngspice_path: optional full path to ngspice or ngspice_con.exe
    - vector_names: names for the data vectors passed to 'wrdata' after 'time'
                    e.g., if wrdata has 'time v(1) v(2) v(3)', pass ['v(1)','v(2)','v(3)']
    Returns: time list, dict of signal->list, and the working folder path where results.csv lives.
    """
    ng = _pick_ngspice(ngspice_path)

    workdir = tempfile.mkdtemp(prefix="kkf_spice_")
    ckt_path = os.path.join(workdir, "ckt.cir")
    csv_path = os.path.join(workdir, "results.csv")
    log_path = os.path.join(workdir, "stdout.log")

    # Write netlist as UTF-8 to avoid Windows encoding issues
    with open(ckt_path, "w", newline="\n", encoding="utf-8") as f:
        f.write(netlist_text)

    si, cf = _win_hide_startup()

    # Run ngspice in batch; capture output in a log file
    proc = subprocess.run(
        [ng, "-b", "-o", log_path, ckt_path],
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=si,
        creationflags=cf,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ngspice failed.\nSTDOUT:\n{}\nSTDERR:\n{}\nSee log: {}".format(proc.stdout, proc.stderr, log_path)
        )

    if not os.path.exists(csv_path):
        raise RuntimeError("ngspice ran but 'results.csv' was not created. Check your .control/wrdata block. See log: {}".format(log_path))

    t, sigs = _parse_wrdata_table(csv_path, vector_names=vector_names)
    # Convert to plain lists for easy printing/plotting
    t_list = t.tolist()
    sigs_list = {k: v.tolist() for k, v in sigs.items()}
    return t_list, sigs_list, workdir


# ---------------------------
# DC runner (.op) — optional, for Tom's DC-only milestone
# ---------------------------
def run_spice_netlist_dc(
    netlist_text: str,
    ngspice_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run ngspice in batch mode on a DC .op netlist and parse node voltages
    printed by 'print v(n) ...' into stdout.log.

    Returns a dict like {'v(1)': 0.8, 'v(2)': 0.73, ...} (all keys in lower case).
    """
    ng = _pick_ngspice(ngspice_path)

    workdir = tempfile.mkdtemp(prefix="kkf_spice_dc_")
    ckt_path = os.path.join(workdir, "ckt.cir")
    log_path = os.path.join(workdir, "stdout.log")

    with open(ckt_path, "w", newline="\n", encoding="utf-8") as f:
        f.write(netlist_text)

    si, cf = _win_hide_startup()

    p = subprocess.run(
        [ng, "-b", "-o", log_path, ckt_path],
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=si,
        creationflags=cf,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"ngspice failed.\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}\nSee: {log_path}"
        )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    # Prefer the block after our marker if present
    m = re.search(r"--- DC OP RESULTS ---(.+?)\n\n", txt, flags=re.S)
    block = m.group(1) if m else txt

    # Parse lines like: v(2) = 7.123e-01
    vals: Dict[str, float] = {}
    for name, val in re.findall(r"(v\([^)]+\))\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", block):
        vals[name.lower()] = float(val)

    if not vals:
        # As a fallback, search the whole log
        for name, val in re.findall(r"(v\([^)]+\))\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", txt):
            vals[name.lower()] = float(val)

    if not vals:
        raise RuntimeError(f"Could not find DC node voltages in {log_path}")

    return vals
