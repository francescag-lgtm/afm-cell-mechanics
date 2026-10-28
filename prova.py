"""
prova.py – Utility functions not related to fitting.

Contains:
  - file/folder discovery helpers
  - GUI folder picker
  - signal pre-processing (PoC alignment, viscous-drag correction)
"""

import os
import re

import numpy as np

SUPPORTED_FORCE_EXTENSIONS = (
    ".jpk-force",
    ".jpk-force-map",
    ".jpk-qi-data",
    ".ibw",
    ".nid",
    ".spm",
)


def _natural_sort_key(value):
    """Sort helper that keeps `cell2` before `cell10`."""
    text = os.path.normpath(str(value)).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def select_data_folder_gui(initial_dir=None, title="Seleziona la cartella dati AFM"):
    """Open a graphical folder picker and return the selected directory path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Tkinter non disponibile: impossibile aprire la finestra grafica.") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    root.update()

    selected = filedialog.askdirectory(
        initialdir=initial_dir or os.getcwd(),
        title=title,
        mustexist=True,
    )
    root.destroy()
    return os.path.abspath(selected) if selected else ""


def collect_fd_folders(input_path):
    """Find all `FD` folders under an experiment/date directory."""
    input_path = os.path.abspath(os.path.expanduser(str(input_path)))

    if os.path.isfile(input_path):
        return []
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Path not found: {input_path}")
    if os.path.basename(input_path).lower() == "fd":
        return [input_path]

    fd_folders = []
    for root, dirs, _ in os.walk(input_path):
        dirs[:] = sorted(dirs, key=_natural_sort_key)
        for dirname in dirs:
            if dirname.lower() == "fd":
                fd_folders.append(os.path.join(root, dirname))

    return sorted(set(fd_folders), key=_natural_sort_key)


def collect_curve_files(input_path, recursive=True, prefer_fd_subfolders=True):
    """Return all supported force-curve files found in a file or directory path."""
    input_path = os.path.abspath(os.path.expanduser(str(input_path)))

    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Path not found: {input_path}")

    search_roots = [input_path]
    if prefer_fd_subfolders:
        fd_folders = collect_fd_folders(input_path)
        if fd_folders:
            search_roots = fd_folders

    curve_files = []
    seen = set()
    for search_root in search_roots:
        for root, dirs, files in os.walk(search_root):
            dirs[:] = sorted(dirs, key=_natural_sort_key)
            for name in sorted(files, key=_natural_sort_key):
                if name.lower().endswith(SUPPORTED_FORCE_EXTENSIONS):
                    path = os.path.join(root, name)
                    if path not in seen:
                        curve_files.append(path)
                        seen.add(path)
            if not recursive:
                break

    return sorted(curve_files, key=_natural_sort_key)


def _cut_at_poc_and_align_baselines(
    trace_signal,
    retrace_signal,
    trace_coord,
    retrace_coord,
):
    """
    Taglia trace e retrace al PoC (indentazione >= 0) senza alcun fill artificiale.
    Calcola la media delle baseline non-contatto di trace e retrace e
    allinea il retrace in modo che le due baseline coincidano.

    Restituisce:
        trace_contact   : segnale trace solo zona contatto (indentazione >= 0)
        retrace_contact : segnale retrace solo zona contatto (indentazione >= 0), allineato
        trace_keep_n    : numero di punti trace da tenere (zona contatto)
        retrace_keep_n  : numero di punti retrace da tenere (zona contatto)
        baseline_shift  : shift applicato al retrace (retrace_baseline - trace_baseline)
        info            : dizionario diagnostico
    """
    trace_signal = np.asarray(trace_signal, dtype=float).reshape(-1)
    retrace_signal = np.asarray(retrace_signal, dtype=float).reshape(-1)
    x_trace = np.asarray(trace_coord, dtype=float).reshape(-1)
    x_ret = np.asarray(retrace_coord, dtype=float).reshape(-1)

    info = {
        "mode": "poc_cut_baseline_mean_alignment",
        "applied": False,
        "trace_keep_n": int(trace_signal.size),
        "retrace_keep_n": int(retrace_signal.size),
        "trace_baseline_mean": float("nan"),
        "retrace_baseline_mean": float("nan"),
        "baseline_shift": 0.0,
    }

    # --- Zona non-contatto: indentazione < 0 ---
    trace_nc_mask = np.isfinite(x_trace) & (x_trace < 0) & np.isfinite(trace_signal)
    ret_nc_mask = np.isfinite(x_ret) & (x_ret < 0) & np.isfinite(retrace_signal)

    if np.count_nonzero(trace_nc_mask) >= 3 and np.count_nonzero(ret_nc_mask) >= 3:
        trace_baseline_mean = float(np.nanmean(trace_signal[trace_nc_mask]))
        retrace_baseline_mean = float(np.nanmean(retrace_signal[ret_nc_mask]))
    else:
        # Fallback: usa i primi/ultimi 10% come baseline
        n_tr = trace_signal.size
        n_ret = retrace_signal.size
        n_bl = max(5, min(n_tr, n_ret) // 10)
        trace_baseline_mean = float(np.nanmedian(trace_signal[:n_bl]))
        retrace_baseline_mean = float(np.nanmedian(retrace_signal[-n_bl:]))

    baseline_shift = retrace_baseline_mean - trace_baseline_mean

    # --- Taglia al PoC: solo indentazione >= 0 ---
    trace_contact_mask = np.isfinite(x_trace) & (x_trace >= 0)
    ret_contact_mask = np.isfinite(x_ret) & (x_ret >= 0)

    if np.count_nonzero(trace_contact_mask) < 3 or np.count_nonzero(ret_contact_mask) < 3:
        # Nessun taglio possibile: ritorna segnali originali senza shift
        return (
            trace_signal.copy(),
            retrace_signal.copy(),
            int(trace_signal.size),
            int(retrace_signal.size),
            0.0,
            info,
        )

    # Per la trace: il PoC è verso la fine (approach), teniamo dal PoC in avanti
    # x_trace è già ordinato: prima non-contatto (neg), poi contatto (pos)
    trace_poc_idx = int(np.where(trace_contact_mask)[0][0])  # primo punto a contatto
    trace_keep_n = int(trace_signal.size) - trace_poc_idx  # punti dal PoC in poi

    # Per la retrace: il PoC è all'inizio (retract), i punti a contatto vengono prima
    ret_keep_n = int(np.where(ret_contact_mask)[0][-1] + 1)  # dal primo punto fino all'ultimo a contatto

    trace_contact = trace_signal[trace_poc_idx:].copy()
    retrace_contact = retrace_signal[:ret_keep_n].copy() - baseline_shift

    info.update({
        "applied": True,
        "trace_poc_idx": trace_poc_idx,
        "trace_keep_n": trace_keep_n,
        "retrace_keep_n": ret_keep_n,
        "trace_baseline_mean": trace_baseline_mean,
        "retrace_baseline_mean": retrace_baseline_mean,
        "baseline_shift": baseline_shift,
    })

    return trace_contact, retrace_contact, trace_keep_n, int(ret_keep_n), baseline_shift, info


def _safe_correct_viscous_drag(
    ind_approach,
    force_approach,
    ind_retract,
    force_retract,
    poly_order=2,
    speed=None,
    return_diagnostics=False,
):
    """Robust force-drag correction inspired by `pyfmrheo`, safe for unequal trace/retrace lengths."""
    ind_approach = np.asarray(ind_approach, dtype=float)
    force_approach = np.asarray(force_approach, dtype=float)
    ind_retract = np.asarray(ind_retract, dtype=float)
    force_retract = np.asarray(force_retract, dtype=float)

    info = {
        "enabled": True,
        "applied": False,
        "status": "not_applied",
        "method": "poly_baseline_average",
        "correction_N": 0.0,
        "baseline_gap_before_N": float("nan"),
        "baseline_gap_after_N": float("nan"),
        "poly_order": int(poly_order),
    }

    base0 = float(force_approach[0]) if force_approach.size else 0.0
    fallback_app = force_approach - base0
    fallback_ret = force_retract - base0

    min_pts = max(int(poly_order) + 1, 5)
    mask_app = np.isfinite(ind_approach) & np.isfinite(force_approach) & (ind_approach < 0)
    mask_ret = np.isfinite(ind_retract) & np.isfinite(force_retract) & (ind_retract < 0)

    if np.count_nonzero(mask_app) < min_pts or np.count_nonzero(mask_ret) < min_pts:
        info["status"] = "skipped_not_enough_noncontact_points"
        return (fallback_app, fallback_ret, info) if return_diagnostics else (fallback_app, fallback_ret)

    try:
        deg_app = min(int(poly_order), np.count_nonzero(mask_app) - 1)
        deg_ret = min(int(poly_order), np.count_nonzero(mask_ret) - 1)
        approach_pol = np.poly1d(np.polyfit(ind_approach[mask_app], force_approach[mask_app], deg_app))
        retract_pol = np.poly1d(np.polyfit(ind_retract[mask_ret], force_retract[mask_ret], deg_ret))

        app_x = ind_approach[mask_app]
        ret_x = ind_retract[mask_ret]
        x_min = max(np.nanmin(app_x), np.nanmin(ret_x))
        x_max = min(np.nanmax(app_x), np.nanmax(ret_x))
        sample_x = app_x[(app_x >= x_min) & (app_x <= x_max)]
        if sample_x.size < min_pts:
            sample_x = ret_x[(ret_x >= x_min) & (ret_x <= x_max)]
        if sample_x.size < min_pts:
            sample_x = app_x

        approach_vals = approach_pol(sample_x)
        retract_vals = retract_pol(sample_x)
        median_gap = float(np.nanmedian(approach_vals - retract_vals))
        if not np.isfinite(median_gap):
            info["status"] = "skipped_nonfinite_gap"
            return (fallback_app, fallback_ret, info) if return_diagnostics else (fallback_app, fallback_ret)

        if speed not in (None, 0):
            correction = (median_gap / 2.0) / abs(float(speed))
        else:
            correction = median_gap / 2.0

        corrected_app_force = force_approach - correction
        corrected_ret_force = force_retract + correction
        offset0 = float(corrected_app_force[0]) if corrected_app_force.size else 0.0
        corrected_app_force = corrected_app_force - offset0
        corrected_ret_force = corrected_ret_force - offset0

        info.update({
            "applied": True,
            "status": "applied",
            "correction_N": float(correction),
            "baseline_gap_before_N": float(median_gap),
            "baseline_gap_after_N": float(np.nanmedian((approach_vals - correction) - (retract_vals + correction))),
        })
        return (corrected_app_force, corrected_ret_force, info) if return_diagnostics else (corrected_app_force, corrected_ret_force)
    except Exception as exc:
        info["status"] = f"failed: {exc}"
        return (fallback_app, fallback_ret, info) if return_diagnostics else (fallback_app, fallback_ret)
