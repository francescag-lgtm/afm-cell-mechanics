import os
import json
import shutil
import tempfile
import textwrap
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pyfmreader import loadfile
from scipy.ndimage import median_filter
from scipy.optimize import OptimizeWarning, curve_fit

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):
        print(obj)

try:
    from sklearn.metrics import r2_score
except Exception:
    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_LABEL_MAP = {
    "mono": "SLS",
    "plr": "PLR puro",
    "bi": "Bi-exp GM",
    "poro": "Poroelastico",
}


def make_odd(value):
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def smooth_signal(signal, window):
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) < 5:
        return signal.copy()

    max_window = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    window = min(make_odd(window), max_window)
    if window < 5:
        return signal.copy()

    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(signal, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def despike_signal(signal, window=9, z_thresh=3.0):
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) < 5:
        return signal.copy(), 0

    if window is None or int(window) < 3 or z_thresh is None or float(z_thresh) <= 0:
        return signal.copy(), 0

    window = min(make_odd(window), len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    if window < 3:
        return signal.copy(), 0

    # Median filter in C/Fortran path is much faster than per-sample Python loops.
    try:
        median_vals = median_filter(signal, size=window, mode="nearest")
    except Exception:
        pad = window // 2
        padded = np.pad(signal, (pad, pad), mode="edge")
        median_vals = np.empty_like(signal)
        for idx in range(len(signal)):
            median_vals[idx] = np.median(padded[idx : idx + window])

    residual = signal - median_vals
    mad = float(np.median(np.abs(residual)))
    if not np.isfinite(mad) or mad <= 1e-12:
        return signal.copy(), 0

    robust_sigma = 1.4826 * mad
    spike_mask = np.abs(residual) > (float(z_thresh) * robust_sigma)
    cleaned = signal.copy()
    cleaned[spike_mask] = median_vals[spike_mask]
    return cleaned.astype(np.float32), int(np.sum(spike_mask))


def robust_inlier_mask(values, z_thresh=3.5):
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)

    if values.ndim != 2 or values.shape[0] < 3 or z_thresh is None or z_thresh <= 0:
        return valid

    median = np.nanmedian(values, axis=0)
    abs_dev = np.abs(values - median)
    mad = np.nanmedian(abs_dev, axis=0)
    robust_sigma = 1.4826 * mad
    std_sigma = np.nanstd(values, axis=0)

    spread = np.where(
        np.isfinite(robust_sigma) & (robust_sigma > 1e-12),
        robust_sigma,
        np.where(np.isfinite(std_sigma) & (std_sigma > 1e-12), std_sigma, np.nan),
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        robust_z = abs_dev / spread

    return valid & ((robust_z <= z_thresh) | ~np.isfinite(robust_z))


def estimate_indentation_from_extend_segment(ext_seg, baseline_defl, k):
    try:
        z_ext = np.asarray(ext_seg.zheight, dtype=float)
        v_ext = np.asarray(ext_seg.vdeflection, dtype=float)
        if len(z_ext) < 5 or len(v_ext) < 5 or not np.isfinite(k) or k <= 0:
            return np.nan

        force_ext = (v_ext - baseline_defl) * k
        finite_mask = np.isfinite(z_ext) & np.isfinite(force_ext)
        z_ext = z_ext[finite_mask]
        force_ext = force_ext[finite_mask]
        if len(z_ext) < 5:
            return np.nan

        head = min(max(10, len(force_ext) // 50), len(force_ext))
        if np.nanmedian(force_ext[:head]) < 0:
            force_ext = -force_ext

        idx_peak = int(np.nanargmax(force_ext))
        if not np.isfinite(force_ext[idx_peak]) or force_ext[idx_peak] <= 0:
            return np.nan

        # Stima il punto di contatto dalla trace come primo superamento robusto
        # della baseline di forza (media + 5*sigma). Questa definizione evita di
        # includere il tratto non-contatto nella stima dell'indentazione.
        n_base = min(max(20, len(force_ext) // 8), len(force_ext))
        base_force = force_ext[:n_base]
        base_mu = float(np.nanmean(base_force))
        base_sigma = float(np.nanstd(base_force))
        force_threshold = base_mu + 5.0 * max(base_sigma, 1e-12)
        contact_candidates = np.where(force_ext > force_threshold)[0]
        idx_contact = int(contact_candidates[0]) if contact_candidates.size else 0

        piezo_disp_m = abs(float(z_ext[idx_peak]) - float(z_ext[idx_contact]))
        cantilever_deflection_m = float(force_ext[idx_peak]) / float(k)
        if not np.isfinite(piezo_disp_m) or not np.isfinite(cantilever_deflection_m):
            return np.nan

        indentation_m = max(piezo_disp_m - cantilever_deflection_m, 0.0)
        return float(indentation_m)
    except Exception:
        return np.nan


def get_force_curve_and_segment(file_path, segment_index=None):
    afm_file = loadfile(file_path)
    force_curve = afm_file.getcurve(0)
    meta = afm_file.filemetadata

    defl_sens = meta["defl_sens_nmbyV"] * 1e-9
    k = meta["spring_const_Nbym"]
    hch = meta["height_channel_key"]

    force_curve.preprocess_force_curve(defl_sens, hch)
    if meta["file_type"] in ("jpk-force", "jpk-force-map", "jpk-qi-data"):
        force_curve.shift_height()

    all_segments = force_curve.get_segments()
    if not all_segments:
        raise ValueError(f"Nessun segmento disponibile in {file_path}")

    if segment_index is not None:
        if len(all_segments) <= segment_index:
            raise ValueError(f"Segmento {segment_index} non disponibile in {file_path}")
        selected_index = int(segment_index)
        segment_name, pause_seg = all_segments[selected_index]
    elif getattr(force_curve, "pause_segments", None):
        segment_name, pause_seg = force_curve.pause_segments[0]
        selected_index = next(i for i, (_, segment_obj) in enumerate(all_segments) if segment_obj is pause_seg)
    else:
        durations = [float(seg.time[-1] - seg.time[0]) for _, seg in all_segments]
        selected_index = int(np.argmax(durations))
        segment_name, pause_seg = all_segments[selected_index]

    ext_seg = force_curve.extend_segments[0][1] if force_curve.extend_segments else all_segments[0][1]
    n_base = max(10, int(0.2 * len(ext_seg.zheight)))
    baseline_defl = np.mean(ext_seg.vdeflection[:n_base])
    estimated_indentation_m = estimate_indentation_from_extend_segment(ext_seg, baseline_defl, k)

    time_data = np.asarray(pause_seg.time - pause_seg.time[0], dtype=np.float32)
    force_data = (np.asarray(pause_seg.vdeflection, dtype=np.float32) - baseline_defl) * k
    distance_data = np.asarray(pause_seg.zheight, dtype=np.float32)

    finite_mask = np.isfinite(time_data) & np.isfinite(force_data) & np.isfinite(distance_data)
    time_data = time_data[finite_mask]
    force_data = force_data[finite_mask]
    distance_data = distance_data[finite_mask]

    sort_index = np.argsort(time_data)
    time_data = time_data[sort_index]
    force_data = force_data[sort_index]
    distance_data = distance_data[sort_index]

    time_data, unique_index = np.unique(time_data, return_index=True)
    force_data = force_data[unique_index]
    distance_data = distance_data[unique_index]

    head = min(max(20, len(force_data) // 100), len(force_data))
    if np.nanmedian(force_data[:head]) < 0:
        force_data = -force_data

    return {
        "time": time_data.astype(np.float32),
        "force": force_data.astype(np.float32),
        "distance": distance_data.astype(np.float32),
        "estimated_indentation_m": float(estimated_indentation_m),
        "segment_index": selected_index,
        "segment_name": segment_name,
    }


def get_reference_force(force_data, normalization_mode="peak", reference_points=200):
    force_data = np.asarray(force_data, dtype=np.float32)
    n_ref = min(max(5, int(reference_points)), len(force_data))
    head_force = force_data[:n_ref]

    if normalization_mode == "first":
        f_ref = float(head_force[0])
    elif normalization_mode == "mean":
        f_ref = float(np.mean(head_force))
    else:
        # Mode "peak": normalizza al PRIMO punto della curva (il picco nel dwell in SR)
        # Questo garantisce che la curva normalizzata parta ESATTAMENTE da 1.0
        f_ref = float(head_force[0])

    if abs(f_ref) < 1e-15:
        raise ValueError("Impossibile normalizzare: forza di riferimento troppo vicina a zero")

    return f_ref


def load_stress_relaxation_curve(file_path, segment_index=None, normalization_mode="peak", reference_points=200):
    try:
        curve = get_force_curve_and_segment(file_path, segment_index=segment_index)
    except Exception as exc:
        raise ValueError(f"Impossibile leggere la curva SR: {file_path}") from exc
    f_ref = get_reference_force(
        curve["force"],
        normalization_mode=normalization_mode,
        reference_points=reference_points,
    )
    curve["reference_force"] = f_ref
    curve["normalized_force"] = (curve["force"] / f_ref).astype(np.float32)
    return curve


def load_valid_stress_relaxation_curves(
    file_paths,
    segment_index=None,
    normalization_mode="peak",
    reference_points=200,
    min_curve_duration_s=None,
):
    curves_data = []
    valid_paths = []
    skipped = []

    for file_path in file_paths:
        try:
            curve = load_stress_relaxation_curve(
                str(file_path),
                segment_index=segment_index,
                normalization_mode=normalization_mode,
                reference_points=reference_points,
            )
        except Exception as exc:
            skipped.append((str(file_path), str(exc)))
            continue

        curves_data.append(curve)
        valid_paths.append(file_path)

    if min_curve_duration_s is not None:
        min_curve_duration_s = float(min_curve_duration_s)
        filtered_pairs = [
            (path_obj, curve)
            for path_obj, curve in zip(valid_paths, curves_data)
            if float(curve["time"][-1] - curve["time"][0]) >= min_curve_duration_s
        ]
        dropped_pairs = [
            (path_obj, curve)
            for path_obj, curve in zip(valid_paths, curves_data)
            if float(curve["time"][-1] - curve["time"][0]) < min_curve_duration_s
        ]
        if dropped_pairs:
            for path_obj, curve in dropped_pairs:
                duration_s = float(curve["time"][-1] - curve["time"][0])
                skipped.append((str(path_obj), f"durata troppo breve ({duration_s:.3f} s < {min_curve_duration_s:.3f} s)"))
        valid_paths = [path_obj for path_obj, _ in filtered_pairs]
        curves_data = [curve for _, curve in filtered_pairs]

    if not curves_data:
        raise ValueError("Nessuna curva SR valida caricata dopo aver escluso i file non leggibili")

    return valid_paths, curves_data, skipped


def interpolate_relaxation_curves(curves_data):
    if not curves_data:
        raise ValueError("Nessuna curva fornita")

    n_points = max(len(curve["time"]) for curve in curves_data)
    overlap_min = max(float(np.min(curve["time"])) for curve in curves_data)
    overlap_max = min(float(np.max(curve["time"])) for curve in curves_data)

    if overlap_min >= overlap_max:
        raise ValueError("Le curve non hanno un intervallo temporale comune")

    time_grid = np.linspace(overlap_min, overlap_max, n_points, dtype=np.float32)
    interpolated_forces = []
    interpolated_distances = []

    for curve in curves_data:
        interpolated_forces.append(np.interp(time_grid, curve["time"], curve["normalized_force"]))
        interpolated_distances.append(np.interp(time_grid, curve["time"], curve["distance"]))

    return time_grid, np.asarray(interpolated_forces, dtype=np.float32), np.asarray(interpolated_distances, dtype=np.float32)


def calculate_mean_curve(
    curves_data,
    outlier_sigma=3.0,
    min_valid_curves=2,
    smoothing_window=None,
    spike_filter_window=None,
    spike_filter_z=3.0,
):
    time_grid, interpolated_forces, interpolated_distances = interpolate_relaxation_curves(curves_data)

    n_spike_values = 0
    if spike_filter_window is not None and int(spike_filter_window) >= 3:
        cleaned_rows = []
        for force_row in interpolated_forces:
            cleaned_row, n_spikes = despike_signal(
                force_row,
                window=int(spike_filter_window),
                z_thresh=float(spike_filter_z),
            )
            cleaned_rows.append(cleaned_row)
            n_spike_values += n_spikes
        interpolated_forces = np.asarray(cleaned_rows, dtype=np.float32)

    if smoothing_window is not None and int(smoothing_window) >= 5:
        window = int(smoothing_window)
        interpolated_forces = np.asarray(
            [smooth_signal(force_row, window) for force_row in interpolated_forces],
            dtype=np.float32,
        )

    keep_mask = robust_inlier_mask(interpolated_forces, z_thresh=outlier_sigma)
    cleaned_forces = np.where(keep_mask, interpolated_forces, np.nan)
    cleaned_distances = np.where(keep_mask, interpolated_distances, np.nan)

    min_required = min(max(1, int(min_valid_curves)), len(curves_data))
    n_valid = np.sum(np.isfinite(cleaned_forces), axis=0)
    common_mask = n_valid >= min_required

    if not np.any(common_mask):
        raise ValueError("Dopo il filtraggio non restano abbastanza punti comuni tra le curve")

    mean_force = np.nanmean(cleaned_forces[:, common_mask], axis=0)
    std_force = np.nanstd(cleaned_forces[:, common_mask], axis=0)
    mean_distance = np.nanmean(cleaned_distances[:, common_mask], axis=0)

    # Rienormalizza la curva media al primo punto per garantire che parta esattamente da 1.0
    if len(mean_force) > 0 and np.isfinite(mean_force[0]) and abs(mean_force[0]) > 1e-12:
        f_first = float(mean_force[0])
        mean_force = mean_force / f_first
        std_force = std_force / abs(f_first)

    stats = {
        "t_min": float(time_grid[common_mask][0]),
        "t_max": float(time_grid[common_mask][-1]),
        "n_timepoints_common": int(np.sum(common_mask)),
        "n_spike_values": int(n_spike_values),
        "n_outlier_values": int(np.sum(~keep_mask)),
        "min_valid_curves": int(min_required),
        "smoothing_window": int(smoothing_window) if smoothing_window is not None else None,
        "spike_filter_window": int(spike_filter_window) if spike_filter_window is not None else None,
        "spike_filter_z": float(spike_filter_z) if spike_filter_window is not None else None,
    }

    return (
        time_grid[common_mask].astype(np.float32),
        mean_force.astype(np.float32),
        std_force.astype(np.float32),
        mean_distance.astype(np.float32),
        stats,
    )


def save_mean_curve_csv(csv_path, time_data, force_data, std_data):
    data = np.column_stack((time_data, force_data, std_data))
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="time (s),mean_normalized_force,std_normalized_force",
        comments="",
        fmt="%.10e",
    )


def save_jpk_force(template_jpk, output_path, force_data, distance_data, segment_index=0):
    force_data = np.asarray(force_data, dtype=np.float32)
    distance_data = np.asarray(distance_data, dtype=np.float32)
    temp_dir = Path(tempfile.mkdtemp())

    try:
        extract_dir = temp_dir / "extract"
        extract_dir.mkdir()

        with zipfile.ZipFile(template_jpk, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        channels_dir = extract_dir / "segments" / str(segment_index) / "channels"
        force_channel = channels_dir / "vDeflection.dat"
        distance_channel = channels_dir / "measuredHeight.dat"

        if not force_channel.exists() or not distance_channel.exists():
            raise ValueError("Canali richiesti non trovati nel template .jpk-force")

        template_points = force_channel.stat().st_size // np.dtype(np.float32).itemsize
        if len(force_data) != template_points or len(distance_data) != template_points:
            orig_idx = np.linspace(0, len(force_data) - 1, template_points)
            src_idx = np.arange(len(force_data))
            force_data = np.interp(orig_idx, src_idx, force_data).astype(np.float32)
            distance_data = np.interp(orig_idx, src_idx, distance_data).astype(np.float32)

        force_channel.write_bytes(force_data.tobytes())
        distance_channel.write_bytes(distance_data.tobytes())

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
            for dir_path in sorted(path for path in extract_dir.rglob("*") if path.is_dir()):
                arcname = f"{dir_path.relative_to(extract_dir).as_posix()}/"
                zip_ref.write(dir_path, arcname)

            for file_path in sorted(path for path in extract_dir.rglob("*") if path.is_file()):
                arcname = file_path.relative_to(extract_dir).as_posix()
                zip_ref.write(file_path, arcname)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def prepare_mean_curve_from_folder(
    folder_path,
    num_curves=None,
    selected_curve_names=None,
    segment_index=None,
    normalization_mode="peak",
    reference_points=200,
    outlier_sigma=3.0,
    min_valid_curves=2,
    smoothing_window=None,
    spike_filter_window=None,
    spike_filter_z=3.0,
    min_curve_duration_s=None,
    indentation_auto_mode="mean_robust",
    save_preprocessed_mean=True,
    save_mean_csv=False,
):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Cartella non trovata: {folder_path}")

    _EXCLUDE_TAGS = ["-mean", "-media", "-norm"]
    all_jpk_files = sorted(
        p for pattern in ("*.jpk-force", "*.jpk")
        for p in folder.rglob(pattern)
    )
    usable_files = [
        path for path in all_jpk_files
        if not any(tag in path.stem.lower() for tag in _EXCLUDE_TAGS)
        and (
            path.parent == folder
            or path.parent.name.upper() == "SR"
            or "_SR-" in path.name.upper()
        )
    ]

    if not usable_files:
        raise FileNotFoundError(
            f"Nessuna curva `.jpk-force` / `.jpk` grezza trovata in {folder_path}. "
            "La cartella deve contenere file SR direttamente oppure in sottocartelle `cellXX/SR`."
        )

    if selected_curve_names is not None:
        requested_names = [str(name).strip().lower() for name in selected_curve_names if str(name).strip()]
        if not requested_names:
            raise ValueError("selected_curve_names e' stato fornito ma non contiene nomi validi")

        requested_set = set(requested_names)

        def _matches_requested(path_obj):
            path_name = path_obj.name.lower()
            path_stem = path_obj.stem.lower()
            return (path_name in requested_set) or (path_stem in requested_set)

        filtered_files = [path for path in usable_files if _matches_requested(path)]
        if not filtered_files:
            raise FileNotFoundError(
                "Nessuna curva corrisponde ai nomi richiesti in selected_curve_names"
            )

        missing_names = [
            requested
            for requested in requested_set
            if not any(
                (path.name.lower() == requested) or (path.stem.lower() == requested)
                for path in usable_files
            )
        ]
        if missing_names:
            print("Attenzione: alcune curve richieste non sono state trovate:")
            for missing_name in sorted(missing_names):
                print(f"  - {missing_name}")

        usable_files = filtered_files

    selected_paths = usable_files if num_curves in (None, 0) else usable_files[: int(num_curves)]
    if not selected_paths:
        raise ValueError("Nessuna curva selezionata per la media")

    print("Cartella rilevata → genero automaticamente la curva media normalizzata")
    print(f"- Cartella dati: {folder_path}")
    if selected_curve_names is None:
        print("- Selezione curve: tutte le curve grezze disponibili")
    else:
        print(f"- Selezione curve richiesta: {len(selected_curve_names)}")
    print(f"- Curve candidate: {len(selected_paths)}")
    for file_path in selected_paths:
        print(f"  • {file_path.name}")

    selected_paths, curves_data, skipped_files = load_valid_stress_relaxation_curves(
        selected_paths,
        segment_index=segment_index,
        normalization_mode=normalization_mode,
        reference_points=reference_points,
        min_curve_duration_s=min_curve_duration_s,
    )
    if skipped_files:
        print(f"- Curve escluse per errore di lettura: {len(skipped_files)}")
        for file_path, reason in skipped_files:
            print(f"  ! {Path(file_path).name}: {reason}")
    print(f"- Curve valide usate per la media: {len(selected_paths)}")

    f_peak_values_N = [
        float(c["reference_force"])
        for c in curves_data
        if np.isfinite(float(c.get("reference_force", np.nan)))
    ]
    f_peak_mean_N = float(np.mean(f_peak_values_N)) if f_peak_values_N else np.nan
    indentation_values_m = [
        float(c.get("estimated_indentation_m", np.nan))
        for c in curves_data
        if np.isfinite(float(c.get("estimated_indentation_m", np.nan)))
    ]
    indentation_mean_m = np.nan
    if indentation_values_m:
        ind_arr = np.asarray(indentation_values_m, dtype=float)
        mode = str(indentation_auto_mode or "mean_robust").strip().lower()
        if mode not in {"mean_robust", "median", "mean"}:
            mode = "mean_robust"

        if mode == "median":
            indentation_mean_m = float(np.nanmedian(ind_arr))
        elif mode == "mean":
            indentation_mean_m = float(np.nanmean(ind_arr))
        else:
            ind_med = float(np.nanmedian(ind_arr))
            mad = float(np.nanmedian(np.abs(ind_arr - ind_med)))
            robust_sigma = 1.4826 * mad
            if np.isfinite(robust_sigma) and robust_sigma > 1e-12:
                inlier_mask = np.abs(ind_arr - ind_med) <= 3.0 * robust_sigma
                if np.any(inlier_mask):
                    indentation_mean_m = float(np.nanmean(ind_arr[inlier_mask]))
                else:
                    indentation_mean_m = ind_med
            else:
                indentation_mean_m = ind_med

    mean_time, mean_force, std_force, mean_distance, clean_stats = calculate_mean_curve(
        curves_data,
        outlier_sigma=outlier_sigma,
        min_valid_curves=min_valid_curves,
        smoothing_window=smoothing_window,
        spike_filter_window=spike_filter_window,
        spike_filter_z=spike_filter_z,
    )

    first_stem = selected_paths[0].stem
    base_stem = first_stem
    for suffix in ("-media-norm", "-mean-norm", "-media", "-mean", "-norm"):
        if base_stem.endswith(suffix):
            base_stem = base_stem[: -len(suffix)]
    output_stem = f"{base_stem}-media-norm"

    csv_path = folder / f"{output_stem}.csv"
    jpk_output_path = folder / f"{output_stem}.jpk-force"

    if save_mean_csv:
        save_mean_curve_csv(csv_path, mean_time, mean_force, std_force)
    elif csv_path.exists():
        try:
            csv_path.unlink()
        except OSError:
            pass

    if save_preprocessed_mean:
        save_jpk_force(
            str(selected_paths[0]),
            str(jpk_output_path),
            mean_force,
            mean_distance,
            segment_index=curves_data[0]["segment_index"],
        )

    print(f"- Intervallo comune usato: {clean_stats['t_min']:.4f} s - {clean_stats['t_max']:.4f} s")
    print(f"- Punti temporali mantenuti: {clean_stats['n_timepoints_common']}")
    print(f"- Spike sottili corretti (mediana robusta): {clean_stats['n_spike_values']}")
    print(f"- Valori scartati come outlier: {clean_stats['n_outlier_values']}")
    if np.isfinite(indentation_mean_m):
        print(
            f"- Indentazione stimata pre-normalizzazione ({mode}): "
            f"{indentation_mean_m:.3e} m"
        )
    if save_preprocessed_mean:
        print(f"- JPK medio generato: {jpk_output_path}")
    if save_mean_csv:
        print(f"- CSV medio generato: {csv_path}")
    else:
        print("- CSV medio non salvato: analisi eseguita direttamente in memoria")

    return {
        "is_generated_mean": True,
        "folder_path": str(folder),
        "base_name": output_stem,
        "display_name": f"{output_stem} (media in memoria)",
        "time": mean_time.astype(float),
        "force": mean_force.astype(float),
        "std": std_force.astype(float),
        "distance": mean_distance.astype(float),
        "csv_path": str(csv_path),
        "jpk_path": str(jpk_output_path) if save_preprocessed_mean else None,
        "f_peak_mean_N": f_peak_mean_N,
        "estimated_indentation_m": indentation_mean_m,
    }


def prepare_area_means_from_day_folder(
    day_folder_path,
    num_curves=None,
    selected_curve_names=None,
    segment_index=None,
    normalization_mode="peak",
    reference_points=200,
    outlier_sigma=3.5,
    min_valid_curves=2,
    smoothing_window=None,
    spike_filter_window=None,
    spike_filter_z=3.0,
    min_curve_duration_s=None,
    indentation_auto_mode="mean_robust",
    save_preprocessed_mean=True,
    save_mean_csv=True,
):
    day_folder = Path(day_folder_path)
    if not day_folder.is_dir():
        raise FileNotFoundError(f"Cartella giornata non trovata: {day_folder_path}")

    def _count_usable_sr_files(folder):
        exclude_tags = ["-mean", "-media", "-norm"]
        all_jpk_files = sorted(
            p for pattern in ("*.jpk-force", "*.jpk")
            for p in folder.rglob(pattern)
        )
        usable_files = [
            path for path in all_jpk_files
            if not any(tag in path.stem.lower() for tag in exclude_tags)
            and (
                path.parent == folder
                or path.parent.name.upper() == "SR"
                or "_SR-" in path.name.upper()
            )
        ]
        return len(usable_files)

    area_dirs = [
        path for path in sorted(day_folder.iterdir())
        if path.is_dir() and _count_usable_sr_files(path) > 0
    ]

    if not area_dirs:
        raise FileNotFoundError(
            "Nessuna area valida trovata nella cartella giornata. "
            "Atteso: sottocartelle area contenenti curve SR grezze."
        )

    print("Modalita area-only: calcolo medie separate per ogni area (nessuna media finale di giornata)")
    print(f"- Giornata: {day_folder}")
    print(f"- Aree candidate: {len(area_dirs)}")

    area_results = []
    skipped_areas = []

    for area_dir in area_dirs:
        print(f"\n=== Area: {area_dir.name} ===")
        try:
            area_mean = prepare_mean_curve_from_folder(
                str(area_dir),
                num_curves=num_curves,
                selected_curve_names=selected_curve_names,
                segment_index=segment_index,
                normalization_mode=normalization_mode,
                reference_points=reference_points,
                outlier_sigma=outlier_sigma,
                min_valid_curves=min_valid_curves,
                smoothing_window=smoothing_window,
                spike_filter_window=spike_filter_window,
                spike_filter_z=spike_filter_z,
                min_curve_duration_s=min_curve_duration_s,
                indentation_auto_mode=indentation_auto_mode,
                save_preprocessed_mean=save_preprocessed_mean,
                save_mean_csv=save_mean_csv,
            )
            area_mean["area_name"] = area_dir.name
            area_results.append(area_mean)
        except Exception as exc:
            skipped_areas.append((area_dir.name, str(exc)))
            print(f"[SKIP] {area_dir.name}: {exc}")

    if not area_results:
        raise ValueError("Nessuna media di area generata con successo")

    summary_rows = []
    for item in area_results:
        summary_rows.append(
            {
                "area": item.get("area_name", "n.d."),
                "folder": item.get("folder_path"),
                "n_points": len(item.get("time", [])),
                "jpk_path": item.get("jpk_path"),
                "csv_path": item.get("csv_path") if save_mean_csv else None,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("area").reset_index(drop=True)

    print("\nRiepilogo medie area generate:")
    display(summary_df)
    if skipped_areas:
        print("\nAree saltate:")
        for area_name, reason in skipped_areas:
            print(f"- {area_name}: {reason}")

    return {
        "is_generated_area_means": True,
        "day_folder": str(day_folder),
        "area_results": area_results,
        "area_summary_df": summary_df,
        "skipped_areas": skipped_areas,
    }


def resolve_sr_input(
    filepath,
    num_curves=None,
    selected_curve_names=None,
    segment_index=None,
    normalization_mode="peak",
    reference_points=200,
    outlier_sigma=3.0,
    min_valid_curves=2,
    smoothing_window=None,
    spike_filter_window=None,
    spike_filter_z=3.0,
    min_curve_duration_s=None,
    indentation_auto_mode="mean_robust",
    save_preprocessed_mean=True,
    save_mean_csv=False,
):
    if os.path.isdir(filepath):
        return prepare_mean_curve_from_folder(
            filepath,
            num_curves=num_curves,
            selected_curve_names=selected_curve_names,
            segment_index=segment_index,
            normalization_mode=normalization_mode,
            reference_points=reference_points,
            outlier_sigma=outlier_sigma,
            min_valid_curves=min_valid_curves,
            smoothing_window=smoothing_window,
            spike_filter_window=spike_filter_window,
            spike_filter_z=spike_filter_z,
            min_curve_duration_s=min_curve_duration_s,
            indentation_auto_mode=indentation_auto_mode,
            save_preprocessed_mean=save_preprocessed_mean,
            save_mean_csv=save_mean_csv,
        )
    return filepath


def load_sr_curve(filepath, input_force_mode="auto", normalized_reference="Fpicco,init"):
    if input_force_mode not in {"auto", "raw", "normalized"}:
        raise ValueError("input_force_mode deve essere 'auto', 'raw' o 'normalized'")

    if isinstance(filepath, dict) and filepath.get("is_generated_mean"):
        t_pause = np.asarray(filepath["time"], dtype=float)
        force_pause = np.asarray(filepath["force"], dtype=float)
        if len(force_pause) == 0:
            raise ValueError("Curva media generata vuota")

        head = min(max(10, len(force_pause) // 50), len(force_pause))
        if np.nanmedian(force_pause[:head]) < 0:
            force_pause = -force_pause

        print(
            f"Nota: curva media generata in memoria → fit eseguito su F(t)/{normalized_reference} "
            f"(nessun CSV salvato)"
        )
        return {
            "FC": None,
            "meta": None,
            "k": np.nan,
            "defl_sens": np.nan,
            "is_normalized": True,
            "normalized_reference": normalized_reference,
            "normalized_source": "generated_mean_in_memory",
            "force_scale": 1.0,
            "force_unit": "adim.",
            "force_axis_label": f"F(t) / {normalized_reference}",
            "f_peak_mean_N": float(filepath.get("f_peak_mean_N", np.nan)),
            "estimated_indentation_m": float(filepath.get("estimated_indentation_m", np.nan)),
            "z_ext": np.array([]),
            "force_ext": np.array([]),
            "t_ext": np.array([]),
            "z_pause": np.full_like(t_pause, np.nan, dtype=float),
            "force_pause": force_pause,
            "t_pause": t_pause,
            "z_ret": np.array([]),
            "force_ret": np.array([]),
            "t_ret": np.array([]),
        }

    file_name = os.path.basename(filepath).lower()
    file_root, file_ext = os.path.splitext(filepath)
    file_ext = file_ext.lower()

    is_normalized = input_force_mode == "normalized" or (
        input_force_mode == "auto" and ("-norm" in file_name or file_ext == ".csv")
    )

    if is_normalized and file_ext == ".csv":
        csv_data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        if csv_data.ndim != 2 or csv_data.shape[1] < 2:
            raise ValueError("CSV normalizzato non valido: servono almeno le colonne tempo e forza normalizzata")

        t_pause = np.asarray(csv_data[:, 0], dtype=float)
        force_pause = np.asarray(csv_data[:, 1], dtype=float)
        if len(force_pause) == 0:
            raise ValueError("CSV normalizzato vuoto")

        head = min(max(10, len(force_pause) // 50), len(force_pause))
        if np.nanmedian(force_pause[:head]) < 0:
            force_pause = -force_pause

        print(f"Nota: curva normalizzata caricata da CSV → fit eseguito su F(t)/{normalized_reference}")
        return {
            "FC": None,
            "meta": None,
            "k": np.nan,
            "defl_sens": np.nan,
            "is_normalized": True,
            "normalized_reference": normalized_reference,
            "normalized_source": "csv",
            "force_scale": 1.0,
            "force_unit": "adim.",
            "force_axis_label": f"F(t) / {normalized_reference}",
            "f_peak_mean_N": np.nan,
            "z_ext": np.array([]),
            "force_ext": np.array([]),
            "t_ext": np.array([]),
            "z_pause": np.full_like(t_pause, np.nan, dtype=float),
            "force_pause": force_pause,
            "t_pause": t_pause,
            "z_ret": np.array([]),
            "force_ret": np.array([]),
            "t_ret": np.array([]),
        }

    afm_file = loadfile(filepath)
    force_curve = afm_file.getcurve(0)
    meta = afm_file.filemetadata

    defl_sens = meta["defl_sens_nmbyV"] * 1e-9
    k = meta["spring_const_Nbym"]
    hch = meta["height_channel_key"]

    force_curve.preprocess_force_curve(defl_sens, hch)
    if meta["file_type"] in ("jpk-force", "jpk-force-map", "jpk-qi-data"):
        force_curve.shift_height()

    ext_seg = force_curve.extend_segments[0][1]
    pause_seg = force_curve.pause_segments[0][1]
    ret_seg = force_curve.retract_segments[-1][1]

    n_base = max(10, int(0.2 * len(ext_seg.zheight)))
    baseline_defl = np.mean(ext_seg.vdeflection[:n_base])

    force_ext_raw = (ext_seg.vdeflection - baseline_defl) * k
    force_ret_raw = (ret_seg.vdeflection - baseline_defl) * k
    estimated_indentation_m = estimate_indentation_from_extend_segment(ext_seg, baseline_defl, k)
    t_pause_jpk = np.asarray(pause_seg.time - pause_seg.time[0], dtype=float)

    if is_normalized:
        csv_candidate = file_root + ".csv"
        if os.path.exists(csv_candidate):
            csv_data = np.loadtxt(csv_candidate, delimiter=",", skiprows=1)
            t_pause = np.asarray(csv_data[:, 0], dtype=float)
            force_pause = np.asarray(csv_data[:, 1], dtype=float)
            z_pause = np.full_like(t_pause, np.nan, dtype=float)
            normalized_source = "csv"
        else:
            raise FileNotFoundError(
                f"Per il file normalizzato {filepath} manca il CSV gemello con i dati quantitativi"
            )

        head = min(max(10, len(force_pause) // 50), len(force_pause))
        if np.nanmedian(force_pause[:head]) < 0:
            force_pause = -force_pause

        force_ext = np.full_like(force_ext_raw, np.nan, dtype=float)
        force_ret = np.full_like(force_ret_raw, np.nan, dtype=float)
        force_scale = 1.0
        force_unit = "adim."
        force_axis_label = f"F(t) / {normalized_reference}"
        print(
            f"Nota: curva normalizzata rilevata → fit eseguito su F(t)/{normalized_reference} "
            f"(sorgente: {normalized_source})"
        )
    else:
        force_pause = (pause_seg.vdeflection - baseline_defl) * k
        t_pause = t_pause_jpk
        z_pause = pause_seg.zheight
        force_ext = force_ext_raw
        force_ret = force_ret_raw
        force_scale = 1e12
        force_unit = "pN"
        force_axis_label = "Forza [pN]"
        normalized_source = None

    return {
        "FC": force_curve,
        "meta": meta,
        "k": k,
        "defl_sens": defl_sens,
        "is_normalized": bool(is_normalized),
        "normalized_reference": normalized_reference if is_normalized else None,
        "normalized_source": normalized_source,
        "force_scale": force_scale,
        "force_unit": force_unit,
        "force_axis_label": force_axis_label,
        "f_peak_mean_N": np.nan,
        "estimated_indentation_m": float(estimated_indentation_m),
        "z_ext": ext_seg.zheight,
        "force_ext": force_ext,
        "t_ext": ext_seg.time,
        "z_pause": z_pause,
        "force_pause": np.asarray(force_pause, dtype=float),
        "t_pause": np.asarray(t_pause, dtype=float),
        "z_ret": ret_seg.zheight,
        "force_ret": force_ret,
        "t_ret": ret_seg.time,
    }


def r2(y, y_pred):
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def information_criteria(y, y_pred, n_params):
    mask = np.isfinite(y) & np.isfinite(y_pred)
    y_valid = np.asarray(y)[mask]
    y_pred_valid = np.asarray(y_pred)[mask]
    n_obs = len(y_valid)
    if n_obs <= n_params:
        return float("nan"), float("nan")
    rss = np.sum((y_valid - y_pred_valid) ** 2)
    rss = max(rss, np.finfo(float).eps)
    aic = n_obs * np.log(rss / n_obs) + 2 * n_params
    bic = n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)
    return aic, bic


def mono_exp(t, F_inf, tau):
    return F_inf + (1.0 - F_inf) * np.exp(-t / tau)


def plr(t, F0, t0_plr, alpha):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = F0 * (t / t0_plr) ** (-alpha)
    return np.where(t > 0, result, np.nan)


def bi_exp(t, F_inf, w, tau1, tau2):
    w_clip = np.clip(w, 0.0, 1.0)
    return F_inf + (1.0 - F_inf) * (w_clip * np.exp(-t / tau1) + (1.0 - w_clip) * np.exp(-t / tau2))


def poro_relax(t, F_inf, tau_p):
    t_clip = np.maximum(t, 0.0)
    return F_inf + (1.0 - F_inf) * np.exp(-np.sqrt(t_clip / tau_p))


def format_parametri(model_key, params, force_unit=""):
    if params is None:
        return "n.d."
    u = f" {force_unit}" if force_unit else ""
    if model_key == "mono":
        return f"F∞={params[0]:.4f}, τ={params[1]:.4f} s"
    if model_key == "plr":
        return f"t0={params[1]:.4f} s, β={params[2]:.4f}"
    if model_key == "bi":
        return f"F∞={params[0]:.4f}, w={params[1]:.4f}, τ1={params[2]:.4f} s, τ2={params[3]:.4f} s"
    if model_key == "poro":
        return f"F∞={params[0]:.4f}, τp={params[1]:.4f} s"
    return "n.d."


def run_sr_analysis(
    sr,
    analysis_input_path,
    save_outputs=True,
    open_output_folder=False,
    plot_max_time_s=None,
    fit_spike_filter_window=None,
    fit_spike_filter_z=3.0,
    fit_smoothing_window=None,
    plr_fixed_t0=None,
    transition_time_s=0.5,
    auto_optimize_transition=False,
    transition_search_range_s=None,
    transition_search_steps=31,
    transition_max_join_gap=0.03,
    transition_max_local_rmse=0.05,
    transition_late_models=("plr", "bi", "mono"),
    transition_min_late_r2=0.90,
    show_transition_diagnostic_plot=False,
    hertz_tip_radius_m=5.0e-06,
    hertz_indentation_m=None,
    hertz_poisson=0.49,
):
    analysis_path = analysis_input_path
    t_raw = np.asarray(sr["t_pause"], dtype=float)
    Ft_raw = np.asarray(sr["force_pause"], dtype=float)
    valid_mask = np.isfinite(t_raw) & np.isfinite(Ft_raw)
    t_full = t_raw[valid_mask]
    Ft_full = Ft_raw[valid_mask]

    if len(t_full) < 20:
        raise ValueError("Troppi pochi punti validi nella curva dwell per eseguire l'analisi")

    # Preprocessing sulla curva completa (usata per grafico SR completo e statistiche globali)
    Ft_processed_full = Ft_full.copy()
    n_fit_spikes = 0
    if fit_spike_filter_window is not None and int(fit_spike_filter_window) >= 3:
        Ft_processed_full, n_fit_spikes = despike_signal(
            Ft_processed_full,
            window=int(fit_spike_filter_window),
            z_thresh=float(fit_spike_filter_z),
        )

    if fit_smoothing_window is not None and int(fit_smoothing_window) >= 5:
        Ft_processed_full = smooth_signal(Ft_processed_full, int(fit_smoothing_window)).astype(float)

    # Rienormalizza al primo punto DOPO lo smoothing per garantire che F_peak parta da 1.0
    if len(Ft_processed_full) > 0 and np.isfinite(Ft_processed_full[0]) and abs(Ft_processed_full[0]) > 1e-12:
        Ft_processed_full = Ft_processed_full / Ft_processed_full[0]

    # Dati usati nei fit (eventualmente limitati nel tempo)
    t = t_full.copy()
    Ft_processed = Ft_processed_full.copy()

    t_limit = None
    if plot_max_time_s is not None and float(plot_max_time_s) > 0:
        t_limit = float(plot_max_time_s)
        time_mask = t <= t_limit
        if np.sum(time_mask) < 20:
            raise ValueError(
                f"Con plot_max_time_s={t_limit:.2f} s restano meno di 20 punti: aumenta il limite o disattivalo"
            )
        t = t[time_mask]
        Ft_processed = Ft_processed[time_mask]

    t_max = float(t[-1])
    force_scale = sr["force_scale"]
    force_unit = sr["force_unit"]
    force_axis_label = sr["force_axis_label"]

    if isinstance(analysis_path, dict):
        save_dir = analysis_path["folder_path"]
        base_name = analysis_path["base_name"]
        analysis_label = analysis_path["display_name"]
    else:
        save_dir = os.path.dirname(os.path.abspath(analysis_path))
        base_name = os.path.splitext(os.path.basename(analysis_path))[0]
        analysis_label = os.path.basename(analysis_path)

    # Usa la curva completa per il calcolo del rilassamento (fino alla durata totale, es. 15 s)
    F_peak = float(Ft_processed_full[0])
    avg_pts = min(50, len(Ft_processed_full))
    F_relax = float(np.mean(Ft_processed_full[-avg_pts:]))
    delta_F = F_peak - F_relax
    delta_F_pct = 100.0 * delta_F / F_peak if abs(F_peak) > 1e-12 else np.nan

    print("=" * 45)
    print(f"  F_peak            = {F_peak * force_scale:>10.4f}  {force_unit}")
    print(f"  F_relax (plateau) = {F_relax * force_scale:>10.4f}  {force_unit}")
    print(f"  ΔF  (assoluto)    = {delta_F * force_scale:>10.4f}  {force_unit}")
    print(f"  ΔF% (relativo)    = {delta_F_pct:>10.2f}  %")
    print("=" * 45)

    if n_fit_spikes > 0:
        print(f"Picchi sottili corretti nel fit finale: {n_fit_spikes}")
    if fit_smoothing_window is not None and int(fit_smoothing_window) >= 5:
        print(f"Smoothing applicato nel fit finale: finestra = {int(fit_smoothing_window)}")

    N_fit = 1800
    if len(t) > N_fit:
        idx = np.unique(np.concatenate([[0], np.round(np.logspace(0, np.log10(len(t) - 1), N_fit - 1)).astype(int)]))
        idx = np.clip(idx, 0, len(t) - 1)
        t_fit = t[idx]
        Ft_fit = Ft_processed[idx]
    else:
        t_fit = t
        Ft_fit = Ft_processed

    Ft_fit_scaled = Ft_fit * force_scale

    fit_results = {}
    fit_covariances = {}
    fit_meta = {}
    fit_diagnostics = {}
    min_points_per_phase = 20

    # Se richiesto, ottimizza automaticamente il punto di transizione tra early-phase
    # (fit poroelastico) e fase globale, minimizzando discontinuita e errore locale.
    t_min_allowed = float(t_fit[min_points_per_phase - 1])
    t_max_allowed = float(t_fit[-min_points_per_phase])
    if not np.isfinite(t_min_allowed) or not np.isfinite(t_max_allowed) or t_max_allowed <= t_min_allowed:
        raise ValueError("Curva troppo corta per stimare una transizione robusta tra i due regimi")

    transition_default = float(transition_time_s)
    transition_selected = float(np.clip(transition_default, t_min_allowed, t_max_allowed))
    transition_search_info = {
        "auto_enabled": bool(auto_optimize_transition),
        "requested_transition_time_s": transition_default,
        "selected_transition_time_s": transition_selected,
        "search_range_s": None,
        "search_steps": int(max(5, transition_search_steps)),
        "acceptance_max_join_gap": float(transition_max_join_gap),
        "acceptance_max_local_rmse": float(transition_max_local_rmse),
        "acceptance_min_late_r2": float(transition_min_late_r2),
        "late_models": list(transition_late_models),
        "best_score": np.nan,
        "n_candidates": 0,
        "n_valid_candidates": 0,
        "n_accepted_candidates": 0,
        "used_fallback_transition": False,
        "selection_reason": "default_manual",
        "selected_late_model": None,
        "selected_late_model_r2": np.nan,
        "best_transition_time_by_model_s": {},
        "best_transition_score_by_model": {},
    }
    transition_eval_rows = []
    model_best_tracker = {
        str(m).lower().strip(): {"t": np.nan, "score": np.inf}
        for m in transition_late_models
        if str(m).lower().strip() in {"plr", "bi", "mono"}
    }

    if auto_optimize_transition:
        if transition_search_range_s is None:
            search_min = max(t_min_allowed, transition_selected * 0.6)
            search_max = min(t_max_allowed, transition_selected * 1.8)
        else:
            if not isinstance(transition_search_range_s, (list, tuple)) or len(transition_search_range_s) != 2:
                raise ValueError("transition_search_range_s deve essere una coppia (t_min, t_max) in secondi")
            search_min = float(transition_search_range_s[0])
            search_max = float(transition_search_range_s[1])
            if search_max < search_min:
                search_min, search_max = search_max, search_min
            search_min = max(t_min_allowed, search_min)
            search_max = min(t_max_allowed, search_max)

        if search_max > search_min:
            n_steps = int(max(5, transition_search_steps))
            transition_candidates = np.linspace(search_min, search_max, n_steps)
            transition_candidates = np.unique(
                np.clip(np.append(transition_candidates, transition_selected), t_min_allowed, t_max_allowed)
            )
        else:
            transition_candidates = np.array([transition_selected], dtype=float)

        transition_search_info["search_range_s"] = [float(search_min), float(search_max)]
        transition_search_info["n_candidates"] = int(len(transition_candidates))

        best_candidate = None
        best_score = np.inf
        best_late_model = None
        best_late_r2 = np.nan

        for candidate_t in transition_candidates:
            early_mask_c = (t_fit >= 0.0) & (t_fit <= candidate_t)
            late_mask_c = t_fit >= candidate_t
            if np.sum(early_mask_c) < min_points_per_phase or np.sum(late_mask_c) < min_points_per_phase:
                continue

            t_early_c = t_fit[early_mask_c]
            F_early_c_norm = (Ft_fit_scaled[early_mask_c] / force_scale).astype(float)
            t_late_c = t_fit[late_mask_c]
            F_late_c_norm = (Ft_fit_scaled[late_mask_c] / force_scale).astype(float)

            try:
                f_inf0_poro_c = float(np.clip(np.nanmedian(F_early_c_norm[-min(30, len(F_early_c_norm)):]), 0.0, 1.2))
                p0_poro_c = [f_inf0_poro_c, max(t_early_c[-1] / 4, 1e-4)]
                lb_poro_c = [0.0, 1e-6]
                ub_poro_c = [1.2, max(t_early_c[-1] * 10, 1e-3)]
                popt_poro_c, _ = curve_fit(
                    poro_relax,
                    t_early_c,
                    F_early_c_norm,
                    p0=p0_poro_c,
                    bounds=(lb_poro_c, ub_poro_c),
                    maxfev=20000,
                )

                y_join_early = float(poro_relax(np.array([candidate_t]), *popt_poro_c)[0])
                local_window = max(0.03, 0.1 * candidate_t)
                tail_mask = t_early_c >= max(0.0, candidate_t - local_window)
                err_early = F_early_c_norm[tail_mask] - poro_relax(t_early_c[tail_mask], *popt_poro_c)
                rmse_early = float(np.sqrt(np.nanmean(err_early ** 2))) if np.any(np.isfinite(err_early)) else np.nan
                rmse_early = 0.0 if not np.isfinite(rmse_early) else rmse_early

                late_model_results = []
                for late_model in transition_late_models:
                    try:
                        late_model = str(late_model).lower().strip()
                        if late_model == "mono":
                            f_inf0_mono_c = float(np.clip(np.nanmedian(F_late_c_norm[-min(30, len(F_late_c_norm)):]), 0.0, 1.2))
                            p0_mono_c = [f_inf0_mono_c, max(t_max / 5, 1e-4)]
                            popt_late_c, _ = curve_fit(
                                mono_exp,
                                t_late_c,
                                F_late_c_norm,
                                p0=p0_mono_c,
                                maxfev=20000,
                                bounds=([0.0, 1e-4], [1.2, np.inf]),
                            )
                            late_pred_full = mono_exp(t_late_c, *popt_late_c)
                            y_join_late = float(mono_exp(np.array([candidate_t]), *popt_late_c)[0])
                            head_mask = t_late_c <= (candidate_t + local_window)
                            err_late = F_late_c_norm[head_mask] - mono_exp(t_late_c[head_mask], *popt_late_c)
                        elif late_model == "plr":
                            p0_plr_c = [1.0, max(float(candidate_t), 1e-6), 0.2]
                            lb_plr_c = [0.2, 1e-8, 0.01]
                            ub_plr_c = [2.0, max(float(t_late_c[-1]), 1e-4), 2.0]
                            popt_late_c, _ = curve_fit(
                                plr,
                                t_late_c,
                                F_late_c_norm,
                                p0=p0_plr_c,
                                bounds=(lb_plr_c, ub_plr_c),
                                maxfev=30000,
                            )
                            late_pred_full = plr(t_late_c, *popt_late_c)
                            y_join_late = float(plr(np.array([candidate_t]), *popt_late_c)[0])
                            head_mask = t_late_c <= (candidate_t + local_window)
                            err_late = F_late_c_norm[head_mask] - plr(t_late_c[head_mask], *popt_late_c)
                        elif late_model == "bi":
                            f_inf0_bi_c = float(np.clip(np.nanmedian(F_late_c_norm[-min(30, len(F_late_c_norm)):]), 0.0, 1.2))
                            p0_bi_c = [f_inf0_bi_c, 0.5, max(t_late_c[-1] / 20, 1e-4), max(t_late_c[-1] / 2, 1e-3)]
                            lb_bi_c = [0.0, 0.0, 1e-4, 1e-4]
                            ub_bi_c = [1.2, 1.0, max(t_late_c[-1] * 10, 1e-2), max(t_late_c[-1] * 10, 1e-2)]
                            popt_late_c, _ = curve_fit(
                                bi_exp,
                                t_late_c,
                                F_late_c_norm,
                                p0=p0_bi_c,
                                bounds=(lb_bi_c, ub_bi_c),
                                maxfev=50000,
                            )
                            late_pred_full = bi_exp(t_late_c, *popt_late_c)
                            y_join_late = float(bi_exp(np.array([candidate_t]), *popt_late_c)[0])
                            head_mask = t_late_c <= (candidate_t + local_window)
                            err_late = F_late_c_norm[head_mask] - bi_exp(t_late_c[head_mask], *popt_late_c)
                        else:
                            continue

                        rmse_late = float(np.sqrt(np.nanmean(err_late ** 2))) if np.any(np.isfinite(err_late)) else np.nan
                        rmse_late = 0.0 if not np.isfinite(rmse_late) else rmse_late
                        join_gap = abs(y_join_early - y_join_late)
                        late_r2 = float(r2(F_late_c_norm, late_pred_full)) if np.any(np.isfinite(late_pred_full)) else np.nan
                        score = join_gap + 0.5 * (rmse_early + rmse_late)
                        accepted_model = bool(
                            np.isfinite(score)
                            and join_gap <= float(transition_max_join_gap)
                            and rmse_early <= float(transition_max_local_rmse)
                            and rmse_late <= float(transition_max_local_rmse)
                            and np.isfinite(late_r2)
                            and late_r2 >= float(transition_min_late_r2)
                        )
                        late_model_results.append(
                            {
                                "late_model": late_model,
                                "late_r2": late_r2,
                                "join_gap": float(join_gap),
                                "rmse_late": float(rmse_late),
                                "score": float(score),
                                "accepted": accepted_model,
                            }
                        )
                    except Exception:
                        continue

                if not late_model_results:
                    continue

                best_late_for_candidate = min(late_model_results, key=lambda x: x["score"])
                join_gap = float(best_late_for_candidate["join_gap"])
                rmse_late = float(best_late_for_candidate["rmse_late"])
                score = float(best_late_for_candidate["score"])
                late_model_selected = str(best_late_for_candidate["late_model"])
                late_r2_selected = float(best_late_for_candidate["late_r2"])
                accepted = bool(
                    np.isfinite(score)
                    and join_gap <= float(transition_max_join_gap)
                    and rmse_early <= float(transition_max_local_rmse)
                    and rmse_late <= float(transition_max_local_rmse)
                    and np.isfinite(late_r2_selected)
                    and late_r2_selected >= float(transition_min_late_r2)
                )

                transition_eval_rows.append(
                    {
                        "candidate_t_s": float(candidate_t),
                        "join_gap": float(join_gap),
                        "rmse_early": float(rmse_early),
                        "rmse_late": float(rmse_late),
                        "score": float(score),
                        "late_model": late_model_selected,
                        "late_r2": float(late_r2_selected),
                        "accepted": accepted,
                        "valid": bool(np.isfinite(score)),
                    }
                )

                transition_search_info["n_valid_candidates"] += 1
                if accepted:
                    transition_search_info["n_accepted_candidates"] += 1
                if accepted and score < best_score:
                    best_score = float(score)
                    best_candidate = float(candidate_t)
                    best_late_model = late_model_selected
                    best_late_r2 = float(late_r2_selected)

                for lm_result in late_model_results:
                    lm_name = str(lm_result.get("late_model", "")).lower().strip()
                    if lm_name not in model_best_tracker:
                        continue
                    lm_score = float(lm_result.get("score", np.nan))
                    lm_accepted = bool(lm_result.get("accepted", False))
                    if lm_accepted and np.isfinite(lm_score) and lm_score < float(model_best_tracker[lm_name]["score"]):
                        model_best_tracker[lm_name]["score"] = lm_score
                        model_best_tracker[lm_name]["t"] = float(candidate_t)
            except Exception:
                continue

        if model_best_tracker:
            transition_search_info["best_transition_time_by_model_s"] = {
                model_name: (
                    float(model_info["t"]) if np.isfinite(model_info["t"]) else np.nan
                )
                for model_name, model_info in model_best_tracker.items()
            }
            transition_search_info["best_transition_score_by_model"] = {
                model_name: (
                    float(model_info["score"])
                    if np.isfinite(model_info["score"])
                    else np.nan
                )
                for model_name, model_info in model_best_tracker.items()
            }

        if best_candidate is not None:
            transition_selected = float(best_candidate)
            transition_search_info["selected_transition_time_s"] = transition_selected
            transition_search_info["best_score"] = float(best_score)
            transition_search_info["selection_reason"] = "auto_accepted"
            transition_search_info["selected_late_model"] = best_late_model
            transition_search_info["selected_late_model_r2"] = float(best_late_r2)
        else:
            transition_search_info["used_fallback_transition"] = True
            transition_search_info["selection_reason"] = "fallback_manual_no_accepted_candidate"
            print(
                "[Transizione poro/visco] Nessun candidato con raccordo accettabile "
                f"(gap<={float(transition_max_join_gap):.3f}, rmse<={float(transition_max_local_rmse):.3f}, "
                f"R²late>={float(transition_min_late_r2):.3f}). "
                f"Uso tempo manuale/clippato: {transition_selected:.3f} s"
            )

    transition_search_df = pd.DataFrame(transition_eval_rows)
    if not transition_search_df.empty:
        transition_search_df = transition_search_df.sort_values("candidate_t_s").reset_index(drop=True)

    poro_fit_max_time = transition_selected
    t_poro_stop = min(poro_fit_max_time, t_max)
    post_domain_label = f"post-transizione (t≥{t_poro_stop:.3f}s)"
    early_domain_label = f"early-phase (0-{t_poro_stop:.3f}s)"

    # Fit globali escludono la fase iniziale (< t_poro_stop)
    global_fit_mask = t_fit >= t_poro_stop
    t_fit_global = t_fit[global_fit_mask]
    Ft_fit_global = Ft_fit_scaled[global_fit_mask]
    Ft_fit_global_norm = Ft_fit_global / force_scale

    if len(t_fit_global) < min_points_per_phase:
        raise ValueError(f"Troppi pochi punti disponibili dopo {t_poro_stop:.3f} s per i fit globali")

    t_fit_log = t_fit_global[t_fit_global > 0]
    Ft_fit_log = (Ft_fit_global / force_scale)[t_fit_global > 0]

    print(f"Transizione early/global selezionata: t = {t_poro_stop:.3f} s")
    if auto_optimize_transition:
        print(
            "Ricerca automatica transizione: "
            f"{transition_search_info['n_valid_candidates']}/{transition_search_info['n_candidates']} candidati validi"
        )
        print(
            "Candidati ben raccordati: "
            f"{transition_search_info['n_accepted_candidates']} "
            f"(gap<={float(transition_max_join_gap):.3f}, rmse<={float(transition_max_local_rmse):.3f}, "
            f"R²late>={float(transition_min_late_r2):.3f})"
        )
        if np.isfinite(transition_search_info.get("best_score", np.nan)):
            print(f"Miglior score di raccordo: {transition_search_info['best_score']:.6f}")
        if transition_search_info.get("selected_late_model") is not None:
            print(
                "Modello late che guida la congiunzione: "
                f"{MODEL_LABEL_MAP.get(transition_search_info['selected_late_model'], transition_search_info['selected_late_model'])} "
                f"(R²late={float(transition_search_info.get('selected_late_model_r2', np.nan)):.4f})"
            )
        if transition_search_info.get("used_fallback_transition", False):
            print("Selezione transizione: fallback al valore manuale (nessun candidato accettato)")
        else:
            print("Selezione transizione: automatica su candidati accettati")
    print(
        f"Punti usati per il fit globale: {len(t_fit_global)} / {len(t)}  |  "
        f"intervallo: {t_poro_stop:.3f} - {t_max:.2f} s  |  unità forza: {force_unit}"
    )
    print("Nota: SLS, PLR e Bi-exp sono tutti fittati sul tratto globale post-transizione.")
    print(f"Fit poroelastico limitato a: 0 - {t_poro_stop:.2f} s")

    try:
        f_inf0 = float(np.clip(np.nanmedian(Ft_fit_global_norm[-min(30, len(Ft_fit_global_norm)):]), 0.0, 1.2))
        p0 = [f_inf0, max(t_max / 5, 1e-4)]
        popt_m, pcov_m = curve_fit(
            mono_exp,
            t_fit_global,
            Ft_fit_global_norm,
            p0=p0,
            maxfev=10000,
            bounds=([0.0, 1e-4], [1.2, np.inf]),
        )
        r2_m = r2(Ft_fit_global_norm, mono_exp(t_fit_global, *popt_m))
        fit_results["mono"] = popt_m
        fit_covariances["mono"] = pcov_m
        print(f"[1] Mono-exp (SLS): F∞ = {popt_m[0]:.3f}   τ = {popt_m[1]:.3f} s   R² = {r2_m:.4f}")
    except Exception as exc:
        popt_m, pcov_m, r2_m = None, None, float("nan")
        fit_results["mono"] = None
        fit_covariances["mono"] = None
        print(f"[1] Mono-exp fallito: {exc}")

    try:
        # t_fit_global parte già da >= t_poro_stop; si usano tutti i punti validi
        sign_plr = -1.0 if np.nanmedian(Ft_fit_global_norm) < 0 else 1.0
        y_plr_abs_all = sign_plr * Ft_fit_global_norm
        mask_valid_plr = np.isfinite(y_plr_abs_all) & (y_plr_abs_all > 0)
        if not np.any(mask_valid_plr):
            raise ValueError("nessun punto positivo valido per PLR dopo correzione di segno")

        t_plr = t_fit_global[mask_valid_plr]
        y_plr_abs = y_plr_abs_all[mask_valid_plr]

        # A_PLR è fissato a 1.0: si fitta solo (t/t0)^(-beta)
        if plr_fixed_t0 is None:
            def _plr_free_t0_model(time_values, t0_value, beta_value):
                return plr(time_values, 1.0, t0_value, beta_value)

            p0 = [max(float(t_plr[0]), 1e-6), 0.2]
            popt_plr_red, pcov_plr_abs = curve_fit(
                _plr_free_t0_model,
                t_plr,
                y_plr_abs,
                p0=p0,
                maxfev=20000,
                bounds=([1e-6, 0.01], [np.inf, 2.0]),
            )
            y_pred_abs = _plr_free_t0_model(t_plr, *popt_plr_red)
            plr_t0_value = float(popt_plr_red[0])
            popt_plr_abs = np.array([1.0, popt_plr_red[0], popt_plr_red[1]], dtype=float)
            fit_meta["plr"] = {"fit_param_names": ["t0", "beta"], "fixed_params": {"A_PLR": 1.0}}
        else:
            plr_t0_value = t_poro_stop if plr_fixed_t0 == "fit_start" else float(plr_fixed_t0)

            def _plr_fixed_t0_model(time_values, beta_value):
                return plr(time_values, 1.0, plr_t0_value, beta_value)

            p0 = [0.2]
            popt_plr_red, pcov_plr_abs = curve_fit(
                _plr_fixed_t0_model,
                t_plr,
                y_plr_abs,
                p0=p0,
                maxfev=20000,
                bounds=([0.01], [2.0]),
            )
            y_pred_abs = _plr_fixed_t0_model(t_plr, *popt_plr_red)
            popt_plr_abs = np.array([1.0, plr_t0_value, popt_plr_red[0]], dtype=float)
            fit_meta["plr"] = {"fit_param_names": ["beta"], "fixed_params": {"A_PLR": 1.0, "t0": plr_t0_value}}

        r2_plr = r2(y_plr_abs, y_pred_abs)
        popt_plr = popt_plr_abs  # A_PLR è sempre 1.0
        fit_results["plr"] = popt_plr
        fit_covariances["plr"] = pcov_plr_abs
        if "t0" in fit_meta["plr"]["fixed_params"]:
            print(f"[2] PLR puro: β = {popt_plr[2]:.3f}   t0 fisso = {popt_plr[1]:.4f} s   R² = {r2_plr:.4f}")
        else:
            print(f"[2] PLR puro: β = {popt_plr[2]:.3f}   t0 = {popt_plr[1]:.4f} s   R² = {r2_plr:.4f}")
    except Exception as exc:
        popt_plr, pcov_plr_abs, r2_plr = None, None, float("nan")
        fit_results["plr"] = None
        fit_covariances["plr"] = None
        fit_meta["plr"] = {"fit_param_names": ["t0", "beta"], "fixed_params": {"A_PLR": 1.0}}
        print(f"[2] PLR puro fallito: {exc}")

    # --- Modulo di Young dal fit PLR (Hertz sferico, indentazione costante) ---
    young_modulus_pa = np.nan
    young_modulus_kpa = np.nan
    young_message = "non calcolato"
    young_diagnostics = {
        "is_normalized_input": bool(sr.get("is_normalized", False)),
        "A_PLR_fit": 1.0,
        "f_peak_mean_N": float("nan"),
        "F0_from_plr_N": float("nan"),
        "F0_from_norm_only_N": float("nan"),
        "tip_radius_m": float("nan"),
        "indentation_m": float("nan"),
        "nu": float("nan"),
        "geom_term": float("nan"),
        "young_prefactor": float("nan"),
        "young_pa": float("nan"),
        "young_if_delta_half_pa": float("nan"),
        "young_if_delta_double_pa": float("nan"),
    }
    auto_indentation_m = float(sr.get("estimated_indentation_m", np.nan))
    indentation_for_young_m = hertz_indentation_m
    if indentation_for_young_m is None and np.isfinite(auto_indentation_m) and auto_indentation_m > 0:
        indentation_for_young_m = auto_indentation_m
        print(
            "    Indentazione auto dal carico pre-normalizzazione: "
            f"δ0 = {auto_indentation_m * 1e9:.3f} nm"
        )

    if hertz_tip_radius_m is None or indentation_for_young_m is None:
        young_message = "non calcolato: imposta hertz_tip_radius_m e hertz_indentation_m"
    elif popt_plr is None:
        young_message = "non calcolato: fit PLR non riuscito"
    else:
        try:
            tip_r = float(hertz_tip_radius_m)
            delta0 = float(indentation_for_young_m)
            nu = float(hertz_poisson)
            young_diagnostics["tip_radius_m"] = tip_r
            young_diagnostics["indentation_m"] = delta0
            young_diagnostics["nu"] = nu
            if tip_r <= 0 or delta0 <= 0:
                raise ValueError("tip radius e indentazione devono essere > 0")
            if not (-0.99 < nu < 0.5):
                raise ValueError("poisson deve essere nell'intervallo (-0.99, 0.5)")
            # A_PLR=1 fisso: F0 dipende dal tipo di curva
            if sr.get("is_normalized", False):
                # Curva normalizzata: serve il picco reale pre-normalizzazione
                f_peak_mean_N = float(sr.get("f_peak_mean_N", np.nan))
                young_diagnostics["f_peak_mean_N"] = f_peak_mean_N
                if not np.isfinite(f_peak_mean_N):
                    raise ValueError(
                        "f_peak_mean_N non disponibile: rielabora la cartella partendo dai file .jpk-force grezzi"
                    )
                F0_N = f_peak_mean_N
                young_diagnostics["F0_from_norm_only_N"] = f_peak_mean_N
            else:
                # Curva grezza: F_peak è già in Newton (force_scale serve solo per la visualizzazione)
                F0_N = float(F_peak)
                young_diagnostics["f_peak_mean_N"] = F0_N
                young_diagnostics["F0_from_norm_only_N"] = F0_N
            geom = (4.0 / 3.0) * np.sqrt(tip_r) * (delta0 ** 1.5)
            young_prefactor = (1.0 - nu ** 2) / geom
            young_modulus_pa = F0_N * young_prefactor
            young_modulus_kpa = young_modulus_pa / 1e3
            young_message = "calcolato"
            young_diagnostics["F0_from_plr_N"] = float(F0_N)
            young_diagnostics["geom_term"] = float(geom)
            young_diagnostics["young_prefactor"] = float(young_prefactor)
            young_diagnostics["young_pa"] = float(young_modulus_pa)
            young_diagnostics["young_if_delta_half_pa"] = float(
                (F0_N * (1.0 - nu ** 2))
                / ((4.0 / 3.0) * np.sqrt(tip_r) * ((delta0 * 0.5) ** 1.5))
            )
            young_diagnostics["young_if_delta_double_pa"] = float(
                (F0_N * (1.0 - nu ** 2))
                / ((4.0 / 3.0) * np.sqrt(tip_r) * ((delta0 * 2.0) ** 1.5))
            )
            print(f"    Young (Hertz sferico, PLR): E = {young_modulus_pa:.3e} Pa  ({young_modulus_kpa:.3f} kPa)")
        except Exception as _young_exc:
            young_message = f"non calcolato: {_young_exc}"
            young_modulus_pa = np.nan
            young_modulus_kpa = np.nan

    try:
        f_inf0_bi = float(np.clip(np.nanmedian(Ft_fit_global_norm[-min(30, len(Ft_fit_global_norm)):]), 0.0, 1.2))
        p0 = [f_inf0_bi, 0.5, max(t_max / 20, 1e-4), max(t_max / 3, 1e-4)]
        lb = [0.0, 0.0, 1e-4, 1e-4]
        ub = [1.2, 1.0, t_max, np.inf]
        popt_b, pcov_b = curve_fit(bi_exp, t_fit_global, Ft_fit_global_norm, p0=p0, maxfev=30000, bounds=(lb, ub))
        if popt_b[2] > popt_b[3]:
            popt_b = np.array([popt_b[0], 1.0 - popt_b[1], popt_b[3], popt_b[2]])
            if pcov_b is not None and np.shape(pcov_b) == (4, 4):
                order = [0, 1, 3, 2]
                pcov_b = pcov_b[np.ix_(order, order)]
        r2_b = r2(Ft_fit_global_norm, bi_exp(t_fit_global, *popt_b))
        fit_results["bi"] = popt_b
        fit_covariances["bi"] = pcov_b
        print(
            f"[3] Bi-exp (GM): F∞ = {popt_b[0]:.3f}   w = {popt_b[1]:.3f}   "
            f"τ1 = {popt_b[2]:.3f} s   τ2 = {popt_b[3]:.3f} s   R² = {r2_b:.4f}"
        )
    except Exception as exc:
        popt_b, pcov_b, r2_b = None, None, float("nan")
        fit_results["bi"] = None
        fit_covariances["bi"] = None
        print(f"[3] Bi-exp fallito: {exc}")

    early_mask = (t_fit >= 0.0) & (t_fit <= t_poro_stop)
    if np.sum(early_mask) < min_points_per_phase:
        raise ValueError(f"Troppi pochi punti disponibili entro {t_poro_stop:.2f} s per il fit poroelastico")

    t_early = t_fit[early_mask]
    F_early = Ft_fit_scaled[early_mask]
    F_early_norm = F_early / force_scale
    print(f"Intervallo early-phase: 0 - {t_early[-1]:.3f} s  ({len(t_early)} punti)")

    try:
        f_inf0_poro = float(np.clip(np.nanmedian(F_early_norm[-min(30, len(F_early_norm)):]), 0.0, 1.2))
        p0 = [f_inf0_poro, max(t_early[-1] / 4, 1e-4)]
        lb = [0.0, 1e-6]
        ub = [1.2, max(t_early[-1] * 10, 1e-3)]
        popt_poro, pcov_poro = curve_fit(
            poro_relax,
            t_early,
            F_early_norm,
            p0=p0,
            bounds=(lb, ub),
            maxfev=30000,
        )
        r2_poro_early = r2(F_early_norm, poro_relax(t_early, *popt_poro))
        fit_results["poro"] = popt_poro
        fit_covariances["poro"] = pcov_poro
        print(f"[4] Poroelastico: F∞ = {popt_poro[0]:.3f}   τp = {popt_poro[1]:.4f} s   R²(early) = {r2_poro_early:.4f}")
    except Exception as exc:
        popt_poro, pcov_poro, r2_poro_early = None, None, float("nan")
        fit_results["poro"] = None
        fit_covariances["poro"] = None
        print(f"[4] Poroelastico fallito: {exc}")

    global_metrics = {k: {"R2": np.nan, "AIC": np.nan, "BIC": np.nan} for k in ["mono", "plr", "bi", "poro"]}
    early_metrics = {k: {"R2": np.nan, "AIC": np.nan, "BIC": np.nan} for k in ["mono", "plr", "bi", "poro"]}

    if fit_results.get("mono") is not None:
        pred = mono_exp(t_fit_global, *fit_results["mono"])
        aic, bic = information_criteria(Ft_fit_global_norm, pred, 2)
        global_metrics["mono"] = {"R2": r2_m, "AIC": aic, "BIC": bic}

    if fit_results.get("plr") is not None:
        pred = plr(t_fit_global, *fit_results["plr"])
        mask = np.isfinite(pred)
        n_plr_params = len(fit_meta.get("plr", {}).get("fit_param_names", ["t0", "beta"]))
        aic, bic = information_criteria(Ft_fit_global_norm[mask], pred[mask], n_plr_params)
        global_metrics["plr"] = {"R2": r2_plr, "AIC": aic, "BIC": bic}

    if fit_results.get("bi") is not None:
        pred = bi_exp(t_fit_global, *fit_results["bi"])
        aic, bic = information_criteria(Ft_fit_global_norm, pred, 4)
        global_metrics["bi"] = {"R2": r2_b, "AIC": aic, "BIC": bic}

    if fit_results.get("poro") is not None:
        pred = poro_relax(t_early, *fit_results["poro"])
        aic, bic = information_criteria(F_early_norm, pred, 2)
        global_metrics["poro"] = {"R2": r2_poro_early, "AIC": aic, "BIC": bic}
        early_metrics["poro"] = {"R2": r2_poro_early, "AIC": aic, "BIC": bic}

    global_comp_metrics = {k: global_metrics[k] for k in ["mono", "plr", "bi"]}
    early_comp_metrics = {k: early_metrics[k] for k in ["poro"]}
    fit_diagnostics = {}
    model_recommendations = {}

    print(f"\nMetriche nella sola fase iniziale (0 → {t_poro_stop:.3f} s):")
    print("  (in early-phase viene valutato solo il poroelastico)")
    met = early_metrics["poro"]
    if np.isfinite(met["R2"]):
        print(f"  {MODEL_LABEL_MAP['poro']:12s}: R²={met['R2']:.4f}   AIC={met['AIC']:.2f}   BIC={met['BIC']:.2f}")
    else:
        print(f"  {MODEL_LABEL_MAP['poro']:12s}: n.d.")

    summary_rows = []
    for domain_label, metrics_source, keys in [
        (post_domain_label, global_comp_metrics, ["mono", "plr", "bi"]),
        (early_domain_label, early_comp_metrics, ["poro"]),
    ]:
        for key in keys:
            met = metrics_source.get(key, {})
            if not np.isfinite(met.get("R2", np.nan)):
                continue
            summary_rows.append(
                {
                    "dominio": domain_label,
                    "modello": MODEL_LABEL_MAP[key],
                    "parametri": format_parametri(key, fit_results.get(key), force_unit),
                    "R2": float(met["R2"]),
                }
            )

    tabella_finale_df = pd.DataFrame(summary_rows)
    tabella_finale_df = tabella_finale_df.sort_values(["dominio", "R2"], ascending=[True, False]).reset_index(drop=True)
    tabella_finale_display_df = tabella_finale_df[["dominio", "modello", "parametri", "R2"]].copy()
    tabella_finale_display_df["R2"] = tabella_finale_display_df["R2"].round(4)

    fig_main, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]

    if not sr["is_normalized"] and len(sr["t_ext"]) > 0 and len(sr["t_ret"]) > 0:
        t_ext_plot = sr["t_ext"]
        t_pause_plot = t_ext_plot[-1] + t_full
        t_ret_plot = t_pause_plot[-1] + sr["t_ret"]
        ax.plot(t_ext_plot, sr["force_ext"] * force_scale, color="steelblue", lw=1.2, label="Avvicinamento")
        ax.plot(t_pause_plot, Ft_processed_full * force_scale, color="darkorange", lw=1.2, label="Dwell (SR)")
        ax.plot(t_ret_plot, sr["force_ret"] * force_scale, color="seagreen", lw=1.2, label="Retract")
        f_rel = np.concatenate([Ft_processed_full * force_scale, sr["force_ret"] * force_scale])
    else:
        ax.plot(
            t_full,
            Ft_processed_full * force_scale,
            color="darkorange",
            lw=1.4,
            label="Dwell normalizzato" if sr["is_normalized"] else "Dwell (SR)",
        )
        f_rel = Ft_processed_full * force_scale

    ax.axhline(F_peak * force_scale, color="red", ls="--", lw=0.8, label=f"F_peak = {F_peak * force_scale:.3f} {force_unit}")
    ax.axhline(F_relax * force_scale, color="purple", ls="--", lw=0.8, label=f"F_relax = {F_relax * force_scale:.3f} {force_unit}")
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel(force_axis_label)
    ax.set_title("Curva SR completa")
    ax.legend(fontsize=9)

    f_lo, f_hi = np.nanmin(f_rel), np.nanmax(f_rel)
    margin = (f_hi - f_lo) * 0.25 if f_hi > f_lo else 0.2
    ax.set_ylim(f_lo - margin, f_hi + margin)
    ax.set_xlim(0.0, float(t_full[-1]))

    ax2 = axes[1]
    t_global_start = float(t_fit_global[0])
    data_mask = (t >= 0.0) & (t <= t_fit_global[-1])
    ax2.plot(
        t[data_mask],
        Ft_processed[data_mask] * force_scale,
        color="darkorange",
        lw=1.0,
        alpha=0.55,
        label="Dati nel tratto di fit",
    )

    t_fine = np.linspace(t_global_start, t_fit_global[-1], 1500)
    t_fine_poro = np.linspace(t_early[0], t_early[-1], 500)
    model_specs = [
        ("mono", "royalblue", "-", t_fine, lambda tt, p: mono_exp(tt, *p), "SLS"),
        ("plr", "chocolate", "-", t_fine, lambda tt, p: plr(tt, *p), "PLR"),
        ("bi", "teal", "-", t_fine, lambda tt, p: bi_exp(tt, *p), "Bi-exp"),
        ("poro", "firebrick", "--", t_fine_poro, lambda tt, p: poro_relax(tt, *p), "Poroelastico"),
    ]
    r2_map = {"mono": r2_m, "plr": r2_plr, "bi": r2_b, "poro": early_metrics["poro"]["R2"]}

    for key, color, ls, t_plot, fn, legend_label in model_specs:
        if fit_results.get(key) is not None:
            y_plot = fn(t_plot, fit_results[key]) * force_scale
            mask_plot = np.isfinite(y_plot)
            ax2.plot(t_plot[mask_plot], y_plot[mask_plot], ls=ls, color=color, lw=2.2, label=legend_label)

    ax2.axhline(F_peak * force_scale, color="red", ls="--", lw=0.8, alpha=0.6)
    ax2.axhline(F_relax * force_scale, color="purple", ls="--", lw=0.8, alpha=0.6)
    ax2.axvline(
        t_poro_stop,
        color="firebrick",
        ls=":",
        lw=1.0,
        alpha=0.8,
        label=f"Fine early-phase = {t_poro_stop:.4f} s",
    )
    ax2.set_xlim(0.0, t_fit_global[-1])
    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel(force_axis_label)
    ax2.set_title("Confronto modelli SR (post-transizione)")
    ax2.legend(fontsize=8, loc="best")
    fig_main.tight_layout()
    plt.show()

    fig_log = None
    loglog_plr_df = pd.DataFrame()
    if len(t_fit_log) > 0:
        y_data_log = Ft_fit_log * force_scale
        y_plr_log = plr(t_fit_log, *popt_plr) if popt_plr is not None else np.full_like(y_data_log, np.nan, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            loglog_plr_df = pd.DataFrame(
                {
                    "tempo_s": t_fit_log,
                    "forza_dati": y_data_log,
                    "forza_plr_fit": y_plr_log,
                    "log10_tempo_s": np.log10(t_fit_log),
                    "log10_forza_dati": np.log10(np.where(y_data_log > 0, y_data_log, np.nan)),
                    "log10_forza_plr": np.log10(np.where(y_plr_log > 0, y_plr_log, np.nan)),
                }
            )

        fig_log, ax_log = plt.subplots(figsize=(7, 5))
        ax_log.loglog(t_fit_log, y_data_log, "o", ms=3, alpha=0.6, label="Dati (campionati)")
        if popt_plr is not None:
            ax_log.loglog(t_fit_log, y_plr_log, "-", lw=2.2, color="forestgreen", label="PLR puro")
        ax_log.set_xlabel("Tempo [s]")
        ax_log.set_ylabel(f"Forza [{force_unit}]")
        ax_log.set_title("Stress relaxation in scala log-log — solo PLR")
        ax_log.legend()
        ax_log.grid(True, which="both", ls=":")
        fig_log.tight_layout()
        plt.show()

    fig_transition = None
    transition_top_df = pd.DataFrame()
    if auto_optimize_transition and not transition_search_df.empty:
        best_t = float(transition_search_info.get("selected_transition_time_s", t_poro_stop))
        selected_df = transition_search_df[np.isclose(transition_search_df["candidate_t_s"], best_t, atol=1e-9)].copy()
        if selected_df.empty:
            selected_df = transition_search_df.nsmallest(1, "score").copy()

        model_time_map = transition_search_info.get("best_transition_time_by_model_s", {}) or {}
        requested_models = [m for m in ["bi", "plr"] if str(m) in set(transition_search_df["late_model"].astype(str).str.lower())]
        model_rows = []
        for model_name in requested_models:
            model_t = model_time_map.get(model_name, np.nan)
            if not np.isfinite(model_t):
                continue
            model_df = transition_search_df[
                (transition_search_df["late_model"].astype(str).str.lower() == model_name)
                & np.isclose(transition_search_df["candidate_t_s"], float(model_t), atol=1e-9)
            ].copy()
            if model_df.empty:
                model_df = transition_search_df[
                    transition_search_df["late_model"].astype(str).str.lower() == model_name
                ].nsmallest(1, "score").copy()
            if not model_df.empty:
                model_rows.append(model_df.iloc[0])

        if model_rows:
            transition_top_df = pd.DataFrame(model_rows).copy()
            transition_top_df = transition_top_df.drop_duplicates(subset=["late_model"], keep="first")
            model_order = {"bi": 0, "plr": 1}
            transition_top_df["_model_order"] = transition_top_df["late_model"].astype(str).str.lower().map(model_order).fillna(99)
            transition_top_df = transition_top_df.sort_values(["_model_order", "score"]).drop(columns=["_model_order"])
        else:
            transition_top_df = selected_df.head(1).copy()

        for col in ["candidate_t_s", "late_r2", "score"]:
            if col in transition_top_df.columns:
                transition_top_df[col] = transition_top_df[col].round(6)

        print("\nTempo di transizione poroelastico/viscoelastico:")
        display(
            transition_top_df[
                ["candidate_t_s", "late_model", "late_r2", "score"]
            ]
        )

        if bool(show_transition_diagnostic_plot):
            fig_transition, ax_transition = plt.subplots(figsize=(7, 4.8))
            ax_transition.plot(
                transition_search_df["candidate_t_s"],
                transition_search_df["score"],
                "o-",
                lw=1.6,
                ms=4,
                color="slateblue",
                label="Score raccordo",
            )
            if np.any(np.isfinite(transition_search_df["join_gap"])):
                ax_transition.plot(
                    transition_search_df["candidate_t_s"],
                    transition_search_df["join_gap"],
                    "--",
                    lw=1.2,
                    color="firebrick",
                    alpha=0.85,
                    label="Gap alla giunzione",
                )

            best_score = float(transition_search_info.get("best_score", np.nan))
            ax_transition.axvline(best_t, color="black", ls=":", lw=1.2, alpha=0.9, label=f"t* = {best_t:.3f} s")
            if np.isfinite(best_score):
                ax_transition.scatter([best_t], [best_score], color="black", s=35, zorder=5)

            ax_transition.set_xlabel("Tempo candidato [s]")
            ax_transition.set_ylabel("Score")
            ax_transition.set_title("Diagnostica transizione poro/visco")
            ax_transition.grid(True, alpha=0.25)
            ax_transition.legend(fontsize=8)
            fig_transition.tight_layout()
            plt.show()

    print("\nTabella finale riassuntiva dei fit:")
    display(tabella_finale_display_df)

    loglog_pdf_df = pd.DataFrame()
    if not loglog_plr_df.empty:
        n_log_rows = min(36, len(loglog_plr_df))
        idx_log = np.unique(np.round(np.linspace(0, len(loglog_plr_df) - 1, n_log_rows)).astype(int))
        loglog_pdf_df = loglog_plr_df.iloc[idx_log].reset_index(drop=True)
        for col in loglog_pdf_df.columns:
            if np.issubdtype(loglog_pdf_df[col].dtype, np.number):
                loglog_pdf_df[col] = loglog_pdf_df[col].round(6)

    report_unico_path = os.path.join(save_dir, f"{base_name}_sr_report_unico.pdf")
    young_summary_path = os.path.join(save_dir, f"{base_name}_young_summary.json")

    selected_transition_s = float(t_poro_stop)
    requested_transition_s = float(transition_search_info.get("requested_transition_time_s", transition_time_s))
    delta_vs_requested_s = selected_transition_s - requested_transition_s
    accepted_candidates_df = pd.DataFrame()
    if isinstance(transition_search_df, pd.DataFrame) and not transition_search_df.empty and "accepted" in transition_search_df.columns:
        accepted_candidates_df = transition_search_df[transition_search_df["accepted"] == True].copy()

    transition_ci_10_90_s = [None, None]
    if not accepted_candidates_df.empty:
        q10 = float(np.nanpercentile(accepted_candidates_df["candidate_t_s"], 10))
        q90 = float(np.nanpercentile(accepted_candidates_df["candidate_t_s"], 90))
        transition_ci_10_90_s = [q10, q90]

    if save_outputs:
        legacy_outputs = [
            f"{base_name}_sr_fit_results.json",
            f"{base_name}_sr_fit.png",
            f"{base_name}_sr_report.pdf",
            f"{base_name}_sr_fit_ranking.csv",
            f"{base_name}_sr_fit_ranking.json",
            f"{base_name}_sr_loglog_points.png",
            f"{base_name}_sr_loglog_linear_fit.png",
            f"{base_name}_sr_loglog_linear_fit.json",
            "sr_mean_file_summary.csv",
            "sr_mean_file_summary.json",
        ]
        for file_name in legacy_outputs:
            file_path = os.path.join(save_dir, file_name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

        with PdfPages(report_unico_path) as pdf:
            pdf.savefig(fig_main, bbox_inches="tight")
            if fig_log is not None:
                pdf.savefig(fig_log, bbox_inches="tight")
            if fig_transition is not None:
                pdf.savefig(fig_transition, bbox_inches="tight")
            if not transition_top_df.empty:
                fig_transition_tab, ax_transition_tab = plt.subplots(figsize=(10, 3.2 + 0.5 * len(transition_top_df)))
                ax_transition_tab.axis("off")
                ax_transition_tab.text(
                    0.0,
                    1.02,
                    "Tempo di transizione poroelastico/viscoelastico",
                    transform=ax_transition_tab.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )
                transition_top_pdf_df = transition_top_df[
                    ["candidate_t_s", "late_model", "late_r2", "score"]
                ].copy()
                transition_top_pdf_df.columns = [
                    "tempo_s", "modello_late", "r2_late", "score"
                ]
                table_transition = ax_transition_tab.table(
                    cellText=transition_top_pdf_df.astype(str).values,
                    colLabels=transition_top_pdf_df.columns,
                    cellLoc="center",
                    colLoc="center",
                    colWidths=[0.18, 0.24, 0.18, 0.18],
                    bbox=[0.0, 0.0, 0.98, 0.9],
                )
                table_transition.auto_set_font_size(False)
                table_transition.set_fontsize(9)
                table_transition.scale(1.0, 1.2)
                for (row_idx, _), cell in table_transition.get_celld().items():
                    cell.set_linewidth(0.6)
                    if row_idx == 0:
                        cell.set_facecolor("#e8eef7")
                        cell.set_text_props(weight="bold")
                pdf.savefig(fig_transition_tab, bbox_inches="tight")
                plt.close(fig_transition_tab)

            tabella_pdf_df = tabella_finale_display_df.copy()

            for col_name, width in {
                "dominio": 16,
                "modello": 16,
                "parametri": 44,
            }.items():
                tabella_pdf_df[col_name] = tabella_pdf_df[col_name].apply(lambda x: textwrap.fill(str(x), width=width))

            row_line_counts = [
                max(str(row[col]).count("\n") + 1 for col in tabella_pdf_df.columns)
                for _, row in tabella_pdf_df.iterrows()
            ]
            fig_height = max(7.0, 3.0 + 0.42 * sum(row_line_counts))

            fig_tab = plt.figure(figsize=(18, fig_height))
            ax_tab = fig_tab.add_axes([0.02, 0.02, 0.96, 0.96])
            ax_tab.axis("off")
            summary_lines = [
                f"File medio analizzato: {analysis_label}",
                f"Durata: {t_max:.2f} s   |   Punti: {len(t)}",
                f"F_peak: {F_peak * force_scale:.4f} {force_unit}",
                f"F_relax: {F_relax * force_scale:.4f} {force_unit}",
                f"ΔF: {delta_F * force_scale:.4f} {force_unit} ({delta_F_pct:.2f}%)",
            ]
            ax_tab.text(0.01, 0.985, "\n".join(summary_lines), ha="left", va="top", fontsize=12)

            table = ax_tab.table(
                cellText=tabella_pdf_df.astype(str).values,
                colLabels=tabella_pdf_df.columns,
                cellLoc="center",
                colLoc="center",
                colWidths=[0.27, 0.16, 0.47, 0.10],
                bbox=[0.0, 0.02, 1.0, 0.82],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9.5)
            table.scale(1.0, 1.45)

            base_height = table[(1, 0)].get_height() if len(tabella_pdf_df) > 0 else table[(0, 0)].get_height()
            for (row_idx, col_idx), cell in table.get_celld().items():
                cell.set_linewidth(0.6)
                if row_idx == 0:
                    cell.set_facecolor("#e8eef7")
                    cell.set_text_props(weight="bold", fontsize=10)
                elif col_idx in [0, 2]:
                    cell.set_text_props(ha="left")

            for row_idx, n_lines in enumerate(row_line_counts, start=1):
                row_height = base_height * max(1.0, 0.9 * n_lines)
                for col_idx in range(len(tabella_pdf_df.columns)):
                    table[(row_idx, col_idx)].set_height(row_height)

            pdf.savefig(fig_tab, bbox_inches="tight")
            plt.close(fig_tab)

        young_summary = {
            "analysis_label": analysis_label,
            "data_dir": str(analysis_path) if not isinstance(analysis_path, dict) else analysis_path.get("folder_path"),
            "report_unico_path": report_unico_path,
            "young_modulus_pa": float(young_modulus_pa) if np.isfinite(young_modulus_pa) else None,
            "young_modulus_kpa": float(young_modulus_kpa) if np.isfinite(young_modulus_kpa) else None,
            "young_message": young_message,
            "F_peak": float(F_peak) if np.isfinite(F_peak) else None,
            "F_relax": float(F_relax) if np.isfinite(F_relax) else None,
            "delta_F": float(delta_F) if np.isfinite(delta_F) else None,
            "transition_time_selected_s": selected_transition_s,
            "transition_time_requested_s": requested_transition_s,
            "transition_delta_vs_requested_s": delta_vs_requested_s,
            "transition_selection_reason": transition_search_info.get("selection_reason"),
            "transition_used_fallback": bool(transition_search_info.get("used_fallback_transition", False)),
            "transition_n_candidates": int(transition_search_info.get("n_candidates", 0)),
            "transition_n_valid_candidates": int(transition_search_info.get("n_valid_candidates", 0)),
            "transition_n_accepted_candidates": int(transition_search_info.get("n_accepted_candidates", 0)),
            "transition_selected_late_model": transition_search_info.get("selected_late_model"),
            "transition_selected_late_model_r2": (
                float(transition_search_info.get("selected_late_model_r2"))
                if np.isfinite(transition_search_info.get("selected_late_model_r2", np.nan))
                else None
            ),
            "transition_best_score": (
                float(transition_search_info.get("best_score"))
                if np.isfinite(transition_search_info.get("best_score", np.nan))
                else None
            ),
            "transition_acceptance_max_join_gap": float(transition_search_info.get("acceptance_max_join_gap", np.nan)),
            "transition_acceptance_max_local_rmse": float(transition_search_info.get("acceptance_max_local_rmse", np.nan)),
            "transition_acceptance_min_late_r2": float(transition_search_info.get("acceptance_min_late_r2", np.nan)),
            "transition_ci10_90_s": transition_ci_10_90_s,
        }
        with open(young_summary_path, "w") as f:
            json.dump(young_summary, f, indent=4)

        print(f"\n✅ File medio usato per l'analisi: {analysis_label}")
        print(f"✅ Report unico con grafici principali + tabella fit salvato in: {report_unico_path}")
        print(f"✅ Riepilogo Young salvato in: {young_summary_path}")
        print(
            "✅ Transizione poro/visco: "
            f"richiesta={requested_transition_s:.3f} s, selezionata={selected_transition_s:.3f} s, "
            f"delta={delta_vs_requested_s:+.3f} s"
        )

    if open_output_folder:
        target_to_open = report_unico_path if os.path.exists(report_unico_path) else save_dir
        os.system(f'open "{target_to_open}"')

    return {
        "analysis_label": analysis_label,
        "report_unico_path": report_unico_path,
        "young_summary_path": young_summary_path,
        "tabella_finale_df": tabella_finale_df,
        "tabella_finale_display_df": tabella_finale_display_df,
        "loglog_plr_df": loglog_plr_df,
        "loglog_pdf_df": loglog_pdf_df,
        "transition_search_df": transition_search_df,
        "transition_top_df": transition_top_df,
        "fit_results": fit_results,
        "fit_diagnostics": fit_diagnostics,
        "global_metrics": global_metrics,
        "early_metrics": early_metrics,
        "model_recommendations": model_recommendations,
        "F_peak": F_peak,
        "F_relax": F_relax,
        "delta_F": delta_F,
        "delta_F_pct": delta_F_pct,
        "t_fit": t_fit,
        "Ft_fit": Ft_fit,
        "t_early": t_early,
        "F_early": F_early,
        "transition_time_selected_s": float(t_poro_stop),
        "transition_search_info": transition_search_info,
        "force_scale": force_scale,
        "force_unit": force_unit,
        "force_axis_label": force_axis_label,
        "young_modulus_pa": young_modulus_pa,
        "young_modulus_kpa": young_modulus_kpa,
        "young_message": young_message,
        "young_diagnostics": young_diagnostics,
        "estimated_indentation_m": auto_indentation_m,
        "indentation_used_for_young_m": float(indentation_for_young_m)
        if indentation_for_young_m is not None else np.nan,
    }
