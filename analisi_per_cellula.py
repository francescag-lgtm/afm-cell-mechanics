"""
analisi_per_cellula.py - Analisi AFM con la CELLULA come unita', per linea cellulare.

Lo stesso codice serve per i due sistemi; cambia solo come vengono trovate le cellule:

  sistema = "murine"   la cellula e' una sottocartella: le sue curve (di solito 3)
                       vengono analizzate una per una e poi riassunte per cellula
                       (FD: mediana dei parametri; SR: curva media della cellula);
  sistema = "umane"    ogni curva e' una cellula diversa (6-7 curve per area, una
                       per cellula): ogni curva e' un'unita' a se'. La sottocartella
                       dell'area resta solo come etichetta e NON viene usata per
                       raggruppare.

Struttura delle cartelle attesa (una cartella per giornata e linea):

    <cartella giornata/linea>/
        <cellula o area 1>/FD/*.jpk-force   curve forza-indentazione
        <cellula o area 1>/SR/*.jpk-force   curve di stress-relaxation
        <cellula o area 2>/...

Le curve SR sono riconosciute anche se stanno direttamente nella sottocartella
o se hanno "_SR-" nel nome. I file medi generati in passato (-media, -mean, -norm)
vengono ignorati.

Per ogni linea il programma produce:
  - una tabella con una riga per cellula (FD e SR);
  - la curva SR media della linea, con i suoi fit;
  - il confronto tra le due linee del sistema con il test di Mann-Whitney sulle cellule.
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu

import importlib
import importlib.util
import sys

# I moduli ting_utils, prova e sr_utils vengono caricati SEMPRE dalla stessa
# cartella di questo file: altrimenti Python puo' trovare per primi file con lo
# stesso nome in altre cartelle del percorso di ricerca (es. general/src).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path[:1]:
    sys.path.insert(0, str(_HERE))


def _load_local_module(name):
    path = _HERE / f"{name}.py"
    if not path.is_file():
        module = importlib.import_module(name)
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    print(f"{name}: uso il file {getattr(module, '__file__', '?')}")
    return module


if "prova" not in sys.modules or Path(getattr(sys.modules["prova"], "__file__", "") or "").resolve().parent != _HERE:
    if (_HERE / "prova.py").is_file():
        _load_local_module("prova")
sr_utils = _load_local_module("sr_utils")

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):
        print(obj)

SISTEMI = ("murine", "umane")
FORCE_EXTENSIONS = (".jpk-force", ".jpk")
_GENERATED_TAGS = ("-media", "-mean", "-norm")


# ---------------------------------------------------------------------------
# 1. Ricerca delle curve e delle cellule
# ---------------------------------------------------------------------------

def _natural_key(value):
    text = str(value).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def _in_excluded_folder(path, line_folder):
    """True se la curva sta in una sottocartella da non analizzare: nome che
    inizia con "_" (es. _duplicati) o che contiene "duplicat"."""
    parts = path.relative_to(line_folder).parts[:-1]
    return any(p.startswith("_") or "duplicat" in p.lower() for p in parts)


def _is_raw_curve(path):
    return path.suffix.lower() in FORCE_EXTENSIONS and not any(
        tag in path.stem.lower() for tag in _GENERATED_TAGS
    )


def _curve_kind(path, line_folder):
    """'FD', 'SR' oppure None in base alla posizione o al nome del file."""
    parts = [p.upper() for p in path.relative_to(line_folder).parts[:-1]]
    if "FD" in parts:
        return "FD"
    if "SR" in parts or "_SR-" in path.name.upper():
        return "SR"
    return None


def _unit_folder(path, line_folder):
    """Sottocartella di primo livello (cellula per le murine, area per le umane)."""
    rel = path.relative_to(line_folder)
    if len(rel.parts) < 2:
        return "(radice)"
    return rel.parts[0]


def find_cells(lines, sistema):
    """Trova tutte le cellule di tutte le linee.

    Parametri
    ---------
    lines : dict {nome_linea: [cartelle giornata della linea]}
        Esempio: {"EV": ["/dati/EV_090626", ...], "RAB5": [...]}.
        Il nome della linea e' preso da qui, non dal nome delle cartelle.
    sistema : "murine" oppure "umane".

    Restituisce un DataFrame con una riga per cellula e per tipo di curva:
    linea, giornata, cartella (cellula o area), id_cellula, tipo (FD/SR), curve.
    """
    if sistema not in SISTEMI:
        raise ValueError(f"sistema deve essere uno tra {SISTEMI}")

    rows = []
    for line_name, folders in lines.items():
        for folder in folders:
            line_folder = Path(folder).expanduser().resolve()
            if not line_folder.is_dir():
                print(f"[MANCANTE] {line_folder}")
                continue
            # Etichetta della giornata: nome della cartella; se non contiene una data
            # (6 cifre, es. ".../270326/A1") si aggiunge il nome della cartella superiore.
            day = line_folder.name
            if not re.search(r"\d{6}", day):
                day = f"{line_folder.parent.name}/{day}"
            curves = sorted(
                (p for p in line_folder.rglob("*") if p.is_file() and _is_raw_curve(p) and not _in_excluded_folder(p, line_folder)),
                key=_natural_key,
            )
            grouped = {}
            for path in curves:
                kind = _curve_kind(path, line_folder)
                if kind is None:
                    continue
                unit = _unit_folder(path, line_folder)
                grouped.setdefault((unit, kind), []).append(path)

            for (unit, kind), paths in sorted(grouped.items(), key=lambda x: (_natural_key(x[0][0]), x[0][1])):
                if sistema == "murine":
                    rows.append({
                        "linea": line_name, "giornata": day, "cartella": unit, "tipo": kind,
                        "id_cellula": f"{day}/{unit}", "curve": [str(p) for p in paths],
                    })
                else:
                    for path in paths:
                        rows.append({
                            "linea": line_name, "giornata": day, "cartella": unit, "tipo": kind,
                            "id_cellula": f"{day}/{unit}/{path.stem}", "curve": [str(path)],
                        })

    cells = pd.DataFrame(rows, columns=["linea", "giornata", "cartella", "tipo", "id_cellula", "curve"])
    if cells.empty:
        raise FileNotFoundError("Nessuna curva FD o SR trovata nelle cartelle indicate.")
    cells["n_curve"] = cells["curve"].map(len)
    return cells


def print_cells_overview(cells):
    overview = (
        cells.groupby(["linea", "tipo"])
        .agg(giornate=("giornata", "nunique"), cellule=("id_cellula", "nunique"), curve=("n_curve", "sum"))
        .reset_index()
    )
    print("Cellule trovate:")
    display(overview)
    return overview


# ---------------------------------------------------------------------------
# 2. Curve forza-indentazione (modello di Ting)
# ---------------------------------------------------------------------------

DEFAULT_TING_OPTIONS = {
    "raw_deflection_smooth_win": 5,
    # Resistenza viscosa: "curva" = stimata su ogni curva (distanza tra i rami
    # fuori contatto, come in ting_utils originale); "minimo_cartella" = per ogni
    # cellula/area meta' della distanza minima tra i rami delle sue curve FD (i
    # tether durante il ritorno possono solo aumentare la distanza), poi la
    # mediana di questi valori sulla giornata, usata per tutte le aree della giornata.
    "drag_mode": "curva",
    "param_overrides": {
        "correct_tilt": True,
        "apply_baseline_alignment": False,
        "baseline_align_poc": False,
        "apply_ramp_correction": True,
        "vdragcorr": True,
        # Il trascinamento e' gia' sottratto ai dati (vdragcorr / fixed_drag_N):
        # il termine vdrag*d(delta)/dt del modello di Ting di pyfmrheo va quindi
        # azzerato, altrimenti il trascinamento viene contato due volte
        # (in pyfmrheo le due correzioni sono pensate come alternative).
        "vdrag": 0.0,
        "enforce_hertz_e0_upper_bound": False,
    },
}


def _merge_ting_options(ting_options):
    overrides = (ting_options or {}).get("param_overrides") or {}
    options = {**DEFAULT_TING_OPTIONS, **(ting_options or {})}
    options["param_overrides"] = {
        **DEFAULT_TING_OPTIONS["param_overrides"],
        **overrides,
    }
    return options


# Parametri della tabella per cellula (FD) confrontati tra le linee
FD_PARAMETERS = ["E0_Pa", "betaE", "E_hertz_Pa"]


def _to_float(value):
    """Converte in float; None o valori non numerici diventano NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _plr_from_result(single):
    """(E0, betaE, tc) del fit Ting-PLR, cercati nei punti in cui le varie
    versioni di ting_utils li salvano."""
    for cand in single.get("model_candidates", []) or []:
        if str(cand.get("model")) == "PLR":
            params = cand.get("params", {}) or {}
            return _to_float(params.get("E0")), _to_float(params.get("betaE")), _to_float(params.get("tc"))
    ting = single.get("ting_result")
    if ting is not None:
        values = []
        for name in ("E0", "betaE", "tc"):
            value = getattr(ting, name, None)
            if value is None and hasattr(ting, "best_values"):
                value = ting.best_values.get(name)
            values.append(_to_float(value))
        return tuple(values)
    return np.nan, np.nan, np.nan


def _ting_row(single):
    """Estrae dai risultati di una curva i parametri dei fit usati."""
    row = {"modello_preferito": single.get("selected_model"), "E_hertz_Pa": np.nan}
    hertz = single.get("hertz_result")
    if hertz is not None:
        row["E_hertz_Pa"] = _to_float(getattr(hertz, "E0", np.nan))
    e0, beta, tc = _plr_from_result(single)
    row.update({"E0_Pa": e0, "betaE": beta, "tc_ms": tc * 1e3})
    ting = single.get("ting_result")
    beta_min = _to_float(getattr(ting, "betaE_min", np.nan))
    beta_max = _to_float(getattr(ting, "betaE_max", np.nan))
    row["betaE_at_bound"] = bool(
        np.isfinite(beta)
        and (
            (np.isfinite(beta_min) and abs(beta - beta_min) < 0.005)
            or (np.isfinite(beta_max) and abs(beta - beta_max) < 0.005)
        )
    )
    ramp = single.get("ramp_correction_info") or {}
    row["ramp_applied"] = bool(ramp.get("applied", False))
    row["ramp_status"] = str(ramp.get("status", "not_reported"))
    row["ramp_closure_pN"] = _to_float(ramp.get("closure_shift_N")) * 1e12
    for cand in single.get("model_candidates", []) or []:
        model = str(cand.get("model"))
        if model in ("PLR", "GM2"):
            continue
        for name, value in (cand.get("params", {}) or {}).items():
            row[f"{model}_{name}"] = _to_float(value)
        row[f"{model}_r2"] = _to_float(cand.get("r2"))
    row["PLR_r2"] = _to_float(single.get("r2_vis"))
    # Store the correction actually applied to each branch; SR uses this value
    # to correct its force zero and must not reuse the raw pre-correction gap.
    drag = single.get("force_drag_info") or (single.get("baseline_info") or {}).get("force_drag_info") or {}
    correction = _to_float(drag.get("correction_N"))
    row["drag_N"] = correction if drag.get("applied") and np.isfinite(correction) else np.nan
    return row


def _fd_curve_arrays(single):
    """Curva corretta (indentazione [m], forza [N]) di avvicinamento e allontanamento."""
    for source in (single, single.get("fit_context") or {}):
        keys = ("ext_indentation", "ext_force", "ret_indentation", "ret_force")
        if all(k in source and source[k] is not None for k in keys):
            arrays = [np.asarray(source[k], dtype=float).ravel() for k in keys]
            if all(a.size > 10 for a in arrays):
                ext_ind, ext_force, ret_ind, ret_force = arrays
                # Come nel fit di Ting (force_fit - force_fit[0]): la forza viene
                # riportata a zero al primo punto dell'avvicinamento con delta >= 0.
                contact = np.where(np.isfinite(ext_ind) & (ext_ind >= 0))[0]
                if contact.size:
                    f_contact = float(ext_force[contact[0]])
                    ext_force = ext_force - f_contact
                    ret_force = ret_force - f_contact
                return {"ext_ind": ext_ind, "ext_force": ext_force, "ret_ind": ret_ind, "ret_force": ret_force}
    return None


def _call_ting(ting_utils, curve_path, options):
    """Fit di Ting di una curva, compatibile con le diverse versioni di ting_utils."""
    import contextlib
    import inspect
    import io

    func = getattr(ting_utils, "_run_single_ting_analysis", None) or getattr(ting_utils, "run_ting_analysis")
    accepted = inspect.signature(func).parameters
    kwargs = {
        "print_reports": False,
        "raw_deflection_smooth_win": options["raw_deflection_smooth_win"],
        "param_overrides": options["param_overrides"],
    }
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    show = plt.show
    plt.show = lambda *a, **k: None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return func(curve_path, **kwargs)
    finally:
        plt.show = show
        plt.close("all")


def _pause_duration_s(curve_path):
    """Durata [s] del segmento di pausa piu' lungo nel file (0 se non c'e').
    Serve a riconoscere le curve di stress-relaxation finite nelle cartelle FD."""
    try:
        fc = sr_utils.loadfile(str(curve_path))
        curve = fc.getcurve(0)
        durations = [float(seg.segment_metadata.get("duration", 0.0) or 0.0) for _, seg in curve.pause_segments]
        return max(durations) if durations else 0.0
    except Exception:
        return 0.0  # se non si riesce a leggere, decide il fit di Ting


def far_field_gap_N(curve_path, window_um=(0.5, 1.5)):
    """Distanza [N] tra avvicinamento e ritorno lontano dalla cellula.

    Mediana di (deflessione x k) dell'avvicinamento meno quella del ritorno,
    nel tratto tra 0.5 e 1.5 um dal punto piu' lontano comune ai due rami.
    """
    fc = sr_utils.loadfile(str(curve_path))
    md = fc.filemetadata
    curve = fc.getcurve(0)
    curve.preprocess_force_curve(md["defl_sens_nmbyV"] / 1e9, md.get("height_channel_key", "measuredHeight"))
    k = float(md["spring_const_Nbym"])
    ext = curve.extend_segments[0][1]
    ret = curve.retract_segments[-1][1]
    ze, de = np.asarray(ext.zheight, float), np.asarray(ext.vdeflection, float) * k
    zr, dr = np.asarray(ret.zheight, float), np.asarray(ret.vdeflection, float) * k
    zfar = min(np.nanmax(ze), np.nanmax(zr))
    lo, hi = zfar - window_um[1] * 1e-6, zfar - window_um[0] * 1e-6
    me, mr = (ze >= lo) & (ze <= hi), (zr >= lo) & (zr <= hi)
    if me.sum() < 20 or mr.sum() < 20:
        return np.nan
    return float(np.nanmedian(de[me]) - np.nanmedian(dr[mr]))


def drag_table_min_gap(cells):
    """Resistenza viscosa per cellula/area = meta' della distanza minima tra i rami
    (lontano dalla cellula) tra le sue curve FD. Restituisce (tabella per curva,
    dict per cartella, dict per giornata)."""
    rows = []
    for _, cell in cells[cells["tipo"] == "FD"].iterrows():
        for path in cell["curve"]:
            try:
                gap = far_field_gap_N(path)
            except Exception:
                gap = np.nan
            rows.append({"linea": cell["linea"], "giornata": cell["giornata"], "cartella": cell["cartella"],
                         "curva": Path(path).stem, "distanza_rami_N": gap})
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {}, {}
    ok = table[np.isfinite(table["distanza_rami_N"])]
    # 1) per ogni cellula/area: meta' della distanza minima (curva meno contaminata dai tether)
    area_min = ok.groupby(["linea", "giornata", "cartella"])["distanza_rami_N"].min().clip(lower=0.0) / 2.0
    # 2) per ogni giornata: mediana dei valori delle aree. Nella stessa giornata cantilever e
    #    velocita' sono gli stessi, quindi la resistenza viscosa e' la stessa; la mediana non
    #    risente di un'area con tutte le curve contaminate ne' di una curva anomala.
    day_value = area_min.groupby(level=[0, 1]).median().to_dict()
    by_folder = {key: day_value[key[:2]] for key in area_min.index}
    table = table.merge(area_min.rename("meta_distanza_minima_area_N").reset_index(),
                        on=["linea", "giornata", "cartella"], how="left")
    table["resistenza_viscosa_giornata_N"] = [day_value.get((l, g), np.nan) for l, g in zip(table["linea"], table["giornata"])]
    return table, by_folder, day_value


# ---------------------------------------------------------------------------
# Correzione per lo spessore finito (bottom effect)
# ---------------------------------------------------------------------------

# F = F_Hertz * [1 + c1*chi + c2*chi^2 + c3*chi^3 + c4*chi^4], chi = sqrt(R*delta)/h,
# campione incomprimibile aderente al substrato (paraboloide).
BOTTOM_COEFFS = {
    "dimitriadis": (1.133, 1.283, 0.769, 0.0975),   # Dimitriadis et al. 2002 (Eq. dimitriadis della tesi)
    "garcia": (1.133, 1.497, 1.469, 0.755),         # Garcia & Garcia 2018, Biophys J, Eq. 16a
}
PROBE_RADIUS_M = 5e-6


def bottom_effect_factor(ext_ind_m, height_m, coeffs="dimitriadis", radius_m=PROBE_RADIUS_M):
    """Fattore di cui il modulo apparente sovrastima quello corretto per lo spessore.

    Per un fit F = E * C * delta^1.5 ai minimi quadrati su una curva che segue
    F = E_vero * C * delta^1.5 * f(chi(delta)), il modulo ottenuto e'
    E_vero * sum(delta^3 f) / sum(delta^3): il fattore e' questa media pesata,
    calcolata sui punti di contatto dell'avvicinamento. Vale esattamente per il
    fit di Hertz; per E0 del fit di Ting e' un'approssimazione (vedi tesi).
    """
    d = np.asarray(ext_ind_m, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size < 5 or not np.isfinite(height_m) or height_m <= 0:
        return np.nan
    c1, c2, c3, c4 = BOTTOM_COEFFS[coeffs]
    chi = np.sqrt(radius_m * d) / height_m
    f = 1 + c1 * chi + c2 * chi**2 + c3 * chi**3 + c4 * chi**4
    w = d**3
    return float(np.sum(w * f) / np.sum(w))


def _height_for(altezze_um, line, day):
    """Altezza [m] da usare: altezze_um[(linea, giornata)] se c'e', altrimenti altezze_um[linea]."""
    if not altezze_um:
        return np.nan
    value = altezze_um.get((line, day), altezze_um.get(line))
    return float(value) * 1e-6 if value is not None else np.nan


FD_MAX_PAUSE_S = 1.0   # curve FD con una pausa piu' lunga sono curve di stress-relaxation


def analyze_fd_curves(cells, ting_options=None, verbose=True, drag_by_folder=None, drag_by_day=None,
                      altezze_um=None):
    """Fit di Ting su tutte le curve FD.

    Restituisce (tabella con una riga per curva, curve corrette per la curva media).
    """
    ting_utils = _load_local_module("ting_utils")  # richiede pyfmreader/pyfmrheo

    options = _merge_ting_options(ting_options)

    fd_cells = cells[cells["tipo"] == "FD"]
    n_total = int(fd_cells["n_curve"].sum())
    rows = []
    curves = []
    count = 0
    for _, cell in fd_cells.iterrows():
        for curve_path in cell["curve"]:
            count += 1
            base = {
                "linea": cell["linea"], "giornata": cell["giornata"], "cartella": cell["cartella"],
                "id_cellula": cell["id_cellula"], "curva": Path(curve_path).stem, "path": curve_path,
            }
            if verbose:
                print(f"[FD {count}/{n_total}] {cell['id_cellula']} - {Path(curve_path).name}")
            pause_s = _pause_duration_s(curve_path)
            if pause_s > FD_MAX_PAUSE_S:
                rows.append({**base, "stato": f"protocollo SR: pausa di {pause_s:.1f} s (non e' una curva FD)"})
                continue
            curve_options = options
            if options.get("drag_mode") == "minimo_cartella":
                key = (cell["linea"], cell["giornata"], cell["cartella"])
                drag = (drag_by_folder or {}).get(key, (drag_by_day or {}).get(key[:2]))
                base["drag_fisso_N"] = drag
                if drag is not None and np.isfinite(drag):
                    curve_options = {**options, "param_overrides": {**options["param_overrides"], "fixed_drag_N": float(drag)}}
            try:
                single = _call_ting(ting_utils, curve_path, curve_options)
                row = {**base, "stato": "ok", **_ting_row(single)}
                rows.append(row)
                arrays = _fd_curve_arrays(single)
                if arrays is not None:
                    curves.append({**base, **arrays})
                if altezze_um:
                    h = _height_for(altezze_um, cell["linea"], cell["giornata"])
                    ind = arrays["ext_ind"] if arrays is not None else None
                    k_d = bottom_effect_factor(ind, h, "dimitriadis") if ind is not None else np.nan
                    k_g = bottom_effect_factor(ind, h, "garcia") if ind is not None else np.nan
                    dmax = float(np.nanmax(ind)) if ind is not None and np.size(ind) else np.nan
                    row.update({
                        "altezza_um": h * 1e6, "delta_max_um": dmax * 1e6,
                        "chi_max": np.sqrt(PROBE_RADIUS_M * dmax) / h if np.isfinite(dmax) and dmax > 0 else np.nan,
                        "fattore_bottom": k_d, "fattore_bottom_garcia": k_g,
                        "E0_Pa_corr": row["E0_Pa"] / k_d if np.isfinite(k_d) else np.nan,
                        "E_hertz_Pa_corr": row["E_hertz_Pa"] / k_d if np.isfinite(k_d) else np.nan,
                        "E0_Pa_corr_garcia": row["E0_Pa"] / k_g if np.isfinite(k_g) else np.nan,
                    })
                if count == 1 and not np.isfinite(row["E0_Pa"]):
                    print("Attenzione: E0 non trovato nei risultati di Ting. Chiavi disponibili:",
                          sorted(single.keys()))
            except Exception as exc:
                rows.append({**base, "stato": f"errore: {type(exc).__name__}: {exc}"})
    fd_curves = pd.DataFrame(rows)
    report_errors(fd_curves, "FD")
    if rows and not curves:
        print("FD: le curve corrette non sono nei risultati di Ting: la curva media FD non verra' calcolata.")
    return fd_curves, curves


U_GRID = np.linspace(0.0, 1.0, 201)   # indentazione normalizzata delta/delta_max


def _branch_vs_u(ind_m, force_n, delta_max_m):
    """Forza [nN] di un ramo in funzione di u = delta/delta_max (solo delta >= 0)."""
    ind = np.asarray(ind_m, dtype=float)
    force = np.asarray(force_n, dtype=float) * 1e9
    keep = np.isfinite(ind) & np.isfinite(force) & (ind >= 0)
    if keep.sum() < 5 or not np.isfinite(delta_max_m) or delta_max_m <= 0:
        return np.full(U_GRID.size, np.nan)
    u = ind[keep] / delta_max_m
    f = force[keep]
    order = np.argsort(u)
    u, f = u[order], f[order]
    u, idx = np.unique(u, return_index=True)
    out = np.interp(U_GRID, u, f[idx], left=np.nan, right=np.nan)
    out[U_GRID > u.max() + 1e-9] = np.nan
    return out


def fd_mean_curves(fd_arrays):
    """Ciclo forza-indentazione medio di ogni linea, come curva continua.

    Ogni curva e' espressa in funzione di u = delta/delta_max (0 -> 1 in
    avvicinamento, 1 -> 0 in allontanamento): in questo modo le curve con
    indentazioni massime diverse si possono mediare senza gradini. Si fa la
    media prima tra le curve della stessa cellula, poi tra le cellule della
    linea; l'asse delle indentazioni e' u moltiplicato per la delta_max media
    della linea.
    """
    by_cell = {}
    for item in fd_arrays:
        dmax = np.nanmax(np.asarray(item["ext_ind"], dtype=float))
        key = (item["linea"], item["id_cellula"])
        entry = by_cell.setdefault(key, {"ext": [], "ret": [], "dmax": []})
        entry["ext"].append(_branch_vs_u(item["ext_ind"], item["ext_force"], dmax))
        entry["ret"].append(_branch_vs_u(item["ret_ind"], item["ret_force"], dmax))
        entry["dmax"].append(dmax * 1e6)
    by_line = {}
    for (line, _), data in by_cell.items():
        entry = by_line.setdefault(line, {"ext": [], "ret": [], "dmax": []})
        entry["ext"].append(np.nanmean(data["ext"], axis=0))
        entry["ret"].append(np.nanmean(data["ret"], axis=0))
        entry["dmax"].append(np.nanmean(data["dmax"]))
    out = {}
    for line, data in by_line.items():
        ext = np.asarray(data["ext"])
        ret = np.asarray(data["ret"])
        dmax_mean = float(np.nanmean(data["dmax"]))
        # ciclo continuo: avvicinamento (u 0 -> 1) seguito da allontanamento (u 1 -> 0)
        u_loop = np.concatenate([U_GRID, U_GRID[::-1]])
        mean_loop = np.concatenate([np.nanmean(ext, axis=0), np.nanmean(ret, axis=0)[::-1]])
        sd_loop = np.concatenate([np.nanstd(ext, axis=0), np.nanstd(ret, axis=0)[::-1]])
        out[line] = {
            "n": len(data["ext"]),
            "delta_max_um": dmax_mean,
            "delta_max_sd_um": float(np.nanstd(data["dmax"])),
            "indentazione_um": u_loop * dmax_mean,
            "forza_nN": mean_loop,
            "sd_nN": sd_loop,
        }
    return out


def plot_fd_mean_curve(line, data, color="C0"):
    """Ciclo forza-indentazione medio di UNA linea (curva continua)."""
    x, m, s = data["indentazione_um"], data["forza_nN"], data["sd_nN"]
    n_half = len(x) // 2
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.plot(x, m, color=color, lw=2.2, label=f"curva media ({data['n']} cellule)")
    ax.plot(x, m + s, color=color, lw=0.8, ls=":", label="± 1 DS tra le cellule")
    ax.plot(x, m - s, color=color, lw=0.8, ls=":")
    # frecce: verso del ciclo
    for i, txt in ((n_half // 2, "avvicinamento"), (n_half + n_half // 2, "allontanamento")):
        if np.isfinite(m[i]) and np.isfinite(m[i + 3]):
            ax.annotate("", xy=(x[i + 3], m[i + 3]), xytext=(x[i], m[i]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
            ax.annotate(txt, (x[i], m[i]), textcoords="offset points",
                        xytext=(-60, 10) if txt == "avvicinamento" else (10, -15), fontsize=8, color=color)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Indentazione [µm]")
    ax.set_ylabel("Forza [nN]")
    ax.set_title(f"{line}: ciclo forza-indentazione medio "
                 f"(δmax medio {data['delta_max_um']:.2f} ± {data['delta_max_sd_um']:.2f} µm)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def report_errors(table, label):
    """Stampa quante curve/cellule sono riuscite e i messaggi di errore piu' frequenti."""
    if table.empty or "stato" not in table.columns:
        print(f"{label}: nessun risultato.")
        return
    n_ok = int((table["stato"] == "ok").sum())
    print(f"{label}: {n_ok} riuscite su {len(table)}")
    errors = table.loc[table["stato"] != "ok", "stato"]
    if not errors.empty:
        print(f"{label}: errori piu' frequenti (numero di casi):")
        for message, count in errors.value_counts().head(5).items():
            print(f"  [{count}] {message[:400]}")


def summarize_fd_by_cell(fd_curves):
    """Una riga per cellula: media dei parametri delle sue curve.

    Per le umane ogni cellula ha una sola curva, quindi la riga coincide con la curva.
    """
    ok = fd_curves[fd_curves["stato"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    numeric_cols = [c for c in ok.columns if c not in {"linea", "giornata", "cartella", "id_cellula", "curva", "path", "stato", "modello_preferito"}]
    ok[numeric_cols] = ok[numeric_cols].apply(pd.to_numeric, errors="coerce")
    per_cell = (
        ok.groupby(["linea", "giornata", "cartella", "id_cellula"])[numeric_cols]
        .mean()
        .reset_index()
    )
    per_cell["n_curve"] = ok.groupby("id_cellula").size().reindex(per_cell["id_cellula"]).to_numpy()
    return per_cell


# ---------------------------------------------------------------------------
# 3. Curve di stress-relaxation
# ---------------------------------------------------------------------------

DEFAULT_SR_OPTIONS = {
    # preparazione della curva della cellula
    "normalization_mode": "peak",
    "spike_filter_window": 9,
    "spike_filter_z": 3.0,
    "mean_smoothing_window": None,     # nessuna media mobile sulla curva della cellula
    "min_curve_duration_s": 5.0,
    "baseline_mode": "line",          # zero SR: retta sulla baseline dell'avvicinamento
    "drag_correction": False,         # zero SR: aggiunge il drag effettivamente applicato alle FD, se la correzione e' attiva
    # fit (stessi parametri di Rilassamento.ipynb)
    "plot_max_time_s": 10.0,
    "fit_smoothing_window": 101,
    "transition_time_s": 0.5,
    "auto_optimize_transition": True,
    "transition_search_range_s": (0.25, 1.2),
    "transition_search_steps": 41,
    "transition_max_join_gap": 0.03,
    "transition_max_local_rmse": 0.05,
    "transition_min_late_r2": 0.90,
    "hertz_tip_radius_m": 5.0e-6,
    "hertz_poisson": 0.5,
    "early_phase_comparison": True,
}

# Parametri della tabella per cellula (SR) confrontati tra le linee
SR_PARAMETERS = [
    "F_0.1s", "F_1s", "F_relax", "rilassamento_pct",
    "SLS_F_inf", "SLS_tau_s",
    "PLR_beta", "PLR_t0_s",
    "BI_F_inf", "BI_w", "BI_tau1_s", "BI_tau2_s",
    "PORO_tau_p_s", "t_trans_s",
]


def _prepare_cell_sr_curve(curve_paths, options, drag_force_N=0.0):
    """Curva SR normalizzata di una cellula.

    Una sola curva (umane): la curva stessa, normalizzata al primo punto.
    Piu' curve (murine): curva media delle curve della cellula.
    Restituisce il dizionario che sr_utils.load_sr_curve accetta come input.
    """
    paths, curves, skipped = sr_utils.load_valid_stress_relaxation_curves(
        [Path(p) for p in curve_paths],
        normalization_mode=options["normalization_mode"],
        min_curve_duration_s=options["min_curve_duration_s"],
        baseline_mode=options["baseline_mode"],
        drag_force_N=drag_force_N,
    )
    if len(curves) == 1:
        curve = curves[0]
        force = np.asarray(curve["normalized_force"], dtype=float)
        if options["spike_filter_window"]:
            force, _ = sr_utils.despike_signal(force, window=options["spike_filter_window"], z_thresh=options["spike_filter_z"])
            force = force / force[0]
        time = np.asarray(curve["time"], dtype=float)
        std = np.zeros_like(force)
    else:
        time, force, std, _, _ = sr_utils.calculate_mean_curve(
            curves,
            outlier_sigma=None,          # nessuna esclusione di punti in base al valore
            min_valid_curves=min(2, len(curves)),
            smoothing_window=options["mean_smoothing_window"],
            spike_filter_window=options["spike_filter_window"],
            spike_filter_z=options["spike_filter_z"],
        )
    indentations = [float(c.get("estimated_indentation_m", np.nan)) for c in curves]
    return {
        "is_generated_mean": True,
        "time": np.asarray(time, dtype=float),
        "force": np.asarray(force, dtype=float),
        "std": np.asarray(std, dtype=float),
        "f_peak_mean_N": float(np.nanmean([c["reference_force"] for c in curves])),
        "estimated_indentation_m": float(np.nanmedian(indentations)) if np.any(np.isfinite(indentations)) else np.nan,
        "n_curve_usate": len(curves),
        "curve_escluse": skipped,
    }


SR_FIXED_TIMES_S = (0.1, 1.0)


def _force_at_times(time, force, times=SR_FIXED_TIMES_S, half_window_s=0.005):
    """Forza normalizzata a tempi fissi (media in +-5 ms). Nelle curve con poca
    forza residua questi valori dipendono dallo zero meno di F_relax."""
    time = np.asarray(time, dtype=float)
    force = np.asarray(force, dtype=float)
    out = {}
    for T in times:
        m = np.abs(time - T) <= half_window_s
        out[f"F_{T:g}s"] = float(np.nanmean(force[m])) if m.any() else np.nan
    return out


def _sr_fit_row(result):
    """Parametri dei fit SR di una curva (SLS, PLR, bi-exp, poroelastico)."""
    fits = result.get("fit_results", {}) or {}
    metrics = result.get("global_metrics", {}) or {}
    row = {"t_trans_s": float(result.get("transition_time_selected_s", np.nan))}
    # Come nei vecchi report: forza residua al termine del mantenimento (media degli
    # ultimi 50 punti, curva normalizzata a F(0) = 1) e rilassamento percentuale.
    row["F_relax"] = _to_float(result.get("F_relax"))
    row["rilassamento_pct"] = _to_float(result.get("delta_F_pct"))
    p = fits.get("mono")
    row.update({"SLS_F_inf": p[0], "SLS_tau_s": p[1]} if p is not None else {"SLS_F_inf": np.nan, "SLS_tau_s": np.nan})
    p = fits.get("plr")
    row.update({"PLR_t0_s": p[1], "PLR_beta": p[2]} if p is not None else {"PLR_t0_s": np.nan, "PLR_beta": np.nan})
    p = fits.get("bi")
    row.update(
        {"BI_F_inf": p[0], "BI_w": p[1], "BI_tau1_s": p[2], "BI_tau2_s": p[3]} if p is not None
        else {"BI_F_inf": np.nan, "BI_w": np.nan, "BI_tau1_s": np.nan, "BI_tau2_s": np.nan}
    )
    p = fits.get("poro")
    row.update({"PORO_F_inf": p[0], "PORO_tau_p_s": p[1]} if p is not None else {"PORO_F_inf": np.nan, "PORO_tau_p_s": np.nan})
    for key, label in [("mono", "SLS"), ("plr", "PLR"), ("bi", "BI"), ("poro", "PORO")]:
        row[f"{label}_r2"] = float(metrics.get(key, {}).get("R2", np.nan))
        row[f"{label}_AIC"] = float(metrics.get(key, {}).get("AIC", np.nan))
    for k, v in (result.get("early_comparison_summary") or {}).items():
        row[f"early_{k}"] = v
    return row


def _run_sr_fit(sr_input, label, options, show_plots=False):
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        sr = sr_utils.load_sr_curve(sr_input, input_force_mode="auto")
    kwargs = dict(
        sr=sr,
        analysis_input_path={"folder_path": os.getcwd(), "base_name": label, "display_name": label},
        save_outputs=False,
        plot_max_time_s=options["plot_max_time_s"],
        fit_spike_filter_window=options["spike_filter_window"],
        fit_spike_filter_z=options["spike_filter_z"],
        fit_smoothing_window=options["fit_smoothing_window"],
        transition_time_s=options["transition_time_s"],
        auto_optimize_transition=options["auto_optimize_transition"],
        transition_search_range_s=options["transition_search_range_s"],
        transition_search_steps=options["transition_search_steps"],
        transition_max_join_gap=options["transition_max_join_gap"],
        transition_max_local_rmse=options["transition_max_local_rmse"],
        transition_min_late_r2=options["transition_min_late_r2"],
        hertz_tip_radius_m=options["hertz_tip_radius_m"],
        hertz_poisson=min(float(options["hertz_poisson"]), 0.4999),
        early_phase_comparison=options["early_phase_comparison"],
        show_early_comparison_plot=show_plots,
    )
    if show_plots:
        return sr_utils.run_sr_analysis(**kwargs)
    backend_show = plt.show
    plt.show = lambda *a, **k: None          # niente grafici per ogni cellula
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return sr_utils.run_sr_analysis(**kwargs)
    finally:
        plt.show = backend_show
        plt.close("all")


def _drag_lookup(fd_curves):
    """Resistenza viscosa [N] per cartella (cellula o area) e per giornata, dalle curve FD."""
    if fd_curves is None or fd_curves.empty or "drag_N" not in fd_curves.columns:
        return {}, {}
    ok = fd_curves[(fd_curves["stato"] == "ok") & np.isfinite(pd.to_numeric(fd_curves["drag_N"], errors="coerce"))]
    by_folder = ok.groupby(["linea", "giornata", "cartella"])["drag_N"].median().to_dict()
    by_day = ok.groupby(["linea", "giornata"])["drag_N"].median().to_dict()
    return by_folder, by_day


def analyze_sr_cells(cells, sr_options=None, verbose=True, fd_curves=None, drag_tables=None):
    """Fit SR per ogni cellula. Restituisce (tabella per cellula, curve delle cellule).

    fd_curves: tabella delle curve FD (per la correzione della resistenza viscosa).
    """
    options = dict(DEFAULT_SR_OPTIONS)
    if sr_options:
        options.update(sr_options)
    if not options["drag_correction"]:
        drag_by_folder, drag_by_day = {}, {}
    elif drag_tables is not None:
        drag_by_folder, drag_by_day = drag_tables
    else:
        drag_by_folder, drag_by_day = _drag_lookup(fd_curves)
    if options["drag_correction"] and not drag_by_folder:
        print("SR: attenzione, nessuna resistenza viscosa disponibile dalle curve FD: "
              "lo zero SR non e' corretto per il trascinamento (esegui anche l'analisi FD).")

    sr_cells = cells[cells["tipo"] == "SR"]
    rows = []
    cell_curves = []
    for i, (_, cell) in enumerate(sr_cells.iterrows(), start=1):
        base = {
            "linea": cell["linea"], "giornata": cell["giornata"], "cartella": cell["cartella"],
            "id_cellula": cell["id_cellula"], "n_curve": cell["n_curve"],
        }
        if verbose:
            print(f"[SR {i}/{len(sr_cells)}] {cell['id_cellula']} ({cell['n_curve']} curve)")
        key = (cell["linea"], cell["giornata"], cell["cartella"])
        if key in drag_by_folder:
            drag, drag_src = drag_by_folder[key], "cartella"
        elif key[:2] in drag_by_day:
            drag, drag_src = drag_by_day[key[:2]], "giornata"
        else:
            drag, drag_src = 0.0, "nessuna"
        base.update({"drag_N_usata": drag, "drag_fonte": drag_src})
        try:
            sr_input = _prepare_cell_sr_curve(cell["curve"], options, drag_force_N=drag)
            result = _run_sr_fit(sr_input, cell["id_cellula"].replace("/", "_"), options)
            row = {**base, "stato": "ok", "n_curve_usate": sr_input["n_curve_usate"], **_sr_fit_row(result)}
            row.update(_force_at_times(sr_input["time"], sr_input["force"]))
            row["E_SR_Pa"] = float(result.get("young_modulus_pa", np.nan))
            rows.append(row)
            cell_curves.append({**base, "time": sr_input["time"], "force": sr_input["force"]})
        except Exception as exc:
            rows.append({**base, "stato": f"errore: {type(exc).__name__}: {exc}"})
    sr_table = pd.DataFrame(rows)
    report_errors(sr_table, "SR")
    return sr_table, cell_curves


def line_mean_curves(cell_curves, options=None, show_plots=True):
    """Curva SR media di ogni linea (media delle curve delle cellule) e relativi fit."""
    options = dict(DEFAULT_SR_OPTIONS) if options is None else {**DEFAULT_SR_OPTIONS, **options}
    by_line = {}
    for item in cell_curves:
        by_line.setdefault(item["linea"], []).append(
            {"time": item["time"], "normalized_force": item["force"], "distance": np.zeros_like(item["time"])}
        )
    out = {}
    rows = []
    for line, curves in by_line.items():
        time_grid, forces, _ = sr_utils.interpolate_relaxation_curves(curves)
        mean_force = np.nanmean(forces, axis=0)
        sd_force = np.nanstd(forces, axis=0)
        mean_force = mean_force / mean_force[0]
        sr_input = {
            "is_generated_mean": True, "time": time_grid.astype(float), "force": mean_force.astype(float),
            "f_peak_mean_N": np.nan, "estimated_indentation_m": np.nan,
        }
        if show_plots:
            print(f"\n===== Curva media della linea {line} ({len(curves)} cellule) =====")
        result = _run_sr_fit(sr_input, f"media_{line}", options, show_plots=show_plots)
        out[line] = {"time": time_grid, "mean": mean_force, "sd": sd_force, "n": len(curves), "result": result}
        rows.append({"linea": line, "n_cellule": len(curves), **_force_at_times(time_grid, mean_force),
                     **_sr_fit_row(result)})
    return out, pd.DataFrame(rows)


SR_MODEL_STYLE = [
    ("poro", "Poroelastico (0 - t_trans)", "firebrick", "--"),
    ("mono", "SLS", "royalblue", "-"),
    ("plr", "PLR", "chocolate", "-"),
    ("bi", "Bi-exp", "teal", "-"),
]


def _sr_model_curve(key, params, t):
    func = {"poro": sr_utils.poro_relax, "mono": sr_utils.mono_exp, "plr": sr_utils.plr, "bi": sr_utils.bi_exp}[key]
    return func(t, *params)


def plot_line_mean_curves(line_means, max_time_s=10.0):
    """Curva SR media di ogni linea (un pannello per linea) con i fit:
    poroelastico prima di t_trans, SLS, PLR e bi-exp dopo."""
    lines = list(line_means)
    fig, axes = plt.subplots(1, len(lines), figsize=(max(8.5, 6.2 * len(lines)), 5.6))
    axes = np.atleast_1d(axes)

    for ax, line in zip(axes, lines):
        d = line_means[line]
        res = d["result"]
        fits = res.get("fit_results", {}) or {}
        t_trans = float(res.get("transition_time_selected_s", np.nan))
        keep = (d["time"] > 0) & (d["time"] <= max_time_s)
        ax.plot(d["time"][keep], d["mean"][keep], color="0.45", lw=3, alpha=0.5, label=f"curva media ({d['n']} cellule)")
        ax.fill_between(d["time"][keep], (d["mean"] - d["sd"])[keep], (d["mean"] + d["sd"])[keep],
                        color="0.6", alpha=0.15, label="± 1 DS tra le cellule")
        for key, label, color, ls in SR_MODEL_STYLE:
            params = fits.get(key)
            if params is None:
                continue
            if key == "poro":
                tt = np.logspace(np.log10(max(d["time"][keep][0], 1e-4)), np.log10(t_trans), 200)
            else:
                tt = np.logspace(np.log10(t_trans), np.log10(max_time_s), 300)
            ax.plot(tt, _sr_model_curve(key, params, tt), ls=ls, color=color, lw=1.8, label=label)
        if np.isfinite(t_trans):
            ax.axvline(t_trans, color="0.3", ls=":", lw=1, label=f"t_trans = {t_trans:.2f} s")
        ax.set_xscale("log")
        ax.set_title(f"{line}: curva media e fit (dopo t_trans)", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Tempo [s] (scala log)")
        ax.set_ylabel("Forza normalizzata F(t)/F(0)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def plot_line_mean_fits_from_zero(line_means, max_time_s=10.0):
    """Per ogni linea: curva media in scala log con il poroelastico (0 - t_trans)
    e SLS, PLR e bi-exp fittati su tutta la curva a partire da t = 0, cioe' dallo
    stesso istante da cui parte il poroelastico. Sotto: residui.

    La PLR e' nella forma (1 + t/t0)^(-beta), che vale 1 a t = 0. I fit sono
    eseguiti sulla curva senza media mobile (vedi sr_utils.compare_early_phase_fits).
    """
    lines = [l for l, d in line_means.items() if (d["result"].get("early_comparison") or {}).get("fits")]
    if not lines:
        return None
    fig, axes = plt.subplots(2, len(lines), figsize=(max(8.5, 6.2 * len(lines)), 8.0), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.3]}, sharex="col")
    styles = {"poro": ("Poroelastico (0 - t_trans)", "firebrick", "--"),
              "mono": ("SLS da t = 0", "royalblue", "-"),
              "plr0": ("PLR (1+t/t0)^-β da t = 0", "chocolate", "-"),
              "bi": ("Bi-exp da t = 0", "teal", "-")}
    for col, line in enumerate(lines):
        d = line_means[line]
        comp = d["result"]["early_comparison"]
        t_data = np.asarray(comp.get("t_data", d["time"]), dtype=float)
        f_data = np.asarray(comp.get("F_data", d["mean"]), dtype=float)
        t_trans = float(comp["t_trans"])
        keep = (t_data > 0) & (t_data <= max_time_s) & np.isfinite(f_data)
        tt, ff = t_data[keep], f_data[keep]
        ax, axr = axes[0, col], axes[1, col]
        ax.plot(tt, ff, "-", lw=5, color="0.55", alpha=0.45, label="curva media", zorder=0)
        fits = {"poro": comp["fits"]["finestra_iniziale"].get("poro"), **comp["fits"].get("intera_curva_da_0", {})}
        for key, (label, color, ls) in styles.items():
            params = fits.get(key)
            if params is None:
                continue
            func = sr_utils._early_model_function(key)
            if key == "poro":
                sel = tt <= t_trans
            else:
                sel = np.ones_like(tt, dtype=bool)
            ax.plot(tt[sel], func(tt[sel], *params), ls=ls, color=color, lw=2, label=label)
            axr.plot(tt[sel], 100 * (ff[sel] - func(tt[sel], *params)), color=color, lw=1)
        ax.axvline(t_trans, color="0.3", ls=":", lw=1, label=f"t_trans = {t_trans:.2f} s")
        axr.axvline(t_trans, color="0.3", ls=":", lw=1)
        axr.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{line} (n = {d['n']} cellule): fit a partire da t = 0", fontsize=10)
        ax.set_ylabel("Forza normalizzata F(t)/F(0)")
        axr.set_ylabel("Residuo [% di F(0)]")
        axr.set_xlabel("Tempo [s] (scala log)")
        for a_ in (ax, axr):
            a_.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Riassunto per linea e confronto tra le linee
# ---------------------------------------------------------------------------

def describe_by_line(per_cell, parameters):
    rows = []
    for line, sub in per_cell.groupby("linea"):
        for p in parameters:
            if p not in sub.columns:
                continue
            v = pd.to_numeric(sub[p], errors="coerce").dropna()
            if v.empty:
                continue
            rows.append({
                "linea": line, "parametro": p, "n_cellule": len(v),
                "media": v.mean(), "DS": v.std(ddof=1) if len(v) > 1 else np.nan,
                "mediana": v.median(), "Q1": v.quantile(0.25), "Q3": v.quantile(0.75),
            })
    return pd.DataFrame(rows)


def compare_lines(per_cell, parameters, line_a, line_b):
    """Test di Mann-Whitney a due code tra due linee, una osservazione per cellula."""
    rows = []
    for p in parameters:
        if p not in per_cell.columns:
            continue
        a = pd.to_numeric(per_cell.loc[per_cell["linea"] == line_a, p], errors="coerce").dropna()
        b = pd.to_numeric(per_cell.loc[per_cell["linea"] == line_b, p], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        _, p_value = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "parametro": p, f"n_{line_a}": len(a), f"n_{line_b}": len(b),
            f"mediana_{line_a}": a.median(), f"mediana_{line_b}": b.median(), "p": p_value,
        })
    return pd.DataFrame(rows)


def boxplot_by_line(per_cell, parameters, title, line_order=None):
    params = [p for p in parameters if p in per_cell.columns and per_cell[p].notna().any()]
    if not params:
        return None
    present = list(dict.fromkeys(per_cell["linea"]))
    lines = [l for l in (line_order or []) if l in present] + [l for l in present if l not in (line_order or [])]
    ncol = min(3, len(params))
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, p in zip(axes.ravel(), params):
        data = [pd.to_numeric(per_cell.loc[per_cell["linea"] == l, p], errors="coerce").dropna().to_numpy() for l in lines]
        ax.boxplot(data, widths=0.5, showfliers=False)
        for i, d in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=12, alpha=0.6)
        ax.set_xticks(range(1, len(lines) + 1))
        ax.set_xticklabels(lines)
        ax.set_title({"E0_Pa": "E0 [Pa] (Ting PLR)", "betaE": "βE (Ting PLR)"}.get(p, p), fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(params):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


SUMMARY_FD = [
    ("E0_Pa", "Ting PLR: E0 [Pa]"),
    ("betaE", "Ting PLR: βE"),
    ("E_hertz_Pa", "Hertz: E [Pa]"),
    # presenti solo se si passa altezze_um (correzione per lo spessore finito)
    ("E0_Pa_corr", "Ting PLR: E0 corretto spessore [Pa]"),
    ("E_hertz_Pa_corr", "Hertz: E corretto spessore [Pa]"),
    ("E0_Pa_corr_garcia", "Ting PLR: E0 corretto (coeff. Garcia) [Pa]"),
    ("fattore_bottom", "fattore di correzione (Dimitriadis)"),
    ("delta_max_um", "indentazione massima [µm]"),
]
SUMMARY_SR = [
    ("F_0.1s", "forza a 0,1 s (F/F0)"),
    ("F_1s", "forza a 1 s (F/F0)"),
    ("F_relax", "forza residua F_relax (F/F0)"),
    ("rilassamento_pct", "rilassamento ΔF [%]"),
    ("SLS_F_inf", "SLS: F∞"),
    ("SLS_tau_s", "SLS: τ [s]"),
    ("PLR_beta", "PLR: β"),
    ("BI_F_inf", "Bi-exp: F∞"),
    ("BI_w", "Bi-exp: w"),
    ("BI_tau1_s", "Bi-exp: τ1 [s]"),
    ("BI_tau2_s", "Bi-exp: τ2 [s]"),
    ("PORO_tau_p_s", "Poroelastico: τp [s]"),
    ("t_trans_s", "t_trans [s]"),
]


def _fmt(value):
    if not np.isfinite(value):
        return "-"
    if value == 0:
        return "0"
    return f"{value:.3g}" if abs(value) < 1e4 else f"{value:.3e}"


def summary_table(tables, line_names):
    """Tabella riassuntiva: media ± DS tra le cellule di ogni linea e p di Mann-Whitney.

    tables: lista di (tabella per cellula, elenco (colonna, etichetta)).
    """
    rows = []
    for per_cell, spec in tables:
        if per_cell is None or per_cell.empty:
            continue
        for col, label in spec:
            if col not in per_cell.columns:
                continue
            row = {"grandezza": label}
            values = {}
            for line in line_names:
                v = pd.to_numeric(per_cell.loc[per_cell["linea"] == line, col], errors="coerce").dropna()
                values[line] = v
                sd = v.std(ddof=1) if len(v) > 1 else np.nan
                row[line] = f"{_fmt(v.mean())} ± {_fmt(sd)} (n={len(v)})" if len(v) else "-"
            if len(line_names) == 2 and all(len(values[l]) >= 2 for l in line_names):
                row["p (Mann-Whitney)"] = _fmt(mannwhitneyu(values[line_names[0]], values[line_names[1]], alternative="two-sided")[1])
            rows.append(row)
    return pd.DataFrame(rows)


def table_figure(table, title):
    n = len(table)
    fig, ax = plt.subplots(figsize=(11.5, 1.2 + 0.36 * (n + 1)))
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left")
    tab = ax.table(cellText=table.values, colLabels=list(table.columns), cellLoc="center", loc="upper center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1.0, 1.35)
    tab.auto_set_column_width(list(range(len(table.columns))))
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#e8eef7")
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_text_props(ha="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Flusso completo
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Criteri di esclusione (facoltativi, decisi prima di guardare i confronti)
# ---------------------------------------------------------------------------

# Valori proposti per le linee murine. Gli stessi criteri valgono per tutte e
# due le linee del sistema; le curve/cellule escluse e il motivo vengono salvati
# in esclusi_<sistema>.csv e riportati nel report.
CRITERI_MURINE = {
    # FD (per curva): il fit di Ting deve descrivere la curva
    "fd_r2_min": 0.90,
    # FD (per curva): betaE non deve essere bloccato a un limite del fit.
    "fd_betaE_at_bound": True,
    # FD (per curva): E_Hertz ed E0 devono essere dello stesso ordine di grandezza
    # (rapporto tra 1/10 e 10). Se non lo sono, uno dei due fit non e' affidabile
    # (di solito il fit di Hertz, per un punto di contatto sbagliato).
    "fd_max_rapporto_hertz_E0": 10.0,
    # SR (per cellula): F_relax deve essere fisico (0 < F_relax <= 1)
    "sr_F_relax_fisico": True,
    # SR (per cellula): almeno uno dei modelli della fase tardiva deve descrivere la curva
    "sr_r2_min": 0.90,
    # SR (per cellula): l'esponente beta della PLR non deve essere bloccato al
    # limite superiore del fit (2), dove il suo valore non e' determinato dai dati
    "sr_beta_at_bound": True,
}


# Linea umana: forze piu' piccole (circa 1 nN) e fit di Ting meno buoni; per una
# legge di potenza con beta ~0.2 E_Hertz/E0 atteso e' ~1.5-2, quindi la finestra
# del rapporto e' piu' larga. F_relax non e' usato come criterio perche' nelle
# umane dipende dall'incertezza sullo zero (escludere i valori negativi
# selezionerebbe le cellule in base all'errore sullo zero).
CRITERI_UMANE = {
    "fd_r2_min": 0.70,
    "fd_betaE_at_bound": True,
    "fd_max_rapporto_hertz_E0": 10.0,
    "sr_F_relax_fisico": False,
    "sr_r2_min": 0.70,
    "sr_beta_at_bound": True,
}


def _fd_exclusion_reasons(row, criteri):
    reasons = []
    r2 = _to_float(row.get("PLR_r2"))
    if criteri.get("fd_r2_min") is not None and not (np.isfinite(r2) and r2 >= criteri["fd_r2_min"]):
        reasons.append(f"R2 Ting {r2:.3f} < {criteri['fd_r2_min']}")
    beta_at_bound = row.get("betaE_at_bound", False)
    if criteri.get("fd_betaE_at_bound") and pd.notna(beta_at_bound) and bool(beta_at_bound):
        reasons.append(f"betaE = {_to_float(row.get('betaE')):.3f} al limite del fit")
    k = criteri.get("fd_max_rapporto_hertz_E0")
    eh, e0 = _to_float(row.get("E_hertz_Pa")), _to_float(row.get("E0_Pa"))
    if k is not None:
        if not (np.isfinite(eh) and np.isfinite(e0) and eh > 0 and e0 > 0):
            reasons.append("E_Hertz o E0 non validi")
        elif not (1.0 / k <= eh / e0 <= k):
            reasons.append(f"E_Hertz/E0 = {eh / e0:.2f} fuori da [1/{k:g}, {k:g}]")
    return "; ".join(reasons)


# limite superiore dell'esponente beta nel fit PLR dello stress-relaxation (sr_utils)
SR_PLR_BETA_MAX = 2.0


def _sr_exclusion_reasons(row, criteri):
    reasons = []
    fr = _to_float(row.get("F_relax"))
    if criteri.get("sr_F_relax_fisico") and not (np.isfinite(fr) and 0.0 < fr <= 1.0):
        reasons.append(f"F_relax = {fr:.3f} non fisico")
    r2min = criteri.get("sr_r2_min")
    if r2min is not None:
        best = np.nanmax([_to_float(row.get(c)) for c in ("SLS_r2", "PLR_r2", "BI_r2")] + [-np.inf])
        if not (np.isfinite(best) and best >= r2min):
            reasons.append(f"R2 migliore fase tardiva {best:.3f} < {r2min}")
    if criteri.get("sr_beta_at_bound"):
        beta = _to_float(row.get("PLR_beta"))
        if np.isfinite(beta) and beta >= SR_PLR_BETA_MAX - 1e-3:
            reasons.append(f"beta PLR = {beta:.3f} al limite del fit")
    return "; ".join(reasons)


def apply_exclusions(table, kind, criteri):
    """Restituisce (righe tenute, righe escluse con il motivo)."""
    if table is None or table.empty or not criteri:
        return table, pd.DataFrame()
    func = _fd_exclusion_reasons if kind == "FD" else _sr_exclusion_reasons
    reasons = table.apply(lambda r: func(r, criteri), axis=1)
    excluded = table[reasons != ""].copy()
    excluded.insert(0, "motivo_esclusione", reasons[reasons != ""])
    excluded.insert(0, "tipo", kind)
    return table[reasons == ""].copy(), excluded


def exclusion_summary(fd_all, fd_kept, sr_all, sr_kept, line_names):
    """Tabella con numero di curve/cellule prima e dopo i criteri, per linea."""
    rows = []
    for label, before, after, col in (
        ("FD: curve", fd_all, fd_kept, None),
        ("FD: cellule", fd_all, fd_kept, "id_cellula"),
        ("SR: cellule", sr_all, sr_kept, None),
    ):
        if before is None or before.empty:
            continue
        row = {"": label}
        for line in line_names:
            b = before[before["linea"] == line]
            a = after[after["linea"] == line] if after is not None and not after.empty else after
            nb = b[col].nunique() if col else len(b)
            na = (a[col].nunique() if col else len(a)) if a is not None and not a.empty else 0
            row[line] = f"{na} su {nb}"
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
# Figure per la tesi (stile uniforme, salvate in PDF nella cartella dei risultati)
# =====================================================================
NOMI_BREVI_TESI = {"D2A1": "A1", "D2.0R": "0R", "D20R": "0R"}
COLORI_TESI = ["#0072B2", "#D55E00"]


def _stile_tesi():
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "legend.frameon": False, "pdf.fonttype": 42, "axes.grid": True,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.titlelocation": "left",
        "mathtext.default": "regular",
    })
    return FuncFormatter(lambda x, pos: ("%g" % round(x, 6)).replace("-", "\u2212").replace(".", ","))


def _lettera(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom", ha="right")


def salva_figure_tesi(output_dir, sistema, line_names, fd_cells=None, fd_means=None, line_means=None, larghezza=5.9):
    """Salva in PDF le figure del capitolo Risultati, nello stesso stile delle figure della tesi:
    boxplot di E0 e betaE, cicli FD medi e curve SR medie (scala logaritmica e lineare)."""
    output_dir = Path(output_dir)
    with plt.rc_context():
        virgola = _stile_tesi()
        nomi = [NOMI_BREVI_TESI.get(l, l) for l in line_names]
        salvate = []

        # --- boxplot E0 e betaE, un punto per cellula ---
        if fd_cells is not None and not fd_cells.empty:
            rng = np.random.default_rng(0)
            fig, axs = plt.subplots(1, 2, figsize=(larghezza, 2.6), gridspec_kw=dict(wspace=0.32))
            for ax, (col, ylab, lett) in zip(axs, [("E0_Pa", r"$E_0$ (Pa)", "a"), ("betaE", r"$\beta_E$", "b")]):
                dati = [fd_cells.loc[fd_cells["linea"] == l, col].dropna().values for l in line_names]
                bp = ax.boxplot(dati, positions=range(len(dati)), widths=0.5, showfliers=False, patch_artist=True,
                                medianprops=dict(color="k", lw=1.3), whiskerprops=dict(lw=0.8),
                                capprops=dict(lw=0.8), boxprops=dict(lw=0.8))
                for patch, c in zip(bp["boxes"], COLORI_TESI):
                    patch.set_facecolor(c); patch.set_alpha(0.18)
                for i, (v, c) in enumerate(zip(dati, COLORI_TESI)):
                    ax.scatter(i + rng.uniform(-0.16, 0.16, v.size), v, s=9, color=c, alpha=0.75, lw=0, zorder=3)
                ax.set_xticks(range(len(dati)))
                ax.set_xticklabels([f"{n}\n(n = {len(v)})" for n, v in zip(nomi, dati)])
                ax.set_ylabel(ylab); ax.yaxis.set_major_formatter(virgola); ax.grid(axis="x", visible=False)
                _lettera(ax, lett)
                if col == "E0_Pa":
                    ax.set_ylim(bottom=0)
            path = output_dir / f"fig_tesi_box_{sistema}.pdf"
            fig.savefig(path, bbox_inches="tight"); plt.close(fig); salvate.append(path)

        # --- ciclo forza-indentazione medio, un pannello per linea ---
        if fd_means:
            fig, axs = plt.subplots(1, len(line_names), figsize=(larghezza, 2.5), sharey=True,
                                    gridspec_kw=dict(wspace=0.12))
            for ax, l, n, c, lett in zip(np.atleast_1d(axs), line_names, nomi, COLORI_TESI, "ab"):
                if l not in fd_means:
                    continue
                d = fd_means[l]
                x = np.asarray(d["indentazione_um"], float); y = np.asarray(d["forza_nN"], float)
                sd = np.asarray(d["sd_nN"], float)
                ax.plot(x, y, color=c, lw=1.6, label="ciclo medio")
                ax.plot(x, y + sd, color=c, lw=0.7, ls=":", label="± 1 DS")
                ax.plot(x, y - sd, color=c, lw=0.7, ls=":")
                ax.axhline(0, color="#555555", lw=0.6)
                ax.set_xlabel(r"Indentazione $\delta$ ($\mu$m)"); ax.set_title(n); _lettera(ax, lett)
                ax.xaxis.set_major_formatter(virgola); ax.yaxis.set_major_formatter(virgola)
                ax.legend(loc="upper left")
            np.atleast_1d(axs)[0].set_ylabel("Forza (nN)")
            path = output_dir / f"fig_tesi_FD_media_{sistema}.pdf"
            fig.savefig(path, bbox_inches="tight"); plt.close(fig); salvate.append(path)

        # --- curve medie di stress-relaxation, un pannello per linea ---
        if line_means:
            for scala in ("log", "lineare"):
                fig, axs = plt.subplots(1, len(line_names), figsize=(larghezza, 2.5), sharey=True,
                                        gridspec_kw=dict(wspace=0.12))
                for ax, l, n, c, lett in zip(np.atleast_1d(axs), line_names, nomi, COLORI_TESI, "ab"):
                    if l not in line_means:
                        continue
                    d = line_means[l]
                    t = np.asarray(d["time"], float); m = np.asarray(d["mean"], float); s_ = np.asarray(d["sd"], float)
                    if scala == "log":
                        idx = np.unique(np.searchsorted(t, np.logspace(-4, np.log10(t.max()), 600)))
                        idx = idx[idx < t.size]; tt = np.maximum(t[idx], 1e-4)
                    else:
                        idx = np.unique(np.r_[np.arange(0, min(2000, t.size), 5),
                                              np.linspace(0, t.size - 1, 1500).astype(int)]); tt = t[idx]
                    ax.fill_between(tt, m[idx] - s_[idx], m[idx] + s_[idx], color=c, alpha=0.18, lw=0, label="± 1 DS")
                    ax.plot(tt, m[idx], color=c, lw=1.6, label="curva media")
                    if scala == "log":
                        ax.set_xscale("log"); ax.set_xlim(1e-4, t.max())
                        ax.xaxis.set_major_formatter(virgola)
                    else:
                        ax.set_xlim(0, t.max()); ax.xaxis.set_major_formatter(virgola)
                    ax.set_ylim(0, 1.05); ax.yaxis.set_major_formatter(virgola)
                    ax.set_xlabel("Tempo di mantenimento (s)"); ax.set_title(f"{n} (n = {d['n']})"); _lettera(ax, lett)
                    ax.legend(loc="lower left")
                np.atleast_1d(axs)[0].set_ylabel(r"Forza normalizzata F/F$_0$")
                suff = "" if scala == "log" else "_lineare"
                path = output_dir / f"fig_tesi_SR_media_{sistema}{suff}.pdf"
                fig.savefig(path, bbox_inches="tight"); plt.close(fig); salvate.append(path)

    for path in salvate:
        print("Figura per la tesi salvata:", path.name)
    return salvate



def run_full_analysis(lines, sistema, output_dir, run_fd=True, run_sr=True,
                      ting_options=None, sr_options=None, verbose=True, criteri=None,
                      escludi_giornate=None, altezze_um=None):
    """Esegue tutta l'analisi e salva tabelle (CSV) e report (PDF) in output_dir.

    Report PDF: tabella dei valori FD, ciclo forza-indentazione medio di ogni
    linea (un grafico per linea), boxplot di E0 e βE (un punto per cellula),
    tabella dei valori SR, curva SR media di ogni linea con i fit dopo t_trans,
    e con SLS/PLR/bi-exp fittati da t = 0.

    criteri: dizionario dei criteri di esclusione (es. CRITERI_MURINE) oppure
    None per non escludere nulla oltre alle analisi fallite.
    escludi_giornate: elenco di giornate da non analizzare (basta una parte del
    nome, es. ["100626"]); vengono riportate nel report.
    altezze_um: altezza delle cellule [um] per la correzione dello spessore finito,
    {linea: h} oppure {(linea, giornata): h}; None = nessuna correzione.
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    line_names = list(lines)
    if len(line_names) != 2:
        print("Attenzione: il confronto statistico richiede esattamente due linee.")
    sr_opts = {**DEFAULT_SR_OPTIONS, **(sr_options or {})}

    cells = find_cells(lines, sistema)
    giornate_escluse = []
    if escludi_giornate:
        drop = cells["giornata"].astype(str).apply(lambda g: any(str(x) in g for x in escludi_giornate))
        giornate_escluse = sorted(cells.loc[drop, "giornata"].unique())
        print("Giornate escluse dall'analisi:", giornate_escluse)
        cells = cells[~drop].reset_index(drop=True)
    ting_opts = _merge_ting_options(ting_options)
    drag_tables = None
    if ting_opts.get("drag_mode") == "minimo_cartella":
        print("\nResistenza viscosa: distanza minima tra i rami per area, mediana per giornata (lettura dei file FD)...")
        drag_curves, drag_by_folder, drag_by_day = drag_table_min_gap(cells)
        drag_curves.to_csv(output_dir / f"distanza_rami_FD_{sistema}.csv", index=False)
        drag_tables = (drag_by_folder, drag_by_day)
        for (line, day), val in sorted(drag_by_day.items()):
            print(f"  {line} {day}: {val * 1e9:.3f} nN")
    print_cells_overview(cells)
    results = {"cells": cells}
    fd_cells = pd.DataFrame()
    sr_ok = pd.DataFrame()
    fig_fd_box = None
    figs_sr = []
    fig_fd_curves = []
    excluded_tables = []
    fd_counts = (None, None)
    sr_counts = (None, None)

    # --- Forza-indentazione ---
    if run_fd and (cells["tipo"] == "FD").any():
        print("\n=== Curve forza-indentazione ===")
        fd_curves, fd_arrays = analyze_fd_curves(
            cells, ting_options=ting_options, verbose=verbose,
            drag_by_folder=drag_tables[0] if drag_tables else None,
            drag_by_day=drag_tables[1] if drag_tables else None,
            altezze_um=altezze_um,
        )
        fd_curves.to_csv(output_dir / f"FD_curve_{sistema}.csv", index=False)
        fd_ok_all = fd_curves[fd_curves["stato"] == "ok"] if "stato" in fd_curves.columns else fd_curves
        fd_kept, fd_excl = apply_exclusions(fd_ok_all, "FD", criteri)
        excluded_tables.append(fd_excl)
        if "stato" in fd_curves.columns:
            wrong = fd_curves[fd_curves["stato"].astype(str).str.startswith("protocollo SR")].copy()
            if not wrong.empty:
                wrong.insert(0, "motivo_esclusione", wrong["stato"])
                wrong.insert(0, "tipo", "FD")
                excluded_tables.append(wrong)
        fd_counts = (fd_ok_all, fd_kept)
        if not fd_excl.empty:
            kept_paths = set(fd_kept["path"])
            fd_arrays = [a for a in fd_arrays if a.get("path") in kept_paths]
        fd_cells = summarize_fd_by_cell(fd_kept)
        results.update(fd_curves=fd_curves)
        if fd_cells.empty:
            print("FD: nessuna curva analizzata con successo (vedi gli errori sopra).")
        else:
            fd_cells.to_csv(output_dir / f"FD_per_cellula_{sistema}.csv", index=False)
            results["fd_cells"] = fd_cells
            if fd_arrays:
                fd_means = fd_mean_curves(fd_arrays)
                results["fd_means"] = fd_means
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                for i, line in enumerate([l for l in line_names if l in fd_means]):
                    fig_fd_curves.append(plot_fd_mean_curve(line, fd_means[line], colors[i % len(colors)]))
                    plt.show()
                rows = []
                for line, d in fd_means.items():
                    rows.append(pd.DataFrame({"linea": line, "indentazione_um": d["indentazione_um"],
                                              "forza_media_nN": d["forza_nN"], "sd_nN": d["sd_nN"]}))
                pd.concat(rows).to_csv(output_dir / f"FD_curva_media_linea_{sistema}.csv", index=False)
            fig_fd_box = boxplot_by_line(fd_cells, ["E0_Pa", "betaE"], "Curve forza-indentazione: un punto per cellula", line_names)
            plt.show()

    # --- Stress-relaxation ---
    if run_sr and (cells["tipo"] == "SR").any():
        print("\n=== Curve di stress-relaxation ===")
        sr_cells, cell_curves = analyze_sr_cells(cells, sr_options=sr_options, verbose=verbose, drag_tables=drag_tables,
                                                 fd_curves=results.get("fd_curves"))
        sr_cells.to_csv(output_dir / f"SR_per_cellula_{sistema}.csv", index=False)
        results["sr_cells"] = sr_cells
        sr_ok = sr_cells[sr_cells["stato"] == "ok"] if "stato" in sr_cells.columns else sr_cells
        sr_ok_all = sr_ok
        sr_ok, sr_excl = apply_exclusions(sr_ok, "SR", criteri)
        excluded_tables.append(sr_excl)
        sr_counts = (sr_ok_all, sr_ok)
        if not sr_excl.empty:
            kept_ids = set(sr_ok["id_cellula"])
            cell_curves = [c for c in cell_curves if c["id_cellula"] in kept_ids]
        if sr_ok.empty:
            print("SR: nessuna cellula analizzata con successo (vedi gli errori sopra).")
        elif cell_curves:
            line_means, line_fit = line_mean_curves(cell_curves, sr_options, show_plots=False)
            line_fit.to_csv(output_dir / f"SR_fit_curva_media_linea_{sistema}.csv", index=False)
            # curve medie SR (tempo, media, deviazione standard) per le figure della tesi
            pd.concat([pd.DataFrame({"linea": line, "tempo_s": d["time"], "forza_norm_media": d["mean"],
                                     "sd": d["sd"], "n_cellule": d["n"]})
                       for line, d in line_means.items()]).to_csv(
                output_dir / f"SR_curva_media_linea_{sistema}.csv", index=False)
            results.update(line_means=line_means, line_fit=line_fit)
            # una pagina per linea: fit dopo t_trans, poi fit a partire da t = 0
            for line in [l for l in line_names if l in line_means]:
                single = {line: line_means[line]}
                figs_sr.append(plot_line_mean_curves(single, sr_opts["plot_max_time_s"]))
                plt.show()
                fig_zero = plot_line_mean_fits_from_zero(single, sr_opts["plot_max_time_s"])
                if fig_zero is not None:
                    figs_sr.append(fig_zero)
                    plt.show()
            cols = ["linea", "n_cellule", "F_0.1s", "F_1s", "F_relax", "rilassamento_pct", "SLS_F_inf", "SLS_tau_s",
                    "PLR_beta", "PLR_t0_s", "BI_F_inf", "BI_w", "BI_tau1_s", "BI_tau2_s", "PORO_tau_p_s", "t_trans_s"]
            print("SR - fit sulla curva media di ciascuna linea:")
            display(line_fit[[c for c in cols if c in line_fit.columns]].round(4))

    # --- Esclusioni ---
    excl_summary = pd.DataFrame()
    if criteri or any(e is not None and not e.empty for e in excluded_tables):
        excl = [e for e in excluded_tables if e is not None and not e.empty]
        excl_all = pd.concat(excl, ignore_index=True) if excl else pd.DataFrame()
        keep_cols = [c for c in ["tipo", "motivo_esclusione", "linea", "giornata", "cartella", "id_cellula", "curva"]
                     if c in excl_all.columns]
        if not excl_all.empty:
            excl_all = excl_all[keep_cols + [c for c in excl_all.columns if c not in keep_cols]]
        excl_all.to_csv(output_dir / f"esclusi_{sistema}.csv", index=False)
        excl_summary = exclusion_summary(fd_counts[0], fd_counts[1], sr_counts[0], sr_counts[1], line_names)
        results.update(excluded=excl_all, exclusion_summary=excl_summary)
        print("\nCriteri di esclusione:", criteri)
        print("Tenute dopo i criteri (dettaglio in esclusi_%s.csv):" % sistema)
        display(excl_summary)

    # --- Due tabelle riassuntive (media ± DS tra le cellule, p di Mann-Whitney) ---
    fd_summary = summary_table([(fd_cells, SUMMARY_FD)], line_names)
    sr_summary = summary_table([(sr_ok, SUMMARY_SR)], line_names)
    fd_summary.to_csv(output_dir / f"riassunto_FD_{sistema}.csv", index=False)
    sr_summary.to_csv(output_dir / f"riassunto_SR_{sistema}.csv", index=False)
    results.update(fd_summary=fd_summary, sr_summary=sr_summary)
    for title, table in (("forza-indentazione", fd_summary), ("stress-relaxation", sr_summary)):
        if not table.empty:
            print(f"\nRiassunto {title} (media ± DS tra le cellule; p del test di Mann-Whitney sulle cellule):")
            display(table)

    # --- Figure per la tesi ---
    try:
        salva_figure_tesi(output_dir, sistema, line_names, fd_cells=fd_cells,
                          fd_means=results.get("fd_means"), line_means=results.get("line_means"))
    except Exception as exc:
        print("Figure per la tesi non salvate:", exc)

    pdf_path = output_dir / f"report_{sistema}.pdf"
    with PdfPages(pdf_path) as pdf:
        pages = []
        if giornate_escluse:
            pages.append(table_figure(pd.DataFrame({"giornata esclusa": giornate_escluse}),
                                      "Giornate non analizzate"))
        if not excl_summary.empty:
            crit_txt = ", ".join(f"{k} = {v}" for k, v in (criteri or {}).items()) or "nessun criterio"
            pages.append(table_figure(excl_summary, f"Curve e cellule tenute dopo i criteri di esclusione ({crit_txt})"))
        if not fd_summary.empty:
            pages.append(table_figure(fd_summary, f"Curve forza-indentazione - {sistema}: valori per linea (unità = cellula)"))
        pages += fig_fd_curves + [fig_fd_box]
        if not sr_summary.empty:
            pages.append(table_figure(sr_summary, f"Curve di stress-relaxation - {sistema}: valori per linea (unità = cellula)"))
        pages += figs_sr
        for fig in pages:
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
    print(f"\nTabelle e report salvati in: {output_dir}")
    return results
