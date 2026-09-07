import copy
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyfmrheo.routines.TingFit as _pyfmrheo_tingfit_module
from matplotlib.backends.backend_pdf import PdfPages
from lmfit import Model
from pyfmreader import loadfile
from pyfmrheo.models.ting import TingModel
from pyfmrheo.routines.HertzFit import doHertzFit
from pyfmrheo.routines.TingFit import doTingFit as _pyfmrheo_doTingFit
from pyfmrheo.utils import force_curves as _pyfmrheo_force_curves
from pyfmrheo.utils.force_curves import (
    correct_offset,
    correct_tilt,
    get_poc_RoV_method,
    get_poc_regulaFalsi_method,
)
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from prova import (
    select_data_folder_gui,
    collect_fd_folders,
    collect_curve_files,
    _safe_correct_viscous_drag,
)

try:
    from pyfmgui.ting_model import TingFitter, ting_force
except ModuleNotFoundError:
    _LOCAL_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "general", "src")
    if _LOCAL_SRC not in sys.path:
        sys.path.insert(0, _LOCAL_SRC)
    from pyfmgui.ting_model import TingFitter, ting_force

__all__ = [
    "select_data_folder_gui",
    "collect_fd_folders",
    "collect_curve_files",
    "process_curve_ting",
    "run_ting_fit",
    "compare_ting_models",
    "run_ting_batch_analysis",
    "run_ting_analysis",
    "save_ting_outputs",
]

_LAST_DRAG_INFO: dict = {}


def _intercepting_drag_correction(
    ind_approach,
    force_approach,
    ind_retract,
    force_retract,
    poly_order=2,
    speed=None,
    **_unused_kwargs,
):
    """Thin wrapper that forwards to _safe_correct_viscous_drag and caches diagnostics."""
    f_app, f_ret, info = _safe_correct_viscous_drag(
        ind_approach,
        force_approach,
        ind_retract,
        force_retract,
        poly_order=poly_order,
        speed=speed,
        return_diagnostics=True,
    )
    _LAST_DRAG_INFO.update(info)
    return f_app, f_ret  # doTingFit expects a 2-tuple


_pyfmrheo_force_curves.correct_viscous_drag = _intercepting_drag_correction
_pyfmrheo_tingfit_module.correct_viscous_drag = _intercepting_drag_correction


def _doTingFit_decoupled_tc(fdc, param_dict):
    """Local wrapper around pyfmrheo TingFit with independent tc-bound downsampling."""
    ext_data = fdc.extend_segments[0][1]
    ret_data = fdc.retract_segments[-1][1]
    fdc_hertz_fit = copy.deepcopy(fdc)

    height = np.r_[ext_data.zheight, ret_data.zheight]
    deflection = np.r_[ext_data.vdeflection, ret_data.vdeflection]
    idx = len(ext_data.zheight)
    if param_dict["offset_type"] == "percentage":
        deltaz = height.max() - height.min()
        maxoffset = height.min() + deltaz * param_dict["max_offset"]
        minoffset = height.min() + deltaz * param_dict["min_offset"]
    else:
        maxoffset = param_dict["max_offset"]
        minoffset = param_dict["min_offset"]

    if param_dict["correct_tilt"]:
        corr_defl = correct_tilt(height, deflection, maxoffset, minoffset)
    else:
        corr_defl = correct_offset(height, deflection, maxoffset, minoffset)
    ext_data.vdeflection = corr_defl[:idx]
    ret_data.vdeflection = corr_defl[idx:]

    if param_dict["poc_method"] == "RoV":
        comp_PoC = get_poc_RoV_method(ext_data.zheight, ext_data.vdeflection, param_dict["poc_win"])
    else:
        comp_PoC = get_poc_regulaFalsi_method(ext_data.zheight, ext_data.vdeflection, param_dict["sigma"])
    poc = [comp_PoC[0], 0]

    hertz_result = doHertzFit(fdc_hertz_fit, param_dict)
    hertz_d0 = hertz_result.delta0
    hertz_E0 = hertz_result.E0
    z_range = np.abs(height.max() - height.min())
    if np.abs(hertz_d0) > z_range:
        print(
            f"[TingFit] Warning: Hertz delta0={hertz_d0 * 1e9:.1f}nm diverged "
            f"(z_range={z_range * 1e9:.1f}nm). Using RoV PoC only (delta0=0).",
            flush=True,
            file=sys.stderr,
        )
        hertz_d0 = 0.0

    poc[0] += hertz_d0
    poc[1] = 0
    fdc.get_force_vs_indentation(poc, param_dict["k"])

    ext_indentation = ext_data.indentation
    ext_force = ext_data.force
    ext_time = ext_data.time
    ret_indentation = ret_data.indentation
    ret_force = ret_data.force
    ret_time = ret_data.time

    # Correzione baseline: usa gli ultimi N punti pre-contatto (δ < 0) di trace e retrace,
    # calcola la media comune e sposta verticalmente entrambe le curve verso di essa.
    # Attivata solo se baseline_align_poc=True nel param_dict.
    if param_dict.get("baseline_align_poc", False):
        _n_bl = 20
        _nc_ext = np.where(np.isfinite(ext_indentation) & (ext_indentation < 0) & np.isfinite(ext_force))[0]
        _nc_ret = np.where(np.isfinite(ret_indentation) & (ret_indentation < 0) & np.isfinite(ret_force))[0]
        if _nc_ext.size > 0 and _nc_ret.size > 0:
            _mu_ext = float(np.nanmean(ext_force[_nc_ext[-min(_n_bl, _nc_ext.size):]]))
            _mu_ret = float(np.nanmean(ret_force[_nc_ret[:min(_n_bl, _nc_ret.size)]]))
            _mu_common = (_mu_ext + _mu_ret) / 2.0
            ext_force = ext_force - _mu_ext + _mu_common
            ret_force = ret_force - _mu_ret + _mu_common

    t_offset = np.abs(ext_data.zheight[-1] - ret_data.zheight[0]) / (ext_data.velocity * -1e-9)
    dt = np.abs(ext_data.time[1] - ext_data.time[0])
    if t_offset > 2 * dt:
        ret_time = ret_time + t_offset

    if param_dict["vdragcorr"]:
        ext_force, ret_force = _intercepting_drag_correction(
            ext_indentation,
            ext_force,
            ret_indentation,
            ret_force,
            poly_order=param_dict["polyordr"],
            speed=param_dict["rampspeed"],
        )

    if not param_dict["compute_v_flag"]:
        v0t = np.abs(ext_data.zheight.min() - ext_data.zheight.max()) / ext_data.segment_metadata["duration"]
        v0r = np.abs(ret_data.zheight.min() - ret_data.zheight.max()) / ret_data.segment_metadata["duration"]
    else:
        v0t, v0r = None, None

    idx_tc = (np.abs(ext_indentation - 0)).argmin()
    t0 = ext_time[-1]
    indentation = np.r_[ext_indentation, ret_indentation]
    time = np.r_[ext_time, ret_time + t0]
    force = np.r_[ext_force, ret_force]
    fit_mask = indentation > (-1 * param_dict["contact_offset"])
    tc = time[idx_tc]
    ind_fit = indentation[fit_mask]
    force_fit = force[fit_mask]
    force_fit = force_fit - force_fit[0]
    time_fit = time[fit_mask]
    tc_fit = tc - time_fit[0]
    time_fit = time_fit - time_fit[0] - tc_fit
    tc_fit = 0.0

    pts_downsample_fit = max(1, int(param_dict["pts_downsample"]))
    downfactor_fit = max(1, len(time_fit) // pts_downsample_fit)
    idxDown = list(range(0, len(time_fit), downfactor_fit))

    idx_tm = np.argmax(force_fit[idxDown])
    f0idx = np.where(time_fit == 0)[0]
    if v0t is not None:
        F0_init = force_fit[f0idx] - param_dict["vdrag"] * v0t
    else:
        F0_init = force_fit[f0idx]

    tc_bound_pts = max(1, int(param_dict.get("tc_bound_pts_downsample", pts_downsample_fit)))
    tc_bound_downfactor = max(1, len(time_fit) // tc_bound_pts)
    tc_step = tc_bound_downfactor * (time_fit[1] - time_fit[0]) * 10 if len(time_fit) > 1 else 0.0
    tc_max = tc_fit + tc_step
    tc_min = max(0.0, tc_fit - tc_step)
    f0_max = F0_init + 100e-12
    f0_min = F0_init - 100e-12

    if param_dict["auto_init_betaE"]:
        betaE_init = 0.15 if hertz_E0 > 10e3 else 0.25
    else:
        betaE_init = param_dict["fluid_exp"]
    if param_dict["contact_model"] == "paraboloid":
        betaE_min, betaE_max = 0.01, 0.49
    else:
        betaE_min, betaE_max = 0.01, 0.99

    ting_model = TingModel(param_dict["contact_model"], param_dict["tip_param"], param_dict["model_type"])
    ting_model.E0_init = hertz_E0
    if np.isfinite(hertz_E0) and hertz_E0 > 0:
        e0_lower_factor = max(float(param_dict.get("hertz_e0_lower_factor", 1e-3)), 1e-9)
        ting_model.E0_min = max(hertz_E0 * e0_lower_factor, 1e-9)
        if bool(param_dict.get("enforce_hertz_e0_upper_bound", False)):
            e0_upper_factor = max(float(param_dict.get("hertz_e0_upper_factor", 1.0)), e0_lower_factor * 1.01)
            ting_model.E0_max = max(hertz_E0 * e0_upper_factor, ting_model.E0_min * 1.01)
        else:
            ting_model.E0_max = np.inf
    else:
        ting_model.E0_min = 1e-9
        ting_model.E0_max = np.inf
    ting_model.tc_init = tc_fit
    ting_model.tc_min = tc_min
    ting_model.tc_max = tc_max
    ting_model.betaE_init = betaE_init
    ting_model.betaE_min = betaE_min
    ting_model.betaE_max = betaE_max
    ting_model.F0_init = F0_init[0]
    ting_model.F0_min = f0_min[0]
    ting_model.F0_max = f0_max[0]
    ting_model.vdrag = param_dict["vdrag"]

    ting_model.fit(
        time_fit[idxDown],
        force_fit[idxDown],
        ind_fit[idxDown],
        t0=param_dict["t0"],
        idx_tm=idx_tm,
        smooth_w=param_dict["smoothing_win"],
        v0t=v0t,
        v0r=v0r,
    )

    return ting_model, hertz_result


def _run_ting_solver(fdc, param_dict):
    """Dispatch Ting solver: legacy upstream by default, decoupled tc-bounds on demand."""
    if bool(param_dict.get("use_decoupled_tc_bounds", False)):
        return _doTingFit_decoupled_tc(fdc, param_dict)
    return _pyfmrheo_doTingFit(fdc, param_dict)


def process_curve_ting(filepath, maxnoncontact=1e-6, pts_downsample=3000, raw_deflection_smooth_win=0):
    """Load and preprocess a single AFM force curve for Ting analysis."""
    afm_file = loadfile(filepath)
    FC = afm_file.getcurve(0)
    metadata = afm_file.filemetadata

    defl_sens = metadata["defl_sens_nmbyV"] * 1e-9
    spring_constant = metadata["spring_const_Nbym"]
    height_channel = metadata["height_channel_key"]

    FC.preprocess_force_curve(defl_sens, height_channel)

    if metadata["file_type"] in ("jpk-force", "jpk-force-map", "jpk-qi-data"):
        FC.shift_height()

    FC_uncorrected = copy.deepcopy(FC)

    seg_id, segment = FC.get_segments()[0]

    idx_max = np.argmax(segment.vdeflection)
    z_setpoint = segment.zheight[idx_max]
    FC.z_at_setpoint = z_setpoint

    z = segment.zheight
    deflection = segment.vdeflection

    baseline_info = {
        "trace_baseline_defl_m": float("nan"),
        "retrace_baseline_defl_m": float("nan"),
        "shift_retrace_defl_m": 0.0,
        "shift_retrace_force_N": 0.0,
        "retrace_points_removed": 0,
        "status": "not_applied",
    }

    try:
        ext_seg = FC.extend_segments[0][1]
        ret_seg = FC.retract_segments[-1][1]

        ext_defl = np.atleast_1d(np.asarray(ext_seg.vdeflection, dtype=float)).astype(float, copy=True)
        ret_defl = np.atleast_1d(np.asarray(ret_seg.vdeflection, dtype=float)).astype(float, copy=True)
        ext_ind = np.atleast_1d(np.asarray(ext_seg.indentation, dtype=float)).astype(float, copy=True)
        ret_ind = np.atleast_1d(np.asarray(ret_seg.indentation, dtype=float)).astype(float, copy=True)

        # Apply Savitzky-Golay filter to raw deflection to reduce high-frequency noise
        _sg_win = int(raw_deflection_smooth_win) if raw_deflection_smooth_win else 0
        if _sg_win >= 5 and _sg_win % 2 == 1:
            _sg_win_tr = min(_sg_win, ext_defl.size - 1 if ext_defl.size % 2 == 0 else ext_defl.size)
            _sg_win_tr = _sg_win_tr if _sg_win_tr % 2 == 1 else _sg_win_tr - 1
            _sg_win_ret = min(_sg_win, ret_defl.size - 1 if ret_defl.size % 2 == 0 else ret_defl.size)
            _sg_win_ret = _sg_win_ret if _sg_win_ret % 2 == 1 else _sg_win_ret - 1
            if _sg_win_tr >= 5:
                ext_defl = savgol_filter(ext_defl, window_length=_sg_win_tr, polyorder=3)
                ext_seg.vdeflection = ext_defl
            if _sg_win_ret >= 5:
                ret_defl = savgol_filter(ret_defl, window_length=_sg_win_ret, polyorder=3)
                ret_seg.vdeflection = ret_defl

        # Baseline non-contatto (indentazione < 0) di deflection e forza
        nc_ext = np.isfinite(ext_ind) & (ext_ind < 0) & np.isfinite(ext_defl)
        nc_ret = np.isfinite(ret_ind) & (ret_ind < 0) & np.isfinite(ret_defl)
        n_bl = max(5, min(ext_defl.size, ret_defl.size) // 10)
        trace_baseline_defl = (
            float(np.nanmean(ext_defl[nc_ext])) if np.count_nonzero(nc_ext) >= 3
            else float(np.nanmedian(ext_defl[:n_bl]))
        )
        retrace_baseline_defl = (
            float(np.nanmean(ret_defl[nc_ret])) if np.count_nonzero(nc_ret) >= 3
            else float(np.nanmedian(ret_defl[-n_bl:]))
        )

        # Nessuna correzione baseline in process_curve_ting: correct_tilt la annullerebbe.
        # La correzione reale avviene in _doTingFit_decoupled_tc dopo correct_tilt.
        ext_defl_shift = 0.0
        ret_defl_shift = 0.0
        retrace_shift = 0.0

        baseline_info = {
            "trace_baseline_defl_m": trace_baseline_defl,
            "retrace_baseline_defl_m": retrace_baseline_defl,
            "shift_trace_defl_m": ext_defl_shift,
            "shift_retrace_defl_m": ret_defl_shift,
            "shift_retrace_force_N": 0.0,
            "retrace_points_removed": 0,
            "retrace_points_filled": 0,
            "baseline_strategy": "poc_proximity_20pts_common_mean",
            "contact_line_poc_idx": -1,
            "contact_line_intersection_idx": -1,
            "status": "deferred_to_tingfit",
        }
    except Exception as exc:
        baseline_info["status"] = f"failed: {exc}"

    param_dict = {
        "height_channel": height_channel,
        "def_sens": defl_sens,
        "k": spring_constant,
        "contact_model": "paraboloid",
        "tip_param": 5e-06,
        "curve_seg": "extend",
        "correct_tilt": False,
        "tilt_min_offset": 1e-08,
        "tilt_max_offset": 1e-06,
        "poisson": 0.5,
        "poc_method": "RoV",
        "poc_win": 4e-07,
        "sigma": 2,
        "max_ind": 0.0,
        "min_ind": 0.0,
        "max_force": 0.0,
        "min_force": 0.0,
        "fit_range_type": "full",
        "vdragcorr": True,
        "polyordr": 2,
        "rampspeed": 0.0,
        "compute_v_flag": False,
        "t0": 1,
        "d0": 0.0,
        "slope": 0.0,
        "auto_init_E0": True,
        "E0": 1000,
        "tc": 0.0,
        "auto_init_betaE": True,
        "fluid_exp": 0.2,
        "f0": 0.0,
        "vdrag": 2.5e-06,
        "model_type": "analytical",
        "smoothing_win": 51,
        "raw_deflection_smooth_win": int(raw_deflection_smooth_win),
        "contact_offset": maxnoncontact,
        "apply_baseline_alignment": False,
        "baseline_align_poc": False,
        "apply_ramp_correction": False,
        "fit_line": False,
        "downsample_flag": True,
        "pts_downsample": int(pts_downsample),
        "tc_bound_pts_downsample": 200,
        "use_decoupled_tc_bounds": True,
        "enforce_hertz_e0_upper_bound": False,
        "hertz_e0_upper_factor": 1.0,
        "hertz_e0_lower_factor": 1e-3,
        "pts_downsample_vis": int(pts_downsample),
        "offset_type": "percentage",
        "max_offset": 0.3,
        "min_offset": 0,
    }

    return {
        "FC_object": FC,
        "segment": segment,
        "segment_id": seg_id,
        "z": z,
        "deflection": deflection,
        "z_setpoint": z_setpoint,
        "spring_constant": spring_constant,
        "defl_sens": defl_sens,
        "metadata": metadata,
        "param_dict": param_dict,
        "baseline_info": baseline_info,
        "FC_uncorrected_object": FC_uncorrected,
    }


def _compute_contact_closure_shift(ext_force, ret_force, ext_ind, ret_ind):
    """Estimate trace-retrace closure gap on the contact region (indentation >= 0)."""
    ext_force = np.atleast_1d(np.asarray(ext_force, dtype=float))
    ret_force = np.atleast_1d(np.asarray(ret_force, dtype=float))
    ext_ind = np.atleast_1d(np.asarray(ext_ind, dtype=float))
    ret_ind = np.atleast_1d(np.asarray(ret_ind, dtype=float))

    force_comb = np.r_[ext_force, ret_force]
    if force_comb.size >= 21:
        force_comb = savgol_filter(force_comb, window_length=21, polyorder=3)
    ext_force_sm = force_comb[:len(ext_force)]
    ret_force_sm = force_comb[len(ext_force):]

    ext_contact = ext_ind >= 0.0
    ret_contact = ret_ind >= 0.0
    n_ext_c = int(np.sum(ext_contact))
    n_ret_c = int(np.sum(ret_contact))
    n_close = min(20, max(3, n_ext_c // 10), max(3, n_ret_c // 10))

    closure_shift = 0.0
    if n_ext_c >= n_close and n_ret_c >= n_close:
        f0 = float(ext_force_sm[ext_contact][0])
        tr_mean = float(np.nanmean(ext_force_sm[ext_contact][0:n_close])) - f0
        ret_mean = float(np.nanmean(ret_force_sm[ret_contact][-n_close:])) - f0
        closure_shift = tr_mean - ret_mean

    return float(closure_shift), ret_contact


def _apply_retrace_ramp_to_vdefl(ret_seg, ret_contact_mask, closure_shift, spring_constant):
    """Apply linear ramp correction to retrace vdeflection on contact points only."""
    if abs(float(closure_shift)) <= 1e-14:
        return False

    contact_idx = np.where(np.asarray(ret_contact_mask, dtype=bool))[0]
    if contact_idx.size == 0:
        return False

    ret_vdefl = np.atleast_1d(np.asarray(ret_seg.vdeflection, dtype=float)).copy()
    ret_vdefl[contact_idx] += np.linspace(0.0, float(closure_shift) / float(spring_constant), contact_idx.size)
    ret_seg.vdeflection = ret_vdefl
    return True


def _build_ramp_corrected_curve(FC_source, param_dict):
    """Return a deepcopy of the curve with ramp correction applied when requested."""
    ramp_info = {
        "applied": False,
        "status": "disabled",
        "closure_shift_N": 0.0,
    }

    if not bool(param_dict.get("apply_ramp_correction", False)):
        return copy.deepcopy(FC_source), ramp_info

    FC_corrected = copy.deepcopy(FC_source)
    FC_prelim = copy.deepcopy(FC_source)
    try:
        _run_ting_solver(FC_prelim, dict(param_dict))
        prelim_ext = FC_prelim.extend_segments[0][1]
        prelim_ret = FC_prelim.retract_segments[-1][1]

        closure_shift, ret_contact = _compute_contact_closure_shift(
            np.atleast_1d(np.asarray(prelim_ext.force, dtype=float)),
            np.atleast_1d(np.asarray(prelim_ret.force, dtype=float)),
            np.atleast_1d(np.asarray(prelim_ext.indentation, dtype=float)),
            np.atleast_1d(np.asarray(prelim_ret.indentation, dtype=float)),
        )
        ramp_applied = _apply_retrace_ramp_to_vdefl(
            FC_corrected.retract_segments[-1][1],
            ret_contact,
            closure_shift,
            param_dict["k"],
        )
        ramp_info.update({
            "applied": bool(ramp_applied),
            "status": "applied" if ramp_applied else "no_contact_points",
            "closure_shift_N": float(closure_shift),
        })
    except Exception as exc:
        ramp_info.update({
            "status": f"failed: {exc}",
        })

    return FC_corrected, ramp_info


def run_ting_fit(filepath=None, curve_data=None, print_reports=True, raw_deflection_smooth_win=0, param_overrides=None):
    """Run the Ting PLR fit and keep all relevant metadata together."""
    if curve_data is None:
        if filepath is None:
            raise ValueError("Provide either filepath or curve_data")
        curve_data = process_curve_ting(filepath, raw_deflection_smooth_win=raw_deflection_smooth_win)

    FC = curve_data["FC_object"]
    param_dict = dict(curve_data["param_dict"])
    baseline_info = curve_data["baseline_info"]

    param_dict, contact_point_info = _autotune_contact_point_params(FC, param_dict)
    if param_overrides:
        param_dict.update(param_overrides)
    curve_data["param_dict"] = param_dict

    # ── Passata preliminare: calcola PoC e indentazione per stimare il ramp di chiusura ──
    # Attivata solo se apply_ramp_correction=True nel param_dict.
    # Viene usata una copia di FC: il FC originale NON viene modificato qui.
    if param_dict.get("apply_ramp_correction", False):
        FC, ramp_info = _build_ramp_corrected_curve(FC, param_dict)
        curve_data["ramp_correction_info"] = ramp_info

    # ── Passata reale: fit sui dati con ramp già applicato a vdeflection ──
    _LAST_DRAG_INFO.clear()
    ting_result, hertz_result = _run_ting_solver(FC, param_dict)
    force_drag_info = dict(_LAST_DRAG_INFO) if _LAST_DRAG_INFO else {
        "enabled": bool(param_dict.get("vdragcorr", False)),
        "applied": False,
        "status": "not_called",
    }

    # ── Confronto con correzioni attive (se almeno una è disattiva nel fit principale) ──
    _corrections_off = (
        not bool(param_dict.get("baseline_align_poc", False))
        or not bool(param_dict.get("apply_ramp_correction", False))
    )
    correction_comparison = {"available": False}
    if _corrections_off:
        try:
            _FC_corr = copy.deepcopy(curve_data.get("FC_uncorrected_object") or FC)
            _param_corr = dict(param_dict)
            _param_corr["baseline_align_poc"] = True
            _param_corr["apply_ramp_correction"] = True
            _FC_corr, _ramp_info_corr = _build_ramp_corrected_curve(_FC_corr, _param_corr)
            _LAST_DRAG_INFO.clear()
            _ting_corr, _ = _run_ting_solver(_FC_corr, _param_corr)
            correction_comparison = {
                "available": True,
                "E0_no_corr_Pa": float(ting_result.E0),
                "E0_with_corr_Pa": float(_ting_corr.E0),
                "delta_E0_Pa": float(_ting_corr.E0 - ting_result.E0),
                "delta_E0_pct": float((_ting_corr.E0 - ting_result.E0) / max(abs(ting_result.E0), 1e-9) * 100),
                "betaE_no_corr": float(ting_result.betaE),
                "betaE_with_corr": float(_ting_corr.betaE),
                "delta_betaE": float(_ting_corr.betaE - ting_result.betaE),
                "R2_no_corr": float(getattr(ting_result, "Rsquared", float("nan"))),
                "R2_with_corr": float(getattr(_ting_corr, "Rsquared", float("nan"))),
            }
        except Exception as _exc:
            correction_comparison = {"available": False, "error": str(_exc)}

    fit_context = _prepare_ting_fit_arrays(FC, param_dict)
    raw_fit_context = None
    FC_uncorrected = curve_data.get("FC_uncorrected_object")
    raw_jpk_z = None
    raw_jpk_force = None
    if FC_uncorrected is not None:
        try:
            # Curva truly raw: F = vdeflection * k vs zheight, PRIMA di correct_tilt.
            # FC_uncorrected è salvato dopo preprocess_force_curve (V→m) ma prima di doTingFit.
            _k = curve_data["param_dict"]["k"]
            _ext_unc = FC_uncorrected.extend_segments[0][1]
            _ret_unc = FC_uncorrected.retract_segments[-1][1]
            _z_raw = np.r_[
                np.atleast_1d(np.asarray(_ext_unc.zheight, dtype=float)),
                np.atleast_1d(np.asarray(_ret_unc.zheight, dtype=float)),
            ]
            _f_raw = np.r_[
                np.atleast_1d(np.asarray(_ext_unc.vdeflection, dtype=float)),
                np.atleast_1d(np.asarray(_ret_unc.vdeflection, dtype=float)),
            ] * _k
            raw_jpk_z = _z_raw
            raw_jpk_force = _f_raw
        except Exception:
            pass

    if FC_uncorrected is not None:
        try:
            raw_fc_for_plot = copy.deepcopy(FC_uncorrected)
            _run_ting_solver(raw_fc_for_plot, dict(param_dict))
            raw_fit_context = _prepare_ting_fit_arrays(raw_fc_for_plot, param_dict)
        except Exception:
            raw_fit_context = None

    baseline_info = dict(baseline_info)
    baseline_info["force_drag_info"] = force_drag_info
    baseline_info["force_drag_applied"] = bool(force_drag_info.get("applied", False))
    curve_data["baseline_info"] = baseline_info

    baseline_comparison: dict[str, float | bool] = {"available": raw_fit_context is not None}
    if raw_fit_context is not None:
        raw_gap_force = float(raw_fit_context.get("retrace_baseline", float("nan")) - raw_fit_context.get("trace_baseline", float("nan")))
        corrected_gap_force = float(raw_gap_force - baseline_info.get("shift_retrace_force_N", 0.0))

        raw_force_ds = np.asarray(raw_fit_context.get("fit_force_raw_ds", []), dtype=float)
        corr_force_ds = np.asarray(fit_context.get("fit_force_raw_ds", []), dtype=float)
        if raw_force_ds.size == 0:
            raw_force_ds = np.asarray(raw_fit_context.get("force_all", []), dtype=float)
        if corr_force_ds.size == 0:
            corr_force_ds = np.asarray(fit_context.get("force_all", []), dtype=float)
        n_cmp = min(raw_force_ds.size, corr_force_ds.size)
        if n_cmp > 0:
            delta_force_ds = corr_force_ds[:n_cmp] - raw_force_ds[:n_cmp]
            max_abs_change_pN = float(np.nanmax(np.abs(delta_force_ds)) * 1e12)
            mean_abs_change_pN = float(np.nanmean(np.abs(delta_force_ds)) * 1e12)
        else:
            max_abs_change_pN = float("nan")
            mean_abs_change_pN = float("nan")

        baseline_comparison.update({
            "raw_gap_pN": raw_gap_force * 1e12,
            "corrected_gap_pN": corrected_gap_force * 1e12,
            "max_abs_change_pN": max_abs_change_pN,
            "mean_abs_change_pN": mean_abs_change_pN,
        })

    closure_refinement = _refine_ting_fit_for_closure(ting_result, fit_context)

    if print_reports:
        if str(baseline_info.get("status", "")).startswith("applied"):
            strategy = baseline_info.get("baseline_strategy", "linear_ramp_to_trace_baseline")
            print(
                "Baseline correction applied before fit | "
                f"strategia={strategy} | "
                f"Δdefl(retrace-trace, coda)={baseline_info['shift_retrace_defl_m'] * 1e9:+.2f} nm | "
                f"Δforce(coda)≈{baseline_info['shift_retrace_force_N'] * 1e12:+.1f} pN"
            )
        else:
            print(f"Baseline correction status: {baseline_info.get('status')}")

        if force_drag_info.get("enabled"):
            if force_drag_info.get("applied"):
                print(
                    "Force-drag correction applicata | "
                    f"gap baseline {force_drag_info.get('baseline_gap_before_N', float('nan')) * 1e12:+.1f}→"
                    f"{force_drag_info.get('baseline_gap_after_N', float('nan')) * 1e12:+.1f} pN | "
                    f"correzione simmetrica ±{abs(force_drag_info.get('correction_N', 0.0)) * 1e12:.1f} pN"
                )
            else:
                print(f"Force-drag correction non applicata: {force_drag_info.get('status', 'n/d')}")

        if baseline_comparison.get("available"):
            print(
                "Confronto curva raw vs corretta | "
                f"gap retrace-trace prima={baseline_comparison.get('raw_gap_pN', float('nan')):+.1f} pN | "
                f"dopo={baseline_comparison.get('corrected_gap_pN', float('nan')):+.1f} pN | "
                f"max|ΔF|={baseline_comparison.get('max_abs_change_pN', float('nan')):.2f} pN | "
                f"trim={baseline_comparison.get('retrace_trim_points', 0)} pt"
            )

        if contact_point_info.get("selected_strategy"):
            print(
                "PoC automatico | "
                f"strategia={contact_point_info['selected_strategy']} | "
                f"Hertz R2 {contact_point_info.get('default_r2', float('nan')):.4f}→{contact_point_info.get('selected_r2', float('nan')):.4f} | "
                f"delta0 {contact_point_info.get('default_delta0_nm', float('nan')):+.1f}→{contact_point_info.get('selected_delta0_nm', float('nan')):+.1f} nm"
            )

        if closure_refinement.get("applied"):
            before = closure_refinement.get("before") or {}
            after = closure_refinement.get("after") or {}
            print(
                "Closure refinement PLR applicato sul retrace | "
                f"tail RMSE {before.get('tail_rmse', float('nan')) * 1e12:.1f}→{after.get('tail_rmse', float('nan')) * 1e12:.1f} pN | "
                f"R2 {before.get('r2', float('nan')):.4f}→{after.get('r2', float('nan')):.4f}"
            )

        print("=== HERTZ (init PoC + E0) ===")
        hertz_result.fit_report()
        print("=== TING PLR (viscoelastico) ===")
        ting_result.fit_report()

        # --- confronto con/senza correzioni baseline ---
        if correction_comparison.get("available"):
            print(
                "[Correzioni baseline] senza correzioni vs con correzioni (baseline_align_poc + ramp) | "
                f"E0: {correction_comparison['E0_no_corr_Pa']:.1f} → {correction_comparison['E0_with_corr_Pa']:.1f} Pa "
                f"(ΔE0={correction_comparison['delta_E0_Pa']:+.1f} Pa, {correction_comparison['delta_E0_pct']:+.1f}%) | "
                f"betaE: {correction_comparison['betaE_no_corr']:.4f} → {correction_comparison['betaE_with_corr']:.4f} "
                f"(Δ={correction_comparison['delta_betaE']:+.4f}) | "
                f"R2: {correction_comparison['R2_no_corr']:.4f} → {correction_comparison['R2_with_corr']:.4f}"
            )
        elif correction_comparison.get("error"):
            print(f"[Correzioni baseline] confronto non disponibile: {correction_comparison['error']}")

        # --- betaE bound diagnostic ---
        betaE_val  = float(getattr(ting_result, "betaE", float("nan")))
        betaE_min_ = float(getattr(ting_result, "betaE_min", 0.01))
        betaE_max_ = float(getattr(ting_result, "betaE_max", 0.99))
        betaE_init_= float(getattr(ting_result, "betaE_init", float("nan")))
        hertz_E0_  = float(getattr(hertz_result, "E0", float("nan")))
        at_lower   = abs(betaE_val - betaE_min_) < 0.005
        at_upper   = abs(betaE_val - betaE_max_) < 0.005
        print(
            f"[betaE diagnostic] valore={betaE_val:.4f} | init={betaE_init_:.4f} | "
            f"bounds=[{betaE_min_:.3f}, {betaE_max_:.3f}] | "
            f"Hertz E0 init={hertz_E0_:.1f} Pa | "
            f"geometry={param_dict.get('contact_model')} | "
            f"smooth_win={param_dict.get('raw_deflection_smooth_win', 0)}"
        )
        if at_lower:
            print(
                "[betaE diagnostic] *** ATTENZIONE: betaE ha raggiunto il BOUND INFERIORE. ***\n"
                "  Cause probabili:\n"
                "  1. Contact point (PoC) stimato troppo in basso → la curva appare troppo elastica\n"
                "  2. Smoothing troppo aggressivo (SMOOTH_WIN alto) → appiattisce la risposta viscosa\n"
                "  3. E0 molto alto (>10 kPa) → betaE_init=0.15, il fit parte già vicino al bound\n"
                "  4. Velocità di rampa eccessiva o curva con poco retrace viscoelastico\n"
                "  Prova: ridurre SMOOTH_WIN, controllare il PoC, verificare il retrace della curva."
            )
        elif at_upper:
            print(
                "[betaE diagnostic] *** ATTENZIONE: betaE ha raggiunto il BOUND SUPERIORE. ***\n"
                "  La cellula appare quasi puramente viscosa. Controllare la qualità della curva."
            )

    out = dict(curve_data)
    out.update({
        "FC": FC,
        "ting_result": ting_result,
        "hertz_result": hertz_result,
        "contact_point_info": contact_point_info,
        "closure_refinement": closure_refinement,
        "raw_fit_context": raw_fit_context,
        "baseline_comparison": baseline_comparison,
        "fit_ind_uncorrected_ds": None if raw_fit_context is None else raw_fit_context.get("fit_ind_ds"),
        "fit_time_uncorrected_ds": None if raw_fit_context is None else raw_fit_context.get("fit_time_ds"),
        "fit_force_uncorrected_ds": None if raw_fit_context is None else raw_fit_context.get("fit_force_raw_ds"),
        "ind_all_uncorrected": None if raw_fit_context is None else raw_fit_context.get("ind_all"),
        "time_all_uncorrected": None if raw_fit_context is None else raw_fit_context.get("time_all"),
        "force_all_uncorrected": None if raw_fit_context is None else raw_fit_context.get("force_all"),
        "raw_jpk_z": raw_jpk_z,
        "raw_jpk_force": raw_jpk_force,
        "fit_context": fit_context,
        "correction_comparison": correction_comparison,
    })
    return out


def _compute_fit_metrics(force_exp, force_pred, n_params):
    """Return RSS, RMSE, R2, adjusted R2, AIC and BIC on finite points only."""
    force_exp = np.asarray(force_exp, dtype=float)
    force_pred = np.asarray(force_pred, dtype=float)
    mask = np.isfinite(force_exp) & np.isfinite(force_pred)
    n_obs = int(np.count_nonzero(mask))

    out = {
        "n_obs": n_obs,
        "rss": float("nan"),
        "rmse": float("nan"),
        "r2": float("nan"),
        "adj_r2": float("nan"),
        "aic": float("nan"),
        "bic": float("nan"),
    }
    if n_obs < max(3, int(n_params) + 1):
        return out

    y = force_exp[mask]
    yhat = force_pred[mask]
    rss = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(rss / max(n_obs, 1)))
    sigma2 = max(rss / max(n_obs, 1), 1e-30)
    aic = float(n_obs * np.log(sigma2) + 2 * int(n_params))
    bic = float(n_obs * np.log(sigma2) + int(n_params) * np.log(max(n_obs, 1)))
    denom = max(n_obs - int(n_params) - 1, 1)
    adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / denom if np.isfinite(r2) else float("nan")

    out.update({
        "rss": rss,
        "rmse": rmse,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "aic": aic,
        "bic": bic,
    })
    return out


def _prepare_ting_fit_arrays(FC, param_dict):
    """Rebuild the arrays used for the Ting fit, optionally skipping baseline alignment."""
    def _as_flat_float_array(values):
        if values is None:
            return np.array([], dtype=float)
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1).astype(float, copy=False) if np.isfinite(arr) else np.array([], dtype=float)
        return arr.reshape(-1).astype(float, copy=False)

    maxnoncontact = float(param_dict.get("contact_offset", 1e-6))
    pts_downsample = int(param_dict.get("pts_downsample_vis", param_dict.get("pts_downsample", 300)))

    ext_data = FC.extend_segments[0][1]
    ret_data = FC.retract_segments[-1][1]

    ext_indentation = _as_flat_float_array(ext_data.indentation)
    ret_indentation = _as_flat_float_array(ret_data.indentation)
    ext_force = _as_flat_float_array(ext_data.force)
    ret_force = _as_flat_float_array(ret_data.force)
    ext_time = _as_flat_float_array(ext_data.time)
    ret_time = _as_flat_float_array(ret_data.time)

    ext_n = min(ext_indentation.size, ext_force.size, ext_time.size)
    ret_n = min(ret_indentation.size, ret_force.size, ret_time.size)
    if ext_n == 0 or ret_n == 0:
        raise ValueError("Force/indentation arrays unavailable for Ting fit reconstruction.")

    ext_indentation = ext_indentation[:ext_n]
    ext_force = ext_force[:ext_n]
    ext_time = ext_time[:ext_n]
    ret_indentation = ret_indentation[:ret_n]
    ret_force = ret_force[:ret_n]
    ret_time = ret_time[:ret_n]

    if bool(param_dict.get("vdragcorr", False)):
        ext_force, ret_force = _safe_correct_viscous_drag(
            ext_indentation,
            ext_force,
            ret_indentation,
            ret_force,
            poly_order=int(param_dict.get("polyordr", 2)),
            speed=param_dict.get("rampspeed", 0.0),
        )

    baseline_margin = max(0.25 * maxnoncontact, 20e-9)
    ext_mask = np.isfinite(ext_indentation) & np.isfinite(ext_force) & (ext_indentation < -baseline_margin)
    ret_mask = np.isfinite(ret_indentation) & np.isfinite(ret_force) & (ret_indentation < -baseline_margin)

    if np.count_nonzero(ext_mask) < 5:
        ext_mask = np.isfinite(ext_indentation) & np.isfinite(ext_force) & (ext_indentation < 0)
    if np.count_nonzero(ret_mask) < 5:
        ret_mask = np.isfinite(ret_indentation) & np.isfinite(ret_force) & (ret_indentation < 0)

    if np.count_nonzero(ext_mask) and np.count_nonzero(ret_mask):
        trace_baseline = float(np.nanmedian(ext_force[ext_mask]))
        retrace_baseline = float(np.nanmedian(ret_force[ret_mask]))
    else:
        ext_head = ext_force[:max(5, ext_force.size // 10)] if ext_force.size else np.array([0.0], dtype=float)
        ret_tail = ret_force[-max(5, ret_force.size // 10):] if ret_force.size else np.array([0.0], dtype=float)
        trace_baseline = float(np.nanmedian(ext_head))
        retrace_baseline = float(np.nanmedian(ret_tail))

    force_noise = float(np.nanmax([
        np.nanstd(ext_force[ext_mask]) if np.count_nonzero(ext_mask) else 0.0,
        np.nanstd(ret_force[ret_mask]) if np.count_nonzero(ret_mask) else 0.0,
        1e-12,
    ]))

    retrace_baseline_shift = 0.0
    retrace_trim_info = {}
    retrace_trim_points = 0

    ext_zheight = np.atleast_1d(np.asarray(ext_data.zheight, dtype=float)).astype(float, copy=False)
    ret_zheight = np.atleast_1d(np.asarray(ret_data.zheight, dtype=float)).astype(float, copy=False)
    velocity = float(np.atleast_1d(np.asarray(ext_data.velocity, dtype=float))[0]) if np.size(ext_data.velocity) else 0.0
    t_offset = np.abs(ext_zheight[-1] - ret_zheight[0]) / (velocity * -1e-9) if velocity not in (0, 0.0) else 0.0
    dt = np.abs(ext_time[1] - ext_time[0]) if ext_time.size > 1 else 0.0
    if dt > 0 and t_offset > 2 * dt:
        ret_time = ret_time + t_offset

    idx_tc = int(np.argmin(np.abs(ext_indentation)))
    t0 = ext_time[-1]
    ind_all = np.r_[ext_indentation, ret_indentation]
    time_all = np.r_[ext_time, ret_time + t0]
    force_all = np.r_[ext_force, ret_force]

    # ── Taglio al PoC: la regione di non-contatto (δ < 0) viene rimossa. ──────
    fit_mask = ind_all >= 0.0
    tc = time_all[idx_tc] if time_all.size else 0.0
    ind_fit = ind_all[fit_mask]
    force_fit = force_all[fit_mask]
    time_fit = time_all[fit_mask]

    # Normalizzazione: la trace parte da F=0 al PoC.
    if force_fit.size > 0:
        force_fit = force_fit - force_fit[0]

    # Il ramp di chiusura è già stato applicato a vdeflection prima del fit:
    # non serve applicarlo di nuovo qui.
    _n_ext_in_fit = int(np.sum(fit_mask[:len(ext_indentation)]))
    if time_fit.size > 0:
        tc_fit = tc - time_fit[0]
        time_fit = time_fit - time_fit[0] - tc_fit
    else:
        tc_fit = 0.0

    downfactor = max(1, time_fit.size // max(1, pts_downsample))
    idxDown = list(range(0, time_fit.size, downfactor))

    return {
        "maxnoncontact": maxnoncontact,
        "pts_downsample": pts_downsample,
        "ext_data": ext_data,
        "ret_data": ret_data,
        "ext_indentation": ext_indentation,
        "ret_indentation": ret_indentation,
        "ext_force": ext_force,
        "ret_force": ret_force,
        "ext_time": ext_time,
        "ret_time": ret_time,
        "baseline_margin": baseline_margin,
        "trace_baseline": trace_baseline,
        "retrace_baseline": retrace_baseline,
        "retrace_baseline_shift": retrace_baseline_shift,
        "retrace_trim_points": retrace_trim_points,
        "retrace_trim_info": retrace_trim_info,
        "t_offset": t_offset,
        "dt": dt,
        "idx_tc": idx_tc,
        "t0": t0,
        "ind_all": ind_all,
        "time_all": time_all,
        "force_all": force_all,
        "fit_mask": fit_mask,
        "tc": tc,
        "ind_fit": ind_fit,
        "force_fit": force_fit,
        "time_fit": time_fit,
        "tc_fit": tc_fit,
        "downfactor": downfactor,
        "idxDown": idxDown,
        "fit_time_ds": time_fit[idxDown],
        "fit_ind_ds": ind_fit[idxDown],
        "fit_force_raw_ds": force_fit[idxDown],
        "force_noise": force_noise,
    }


def _autotune_contact_point_params(FC, param_dict):
    """Automatically choose a contact-point strategy without degrading the Ting fit."""
    base_params = dict(param_dict)
    candidate_specs = [
        ("current_default", {}),
        ("regulaFalsi-sigma2", {"poc_method": "regulaFalsi", "sigma": 2, "correct_tilt": True, "fit_line": True}),
        ("regulaFalsi-sigma4", {"poc_method": "regulaFalsi", "sigma": 4, "correct_tilt": True, "fit_line": True}),
        ("regulaFalsi-sigma6", {"poc_method": "regulaFalsi", "sigma": 6, "correct_tilt": True, "fit_line": True}),
    ]

    diagnostics = {
        "selected_strategy": "current_default",
        "selected_reason": "fallback_to_default",
        "default_r2": float("nan"),
        "selected_r2": float("nan"),
        "default_ting_r2": float("nan"),
        "selected_ting_r2": float("nan"),
        "default_delta0_nm": float("nan"),
        "selected_delta0_nm": float("nan"),
        "candidates": [],
    }

    valid_rows = []
    for label, updates in candidate_specs:
        trial_params = dict(base_params)
        trial_params.update(updates)
        try:
            ting_trial, hertz_trial = _run_ting_solver(copy.deepcopy(FC), trial_params)
            row = {
                "strategy": label,
                "ok": True,
                "hertz_r2": float(getattr(hertz_trial, "Rsquared", float("nan"))),
                "ting_r2": float(getattr(ting_trial, "Rsquared", float("nan"))),
                "delta0_nm": float(getattr(hertz_trial, "delta0", float("nan")) * 1e9),
                "E0_Pa": float(getattr(hertz_trial, "E0", float("nan"))),
                "updates": dict(updates),
            }
            valid_rows.append(row)
        except Exception as exc:
            row = {
                "strategy": label,
                "ok": False,
                "error": str(exc),
                "updates": dict(updates),
            }
        diagnostics["candidates"].append(row)

    if not valid_rows:
        return base_params, diagnostics

    default_row = next((row for row in valid_rows if row["strategy"] == "current_default"), valid_rows[0])
    best_ting_r2 = max(row["ting_r2"] for row in valid_rows if np.isfinite(row.get("ting_r2", float("nan"))))
    near_best = [
        row for row in valid_rows
        if np.isfinite(row.get("ting_r2", float("nan"))) and row["ting_r2"] >= (best_ting_r2 - 0.002)
    ]
    if not near_best:
        near_best = valid_rows

    selected_row = sorted(
        near_best,
        key=lambda row: (
            abs(row.get("delta0_nm", float("inf"))),
            1 if row.get("strategy") == "current_default" else 0,
            -row.get("ting_r2", float("-inf")),
            -row.get("hertz_r2", float("-inf")),
        ),
    )[0]

    tuned_params = dict(base_params)
    tuned_params.update(selected_row.get("updates", {}))

    diagnostics.update({
        "selected_strategy": selected_row.get("strategy", "current_default"),
        "selected_reason": "near-best Ting fit with smaller |delta0|",
        "default_r2": float(default_row.get("hertz_r2", float("nan"))),
        "selected_r2": float(selected_row.get("hertz_r2", float("nan"))),
        "default_ting_r2": float(default_row.get("ting_r2", float("nan"))),
        "selected_ting_r2": float(selected_row.get("ting_r2", float("nan"))),
        "default_delta0_nm": float(default_row.get("delta0_nm", float("nan"))),
        "selected_delta0_nm": float(selected_row.get("delta0_nm", float("nan"))),
    })
    return tuned_params, diagnostics


def _compute_closure_metrics(force_exp, force_pred, indentation, n_params=4):
    """Quantify how well the fitted retrace closes near the unloading tail."""
    force_exp = np.asarray(force_exp, dtype=float)
    force_pred = np.asarray(force_pred, dtype=float)
    indentation = np.asarray(indentation, dtype=float)
    valid = np.isfinite(force_exp) & np.isfinite(force_pred) & np.isfinite(indentation)

    if np.count_nonzero(valid) < 12:
        return {
            "r2": float("nan"),
            "rmse": float("nan"),
            "tail_rmse": float("nan"),
            "tail_bias": float("nan"),
            "end_bias": float("nan"),
            "tail_points": 0,
        }

    y = force_exp[valid]
    yhat = force_pred[valid]
    ind = indentation[valid]
    metrics = _compute_fit_metrics(y, yhat, n_params=n_params)

    idx_peak = int(np.argmax(y))
    unload_idx = np.where((np.arange(len(y)) >= idx_peak) & (ind >= 0.0))[0]
    if unload_idx.size < 6:
        unload_idx = np.where(np.arange(len(y)) >= idx_peak)[0]

    if unload_idx.size >= 8:
        tail_idx = unload_idx[int(0.75 * len(unload_idx)):]
    else:
        tail_idx = unload_idx

    end_count = min(5, len(y))
    end_idx = np.arange(len(y) - end_count, len(y)) if end_count else np.array([], dtype=int)
    tail_err = yhat[tail_idx] - y[tail_idx] if tail_idx.size else np.array([], dtype=float)
    end_err = yhat[end_idx] - y[end_idx] if end_idx.size else np.array([], dtype=float)

    return {
        "r2": float(metrics.get("r2", float("nan"))),
        "rmse": float(metrics.get("rmse", float("nan"))),
        "tail_rmse": float(np.sqrt(np.mean(tail_err ** 2))) if tail_err.size else float("nan"),
        "tail_bias": float(np.mean(tail_err)) if tail_err.size else float("nan"),
        "end_bias": float(np.mean(end_err)) if end_err.size else float("nan"),
        "tail_points": int(tail_idx.size),
    }


def _build_closure_weights(indentation, force):
    """Increase the influence of the unloading tail to encourage better closure."""
    indentation = np.asarray(indentation, dtype=float)
    force = np.asarray(force, dtype=float)
    weights = np.ones_like(force, dtype=float)

    valid = np.isfinite(indentation) & np.isfinite(force)
    if np.count_nonzero(valid) < 8:
        return weights

    y = np.where(valid, force, -np.inf)
    idx_peak = int(np.argmax(y))
    weights[idx_peak:] *= 1.25

    unload_idx = np.where((np.arange(len(force)) >= idx_peak) & np.isfinite(indentation) & (indentation >= 0.0))[0]
    if unload_idx.size < 6:
        unload_idx = np.where(np.arange(len(force)) >= idx_peak)[0]

    if unload_idx.size:
        pos = np.linspace(0.0, 1.0, unload_idx.size)
        weights[unload_idx] *= 1.0 + 0.9 * pos
        finite_unload = np.isfinite(indentation[unload_idx])
        if np.any(finite_unload):
            low_indent_cut = np.nanpercentile(indentation[unload_idx][finite_unload], 35)
            low_indent_idx = unload_idx[indentation[unload_idx] <= low_indent_cut]
            weights[low_indent_idx] *= 1.20

    return np.clip(weights, 1.0, 4.0)


def _refine_ting_fit_for_closure(ting_result, fit_context):
    """Apply a conservative weighted PLR refit when the unloading branch stays visibly open."""
    fit_time_ds = np.asarray(fit_context.get("fit_time_ds", []), dtype=float)
    fit_ind_ds = np.asarray(fit_context.get("fit_ind_ds", []), dtype=float)
    fit_force_raw_ds = np.asarray(fit_context.get("fit_force_raw_ds", []), dtype=float)

    out = {
        "attempted": False,
        "applied": False,
        "reason": "not_needed",
        "before": None,
        "after": None,
    }
    if fit_force_raw_ds.size < 25:
        out["reason"] = "not_enough_points"
        return out

    idx_tm_ds = int(np.argmax(fit_force_raw_ds))
    smooth_w = int(getattr(ting_result, "smooth_w", 5) or 5)
    t0_model = float(getattr(ting_result, "t0", 1))

    pred_before = ting_result.eval(
        fit_time_ds,
        fit_force_raw_ds,
        fit_ind_ds,
        t0=t0_model,
        idx_tm=idx_tm_ds,
        smooth_w=smooth_w,
        v0t=ting_result.v0t,
        v0r=ting_result.v0r,
    )
    before = _compute_closure_metrics(fit_force_raw_ds, pred_before, fit_ind_ds, n_params=4)
    out["before"] = before

    needs_refine = (
        (np.isfinite(before["tail_rmse"]) and before["tail_rmse"] > 30e-12)
        or (np.isfinite(before["tail_bias"]) and abs(before["tail_bias"]) > 25e-12)
        or (np.isfinite(before["end_bias"]) and abs(before["end_bias"]) > 20e-12)
    )
    if not needs_refine:
        out["reason"] = "already_sufficiently_closed"
        return out

    weights = np.ones_like(fit_force_raw_ds, dtype=float)
    contact_mask = fit_ind_ds >= 0.0
    weights[contact_mask] *= 1.08
    weights[idx_tm_ds:] *= 1.35

    unload_idx = np.where((np.arange(len(fit_force_raw_ds)) >= idx_tm_ds) & contact_mask)[0]
    if unload_idx.size:
        pos = np.linspace(0.0, 1.0, unload_idx.size)
        weights[unload_idx] *= 1.0 + 0.7 * pos
        low_indent_cut = np.nanpercentile(fit_ind_ds[unload_idx], 30)
        low_indent_idx = unload_idx[fit_ind_ds[unload_idx] <= low_indent_cut]
        weights[low_indent_idx] *= 1.25

    fixed_params = {
        "t0": t0_model,
        "F": fit_force_raw_ds,
        "delta": fit_ind_ds,
        "modelFt": ting_result.modelFt,
        "vdrag": ting_result.vdrag,
        "smooth_w": smooth_w,
        "idx_tm": idx_tm_ds,
        "v0t": ting_result.v0t,
        "v0r": ting_result.v0r,
    }
    weighted_model = Model(
        lambda time, E0, tc, betaE, F0: ting_result.model(time, E0, tc, betaE, F0, **fixed_params)
    )

    params = ting_result.build_params()
    current_E0 = max(float(getattr(ting_result, "E0", 1.0)), 1e-6)
    e0_min = max(float(getattr(ting_result, "E0_min", 0.0)), current_E0 / 5.0, 1e-6)
    e0_max_attr = float(getattr(ting_result, "E0_max", np.inf))
    e0_max = max(current_E0 * 5.0, current_E0 + 1.0)
    if np.isfinite(e0_max_attr):
        e0_max = min(e0_max, e0_max_attr)

    params["E0"].set(value=current_E0, min=e0_min, max=e0_max)
    params["tc"].set(
        value=float(getattr(ting_result, "tc", 0.0)),
        min=float(getattr(ting_result, "tc_min", -np.inf)),
        max=float(getattr(ting_result, "tc_max", np.inf)),
    )
    params["betaE"].set(
        value=float(getattr(ting_result, "betaE", 0.2)),
        min=float(getattr(ting_result, "betaE_min", 0.01)),
        max=float(getattr(ting_result, "betaE_max", 0.99)),
    )
    params["F0"].set(
        value=float(getattr(ting_result, "F0", 0.0)),
        min=float(getattr(ting_result, "F0_min", -np.inf)),
        max=float(getattr(ting_result, "F0_max", np.inf)),
    )

    out["attempted"] = True
    try:
        result = weighted_model.fit(
            fit_force_raw_ds,
            params,
            time=fit_time_ds,
            weights=weights,
            method="leastsq",
            max_nfev=8000,
        )
    except Exception as exc:
        out["reason"] = f"weighted_refit_failed: {exc}"
        return out

    if (not getattr(result, "success", False)) or (result.best_fit is None):
        out["reason"] = "weighted_refit_unsuccessful"
        return out

    pred_after = np.asarray(result.best_fit, dtype=float)
    after = _compute_closure_metrics(fit_force_raw_ds, pred_after, fit_ind_ds, n_params=4)
    out["after"] = after

    r2_before = float(before.get("r2", float("nan")))
    r2_after = float(after.get("r2", float("nan")))
    tail_before = float(before.get("tail_rmse", float("nan")))
    tail_after = float(after.get("tail_rmse", float("nan")))
    bias_before = abs(float(before.get("tail_bias", float("nan"))))
    bias_after = abs(float(after.get("tail_bias", float("nan"))))

    accept = False
    if np.isfinite(r2_after - r2_before) and (r2_after - r2_before) > 5e-4:
        accept = True
        out["reason"] = "overall_r2_improved"
    elif (
        np.isfinite(tail_before) and tail_before > 0 and np.isfinite(tail_after)
        and (tail_after <= 0.90 * tail_before)
        and np.isfinite(r2_before) and np.isfinite(r2_after)
        and (r2_after >= r2_before - 0.0025)
    ):
        accept = True
        out["reason"] = "tail_closure_improved"
    elif (
        np.isfinite(bias_before) and bias_before > 0 and np.isfinite(bias_after)
        and (bias_after <= 0.80 * bias_before)
        and np.isfinite(r2_before) and np.isfinite(r2_after)
        and (r2_after >= r2_before - 0.0015)
    ):
        accept = True
        out["reason"] = "tail_bias_reduced"
    else:
        out["reason"] = "weighted_refit_not_better_enough"

    if not accept:
        return out

    best = result.best_values
    ting_result.E0 = float(best["E0"])
    ting_result.tc = float(best["tc"])
    ting_result.betaE = float(best["betaE"])
    ting_result.F0 = float(best["F0"])
    ting_result.E0_init = ting_result.E0
    ting_result.tc_init = ting_result.tc
    ting_result.betaE_init = ting_result.betaE
    ting_result.F0_init = ting_result.F0

    model_predictions = ting_result.eval(
        fit_time_ds,
        fit_force_raw_ds,
        fit_ind_ds,
        t0=t0_model,
        idx_tm=idx_tm_ds,
        smooth_w=smooth_w,
        v0t=ting_result.v0t,
        v0r=ting_result.v0r,
    )
    abs_error = model_predictions - fit_force_raw_ds
    ting_result.MAE = float(np.mean(abs_error))
    ting_result.SE = np.square(abs_error)
    ting_result.MSE = float(np.mean(ting_result.SE))
    ting_result.RMSE = float(np.sqrt(ting_result.MSE))
    ting_result.Rsquared = float(1.0 - (np.var(abs_error) / np.var(fit_force_raw_ds)))
    ting_result.chisq = float(ting_result.get_chisq(
        fit_time_ds, fit_force_raw_ds, fit_ind_ds, t0_model, idx_tm_ds, smooth_w, ting_result.v0t, ting_result.v0r
    ))
    ting_result.redchi = float(ting_result.get_red_chisq(
        fit_time_ds, fit_force_raw_ds, fit_ind_ds, t0_model, idx_tm_ds, smooth_w, ting_result.v0t, ting_result.v0r
    ))

    out["applied"] = True
    return out


def _fit_reference_ting_model(model_name, indentation, time, force, probe_radius_nm, poisson):
    """Fit an auxiliary viscoelastic model on the same contact window used for GM2."""
    fallback = {
        "model": model_name,
        "ok": False,
        "n_params": 0,
        "stable": False,
        "warnings": [],
        "params": {},
        "pred": None,
        "diagnostics": None,
        "closure": None,
        "closure_refined": False,
        "n_obs": int(len(force)),
        "rss": float("nan"),
        "rmse": float("nan"),
        "r2": float("nan"),
        "adj_r2": float("nan"),
        "aic": float("nan"),
        "bic": float("nan"),
    }
    if len(time) < 8:
        return fallback

    indentation = np.asarray(indentation, dtype=float)
    time = np.asarray(time, dtype=float)
    force = np.asarray(force, dtype=float)

    fitter = TingFitter(
        model=model_name,
        geometry="sphere",
        probe_size=probe_radius_nm,
        poisson=poisson,
    )
    params, ok = fitter.fit(indentation, time, force)
    if (not ok) or (params is None):
        return fallback

    params = np.asarray(params, dtype=float)
    pred = np.asarray(fitter.get_fitted_force(indentation, time), dtype=float)
    diag = fitter.get_parameter_diagnostics()
    warnings = list(fitter.get_identifiability_warnings(corr_threshold=0.95, rel_ci_threshold=1.0))
    closure = _compute_closure_metrics(force, pred, indentation, n_params=len(params))
    closure_refined = False

    needs_refine = (
        len(force) >= 25 and (
            (np.isfinite(closure.get("tail_rmse", float("nan"))) and closure["tail_rmse"] > 30e-12)
            or (np.isfinite(closure.get("tail_bias", float("nan"))) and abs(closure["tail_bias"]) > 25e-12)
            or (np.isfinite(closure.get("end_bias", float("nan"))) and abs(closure["end_bias"]) > 20e-12)
        )
    )

    if needs_refine:
        weights = _build_closure_weights(indentation, force)
        weighted_params, weighted_ok = fitter.fit(
            indentation,
            time,
            force,
            weights=weights,
            initial_params=params,
        )
        if weighted_ok and (weighted_params is not None):
            weighted_params = np.asarray(weighted_params, dtype=float)
            pred_weighted = np.asarray(fitter.get_fitted_force(indentation, time), dtype=float)
            closure_weighted = _compute_closure_metrics(force, pred_weighted, indentation, n_params=len(weighted_params))

            r2_before = float(closure.get("r2", float("nan")))
            r2_after = float(closure_weighted.get("r2", float("nan")))
            tail_before = float(closure.get("tail_rmse", float("nan")))
            tail_after = float(closure_weighted.get("tail_rmse", float("nan")))
            bias_before = abs(float(closure.get("tail_bias", float("nan"))))
            bias_after = abs(float(closure_weighted.get("tail_bias", float("nan"))))
            end_before = abs(float(closure.get("end_bias", float("nan"))))
            end_after = abs(float(closure_weighted.get("end_bias", float("nan"))))

            accept = False
            if np.isfinite(r2_after - r2_before) and (r2_after - r2_before) > 5e-4:
                accept = True
            elif np.isfinite(tail_before) and tail_before > 0 and np.isfinite(tail_after) and (tail_after <= 0.92 * tail_before) and np.isfinite(r2_after) and np.isfinite(r2_before) and (r2_after >= r2_before - 0.002):
                accept = True
            elif np.isfinite(bias_before) and bias_before > 0 and np.isfinite(bias_after) and (bias_after <= 0.80 * bias_before) and np.isfinite(r2_after) and np.isfinite(r2_before) and (r2_after >= r2_before - 0.0015):
                accept = True
            elif np.isfinite(end_before) and end_before > 0 and np.isfinite(end_after) and (end_after <= 0.80 * end_before) and np.isfinite(r2_after) and np.isfinite(r2_before) and (r2_after >= r2_before - 0.0015):
                accept = True

            if accept:
                params = weighted_params
                pred = pred_weighted
                diag = fitter.get_parameter_diagnostics()
                warnings = list(fitter.get_identifiability_warnings(corr_threshold=0.95, rel_ci_threshold=1.0))
                closure = closure_weighted
                closure_refined = True

    metrics = _compute_fit_metrics(force, pred, len(params))

    fallback.update(metrics)
    fallback.update({
        "ok": True,
        "n_params": int(len(params)),
        "stable": len(warnings) == 0,
        "warnings": list(warnings),
        "params": {
            name: float(value)
            for name, value in zip(fitter.get_parameter_names(), np.asarray(params, dtype=float))
        },
        "pred": pred,
        "diagnostics": diag,
        "closure": closure,
        "closure_refined": bool(closure_refined),
    })
    return fallback


def _select_best_visco_model(candidates):
    """Compare candidate models with an Occam-aware BIC rule plus stability and closure checks."""
    compact_candidates = []
    for item in candidates:
        compact = dict(item)
        compact.pop("pred", None)
        compact.pop("diagnostics", None)
        compact_candidates.append(compact)

    valid = [
        item for item in compact_candidates
        if item.get("ok") and np.isfinite(item.get("bic", float("nan")))
    ]
    if not valid:
        return {
            "selected_model": "PLR",
            "reason": "Fallback su PLR: confronto automatico non disponibile.",
            "candidates": compact_candidates,
        }

    def _closure_penalty(item):
        closure = item.get("closure") or {}
        tail_rmse = abs(float(closure.get("tail_rmse", float("nan")))) * 1e12
        end_bias = abs(float(closure.get("end_bias", float("nan")))) * 1e12
        penalty = 0.0
        if np.isfinite(tail_rmse):
            penalty += max(0.0, (tail_rmse - 25.0) / 15.0)
        if np.isfinite(end_bias):
            penalty += max(0.0, (end_bias - 20.0) / 12.0)
        return float(0.75 * penalty)

    for item in valid:
        item["closure_penalty"] = _closure_penalty(item)
        item["bic_effective"] = float(item["bic"] + item["closure_penalty"])

    valid.sort(key=lambda item: (item["bic_effective"], item["bic"], -item.get("r2", float("-inf"))))
    best_bic = float(valid[0]["bic"])
    finite_aics = [item["aic"] for item in valid if np.isfinite(item.get("aic", float("nan")))]
    best_aic = float(min(finite_aics)) if finite_aics else float("nan")

    for item in valid:
        item["delta_bic"] = float(item["bic"] - best_bic)
        item["delta_aic"] = float(item["aic"] - best_aic) if np.isfinite(best_aic) else float("nan")

    near_tie = [item for item in valid if item.get("delta_bic", float("inf")) <= 4.0]
    if near_tie:
        recommended = sorted(
            near_tie,
            key=lambda item: (
                0 if item.get("stable", True) else 1,
                item.get("closure_penalty", float("inf")),
                item.get("n_params", 99),
                item.get("bic", float("inf")),
                -item.get("r2", float("-inf")),
            ),
        )[0]
    else:
        stable_valid = [item for item in valid if item.get("stable", True)]
        recommended = stable_valid[0] if stable_valid else valid[0]

    reason_parts = [
        f"BIC={recommended['bic']:.2f}",
        f"R2={recommended.get('r2', float('nan')):.4f}",
    ]
    if recommended.get("closure_penalty", 0.0) > 0.0:
        reason_parts.append("scelta penalizzando i modelli che restano più aperti sul retrace")
    elif recommended.get("delta_bic", float("inf")) <= 4.0:
        reason_parts.append("modello più parsimonioso tra quelli quasi equivalenti")
    elif recommended.get("stable", True):
        reason_parts.append("miglior compromesso tra qualità del fit e stabilità")
    else:
        reason_parts.append("fit descrittivo da interpretare con cautela")

    return {
        "selected_model": recommended["model"],
        "reason": " | ".join(reason_parts),
        "candidates": compact_candidates,
    }


def compare_ting_models(FC, param_dict, ting_result, verbose=True, fit_context=None, enable_gm2=False):
    """Rebuild the fit arrays, compare Ting-PLR vs Ting-GM2 and compute diagnostics."""
    if fit_context is None:
        fit_context = _prepare_ting_fit_arrays(FC, param_dict)

    # ── Dati ricostruiti dal fit ──
    maxnoncontact = fit_context["maxnoncontact"]
    pts_downsample = fit_context["pts_downsample"]
    ext_data = fit_context["ext_data"]
    ret_data = fit_context["ret_data"]
    ext_indentation = fit_context["ext_indentation"]
    ret_indentation = fit_context["ret_indentation"]
    ext_force = fit_context["ext_force"]
    ret_force = fit_context["ret_force"]
    ext_time = fit_context["ext_time"]
    ret_time = fit_context["ret_time"]
    baseline_margin = fit_context["baseline_margin"]
    trace_baseline = fit_context["trace_baseline"]
    retrace_baseline = fit_context["retrace_baseline"]
    retrace_baseline_shift = fit_context["retrace_baseline_shift"]
    retrace_trim_points = fit_context["retrace_trim_points"]
    retrace_trim_info = fit_context["retrace_trim_info"]
    t_offset = fit_context["t_offset"]
    dt = fit_context["dt"]
    idx_tc = fit_context["idx_tc"]
    t0 = fit_context["t0"]
    ind_all = fit_context["ind_all"]
    time_all = fit_context["time_all"]
    force_all = fit_context["force_all"]
    fit_mask = fit_context["fit_mask"]
    tc = fit_context["tc"]
    ind_fit = fit_context["ind_fit"]
    force_fit = fit_context["force_fit"]
    time_fit = fit_context["time_fit"]
    tc_fit = fit_context["tc_fit"]
    downfactor = fit_context["downfactor"]
    idxDown = fit_context["idxDown"]
    fit_time_ds = fit_context["fit_time_ds"]
    fit_ind_ds = fit_context["fit_ind_ds"]
    fit_force_raw_ds = fit_context["fit_force_raw_ds"]

    # ── Valutazione del modello Ting-PLR ──
    idx_tm_ds = int(np.argmax(fit_force_raw_ds))
    smooth_w = int(getattr(ting_result, "smooth_w", 5) or 5)
    t0_model = float(getattr(ting_result, "t0", 1))
    fit_force_model_ds = ting_result.eval(
        fit_time_ds,
        fit_force_raw_ds,
        fit_ind_ds,
        t0=t0_model,
        idx_tm=idx_tm_ds,
        smooth_w=smooth_w,
        v0t=ting_result.v0t,
        v0r=ting_result.v0r,
    )

    ss_res = np.sum((fit_force_raw_ds - fit_force_model_ds) ** 2)
    ss_tot = np.sum((fit_force_raw_ds - fit_force_raw_ds.mean()) ** 2)
    r2_vis = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    probe_radius_nm = float(param_dict.get("tip_param", 5e-6)) * 1e9

    # ── Setup del confronto GM2 ──
    shared_mask = np.isfinite(fit_ind_ds) & np.isfinite(fit_force_raw_ds)
    fit_time_shared = fit_time_ds[shared_mask]
    fit_ind_shared = fit_ind_ds[shared_mask]
    fit_force_shared = fit_force_raw_ds[shared_mask]
    # fit_time_shared è già azzerato al PoC da _prepare_ting_fit_arrays (t=0 al PoC).
    # Non ri-azzeriamo: usiamo direttamente fit_time_shared come tempo relativo al PoC.
    fit_time_shared_rel = fit_time_shared
    contact_mask_shared = fit_ind_shared >= 0.0

    gm2_contact_thresholds_nm = [0, 40, 80, 120, 150, 180, 200]

    idx_contact_all = np.where(contact_mask_shared)[0]
    fit_force_gm2_ds = np.full_like(fit_force_raw_ds, np.nan)
    fit_time_gm2 = np.array([], dtype=float)
    fit_ind_gm2 = np.array([], dtype=float)
    fit_force_gm2 = np.array([], dtype=float)
    fit_time_gm2_rel = np.array([], dtype=float)
    gm2_contact_threshold_nm = float("nan")
    gm2_fit_points = 0
    gm2_F0 = float("nan")

    contact_duration = (
        float(fit_time_ds[idx_contact_all[-1]] - fit_time_ds[idx_contact_all[0]])
        if len(idx_contact_all) > 1 else 0.05
    )
    dt_med = np.median(np.diff(fit_time_ds[idx_contact_all])) if len(idx_contact_all) > 2 else 1e-3
    tau_ratio_min = 6.0
    tau_floor = max(5.0 * dt_med, 1e-4)
    tau_ceiling = max(2.0 * contact_duration, 0.02)
    tau1_ceiling = tau_ceiling / tau_ratio_min  # tau1 deve essere ben separato da tau2
    gm2_base_E = max(float(getattr(ting_result, "E0", 250.0)), 50.0)

    gm2_reg_info = {
        "tau_ratio_min": float(tau_ratio_min),
        "tau_floor_s": float(tau_floor),
        "tau_ceiling_s": float(tau_ceiling),
        "contact_threshold_nm": None,
        "retained_points": 0,
        "fit_strategy": "soft_l1_weighted_full-contact-forced-when-valid",
        "n_trials": 0,
        "n_valid": 0,
        "n_rejected": 0,
        "used_soft_fallback": False,
        "used_full_contact": False,
    }

    # ── Ricerca su griglia per GM2 ──
    # Griglia su coppie (tau1, tau2) logaritmicamente separate (tau2 >= tau_ratio_min * tau1)
    _t1_candidates = [max(tau_floor, 0.003), max(tau_floor, 0.005), max(tau_floor, 0.008), max(tau_floor, 0.012)]
    _t2_candidates = [min(tau_ceiling, 0.050), min(tau_ceiling, 0.100), min(tau_ceiling, 0.150), min(tau_ceiling, 0.200)]
    gm2_initial_grid = []
    for _t1 in _t1_candidates:
        for _t2 in _t2_candidates:
            if _t2 < tau_ratio_min * _t1:
                continue
            for _frac in [0.6, 1.0]:
                gm2_initial_grid.append(np.array([
                    gm2_base_E,
                    _frac * gm2_base_E,
                    _t1,
                    (2.0 - _frac) * gm2_base_E,
                    _t2,
                    0.0,
                ], dtype=float))

    scan_rows = []
    best_candidate = None
    best_full_contact_candidate = None
    best_relaxed_candidate = None
    gm2_ok = False
    gm2_params = None
    r2_gm2 = float("nan")
    Einf_gm2 = float("nan")
    A1_gm2 = float("nan")
    tau1_gm2 = float("nan")
    A2_gm2 = float("nan")
    tau2_gm2 = float("nan")

    if enable_gm2 and len(idx_contact_all) > 8:
        min_keep_points = max(40, int(0.35 * len(idx_contact_all)))
        max_thr = max(gm2_contact_thresholds_nm)

        for thr_nm in gm2_contact_thresholds_nm:
            idx_contact = np.where(fit_ind_shared >= (thr_nm * 1e-9))[0]
            if len(idx_contact) < min_keep_points:
                continue

            i0 = int(idx_contact[0])
            i1 = int(idx_contact[-1])
            fit_slice = slice(i0, i1 + 1)

            fit_time_local = fit_time_shared[fit_slice]
            fit_ind_local = fit_ind_shared[fit_slice]
            fit_force_local = fit_force_shared[fit_slice]
            # Il tempo deve essere relativo al PoC (t=0 al PoC), NON all'inizio
            # della finestra tagliata. fit_time_shared è già azzerato al PoC da
            # _prepare_ting_fit_arrays, quindi lo usiamo direttamente.
            fit_time_rel_local = fit_time_local
            force_scale = max(float(np.nanstd(fit_force_local)), 1e-12)

            weights = np.ones_like(fit_force_local, dtype=float)
            if np.max(fit_ind_local) > 0:
                weights += 0.8 * (fit_ind_local / np.max(fit_ind_local))
            idx_peak = int(np.argmax(fit_force_local)) if len(fit_force_local) else 0
            weights[idx_peak:] *= 1.25

            def gm2_residuals(params):
                Einf, A1, tau1, A2, tau2, F0 = [float(v) for v in params]
                if tau1 > tau2:
                    tau1, tau2 = tau2, tau1
                    A1, A2 = A2, A1

                pred = ting_force(
                    fit_ind_local,
                    fit_time_rel_local,
                    "GM2",
                    [Einf, A1, tau1, A2, tau2],
                    geometry="sphere",
                    probe_size=int(round(probe_radius_nm)),
                    poisson=float(param_dict.get("poisson", 0.5)),
                ) + F0

                res = np.sqrt(weights) * (pred - fit_force_local) / force_scale
                pen_tau = 0.05 * max(0.0, tau_ratio_min * tau1 - tau2) / max(tau_floor, 1e-6)
                pen_amp = 0.02 * max(0.0, 10.0 - min(A1, A2)) / 10.0
                return np.r_[res, pen_tau, pen_amp]

            bounds = (
                [0.0, 0.0, max(tau_floor, 1e-4), 0.0, max(tau_ratio_min * tau_floor, 2e-4), -150e-12],
                [1e5, 1e5, tau1_ceiling,         1e5, tau_ceiling,                            150e-12],
            )

            for guess in gm2_initial_grid:
                gm2_reg_info["n_trials"] += 1
                try:
                    result = least_squares(
                        gm2_residuals,
                        guess,
                        bounds=bounds,
                        loss="soft_l1",
                        f_scale=1.0,
                        x_scale="jac",
                        max_nfev=12000,
                        ftol=1e-12,
                        xtol=1e-12,
                        gtol=1e-12,
                    )
                except Exception:
                    gm2_reg_info["n_rejected"] += 1
                    continue

                if (not result.success) or (result.x is None):
                    gm2_reg_info["n_rejected"] += 1
                    continue

                p = np.asarray(result.x, dtype=float).copy()
                Einf, A1, tau1, A2, tau2, F0 = [float(v) for v in p]

                if tau1 > tau2:
                    tau1, tau2 = tau2, tau1
                    A1, A2 = A2, A1

                pred = ting_force(
                    fit_ind_local,
                    fit_time_rel_local,
                    "GM2",
                    [Einf, A1, tau1, A2, tau2],
                    geometry="sphere",
                    probe_size=int(round(probe_radius_nm)),
                    poisson=float(param_dict.get("poisson", 0.5)),
                ) + F0

                ss_tot_gm2 = np.sum((fit_force_local - fit_force_local.mean()) ** 2)
                ss_res_gm2 = np.sum((fit_force_local - pred) ** 2)
                r2_local = 1.0 - ss_res_gm2 / ss_tot_gm2 if ss_tot_gm2 > 0 else float("nan")
                branch_balance = min(A1, A2) / max(A1 + A2, 1e-12)
                score = float(r2_local) - 0.0010 * (thr_nm / max_thr) - 0.0005 * max(0.0, 0.05 - branch_balance)

                candidate = {
                    "params": np.array([Einf, A1, tau1, A2, tau2], dtype=float),
                    "F0": float(F0),
                    "pred": pred.copy(),
                    "r2": float(r2_local),
                    "score": float(score),
                    "fit_slice": fit_slice,
                    "threshold_nm": float(thr_nm),
                    "retained_points": int(len(fit_time_rel_local)),
                    "branch_balance": float(branch_balance),
                }

                if (best_relaxed_candidate is None) or (candidate["score"] > best_relaxed_candidate["score"]):
                    best_relaxed_candidate = candidate

                valid = True
                if tau1 < tau_floor:
                    valid = False
                if tau2 > tau_ceiling:
                    valid = False
                if tau2 < tau_ratio_min * tau1:
                    valid = False

                scan_rows.append({
                    "threshold_nm": float(thr_nm),
                    "retained_points": int(len(fit_time_rel_local)),
                    "score": float(score),
                    "r2": float(r2_local),
                    "ok": bool(valid),
                    "branch_balance": float(branch_balance),
                    "params_full": [float(Einf), float(A1), float(tau1), float(A2), float(tau2), float(F0)],
                })

                if not valid:
                    gm2_reg_info["n_rejected"] += 1
                    continue

                gm2_reg_info["n_valid"] += 1
                if float(thr_nm) <= 0.0:
                    if (best_full_contact_candidate is None) or (candidate["score"] > best_full_contact_candidate["score"]):
                        best_full_contact_candidate = candidate
                if (best_candidate is None) or (candidate["score"] > best_candidate["score"]):
                    best_candidate = candidate

    if best_full_contact_candidate is not None:
        best_candidate = best_full_contact_candidate
        gm2_reg_info["used_full_contact"] = True
    elif (best_candidate is None) and (best_relaxed_candidate is not None):
        best_candidate = best_relaxed_candidate
        gm2_reg_info["used_soft_fallback"] = True

    if best_candidate is not None:
        gm2_ok = True
        gm2_params = best_candidate["params"]
        fit_force_gm2_part = best_candidate["pred"]
        gm2_fit_slice = best_candidate["fit_slice"]
        r2_gm2 = float(best_candidate["r2"])
        gm2_contact_threshold_nm = float(best_candidate["threshold_nm"])
        gm2_fit_points = int(best_candidate["retained_points"])
        gm2_F0 = float(best_candidate["F0"])
        fit_time_gm2 = fit_time_shared[contact_mask_shared]
        fit_ind_gm2 = fit_ind_shared[contact_mask_shared]
        fit_force_gm2 = fit_force_shared[contact_mask_shared]
        fit_time_gm2_rel = fit_time_gm2 - fit_time_gm2[0] if len(fit_time_gm2) else fit_time_gm2
        Einf_gm2, A1_gm2, tau1_gm2, A2_gm2, tau2_gm2 = [float(v) for v in gm2_params]

        fit_force_gm2_full = ting_force(
            fit_ind_shared,
            fit_time_shared_rel,
            "GM2",
            [Einf_gm2, A1_gm2, tau1_gm2, A2_gm2, tau2_gm2],
            geometry="sphere",
            probe_size=int(round(probe_radius_nm)),
            poisson=float(param_dict.get("poisson", 0.5)),
        ) + gm2_F0
        fit_force_gm2_ds[shared_mask] = fit_force_gm2_full
        gm2_reg_info["contact_threshold_nm"] = gm2_contact_threshold_nm
        gm2_reg_info["retained_points"] = gm2_fit_points
        gm2_reg_info["branch_balance"] = float(best_candidate.get("branch_balance", float("nan")))

    fit_force_plr_shared = fit_force_model_ds[shared_mask]
    fit_force_gm2_shared = fit_force_gm2_ds[shared_mask]

    # ── Costruzione dei candidati e selezione finale ──
    contact_mask = contact_mask_shared
    fit_time_contact = fit_time_shared[contact_mask]
    fit_time_contact_rel = fit_time_contact - fit_time_contact[0] if len(fit_time_contact) else fit_time_contact
    fit_ind_contact = fit_ind_shared[contact_mask]
    fit_force_contact = fit_force_shared[contact_mask]
    fit_force_plr_contact = fit_force_plr_shared[contact_mask]
    fit_force_gm2_contact = fit_force_gm2_shared[contact_mask]

    poisson = float(param_dict.get("poisson", 0.5))
    plr_metrics = _compute_fit_metrics(fit_force_shared, fit_force_plr_shared, n_params=4)
    plr_candidate = {
        "model": "PLR",
        "ok": True,
        "n_params": 4,
        "stable": True,
        "warnings": [],
        "params": {
            "E0": float(ting_result.E0),
            "betaE": float(ting_result.betaE),
            "tc": float(ting_result.tc),
            "F0": float(getattr(ting_result, "F0", 0.0)),
        },
        "closure": _compute_closure_metrics(fit_force_contact, fit_force_plr_contact, fit_ind_contact, n_params=4),
        "closure_refined": bool((fit_context.get("closure_refinement") or {}).get("applied", False)),
        **plr_metrics,
    }

    sls_candidate = _fit_reference_ting_model(
        "SLS",
        fit_ind_shared,
        fit_time_shared_rel,
        fit_force_shared,
        probe_radius_nm,
        poisson,
    )
    maxwell_candidate = _fit_reference_ting_model(
        "Maxwell",
        fit_ind_shared,
        fit_time_shared_rel,
        fit_force_shared,
        probe_radius_nm,
        poisson,
    )

    if enable_gm2:
        gm2_metrics = _compute_fit_metrics(fit_force_shared, fit_force_gm2_shared, n_params=6)
        gm2_candidate = {
            "model": "GM2",
            "ok": bool(gm2_ok),
            "n_params": 6,
            "stable": True,
            "warnings": [],
            "params": {
                "Einf": None if not np.isfinite(Einf_gm2) else float(Einf_gm2),
                "A1": None if not np.isfinite(A1_gm2) else float(A1_gm2),
                "tau1": None if not np.isfinite(tau1_gm2) else float(tau1_gm2),
                "A2": None if not np.isfinite(A2_gm2) else float(A2_gm2),
                "tau2": None if not np.isfinite(tau2_gm2) else float(tau2_gm2),
                "F0": None if not np.isfinite(gm2_F0) else float(gm2_F0),
            },
            "closure": _compute_closure_metrics(fit_force_contact, fit_force_gm2_contact, fit_ind_contact, n_params=6),
            "closure_refined": False,
            **gm2_metrics,
        }
        model_candidates = [plr_candidate, sls_candidate, maxwell_candidate, gm2_candidate]
    else:
        gm2_candidate = {"model": "GM2", "ok": False, "n_params": 6, "stable": False, "warnings": ["GM2 disabled"], "params": {}, "r2": float("nan"), "rmse": float("nan"), "aic": float("nan"), "bic": float("nan"), "closure": {}, "closure_refined": False}
        model_candidates = [plr_candidate, sls_candidate, maxwell_candidate]
    model_selection = _select_best_visco_model(model_candidates)
    selected_model = model_selection["selected_model"]
    selection_reason = model_selection["reason"]
    model_candidates = model_selection["candidates"]

    # ── Report compatto a video ──
    if verbose:
        print(f"Punti fit: {len(idxDown)} | R2 Ting-PLR: {r2_vis:.4f}")
        if enable_gm2:
            print(f"R2 Ting-GM2: {r2_gm2:.4f} | tau1={tau1_gm2 * 1000:.2f} ms | tau2={tau2_gm2 * 1000:.2f} ms")
        print(f"Traslazione verticale applicata al retrace: {-retrace_baseline_shift * 1e12:+.1f} pN")
        print(f"Modello consigliato: {selected_model} | {selection_reason}")

    return {
        "maxnoncontact": maxnoncontact,
        "pts_downsample": pts_downsample,
        "ext_data": ext_data,
        "ret_data": ret_data,
        "ext_indentation": ext_indentation,
        "ret_indentation": ret_indentation,
        "ext_force": ext_force,
        "ret_force": ret_force,
        "ext_time": ext_time,
        "ret_time": ret_time,
        "baseline_margin": baseline_margin,
        "trace_baseline": trace_baseline,
        "retrace_baseline": retrace_baseline,
        "retrace_baseline_shift": retrace_baseline_shift,
        "retrace_trim_points": retrace_trim_points,
        "retrace_trim_info": retrace_trim_info,
        "t_offset": t_offset,
        "dt": dt,
        "idx_tc": idx_tc,
        "t0": t0,
        "ind_all": ind_all,
        "time_all": time_all,
        "force_all": force_all,
        "fit_mask": fit_mask,
        "tc": tc,
        "ind_fit": ind_fit,
        "force_fit": force_fit,
        "time_fit": time_fit,
        "tc_fit": tc_fit,
        "downfactor": downfactor,
        "idxDown": idxDown,
        "fit_time_ds": fit_time_ds,
        "fit_ind_ds": fit_ind_ds,
        "fit_force_raw_ds": fit_force_raw_ds,
        "fit_force_model_ds": fit_force_model_ds,
        "r2_vis": float(r2_vis),
        "probe_radius_nm": probe_radius_nm,
        "gm2_contact_thresholds_nm": gm2_contact_thresholds_nm,
        "fit_force_gm2_ds": fit_force_gm2_ds,
        "fit_time_gm2": fit_time_gm2,
        "fit_ind_gm2": fit_ind_gm2,
        "fit_force_gm2": fit_force_gm2,
        "fit_time_gm2_rel": fit_time_gm2_rel,
        "gm2_contact_threshold_nm": gm2_contact_threshold_nm,
        "gm2_fit_points": gm2_fit_points,
        "gm2_F0": gm2_F0,
        "contact_duration": contact_duration,
        "tau_ratio_min": tau_ratio_min,
        "tau_floor": tau_floor,
        "tau_ceiling": tau_ceiling,
        "gm2_base_E": gm2_base_E,
        "gm2_reg_info": gm2_reg_info,
        "gm2_initial_grid": gm2_initial_grid,
        "best_candidate": best_candidate,
        "best_relaxed_candidate": best_relaxed_candidate,
        "gm2_ok": gm2_ok,
        "gm2_params": gm2_params,
        "r2_gm2": r2_gm2,
        "Einf_gm2": Einf_gm2,
        "A1_gm2": A1_gm2,
        "tau1_gm2": tau1_gm2,
        "A2_gm2": A2_gm2,
        "tau2_gm2": tau2_gm2,
        "scan_rows": scan_rows,
        "model_candidates": model_candidates,
        "model_selection": model_selection,
        "selected_model": selected_model,
        "selection_reason": selection_reason,
    }


def _run_single_ting_analysis(data_dir, print_reports=True, raw_deflection_smooth_win=0, param_overrides=None):
    """Full Ting workflow for one curve: preprocess, fit, compare PLR/GM2."""
    fit_results = run_ting_fit(filepath=data_dir, print_reports=print_reports, raw_deflection_smooth_win=raw_deflection_smooth_win, param_overrides=param_overrides)
    compare_results = compare_ting_models(
        fit_results["FC"],
        fit_results["param_dict"],
        fit_results["ting_result"],
        verbose=print_reports,
        fit_context=fit_results.get("fit_context"),
        enable_gm2=False,
    )

    tr_primary = fit_results.get("ting_result")
    r2_primary = float(compare_results.get("r2_vis", float("nan")))
    tc_on_boundary = False
    if tr_primary is not None:
        tc_on_boundary = (
            np.isfinite(getattr(tr_primary, "tc", np.nan))
            and (
                np.isclose(tr_primary.tc, tr_primary.tc_max, rtol=0.0, atol=5e-6)
                or np.isclose(tr_primary.tc, tr_primary.tc_min, rtol=0.0, atol=5e-6)
            )
        )

    compare_results["fit_strategy"] = "legacy_upstream"
    compare_results["fit_strategy_note"] = "Legacy solution kept."

    out = dict(fit_results)
    out.update(compare_results)

    out["force_drag_info"] = out.get("baseline_info", {}).get("force_drag_info", {})
    out["force_drag_applied"] = bool(out.get("force_drag_info", {}).get("applied", False))
    out["is_batch"] = False
    out["source_path"] = os.path.abspath(data_dir)
    out["analysis_results"] = out
    return out


def run_ting_batch_analysis(data_dir, print_reports=False, raw_deflection_smooth_win=0, param_overrides=None):
    """Analyze all supported force curves found inside a selected directory."""
    input_dir = os.path.abspath(data_dir)
    fd_folders = collect_fd_folders(input_dir)
    curve_files = collect_curve_files(input_dir, recursive=True, prefer_fd_subfolders=True)
    if not curve_files:
        raise FileNotFoundError(
            f"Nessuna curva supportata trovata in: {input_dir}"
        )

    if fd_folders:
        print(
            f"Analisi batch: trovate {len(curve_files)} curve in {len(fd_folders)} cartelle FD sotto {input_dir}"
        )
    else:
        print(f"Analisi batch: trovate {len(curve_files)} curve in {input_dir}")

    batch_curve_results = []
    summary_rows = []
    failed_files = []

    for idx, curve_path in enumerate(curve_files, start=1):
        curve_name = os.path.splitext(os.path.basename(curve_path))[0]
        fd_folder = os.path.dirname(curve_path)
        parent_folder = os.path.basename(fd_folder)
        cell_folder = os.path.basename(os.path.dirname(fd_folder)) if parent_folder.lower() == "fd" else parent_folder
        print(f"[{idx}/{len(curve_files)}] Analisi di {curve_name} ({cell_folder})")
        try:
            single = _run_single_ting_analysis(
                curve_path,
                print_reports=print_reports,
                raw_deflection_smooth_win=raw_deflection_smooth_win,
                param_overrides=param_overrides,
            )
            _plr_cand = next((c for c in single.get("model_candidates", []) if c["model"] == "PLR"), None)
            _gm2_cand = next((c for c in single.get("model_candidates", []) if c["model"] == "GM2"), None)
            _hertz_res = single.get("hertz_result")
            _young_hertz_pa = float(getattr(_hertz_res, "E0", float("nan"))) if _hertz_res is not None else None
            summary_rows.append({
                "curve": curve_name,
                "cell_folder": cell_folder,
                "fd_folder": fd_folder,
                "path": curve_path,
                "status": "ok",
                "selected_model": single.get("selected_model"),
                "selection_reason": single.get("selection_reason"),
                # PLR parameters
                "E0_Pa": float(_plr_cand["params"]["E0"]) if _plr_cand else None,
                "betaE": float(_plr_cand["params"]["betaE"]) if _plr_cand else None,
                "betaE_at_bound": (
                    abs(float(_plr_cand["params"]["betaE"]) - 0.01) < 0.005
                ) if _plr_cand else None,
                "tc_ms": float(_plr_cand["params"]["tc"]) * 1000 if _plr_cand else None,
                "rmse_plr_pN": float(_plr_cand["rmse"]) * 1e12 if _plr_cand else None,
                "r2_plr": float(single.get("r2_vis", float("nan"))),
                # GM2 parameters
                "Einf_Pa": float(_gm2_cand["params"]["Einf"]) if _gm2_cand else None,
                "A1_Pa": float(_gm2_cand["params"]["A1"]) if _gm2_cand else None,
                "tau1_ms": float(_gm2_cand["params"]["tau1"]) * 1000 if _gm2_cand else None,
                "A2_Pa": float(_gm2_cand["params"]["A2"]) if _gm2_cand else None,
                "tau2_ms": float(_gm2_cand["params"]["tau2"]) * 1000 if _gm2_cand else None,
                "rmse_gm2_pN": float(_gm2_cand["rmse"]) * 1e12 if _gm2_cand else None,
                "r2_gm2": None if not np.isfinite(single.get("r2_gm2", float("nan"))) else float(single["r2_gm2"]),
                "hertz_E0_init_pa": _young_hertz_pa,
                "hertz_E0_init_kpa": None if _young_hertz_pa is None or not np.isfinite(_young_hertz_pa) else (_young_hertz_pa / 1e3),
                "young_hertz_pa": _young_hertz_pa,
                "young_hertz_kpa": None if _young_hertz_pa is None or not np.isfinite(_young_hertz_pa) else (_young_hertz_pa / 1e3),
                "retrace_trim_points": int(single.get("retrace_trim_points", 0)),
                "closure_refined": bool(single.get("closure_refinement", {}).get("applied", False)),
                "contact_strategy": single.get("contact_point_info", {}).get("selected_strategy"),
                "force_drag_applied": bool(single.get("force_drag_info", {}).get("applied", False)),
            })
            batch_curve_results.append({
                "path": curve_path,
                "curve": curve_name,
                "cell_folder": cell_folder,
                "fd_folder": fd_folder,
                "analysis": single,
            })
        except Exception as exc:
            error_msg = str(exc)
            failed_files.append({"path": curve_path, "error": error_msg})
            summary_rows.append({
                "curve": curve_name,
                "cell_folder": cell_folder,
                "fd_folder": fd_folder,
                "path": curve_path,
                "status": f"failed: {error_msg}",
                "selected_model": None,
                "selection_reason": None,
                "E0_Pa": None,
                "betaE": None,
                "betaE_at_bound": None,
                "tc_ms": None,
                "rmse_plr_pN": None,
                "r2_plr": None,
                "Einf_Pa": None,
                "A1_Pa": None,
                "tau1_ms": None,
                "A2_Pa": None,
                "tau2_ms": None,
                "rmse_gm2_pN": None,
                "r2_gm2": None,
                "hertz_E0_init_pa": None,
                "hertz_E0_init_kpa": None,
                "young_hertz_pa": None,
                "young_hertz_kpa": None,
                "retrace_trim_points": None,
                "closure_refined": None,
                "contact_strategy": None,
                "force_drag_applied": None,
            })
            print(f"   -> errore: {error_msg}")

    # Riepilogo Young (Hertz) per cellula sulle curve valide
    if summary_rows:
        by_cell_young = {}
        for row in summary_rows:
            if row.get("status") != "ok":
                continue
            y_pa = row.get("young_hertz_pa")
            if y_pa is None or not np.isfinite(y_pa):
                continue
            by_cell_young.setdefault(row.get("cell_folder", "n/d"), []).append(float(y_pa))

        if by_cell_young:
            print("\nYoung (Hertz) per cellula:")
            for cell_name in sorted(by_cell_young.keys()):
                vals = np.array(by_cell_young[cell_name], dtype=float)
                print(
                    f"  {cell_name}: E_mean={np.mean(vals):.3e} Pa ({np.mean(vals)/1e3:.3f} kPa)"
                    f" | E_median={np.median(vals):.3e} Pa | n={len(vals)}"
                )

    out = {
        "is_batch": True,
        "input_path": input_dir,
        "fd_folders": fd_folders,
        "n_fd_folders": int(len(fd_folders)),
        "curve_files": curve_files,
        "n_found": int(len(curve_files)),
        "n_success": int(len(batch_curve_results)),
        "n_failed": int(len(failed_files)),
        "summary_rows": summary_rows,
        "batch_curve_results": batch_curve_results,
        "failed_files": failed_files,
    }
    out["analysis_results"] = out
    return out


def run_ting_analysis(data_dir, print_reports=True, raw_deflection_smooth_win=0, param_overrides=None):
    """Analyze either one curve file or all curves contained in a selected folder."""
    data_dir = os.path.abspath(os.path.expanduser(str(data_dir)))
    if os.path.isdir(data_dir):
        return run_ting_batch_analysis(
            data_dir,
            print_reports=False,
            raw_deflection_smooth_win=raw_deflection_smooth_win,
            param_overrides=param_overrides,
        )
    return _run_single_ting_analysis(
        data_dir,
        print_reports=print_reports,
        raw_deflection_smooth_win=raw_deflection_smooth_win,
        param_overrides=param_overrides,
    )


def _save_single_ting_outputs(
    data_dir,
    analysis_results,
    open_output_folder=False,
    show_plots=True,
    save_json=True,
    save_csv=True,
):
    """Save selected outputs (JSON/CSV/PNG) for one Ting analysis."""
    save_dir = os.path.dirname(os.path.abspath(data_dir))
    curve_name = os.path.splitext(os.path.basename(data_dir))[0]
    print(f"Cartella di salvataggio: {save_dir}")

    ting_result = analysis_results["ting_result"]
    hertz_result = analysis_results.get("hertz_result")

    # ── Metadati e file JSON/CSV ──
    results_json = {
        "curve": curve_name,
        "model": "Ting PLR + Ting Generalized Maxwell N=2",
        "baseline_correction": analysis_results.get("baseline_info"),
        "baseline_comparison": analysis_results.get("baseline_comparison"),
        "contact_point_info": analysis_results.get("contact_point_info"),
        "retrace_trim_points": analysis_results.get("retrace_trim_points"),
        "retrace_trim_info": analysis_results.get("retrace_trim_info"),
        "selected_model": analysis_results.get("selected_model"),
        "selection_reason": analysis_results.get("selection_reason"),
        "model_selection": analysis_results.get("model_selection"),
        "closure_refinement": analysis_results.get("closure_refinement"),
        "PLR": {
            "E0_Pa": float(ting_result.E0),
            "betaE": float(ting_result.betaE),
            "tc_s": float(ting_result.tc),
            "F0_N": float(ting_result.F0),
            "v0t_ms": float(ting_result.v0t),
            "v0r_ms": float(ting_result.v0r),
            "R2": float(analysis_results["r2_vis"]),
            "RMSE": float(ting_result.RMSE),
        },
        "GM2": {
            "ok": bool(analysis_results["gm2_ok"]),
            "Einf_Pa": None if not np.isfinite(analysis_results["Einf_gm2"]) else float(analysis_results["Einf_gm2"]),
            "A1_Pa": None if not np.isfinite(analysis_results["A1_gm2"]) else float(analysis_results["A1_gm2"]),
            "tau1_s": None if not np.isfinite(analysis_results["tau1_gm2"]) else float(analysis_results["tau1_gm2"]),
            "A2_Pa": None if not np.isfinite(analysis_results["A2_gm2"]) else float(analysis_results["A2_gm2"]),
            "tau2_s": None if not np.isfinite(analysis_results["tau2_gm2"]) else float(analysis_results["tau2_gm2"]),
            "F0_N": None if not np.isfinite(analysis_results["gm2_F0"]) else float(analysis_results["gm2_F0"]),
            "R2": None if not np.isfinite(analysis_results["r2_gm2"]) else float(analysis_results["r2_gm2"]),
            "regularization": analysis_results["gm2_reg_info"],
        },
        "hertz_E0_Pa": float(hertz_result.E0) if hertz_result is not None else None,
    }

    json_path = None
    if save_json:
        json_path = os.path.join(save_dir, f"{curve_name}_ting_fit_results.json")
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=4)
        print(f"JSON salvato: {os.path.basename(json_path)}")

    csv_path = None
    if save_csv:
        # ── Export tabellare del fit ──
        csv_path = os.path.join(save_dir, f"{curve_name}_ting_fit_data.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "indentation_m", "force_exp_N", "force_plr_N", "force_gm2_N"])
            for t, ind, fexp, fplr, fgm2 in zip(
                analysis_results["fit_time_ds"],
                analysis_results["fit_ind_ds"],
                analysis_results["fit_force_raw_ds"],
                analysis_results["fit_force_model_ds"],
                analysis_results["fit_force_gm2_ds"],
            ):
                writer.writerow([t, ind, fexp, fplr, fgm2])
        print(f"CSV salvato: {os.path.basename(csv_path)}")

    cm = analysis_results["fit_ind_ds"] >= 0.0

    def _arr_first_valid(keys):
        for key in keys:
            val = analysis_results.get(key)
            if val is None:
                continue
            arr = np.asarray(val, dtype=float)
            if arr.size:
                return arr
        return np.asarray([], dtype=float)

    raw_ind = _arr_first_valid(["fit_ind_uncorrected_ds", "ind_all_uncorrected", "fit_ind_ds", "ind_all"])
    raw_time = _arr_first_valid(["fit_time_uncorrected_ds", "time_all_uncorrected", "fit_time_ds", "time_all"])
    raw_force = _arr_first_valid(["fit_force_uncorrected_ds", "force_all_uncorrected", "fit_force_raw_ds", "force_all"])
    path0 = None

    # ── Confronto con la curva non corretta ──
    if raw_ind.size and raw_force.size and raw_time.size:
        valid = np.isfinite(raw_ind) & np.isfinite(raw_time) & np.isfinite(raw_force)
        raw_ind = raw_ind[valid]
        raw_time = raw_time[valid]
        raw_force = raw_force[valid]

    if raw_ind.size and raw_force.size and raw_time.size:
        fig0, (ax0a, ax0b) = plt.subplots(1, 2, figsize=(12, 4.8))
        ax0a.plot(raw_ind * 1e9, raw_force * 1e12, color="#8A8F98", linewidth=1.8, label="Prima della correzione")
        ax0a.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax0a.set_xlabel("Indentation [nm]", fontsize=12)
        ax0a.set_ylabel("Force [pN]", fontsize=12)
        ax0a.set_title("Dati iniziali: forza vs indentazione")
        ax0a.grid(alpha=0.25, linestyle="--")
        ax0a.legend()

        ax0b.plot(raw_time, raw_force * 1e12, color="#8A8F98", linewidth=1.8, label="Prima della correzione")
        ax0b.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax0b.set_xlabel("Time [s]", fontsize=12)
        ax0b.set_ylabel("Force [pN]", fontsize=12)
        ax0b.set_title("Dati iniziali: forza vs tempo")
        ax0b.grid(alpha=0.25, linestyle="--")
        ax0b.legend()

        comp = analysis_results.get("baseline_comparison", {})
        fig0.suptitle(
            "Dati iniziali (non corretti) | "
            f"gap raw={comp.get('raw_gap_pN', float('nan')):+.1f} pN | "
            f"trim={comp.get('retrace_trim_points', 0)} pt",
            fontsize=12,
        )
        fig0.tight_layout()
        path0 = os.path.join(save_dir, f"{curve_name}_baseline_comparison.png")
        fig0.savefig(path0, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig0)
        print(f"PNG salvato: {os.path.basename(path0)}")

    # ── Forza vs indentazione ──
    fig1, ax1 = plt.subplots()
    ax1.plot(
        (analysis_results["fit_ind_ds"] * 1e9)[cm],
        (analysis_results["fit_force_raw_ds"] * 1e12)[cm],
        color="#E72E38",
        linewidth=2.8,
        label="Dati sperimentali",
    )
    ax1.plot(
        (analysis_results["fit_ind_ds"] * 1e9)[cm],
        (analysis_results["fit_force_model_ds"] * 1e12)[cm],
        color="#2B3E51",
        linewidth=2.2,
        linestyle="--",
        label=f"Ting PLR  betaE={ting_result.betaE:.3f}  R2={analysis_results['r2_vis']:.4f}",
    )
    if analysis_results["gm2_ok"]:
        ax1.plot(
            (analysis_results["fit_ind_ds"] * 1e9)[cm],
            (analysis_results["fit_force_gm2_ds"] * 1e12)[cm],
            color="#2D8C4F",
            linewidth=2.0,
            linestyle="-.",
            label=(
                f"Ting GM2  tau1={analysis_results['tau1_gm2'] * 1000:.2f} ms  "
                f"tau2={analysis_results['tau2_gm2'] * 1000:.2f} ms  R2={analysis_results['r2_gm2']:.4f}"
            ),
        )
    ax1.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlabel("Indentation [nm]", fontsize=13)
    ax1.set_ylabel("Force [pN]", fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.25, linestyle="--")
    fig1.tight_layout()
    path1 = os.path.join(save_dir, f"{curve_name}_ting_force_vs_ind.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig1)
    print(f"PNG salvato: {os.path.basename(path1)}")

    # ── Forza vs tempo ──
    fig2, ax2 = plt.subplots()
    ax2.plot(
        analysis_results["fit_time_ds"][cm],
        (analysis_results["fit_force_raw_ds"] * 1e12)[cm],
        color="#E72E38",
        linewidth=2.8,
        label="Dati sperimentali",
    )
    ax2.plot(
        analysis_results["fit_time_ds"][cm],
        (analysis_results["fit_force_model_ds"] * 1e12)[cm],
        color="#2B3E51",
        linewidth=2.2,
        linestyle="--",
        label=f"Ting PLR  betaE={ting_result.betaE:.3f}  R2={analysis_results['r2_vis']:.4f}",
    )
    if analysis_results["gm2_ok"]:
        ax2.plot(
            analysis_results["fit_time_ds"][cm],
            (analysis_results["fit_force_gm2_ds"] * 1e12)[cm],
            color="#2D8C4F",
            linewidth=2.0,
            linestyle="-.",
            label=(
                f"Ting GM2  tau1={analysis_results['tau1_gm2'] * 1000:.2f} ms  "
                f"tau2={analysis_results['tau2_gm2'] * 1000:.2f} ms  R2={analysis_results['r2_gm2']:.4f}"
            ),
        )
    ax2.axvline(0, color="gray", linewidth=0.8, linestyle=":", label="Contact point")
    ax2.set_xlabel("Time [s]", fontsize=13)
    ax2.set_ylabel("Force [pN]", fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.25, linestyle="--")
    fig2.tight_layout()
    path2 = os.path.join(save_dir, f"{curve_name}_ting_force_vs_time.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig2)
    print(f"PNG salvato: {os.path.basename(path2)}")

    if open_output_folder:
        os.system(f'open "{save_dir}"')

    return {
        "save_dir": save_dir,
        "curve_name": curve_name,
        "json_path": json_path,
        "csv_path": csv_path,
        "plot_baseline_comparison": path0,
        "plot_force_vs_indentation": path1,
        "plot_force_vs_time": path2,
    }


def save_ting_outputs(
    data_dir,
    analysis_results,
    open_output_folder=False,
    save_each_curve=True,
    show_plots=True,
    save_json=True,
    save_csv=True,
    save_summary_table=True,
):
    """Save selected outputs for either a single curve or a whole folder analysis."""
    if analysis_results.get("is_batch", False):
        # ── Modalita batch: riepilogo e singole curve ──
        save_dir = os.path.abspath(data_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Cartella batch di salvataggio: {save_dir}")

        summary_rows = list(analysis_results.get("summary_rows", []))
        summary_table_path = None

        if save_summary_table:
            # ── Tabella PDF riepilogativa ──
            summary_table_path = os.path.join(save_dir, "ting_batch_summary_table.pdf")

            col_specs = [
                ("curve", "Curve"),
                ("status", "Status"),
                ("selected_model", "Model"),
                ("E0_Pa", "E0 [Pa]"),
                ("betaE", "betaE"),
                ("tc_ms", "tc [ms]"),
                ("rmse_plr_pN", "RMSE PLR [pN]"),
                ("r2_plr", "R2 PLR"),
                ("hertz_E0_init_kpa", "Hertz E0 init [kPa]"),
                ("young_hertz_kpa", "Young Hertz [kPa]"),
                ("Einf_Pa", "Einf [Pa]"),
                ("A1_Pa", "A1 [Pa]"),
                ("tau1_ms", "tau1 [ms]"),
                ("A2_Pa", "A2 [Pa]"),
                ("tau2_ms", "tau2 [ms]"),
                ("rmse_gm2_pN", "RMSE GM2 [pN]"),
                ("r2_gm2", "R2 GM2"),
            ]

            def _fmt_table_value(v):
                if v is None:
                    return ""
                if isinstance(v, (float, np.floating)):
                    return f"{float(v):.4g}"
                return str(v)

            by_cell = {}
            for row in summary_rows:
                by_cell.setdefault(row.get("cell_folder", "n/d"), []).append(row)

            rows_per_page = 22
            with PdfPages(summary_table_path) as pdf:
                for cell_name in sorted(by_cell.keys()):
                    rows = by_cell[cell_name]
                    for start in range(0, len(rows), rows_per_page):
                        chunk = rows[start:start + rows_per_page]
                        page_idx = (start // rows_per_page) + 1

                        fig, ax = plt.subplots(figsize=(16, 9))
                        ax.axis("off")
                        ax.set_title(
                            f"Ting summary | Cell: {cell_name} | curves: {len(rows)} | page {page_idx}",
                            fontsize=13,
                            pad=18,
                        )

                        col_labels = [label for _, label in col_specs]
                        cell_text = [
                            [_fmt_table_value(row.get(key)) for key, _ in col_specs]
                            for row in chunk
                        ]

                        table = ax.table(
                            cellText=cell_text,
                            colLabels=col_labels,
                            loc="center",
                            cellLoc="center",
                            colLoc="center",
                        )
                        table.auto_set_font_size(False)
                        table.set_fontsize(7)
                        table.scale(1.0, 1.25)

                        for (r, c), cell in table.get_celld().items():
                            if r == 0:
                                cell.set_text_props(weight="bold")
                                cell.set_facecolor("#EAEAEA")

                        fig.tight_layout()
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

            print(f"Tabella batch salvata: {os.path.basename(summary_table_path)}")

        summary_json_path = None
        if save_json:
            # ── Export JSON batch ──
            summary_json_path = os.path.join(save_dir, "ting_batch_summary.json")
            with open(summary_json_path, "w") as f:
                json.dump({
                    "input_path": analysis_results.get("input_path"),
                    "n_found": analysis_results.get("n_found"),
                    "n_success": analysis_results.get("n_success"),
                    "n_failed": analysis_results.get("n_failed"),
                    "summary_rows": summary_rows,
                    "failed_files": analysis_results.get("failed_files", []),
                    "fd_folders": analysis_results.get("fd_folders", []),
                    "n_fd_folders": analysis_results.get("n_fd_folders", 0),
                }, f, indent=4)
            print(f"JSON batch salvato: {os.path.basename(summary_json_path)}")

        summary_csv_path = None
        if save_csv:
            # ── Export CSV batch ──
            summary_csv_path = os.path.join(save_dir, "ting_batch_summary.csv")
            fieldnames = [
                "curve", "cell_folder", "fd_folder", "path", "status", "selected_model", "selection_reason",
                "E0_Pa", "betaE", "tc_ms", "rmse_plr_pN", "r2_plr",
                "hertz_E0_init_pa", "hertz_E0_init_kpa",
                "young_hertz_pa", "young_hertz_kpa",
                "Einf_Pa", "A1_Pa", "tau1_ms", "A2_Pa", "tau2_ms", "rmse_gm2_pN", "r2_gm2",
                "retrace_trim_points",
                "closure_refined", "contact_strategy", "force_drag_applied",
            ]
            with open(summary_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow({key: row.get(key) for key in fieldnames})
            print(f"CSV batch salvato: {os.path.basename(summary_csv_path)}")

        individual_outputs = []
        if save_each_curve:
            # ── Salvataggio delle singole curve ──
            for item in analysis_results.get("batch_curve_results", []):
                try:
                    outputs = _save_single_ting_outputs(
                        item["path"],
                        item["analysis"],
                        open_output_folder=False,
                        show_plots=False,
                        save_json=save_json,
                        save_csv=save_csv,
                    )
                    individual_outputs.append(outputs)
                except Exception as exc:
                    individual_outputs.append({
                        "curve_name": item.get("curve"),
                        "error": str(exc),
                    })

        if open_output_folder:
            os.system(f'open "{save_dir}"')

        return {
            "save_dir": save_dir,
            "summary_table_path": summary_table_path,
            "summary_json_path": summary_json_path,
            "summary_csv_path": summary_csv_path,
            "n_saved_curves": len(individual_outputs),
            "individual_outputs": individual_outputs,
        }

    return _save_single_ting_outputs(
        data_dir,
        analysis_results,
        open_output_folder=open_output_folder,
        show_plots=show_plots,
        save_json=save_json,
        save_csv=save_csv,
    )
