# -*- coding: utf-8 -*-
# plays_app.py
# Streamlit app: Plays (ML + Totais)
# - LÃª ml_artifact.json (ML core v2) para p_map (calibrado) e calcula odds de sÃ©rie (BO1/3/5)
# - LÃª CSV OracleElixir (team rows) para calcular Totais (kills/towers/dragons/barons/inhibitors/time)
# - Calcula odds justas de Over/Under e combos ML+Over/Under (baseline independÃªncia)

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datetime import date, datetime, timedelta
import os
import shutil
import json
import sys
import subprocess

import math
import hashlib
import re
import difflib
import html

import pandas as pd
import numpy as np
import streamlit as st

# --- Streamlit Cloud: ajuste de PATH (mlcore/app_core) ---
# No Streamlit Cloud o app roda em Linux e o repositório pode estar aninhado
# (por ex.: `src/` ou `gabinetelolweb3.1/`). Este bloco procura a pasta que
# contém `mlcore/` e garante que ela esteja no sys.path antes dos imports.
_THIS_DIR = Path(__file__).resolve().parent

# 1) caminhos mais comuns
_candidates = [
    _THIS_DIR,
    _THIS_DIR / 'src',
    _THIS_DIR / 'app',
]

# 2) pais (caso o arquivo esteja em subpasta)
_candidates += list(_THIS_DIR.parents)

def _inject_path_with_mlcore():
    import sys

    # (A) tentar candidatos diretos
    for base in _candidates:
        if (base / 'mlcore').is_dir():
            sys.path.insert(0, str(base))
            return True

    # (B) procurar por `mlcore/artifact.py` em até 4 níveis
    # (isso cobre cenários tipo repo_root/gabinetelolweb3.1/mlcore)
    max_hits = 3
    hits = 0
    try:
        for artifact in _THIS_DIR.glob('**/mlcore/artifact.py'):
            base = artifact.parent.parent
            if base.is_dir():
                sys.path.insert(0, str(base))
                hits += 1
                return True
            if hits >= max_hits:
                break
    except Exception:
        pass

    return False

_injected = _inject_path_with_mlcore()

from mlcore.artifact import MLArtifact

from mlcore.bo import prob_win_series
from mlcore.tiers import tier_base_points, league_base_points, tier_priority
from mlcore.players_layer import PlayersArtifact, compute_delta_and_confidence
from mlcore.achievements import load_achievements, build_team_index, team_boost_points
from mlcore.utils import clip_prob, prob_to_decimal_odds, decimal_odds_to_prob
from mlcore.walkforward_report import run_walkforward
from mlcore.fit_ml_params import fit_params as _fit_ml_params_run
from mlcore.layer_validation import run_layer_validation
from mlcore.calibration_diagnostics import run_calibration_diagnostics
from app_core.market import (
    parse_lines_text,
    parse_time_lines_text,
    matchup_key as build_matchup_key,
    odd_from_prob as market_odd_from_prob,
    prob_from_odd as market_prob_from_odd,
)
from app_core.line_store import load_json_store, persist_json_store, ensure_session_store
from app_core.lineup_state import (
    POS_CANON as _POS_CANON,
    POS_CANON_ML as _POS_CANON_ML,
    safe_key as _safe_key_impl,
    norm_pos as _norm_pos_impl,
    norm_pos_ml as _norm_pos_ml_impl,
    parse_pct as _parse_pct_impl,
    lineup_store as _lineup_store_impl,
    lineup_key as _lineup_key_impl,
    load_lineup_for_matchup as _load_lineup_for_matchup_impl,
    save_lineup_for_matchup as _save_lineup_for_matchup_impl,
    get_lineup_selected as _get_lineup_selected_impl,
)
from app_core.lines_state import load_lines_for_matchup as _load_lines_for_matchup_impl, save_lines_for_matchup as _save_lines_for_matchup_impl
from app_core.navigation import TAB_ALL as _TAB_ALL, pick_tabs as _pick_tabs
from app_core.odds_core import (
    laplace_odds as core_laplace_odds,
    odd_to_prob as core_odd_to_prob,
    prob_to_odd as core_prob_to_odd,
    edge_pp_from_odds as core_edge_pp_from_odds,
    ou_block as core_ou_block,
    event_block as core_event_block,
    tail_df as core_tail_df,
    recency_slices as core_recency_slices,
)
from app_core.resumo_core import (
    wr_laplace as core_wr_laplace,
    counts_ml_totals as core_counts_ml_totals,
    make_pricing_adapters as core_make_pricing_adapters,
    finalize_resumo_dataframe as core_finalize_resumo_dataframe,
    build_ml_rows as core_build_ml_rows,
    build_metric_rows as core_build_metric_rows,
    build_combo_rows as core_build_combo_rows,
)
from app_core.players_core import (
    build_player_means_rows as core_build_player_means_rows,
    build_player_market_rows as core_build_player_market_rows,
)
from app_core.players_state import (
    parse_players_text as core_parse_players_text,
    resolve_players as core_resolve_players,
    build_lane_rows as core_build_lane_rows,
    ensure_players_text_state as core_ensure_players_text_state,
    ensure_player_line_input_state as core_ensure_player_line_input_state,
    apply_players_submission_state as core_apply_players_submission_state,
    build_per_player_lines_by_idx as core_build_per_player_lines_by_idx,
)
from app_core.app_state import (
    bootstrap_state as core_bootstrap_state,
    apply_recommended_preset as core_apply_recommended_preset,
    apply_excellent_value_preset as core_apply_excellent_value_preset,
    apply_twenty_ten_preset as core_apply_twenty_ten_preset,
    get_state_schema as core_get_state_schema,
    get_bool as state_get_bool,
    get_float as state_get_float,
    get_int as state_get_int,
    get_str as state_get_str,
)
from app_core.formatting import format_num as core_format_num, format_pct as core_format_pct
from app_core.settings_core import (
    settings_path as core_settings_path,
    load_settings_from_disk as core_load_settings_from_disk,
    apply_settings_to_session as core_apply_settings_to_session,
    save_settings_to_disk as core_save_settings_to_disk,
    process_pending_settings_actions as core_process_pending_settings_actions,
    init_settings_once as core_init_settings_once,
)
from app_core.settings_ui import render_settings_sidebar as core_render_settings_sidebar
from app_core.trace_core import build_combined_trace as core_build_combined_trace
from app_core.calibration_schedule import build_patch_snapshot as core_build_patch_snapshot, recommend_recalibration as core_recommend_recalibration
from app_core.fusion_core import (
    fuse_with_coherence_guard as core_fuse_with_coherence_guard,
    season_form_blend as core_season_form_blend,
    hybrid_momentum_blend as core_hybrid_momentum_blend,
)
from app_core.consistency_core import (
    panel_title_with_suffix as core_panel_title_with_suffix,
    fmt_metric_value as core_fmt_metric_value,
    team_metric_stats as core_team_metric_stats,
    last_rows_metric as core_last_rows_metric,
    build_overall_tables as core_build_overall_tables,
    h2h_numeric_series as core_h2h_numeric_series,
    build_ml_totals_model_payload as core_build_ml_totals_model_payload,
    build_ml_totals_laplace_payload as core_build_ml_totals_laplace_payload,
    build_audit_base_options as core_build_audit_base_options,
    build_audit_market_options as core_build_audit_market_options,
    build_audit_view as core_build_audit_view,
    build_quick_metric_options as core_build_quick_metric_options,
    quick_team_frame as core_quick_team_frame,
    quick_outcome_probs as core_quick_outcome_probs,
    build_quick_selector_rows as core_build_quick_selector_rows,
)
from mlcore.lines_engine import (
    build_profile,
    filter_team_games,
    matchup_expected_totals,
    prob_over_series_from_sims,
    simulate_series_total,
    total_over_prob,
    normalize_map_mode,
    _combine_mean,
    _combine_sd,
    MatchupTotals,
)

from mlcore.stats_teams import Filters, build_team_games, load_team_rows, apply_filters, recency_weights


# -----------------------------
# UI text normalization (fix mojibake like "ConfiguraÃ§Ãµes")
# -----------------------------
_MOJIBAKE_MARKERS = ("Ã", "â", "�")


def _fix_mojibake_text(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    if not any(m in s for m in _MOJIBAKE_MARKERS):
        return s

    def _score(txt: str) -> tuple[int, int]:
        bad = sum(txt.count(m) for m in _MOJIBAKE_MARKERS)
        return (bad, len(txt))

    def _replace_common(txt: str) -> str:
        rep = {
            "â€”": "—",
            "â€“": "–",
            "â†’": "→",
            "âœ…": "✅",
            "âšï¸": "⚠️",
            "âš ï¸": "⚠️",
            "Î”": "Δ",
            "â€¦": "...",
            "Âª": "ª",
            "Âº": "º",
            "Ã¡": "á",
            "Ã ": "à",
            "Ã¢": "â",
            "Ã£": "ã",
            "Ã¤": "ä",
            "Ã©": "é",
            "Ãª": "ê",
            "Ã­": "í",
            "Ã³": "ó",
            "Ã´": "ô",
            "Ãµ": "õ",
            "Ãº": "ú",
            "Ã§": "ç",
            "Ã": "Á",
            "Ã‰": "É",
            "Ã“": "Ó",
            "Ã‡": "Ç",
            "Ãš": "Ú",
            "ÃŽ": "Î",
            "Ã‘": "Ñ",
            "Ã£o": "ão",
            "NÃ£o": "Não",
            "nÃ£o": "não",
            "SÃ©rie": "Série",
            "sÃ©rie": "série",
            "MÃ©dia": "Média",
            "mÃ©dia": "média",
            "HistÃ³rico": "Histórico",
            "histÃ³rico": "histórico",
            "ConfiguraÃ§Ãµes": "Configurações",
            "CalibraÃ§Ãµes": "Calibrações",
            "ParÃ¢metros": "Parâmetros",
            "parÃ¢metros": "parâmetros",
            "DragÃµes": "Dragões",
            "BarÃµes": "Barões",
            "CampeÃµes": "Campeões",
            "VitÃ³ria": "Vitória",
            "Ãšltimos": "Últimos",
            "vÃ¡lido": "válido",
            "invÃ¡lido": "inválido",
            "rÃ¡pido": "rápido",
            "sessÃ£o": "sessão",
            "aÃ§Ã£o": "ação",
            "forÃ§a": "força",
            "diferenÃ§a": "diferença",
            "MÃ©trica": "Métrica",
        }
        out = txt
        for k, v in rep.items():
            out = out.replace(k, v)
        return out

    candidates = [s, _replace_common(s)]
    for enc in ("latin1", "cp1252"):
        try:
            dec = s.encode(enc).decode("utf-8")
            candidates.append(dec)
            candidates.append(_replace_common(dec))
        except Exception:
            pass

    best = min(candidates, key=_score)
    return best if _score(best) < _score(s) else s


def _patch_streamlit_text_rendering() -> None:
    if getattr(st, "_gabinete_text_patch", False):
        return

    def _fix_obj(x):
        if isinstance(x, str):
            return _fix_mojibake_text(x)
        if isinstance(x, list):
            return [_fix_obj(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_fix_obj(v) for v in x)
        if isinstance(x, dict):
            return {k: _fix_obj(v) for k, v in x.items()}
        return x

    def _wrap_first_arg(fn):
        def _inner(*args, **kwargs):
            if args:
                args = tuple(_fix_obj(a) for a in args)
            if kwargs:
                kwargs = {k: _fix_obj(v) for k, v in kwargs.items()}
            elif "label" in kwargs:
                kwargs["label"] = _fix_mojibake_text(kwargs.get("label"))
            return fn(*args, **kwargs)
        return _inner

    def _wrap_write(fn):
        def _inner(*args, **kwargs):
            fixed = tuple(_fix_mojibake_text(a) if isinstance(a, str) else a for a in args)
            return fn(*fixed, **kwargs)
        return _inner

    def _wrap_tabs(fn):
        def _inner(tabs, *args, **kwargs):
            fixed_tabs = [_fix_mojibake_text(x) if isinstance(x, str) else x for x in tabs]
            return fn(fixed_tabs, *args, **kwargs)
        return _inner

    def _wrap_expander(fn):
        def _inner(label, *args, **kwargs):
            # UX rule: todos os expanders começam minimizados.
            kwargs["expanded"] = False
            return fn(_fix_mojibake_text(label) if isinstance(label, str) else label, *args, **kwargs)
        return _inner

    def _fix_dataframe_obj(obj):
        if isinstance(obj, pd.DataFrame):
            try:
                d = obj.copy()
                d.columns = [(_fix_mojibake_text(c) if isinstance(c, str) else c) for c in d.columns]
                try:
                    d.index = [(_fix_mojibake_text(i) if isinstance(i, str) else i) for i in d.index]
                except Exception:
                    pass
                for c in d.columns:
                    try:
                        if pd.api.types.is_object_dtype(d[c]) or pd.api.types.is_string_dtype(d[c]):
                            d[c] = d[c].map(lambda v: _fix_mojibake_text(v) if isinstance(v, str) else v)
                    except Exception:
                        pass
                return d
            except Exception:
                return obj
        return obj

    def _wrap_dataframe(fn):
        def _inner(data=None, *args, **kwargs):
            data2 = _fix_dataframe_obj(data)
            return fn(data2, *args, **kwargs)
        return _inner

    for name in [
        "markdown", "caption", "title", "header", "subheader", "text",
        "info", "warning", "error", "success", "button", "checkbox",
        "radio", "selectbox", "multiselect", "text_input", "text_area",
        "number_input", "slider", "metric", "download_button",
        "file_uploader", "date_input", "time_input", "expander",
    ]:
        if hasattr(st, name):
            setattr(st, name, _wrap_first_arg(getattr(st, name)))
    if hasattr(st, "write"):
        st.write = _wrap_write(st.write)
    if hasattr(st, "tabs"):
        st.tabs = _wrap_tabs(st.tabs)
    if hasattr(st, "expander"):
        st.expander = _wrap_expander(st.expander)
    if hasattr(st, "dataframe"):
        st.dataframe = _wrap_dataframe(st.dataframe)
    if hasattr(st, "table"):
        st.table = _wrap_dataframe(st.table)

    st._gabinete_text_patch = True


_patch_streamlit_text_rendering()


# -----------------------------
# App settings (persistÃªncia em disco)
# -----------------------------
_SETTINGS_FILE = ".gabinete_settings.json"

# Chaves que vale a pena persistir entre sessÃµes (evitar salvar coisas gigantes do session_state)
_PERSIST_KEYS: List[str] = [
    "cfg_profile",
    # paths / boot
    "cfg_auto_paths",
    "cfg_csv_override",
    "cfg_artifact_override",
    "boot_daily_csv",
    "boot_hist_csvs",
    # filtros / modos comuns
    "window_short",
    "totals_mode",
    "ml_mode",
    "map_mode_ui",
    "league_mode",
    "year_opt",
    "split_opt",
    "playoffs_opt",
    "fixed_league",
    "weight_mode",
    "half_life_days",
    "n_sims",
    "resumo_max_odd",
    "resumo_combo_preset",
    "resumo_filter_preset",
    "resumo_filter_custom",
    "resumo_filter_profiles",
    "resumo_filter_profile_active",
    "resumo_combo_hist_w",
    "resumo_combo_style_w",
    "resumo_combo_form_w",
    "resumo_combo_use_mismatch",
    "ml_engine",
    "ml_engine_compare",
    "ml_engine_alert_pp",
    "ml_microseg_enabled",
    "ml_microseg_strength",
    "ml_microseg_min_n",
    "ml_microseg_shrink_k",
    "ml_microseg_lookback_games",
    "ml_microseg_cap_pp",
    "ml_show_calc_details_default",
    "ml_as_of_date",
    "ui_fast_flow",
    # lineup
    "use_lineup_adjust",
    "players_csv_path",
    "lineup_lambda_points",
    "lineup_delta_mode",
    "lineup_delta_cap",
    "lineup_delta_slope",
    "lineup_min_gp",
    "lineup_shrink_m",
    "ml_players_w_lineup",
    "ml_players_lane_w_top",
    "ml_players_lane_w_jungle",
    "ml_players_lane_w_mid",
    "ml_players_lane_w_adc",
    "ml_players_lane_w_support",
    "ml_players_lg_w_lpl",
    "ml_players_lg_w_lck",
    "ml_players_lg_w_emea",
    "ml_players_lg_w_na",
    "ml_players_lg_w_br",
    "ml_players_lg_w_apac",
    "ml_players_lg_w_other",
    "fusion_team_weight_early",
    "fusion_team_weight_mid",
    "fusion_team_weight_playoffs",
    "fusion_early_season_team_weight",
    "fusion_season_knee_games",
    "fusion_coverage_power",
    "fusion_transfer_boost",
    "fusion_divergence_pp_cap",
    "fusion_divergence_low_coverage",
    "fusion_divergence_shrink",
    "resumo_use_edge_gate",
    "resumo_min_edge_pp",
    "wf_min_train_games",
    "wf_k",
    "wf_scale",
    "wf_apply_ml_correction",
    "wf_ml_corr_strength",
    "lv_min_train_games",
    "lv_min_league_games",
    "resumo_min_sample_liga",
    "resumo_min_sample_h2h",
    "resumo_hist_shrink_n",
    "sim_seed_mode",
    "sim_seed_manual",
    "ml_consistency_guard",
    "ml_consistency_tol_pp",
    "ml_hybrid_blend_enabled",
    "ml_hybrid_blend_max_weight",
    "ml_hybrid_blend_knee_games",
    "ml_hybrid_momentum_games",
    "ml_hybrid_beta",
    "ml_conf_guard_enabled",
    "ml_conf_guard_min_games",
    "ml_conf_guard_max_shrink",
    "ml_series_gamma_use_fitted",
    "cal_diag_bins",
    "cal_diag_min_league_games",
    # elo params (se usar / comparar)
    "elo_team_hl",
    "elo_player_hl",
    "elo_w_players",
    "elo_k_team",
    "elo_k_player",
    "elo_shrink_m",
]

def _settings_path() -> Path:
    return core_settings_path(Path(__file__).resolve().parent, _SETTINGS_FILE)

def _load_app_settings_from_disk() -> Dict[str, Any]:
    return core_load_settings_from_disk(_settings_path(), date_keys=("ml_as_of_date",))

def _apply_settings_to_session(settings: Dict[str, Any], force: bool = False) -> None:
    core_apply_settings_to_session(st.session_state, settings or {}, persist_keys=_PERSIST_KEYS, force=bool(force))

def _save_app_settings_to_disk() -> None:
    core_save_settings_to_disk(_settings_path(), st.session_state, persist_keys=_PERSIST_KEYS)

_STATE_SCHEMA = core_get_state_schema()
_PARAM_DEFAULTS: Dict[str, Any] = dict(_STATE_SCHEMA.get("defaults", {}))
_STATE_MIGRATIONS: Dict[str, str] = dict(_STATE_SCHEMA.get("migrations", {}))
_STATE_CONSTRAINTS: Dict[str, Tuple[float, float]] = dict(_STATE_SCHEMA.get("constraints", {}))

# Bootstrap central de estado (defaults + migrações + saneamento).
core_bootstrap_state(
    st.session_state,
    defaults=_PARAM_DEFAULTS,
    migrations=_STATE_MIGRATIONS,
    constraints=_STATE_CONSTRAINTS,
)

# Defaults locais (não dependem de schema externo).
_LOCAL_DEFAULTS: Dict[str, Any] = {
    "ml_microseg_enabled": True,
    "ml_microseg_strength": 0.60,
    "ml_microseg_min_n": 30,
    "ml_microseg_shrink_k": 90.0,
    "ml_microseg_lookback_games": 1200,
    "ml_microseg_cap_pp": 8.0,
    "resumo_filter_preset": "Custom",
    "resumo_filter_custom": {},
    "resumo_filter_profile_active": "",
    "resumo_filter_profiles": {
        "Conservador": {
            "resumo_max_odd": 1.72,
            "resumo_req_liga": True,
            "resumo_req_h2h": True,
            "resumo_req_model": True,
            "resumo_min_sample_liga": 10,
            "resumo_min_sample_h2h": 4,
        },
        "Balanceado": {
            "resumo_max_odd": 2.00,
            "resumo_req_liga": True,
            "resumo_req_h2h": False,
            "resumo_req_model": True,
            "resumo_min_sample_liga": 8,
            "resumo_min_sample_h2h": 3,
        },
        "Agressivo": {
            "resumo_max_odd": 2.35,
            "resumo_req_liga": False,
            "resumo_req_h2h": False,
            "resumo_req_model": True,
            "resumo_min_sample_liga": 5,
            "resumo_min_sample_h2h": 1,
        },
    },
    "ml_players_lane_w_top": 1.00,
    "ml_players_lane_w_jungle": 1.00,
    "ml_players_lane_w_mid": 1.00,
    "ml_players_lane_w_adc": 1.00,
    "ml_players_lane_w_support": 1.00,
    "ml_players_lg_w_lpl": 1.00,
    "ml_players_lg_w_lck": 1.00,
    "ml_players_lg_w_emea": 1.00,
    "ml_players_lg_w_na": 1.00,
    "ml_players_lg_w_br": 1.00,
    "ml_players_lg_w_apac": 1.00,
    "ml_players_lg_w_other": 1.00,
}
for _k, _v in _LOCAL_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _reset_all_tuning_to_defaults() -> None:
    core_apply_recommended_preset(st.session_state, "PadrÃ£o")
    for k, v in _PARAM_DEFAULTS.items():
        st.session_state[k] = v


def _render_settings_sidebar() -> None:
    core_render_settings_sidebar(
        save_settings_fn=_save_app_settings_to_disk,
        settings_path=_settings_path(),
    )


def _process_pending_settings_actions() -> None:
    core_process_pending_settings_actions(
        st.session_state,
        load_settings_fn=lambda: _load_app_settings_from_disk(),
        apply_settings_fn=lambda data, force: _apply_settings_to_session(data, force=force),
        apply_recommended_preset_fn=lambda prof: core_apply_recommended_preset(st.session_state, profile=prof),
        apply_excellent_preset_fn=lambda: core_apply_excellent_value_preset(st.session_state),
        apply_twenty_ten_preset_fn=lambda: core_apply_twenty_ten_preset(st.session_state),
        reset_all_tuning_fn=_reset_all_tuning_to_defaults,
        param_defaults=_PARAM_DEFAULTS,
    )


def _init_app_settings_once() -> None:
    core_init_settings_once(
        st.session_state,
        load_settings_fn=lambda: _load_app_settings_from_disk(),
        apply_settings_fn=lambda data, force: _apply_settings_to_session(data, force=force),
        process_pending_fn=_process_pending_settings_actions,
    )


# -----------------------------
# Helpers
# -----------------------------
def _file_sig(path: str) -> Tuple[str, int, int]:
    p = Path(path)
    try:
        st_ = p.stat()
        return (str(p.resolve()), int(st_.st_size), int(st_.st_mtime))
    except Exception:
        return (str(p), 0, 0)



@st.cache_data(show_spinner=False)
def _csv_diag(csv_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    """DiagnÃ³stico rÃ¡pido do CSV (para entender por que 'faltam jogos')."""
    csv_path, _size, _mtime = csv_sig
    out: Dict[str, Any] = {}
    try:
        df = pd.read_csv(csv_path, usecols=["gameid", "participantid"], low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if "gameid" not in df.columns or "participantid" not in df.columns:
                return {}
            df = df[["gameid", "participantid"]].copy()
        except Exception:
            return {}
    out["rows"] = int(len(df))
    # gameid vazios
    gid = df["gameid"]
    miss = int(gid.isna().sum())
    try:
        miss += int((gid.astype(str).str.strip() == "").sum())
    except Exception:
        pass
    out["rows_missing_gameid"] = miss

    out["unique_gameids"] = int(df["gameid"].dropna().nunique())

    # team rows
    try:
        pid = pd.to_numeric(df["participantid"], errors="coerce")
    except Exception:
        pid = df["participantid"]
    team_mask = pid.isin([100, 200])
    df_team = df.loc[team_mask].copy()
    out["team_rows"] = int(len(df_team))

    # jogos completos/incompletos (pelo par 100/200)
    df_team = df_team.dropna(subset=["gameid"])
    if df_team.empty:
        out["games_complete"] = 0
        out["games_incomplete"] = 0
        return out

    grp = df_team.groupby("gameid")["participantid"].agg(lambda x: set(pd.to_numeric(pd.Series(list(x)), errors="coerce").dropna().astype(int).tolist()))
    complete = int(grp.apply(lambda s: (100 in s) and (200 in s)).sum())
    incomplete = int(grp.apply(lambda s: ((100 in s) ^ (200 in s))).sum())
    out["games_complete"] = complete
    out["games_incomplete"] = incomplete
    return out



@st.cache_data(show_spinner=False)
def _load_team_games(csv_sig: Tuple[str, int, int]) -> pd.DataFrame:
    csv_path, _size, _mtime = csv_sig
    rows = load_team_rows(csv_path)
    df = build_team_games(rows)

    # Canonicaliza nomes de times para evitar que capitalizaÃ§Ã£o diferente (ex.: "Kia" vs "KIA")
    # faÃ§a o mesmo time aparecer como se fosse outro e caia em liga errada no modo auto.
    maps = _team_canon_maps(csv_sig)
    key_to_best = maps.get("variant_to_best_key") if isinstance(maps, dict) else None
    if isinstance(key_to_best, dict) and len(key_to_best) > 0 and "team" in df.columns:
        def _canonize_name(x: Any) -> str:
            s = ("" if x is None else str(x)).strip()
            s = re.sub(r"\s+", " ", s)
            k = _canon_team_key(s)
            return str(key_to_best.get(k, s))
        try:
            df["team"] = df["team"].map(_canonize_name)
            if "opponent" in df.columns:
                df["opponent"] = df["opponent"].map(_canonize_name)
        except Exception:
            pass
    return df
def _canon_team_key(name: str) -> str:
    """Chave canÃ´nica segura para agrupar variaÃ§Ãµes de capitalizaÃ§Ã£o/whitespace.

    Importante: NÃƒO remove pontuaÃ§Ã£o (pra nÃ£o misturar times diferentes).
    Ex.: "Dplus Kia" e "Dplus KIA" -> mesma chave.
    """
    s = ("" if name is None else str(name)).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


@st.cache_data(show_spinner=False)
def _team_canon_maps(csv_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    """Monta mapeamentos para evitar que o mesmo time apareÃ§a com 2 grafias no CSV.

    Retorna:
      - variant_to_best_key: (canon_key -> best_display_name)
      - best_to_variants: (best_display_name -> [variants...])
      - best_list: [best_display_name...]
    """
    csv_path, _size, _mtime = csv_sig
    rows = load_team_rows(csv_path)
    s = rows.get("teamname")
    if s is None:
        return {"variant_to_best_key": {}, "best_to_variants": {}, "best_list": []}

    names = s.dropna().astype(str).map(lambda x: re.sub(r"\s+", " ", x.strip()))
    if names.empty:
        return {"variant_to_best_key": {}, "best_to_variants": {}, "best_list": []}

    vc = names.value_counts()
    # agrupa por chave canÃ´nica
    key_to_variants: Dict[str, List[str]] = {}
    for v in vc.index.tolist():
        k = _canon_team_key(v)
        key_to_variants.setdefault(k, []).append(v)

    # escolhe "best" como a variante mais frequente
    key_to_best: Dict[str, str] = {}
    best_to_variants: Dict[str, List[str]] = {}
    for k, variants in key_to_variants.items():
        best = max(variants, key=lambda v: int(vc.get(v, 0)))
        key_to_best[k] = best
        best_to_variants[best] = sorted(list(set(variants)))

    best_list = sorted(list(best_to_variants.keys()))
    return {"variant_to_best_key": key_to_best, "best_to_variants": best_to_variants, "best_list": best_list}




@st.cache_data(show_spinner=False)
def _list_teams(csv_sig: Tuple[str, int, int]) -> List[str]:
    maps = _team_canon_maps(csv_sig)
    teams = maps.get("best_list") if isinstance(maps, dict) else None
    if not isinstance(teams, list) or len(teams) == 0:
        # fallback (nÃ£o deveria acontecer)
        csv_path, _size, _mtime = csv_sig
        rows = load_team_rows(csv_path)
        teams = sorted(rows["teamname"].dropna().unique().tolist())
    return teams


@st.cache_data(show_spinner=False)
def _load_player_rows(csv_sig: Tuple[str, int, int]) -> pd.DataFrame:
    """Carrega rows de players (participantid 1..10) do OracleElixir.

    MantÃ©m sÃ³ colunas necessÃ¡rias para a aba CampeÃµes/Players.
    """
    csv_path, _size, _mtime = csv_sig
    usecols = [
        'gameid','date','league','year','split','playoffs','game','patch',
        'participantid','side','position','playerid','playername','teamname','champion',
        'result','kills','deaths','assists','gamelength','teamkills','teamdeaths',
        'ckpm','total cs','cspm','damagetochampions','visionscore',
    ]
    try:
        try:
            df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
        except ValueError as e:
            # older match_data files may not have playerid; retry without it
            if "Usecols do not match columns" in str(e) and "playerid" in str(e):
                usecols2 = [c for c in usecols if c != "playerid"]
                df = pd.read_csv(csv_path, usecols=usecols2, low_memory=False)
            else:
                raise
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False)
        keep = [c for c in usecols if c in df.columns]
        df = df[keep].copy()

    # filtra players
    pid = pd.to_numeric(df.get('participantid'), errors='coerce')
    df = df.loc[pid.between(1, 10)].copy()

    # normaliza
    for c in ['league','split','position','side','playerid','playername','teamname','champion']:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str).str.strip()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=False)

    # auxiliares
    if 'teamname' in df.columns:
        df['team'] = df['teamname']
    if 'playername' in df.columns:
        df['player'] = df['playername']
    if 'champion' in df.columns:
        df['champ'] = df['champion']
    if 'position' in df.columns:
        df['pos'] = df['position']

    # identidade estÃ¡vel (playerid quando disponÃ­vel)
    if 'playerid' in df.columns:
        df['playerid'] = df['playerid'].fillna('').astype(str).str.strip()
    else:
        df['playerid'] = ''

    df['player_key'] = df['playerid']
    df.loc[df['player_key'].astype(str).str.strip().eq(''), 'player_key'] = df.get('player', '').astype(str)

    # posiÃ§Ã£o para o players_layer (Bot)
    if 'pos' in df.columns:
        df['pos_ml'] = df['pos'].apply(_norm_pos_ml)
    else:
        df['pos_ml'] = ''

    return df


def _apply_filters_players(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    out = df
    if filters is None:
        return out
    if getattr(filters, 'year', None) is not None and 'year' in out.columns:
        out = out[pd.to_numeric(out['year'], errors='coerce') == int(filters.year)]
    if getattr(filters, 'league', None) is not None and 'league' in out.columns:
        out = out[out['league'].astype(str) == str(filters.league)]
    if getattr(filters, 'split', None) is not None and 'split' in out.columns:
        out = out[out['split'].astype(str) == str(filters.split)]
    if getattr(filters, 'playoffs', None) is not None and 'playoffs' in out.columns:
        target = 1 if bool(filters.playoffs) else 0
        po = pd.to_numeric(out['playoffs'], errors='coerce').fillna(0).astype(int)
        out = out[(po > 0).astype(int) == target]
    return out


def _champion_vocab(df_players: pd.DataFrame) -> list[str]:
    try:
        return sorted([c for c in df_players['champ'].fillna('').astype(str).unique().tolist() if c])
    except Exception:
        return []


def _fuzzy_match_one(query: str, choices: list[str], cutoff: float = 0.65) -> tuple[str | None, list[str]]:
    q = str(query or '').strip()
    if not q:
        return None, []
    # normalizaÃ§Ã£o leve
    q_norm = re.sub(r"[^A-Za-z0-9]", "", q).lower()
    if not q_norm:
        return None, []

    norm_map = {}
    for ch in choices:
        key = re.sub(r"[^A-Za-z0-9]", "", str(ch)).lower()
        if key and key not in norm_map:
            norm_map[key] = ch

    if q_norm in norm_map:
        return norm_map[q_norm], []

    # sugestÃµes
    cand = difflib.get_close_matches(q_norm, list(norm_map.keys()), n=5, cutoff=cutoff)
    sug = [norm_map[x] for x in cand]
    return (sug[0] if sug else None), sug


def _stable_seed_from_parts(*parts: Any) -> int:
    txt = "||".join([str(x) for x in parts])
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _ml_consistency_signature(
    *,
    season_label: str,
    teamA: str,
    teamB: str,
    bo: int,
    ml_engine: str,
    calc_model_mode: str,
    year_opt: Any,
    split_opt: Any,
    playoffs_opt: Any,
    league_mode: str,
    fixed_league: Any,
    as_of_date: Any,
    artifact_path: str,
    use_lineup_adjust: bool,
    lineup_mode: str,
    lineupA: Optional[dict] = None,
    lineupB: Optional[dict] = None,
) -> int:
    art_sig = _file_sig(str(artifact_path)) if str(artifact_path or "").strip() else ("", 0, 0)
    return _stable_seed_from_parts(
        season_label,
        teamA,
        teamB,
        int(bo),
        ml_engine,
        calc_model_mode,
        year_opt,
        split_opt,
        playoffs_opt,
        league_mode,
        fixed_league,
        str(as_of_date),
        art_sig,
        bool(use_lineup_adjust),
        str(lineup_mode or ""),
        json.dumps(lineupA or {}, sort_keys=True, ensure_ascii=True),
        json.dumps(lineupB or {}, sort_keys=True, ensure_ascii=True),
    )


@st.cache_data(show_spinner=False)
def _series_sims_cached(
    bo: int,
    p_blue_map: float,
    total_obj,
    n_sims: int,
    seed: int = 1337,
) -> np.ndarray:
    """Cache Monte Carlo sims for series totals (per metric)."""
    # total_obj is a frozen dataclass (MatchupTotals), safe to cache by value.
    return simulate_series_total(
        bo=int(bo),
        p_blue_map=float(p_blue_map),
        total=total_obj,
        n_sims=int(n_sims),
        seed=int(seed),
    )


# -----------------------------
# ML core v2: carregar artifact, calcular total_strength, ML e ranking
# -----------------------------

def _resolve_relpath(base_file: str, maybe_rel: str) -> str:
    p = Path(maybe_rel)
    if p.is_absolute():
        return str(p)
    return str((Path(base_file).parent / p).resolve())


def _artifact_sig(path: str) -> Tuple[str, int, int]:
    return _file_sig(path)


@st.cache_resource(show_spinner=False)
def _load_artifact_cached(art_sig: Tuple[str, int, int]) -> MLArtifact:
    art_path, _size, _mtime = art_sig
    return MLArtifact.load_json(art_path)


@st.cache_data(show_spinner=False)
def _load_achievements_cached(ach_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    """Carrega achievements.json com tolerÃ¢ncia a erros.

    Nunca deve derrubar o app (ex.: JSON corrompido). Em caso de falha,
    retorna um dict com '__load_error__' para o chamador decidir o fallback.
    """
    ach_path, _size, _mtime = ach_sig
    try:
        return load_achievements(ach_path)
    except Exception as e:
        return {
            "__load_error__": f"{type(e).__name__}: {e}",
            "__path__": str(ach_path),
        }


@st.cache_data(show_spinner=False)
def _load_players_artifact_cached(p_sig: Tuple[str, int, int]) -> PlayersArtifact:
    p_path, _size, _mtime = p_sig
    return PlayersArtifact.load_json(p_path)

@st.cache_data(show_spinner=False)
def _load_fitted_params_cached(cfg_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    cfg_path, _size, _mtime = cfg_sig
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {}



def _save_achievements_json(path: str, data: Dict[str, Any]) -> None:
    """Salva achievements.json de forma robusta (aceita dataclasses/pydantic/numpy)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _json_default(o: Any):
        # dataclasses (ex.: AchievementEntry)
        try:
            from dataclasses import is_dataclass, asdict as _asdict
            if is_dataclass(o):
                return _asdict(o)
        except Exception:
            pass
        # pydantic
        if hasattr(o, 'model_dump'):
            try:
                return o.model_dump()
            except Exception:
                pass
        if hasattr(o, 'dict'):
            try:
                return o.dict()
            except Exception:
                pass
        # numpy / pandas
        try:
            import numpy as _np
            if isinstance(o, (_np.integer, _np.floating)):
                return o.item()
        except Exception:
            pass
        try:
            import pandas as _pd
            if isinstance(o, _pd.Timestamp):
                return o.isoformat()
        except Exception:
            pass
        # datas
        try:
            from datetime import date as _date, datetime as _dt
            if isinstance(o, (_date, _dt)):
                return o.isoformat()
        except Exception:
            pass
        # pathlib
        try:
            from pathlib import Path as _Path
            if isinstance(o, _Path):
                return str(o)
        except Exception:
            pass
        # fallback: __dict__ ou string
        if hasattr(o, '__dict__'):
            try:
                return dict(o.__dict__)
            except Exception:
                pass
        return str(o)

    # escrita atÃ´mica + backup para evitar JSON corrompido em caso de interrupÃ§Ã£o
    tmp = p.with_suffix(p.suffix + ".tmp")
    if p.exists():
        try:
            bak = p.with_suffix(p.suffix + ".bak")
            shutil.copy2(p, bak)
        except Exception:
            pass

    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            pass
    try:
        os.replace(tmp, p)
    except Exception:
        # fallback (nÃ£o atÃ´mico)
        with open(p, 'w', encoding='utf-8') as f2:
            json.dump(data, f2, ensure_ascii=False, indent=2, default=_json_default)


def _team_meta_from_artifact(art: MLArtifact) -> Dict[str, Dict[str, str]]:
    meta = art.meta.get("team_meta")
    if isinstance(meta, dict) and meta:
        return {str(k): dict(v) for k, v in meta.items() if isinstance(v, dict)}
    # fallback (caso artifact antigo)
    tm = getattr(art.model, "team_meta", None)
    if isinstance(tm, dict) and tm:
        return {str(k): dict(v) for k, v in tm.items() if isinstance(v, dict)}
    return {}


def _ach_layer_for_artifact(art: MLArtifact, artifact_path: str, ach_override_path: Optional[str] = None) -> Dict[str, Any]:
    """Resolve camada de achievements para o cÃ¡lculo de forÃ§a.

    Prioridade:
    1) path vindo do artifact (art.meta['achievements']['path'])
    2) fallback: achievements.json ao lado do ml_artifact.json (mesma pasta)

    weights_points:
    - se o artifact tiver weights_points, usa
    - senÃ£o, tenta config.fixed_weights_points do achievements.json
    - senÃ£o, usa um default simples (ajustÃ¡vel na aba Vencedores)
    """
    DEFAULT_WEIGHTS: Dict[str, float] = {
        # MSI
        "msi:top16": 6.0,
        "msi:top8": 12.0,
        "msi:top4": 20.0,
        "msi:runner_up": 32.0,
        "msi:champion": 45.0,
        # WORLDS
        "worlds:top16": 8.0,
        "worlds:top8": 16.0,
        "worlds:top4": 26.0,
        "worlds:runner_up": 42.0,
        "worlds:champion": 60.0,
        # Regional / Other
        "regional:playoffs": 6.0,
        "regional:top8": 8.0,
        "regional:top4": 12.0,
        "regional:runner_up": 18.0,
        "regional:champion": 24.0,
        "other:champion": 12.0,
        "other:runner_up": 8.0,
        "other:top4": 6.0,
    }

    ach_meta = dict(art.meta.get("achievements") or {})
    ach_path = str(ach_meta.get("path") or "").strip()
    weights_points = dict(ach_meta.get("weights_points") or {})

    # override: se o usuÃ¡rio definiu um caminho no app (aba Vencedores), usa esse arquivo
    ach_path_resolved = ""
    if ach_override_path:
        try:
            cand = str(ach_override_path).strip()
            if cand:
                # permite path relativo ao ml_artifact.json
                cand_res = _resolve_relpath(artifact_path, cand) if not os.path.isabs(cand) else cand
                if Path(cand_res).exists():
                    ach_path_resolved = cand_res
        except Exception:
            ach_path_resolved = ""

    # resolve path (se override nÃ£o setou)
    if not ach_path_resolved:
        if ach_path:
            ach_path_resolved = _resolve_relpath(artifact_path, ach_path)
        else:
            # fallback: achievements.json no mesmo diretÃ³rio do artifact
            try:
                guess = str(Path(artifact_path).with_name("achievements.json"))
                if Path(guess).exists():
                    ach_path_resolved = guess
            except Exception:
                ach_path_resolved = ""

    if not ach_path_resolved or (not Path(ach_path_resolved).exists()):
        return {"enabled": False, "path": "", "team_idx": {}, "weights_points": {}}

    data = _load_achievements_cached(_file_sig(ach_path_resolved))
    # JSON invÃ¡lido / erro de leitura: nÃ£o derruba o app; desativa achievements e reporta no retorno.
    if isinstance(data, dict) and data.get("__load_error__"):
        return {
            "enabled": False,
            "path": ach_path_resolved,
            "team_idx": {},
            "weights_points": {},
            "error": str(data.get("__load_error__")),
        }

    team_idx = build_team_index(data) if data else {}

    # fallback weights_points: fixed_weights_points dentro do arquivo
    if (not weights_points) and isinstance(data, dict):
        cfg = data.get("config") or {}
        fwp = cfg.get("fixed_weights_points") or {}
        if isinstance(fwp, dict) and fwp:
            weights_points = {str(k): float(v) for k, v in fwp.items() if v is not None}

    if not weights_points:
        weights_points = dict(DEFAULT_WEIGHTS)

    return {
        "enabled": True,
        "path": ach_path_resolved,
        "team_idx": team_idx,
        "weights_points": weights_points,
    }


def _p_from_diff_points(diff_points: float, scale_points: float = 400.0) -> float:
    # Elo base-10
    try:
        denom = float(scale_points) if float(scale_points) > 0 else 400.0
        p = float(1.0 / (1.0 + 10 ** (-float(diff_points) / denom)))
        return float(clip_prob(p))
    except Exception:
        return float("nan")


# -----------------------------
# Helpers: infer league/tier for ML core v2 (to avoid artifact mislabeling)
# -----------------------------
_MAJOR_LEAGUES = {
    "LCK","LPL","LEC","LCS",
    "LTA NORTH","LTA SOUTH","LTA",
    "PCS","VCS",
    "CBLOL","LLA",
    "MSI","WORLDS","WC","WORLD CHAMPIONSHIP",
}

def _norm_league(s: str) -> str:
    return str(s or "").strip()

def _norm_league_key(s: str) -> str:
    return _norm_league(s).upper()

def _tier_from_league(lg: str) -> str:
    k = _norm_league_key(lg)
    if not k:
        return "OTHER"
    # Challengers / Academy / CL
    if any(x in k for x in ["CHALLENG", "ACADEMY", "CL", "LCKC", "NACL", "LFL", "LVP", "LDL", "LPL ACADEMY"]):
        return "T2"
    if k in _MAJOR_LEAGUES:
        return "T1"
    # default
    return "MINOR"

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def _infer_team_league_from_csv(_csv_sig: Tuple[str, int, int], team: str, year: Optional[int]) -> str:
    """Infere a liga mais provÃ¡vel do time no CSV ativo (team rows 100/200).

    Regra:
    - por padrÃ£o, usa a liga MAIS frequente;
    - se existir uma liga "major" com volume relevante, prioriza a major
      (evita cair em torneios menores por ruÃ­do).
    """
    csv_path, _size, _mtime = _csv_sig
    if not csv_path or not team:
        return ""

    usecols = ["teamname", "league", "year", "participantid"]
    try:
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            return ""
        keep = [c for c in usecols if c in df.columns]
        if not keep:
            return ""
        df = df[keep].copy()

    if "participantid" in df.columns:
        df = df[df["participantid"].isin([100, 200])]
    if year is not None and "year" in df.columns:
        df = df[df["year"] == int(year)]
    if "teamname" not in df.columns or "league" not in df.columns:
        return ""

    # canonicalizaÃ§Ã£o leve: casefold + strip (nÃ£o remove pontuaÃ§Ã£o pra evitar misturar times diferentes)
    t0 = str(team).strip().casefold()
    s = df["teamname"].astype(str).str.strip().str.casefold()
    d = df.loc[s == t0, "league"].dropna().astype(str).str.strip()
    if d.empty:
        return ""

    vc = d.value_counts()
    # estabiliza desempates
    vc_df = vc.reset_index()
    vc_df.columns = ["league", "count"]
    vc_df["league_norm"] = vc_df["league"].astype(str).apply(_norm_league_key)
    vc_df = vc_df.sort_values(["count", "league_norm"], ascending=[False, True])

    top_league = str(vc_df.iloc[0]["league"])
    top_cnt = int(vc_df.iloc[0]["count"])

    # melhor major (se existir)
    majors = vc_df[vc_df["league_norm"].isin(_MAJOR_LEAGUES)]
    if not majors.empty:
        best_major = majors.iloc[0]  # jÃ¡ estÃ¡ ordenado por count desc / nome
        major_league = str(best_major["league"])
        major_cnt = int(best_major["count"])

        # sÃ³ "forÃ§a major" se tiver volume suficiente
        min_major_games = 5
        if (_norm_league_key(top_league) in _MAJOR_LEAGUES) or (
            major_cnt >= min_major_games and major_cnt >= max(2, int(0.5 * top_cnt))
        ):
            return major_league

    return top_league


@st.cache_data(show_spinner=False)
def _infer_match_league_from_csv(_csv_sig: Tuple[str, int, int], teamA: str, teamB: str, year: Optional[int]) -> str:
    """Infere a liga do confronto (A x B) a partir do CSV ativo.

    1) tenta achar liga diretamente nos H2H (A vs B) no recorte;
    2) se nÃ£o existir H2H, reconcilia a liga de cada time; se divergirem, retorna "".
    """
    if not teamA or not teamB:
        return ""
    try:
        df = _load_team_games(_csv_sig)
    except Exception:
        df = None
    if df is None or getattr(df, "empty", True):
        # fallback direto por time (mais custoso, mas evita quebrar)
        lgA = _infer_team_league_from_csv(_csv_sig, teamA, year)
        lgB = _infer_team_league_from_csv(_csv_sig, teamB, year)
        if lgA and lgB and _norm_league_key(lgA) == _norm_league_key(lgB):
            return lgA
        return lgA or lgB or ""

    dd = df
    if year is not None and "year" in dd.columns:
        dd = dd[dd["year"] == int(year)]

    tA = str(teamA).strip().casefold()
    tB = str(teamB).strip().casefold()
    s_team = dd["team"].astype(str).str.strip().str.casefold()
    s_opp = dd["opponent"].astype(str).str.strip().str.casefold()

    mask = ((s_team == tA) & (s_opp == tB)) | ((s_team == tB) & (s_opp == tA))
    h2h = dd.loc[mask, "league"].dropna().astype(str).str.strip()
    if not h2h.empty:
        vc = h2h.value_counts()
        vc_df = vc.reset_index()
        vc_df.columns = ["league", "count"]
        vc_df["league_norm"] = vc_df["league"].astype(str).apply(_norm_league_key)
        # ordena por count desc, e major primeiro em empate
        vc_df["is_major"] = vc_df["league_norm"].isin(_MAJOR_LEAGUES).astype(int)
        vc_df = vc_df.sort_values(["count", "is_major", "league_norm"], ascending=[False, False, True])
        return str(vc_df.iloc[0]["league"])

    # sem H2H: reconcilia ligas provÃ¡veis
    lgA = _infer_team_league_from_csv(_csv_sig, teamA, year)
    lgB = _infer_team_league_from_csv(_csv_sig, teamB, year)
    if lgA and lgB and _norm_league_key(lgA) == _norm_league_key(lgB):
        return lgA
    return lgA or lgB or ""



def _strength_breakdown(
    *,
    art: MLArtifact,
    team: str,
    as_of: date,
    team_meta: Dict[str, Dict[str, str]],
    ach_layer: Dict[str, Any],
    league_hint: Optional[str] = None,
    tier_hint: Optional[str] = None,
) -> Dict[str, Any]:
    model = getattr(art, "model", None)
    ratings = dict(getattr(model, "ratings", {}) or {}) if model is not None else {}
    # usa residual "efetivo" (com shrink por games_played) para evitar overconfidence em amostras pequenas
    _rating_raw_fn = getattr(model, "rating_raw", None) if model is not None else None
    _residual_eff_fn = getattr(model, "residual_eff", None) if model is not None else None
    residual_raw = float(_rating_raw_fn(team)) if callable(_rating_raw_fn) else float(ratings.get(team, 0.0))
    residual = float(_residual_eff_fn(team)) if callable(_residual_eff_fn) else residual_raw
    _n_games_fn = getattr(model, "n_games", None) if model is not None else None
    games_played = int(_n_games_fn(team)) if callable(_n_games_fn) else int(getattr(model, "games_played", {}).get(team, 0) if model is not None else 0)

    m = dict(team_meta.get(team) or {})
    league_raw = str(m.get("league") or "OTHER")
    tier_raw = str(m.get("tier") or "OTHER")

    league = _norm_league(league_hint) or league_raw
    tier = str(tier_hint or "").strip() or (tier_raw if not _norm_league(league_hint) else _tier_from_league(league))

    tb = float(tier_base_points(tier))
    lb = float(league_base_points(league))

    ach = 0.0
    if ach_layer.get("enabled") and ach_layer.get("weights_points"):
        ach = float(team_boost_points(team, as_of, ach_layer.get("team_idx") or {}, ach_layer.get("weights_points") or {}))

    total = float(residual + tb + lb + ach)

    return {
        "team": team,
        "league": league,
        "tier": tier,
        "league_raw": league_raw,
        "tier_raw": tier_raw,
        "residual_elo": residual,
        "residual_raw": residual_raw,
        "games_played": games_played,
        "tier_base": tb,
        "league_base": lb,
        "ach_boost": ach,
        "total_strength": total,
    }


def _mlcore_p_map_v2(
    artifact_path: str,
    team_blue: str,
    team_red: str,
    as_of: date,
    ach_override_path: Optional[str] = None,
    league_hint_blue: Optional[str] = None,
    league_hint_red: Optional[str] = None,
    tier_hint_blue: Optional[str] = None,
    tier_hint_red: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    p = Path(artifact_path)
    if not p.exists():
        return None

    art = _load_artifact_cached(_artifact_sig(str(p)))
    team_meta = _team_meta_from_artifact(art)
    ach_layer = _ach_layer_for_artifact(art, str(p), ach_override_path=ach_override_path)

    scale_points = float(getattr(getattr(art.model, "params", None), "scale_points", 400.0) or 400.0)

    b = _strength_breakdown(art=art, team=team_blue, as_of=as_of, team_meta=team_meta, ach_layer=ach_layer, league_hint=league_hint_blue, tier_hint=tier_hint_blue)
    r = _strength_breakdown(art=art, team=team_red, as_of=as_of, team_meta=team_meta, ach_layer=ach_layer, league_hint=league_hint_red, tier_hint=tier_hint_red)

    model = getattr(art, "model", None)
    diff = float(b["total_strength"] - r["total_strength"])
    p_raw = float("nan")
    pricing_source = "fallback_diff"
    cal_error = ""

    # Main path: use the trained model directly for ML pricing.
    if model is not None and hasattr(model, "p_blue"):
        try:
            p_raw = float(model.p_blue(
                team_blue,
                team_red,
                blue_league=b.get("league"),
                red_league=r.get("league"),
                blue_tier=b.get("tier"),
                red_tier=r.get("tier"),
            ))
            p_raw = float(clip_prob(p_raw))
            pricing_source = "artifact_model_p_blue"
            # keep a readable diff for UI, compatible with prior structure
            diff = float((float(scale_points) / math.log(10.0)) * math.log(p_raw / (1.0 - p_raw)))
        except Exception:
            p_raw = float("nan")

    # Fallback for old artifacts/edge cases.
    if not math.isfinite(p_raw):
        p_raw = _p_from_diff_points(diff, scale_points=scale_points)
        pricing_source = "reconstructed_diff"

    try:
        p_cal = float(clip_prob(float(art.calibrator.predict_proba(float(p_raw)))))
    except Exception:
        p_cal = float(clip_prob(float(p_raw)))
        cal_error = "calibrator_failed_fallback_to_p_raw"

    return {
        "p_raw": p_raw,
        "p_cal": p_cal,
        "diff_points": diff,
        "scale_points": scale_points,
        "ach_enabled": bool(ach_layer.get("enabled")) if isinstance(ach_layer, dict) else False,
        "ach_path": str(ach_layer.get("path")) if isinstance(ach_layer, dict) else "",
        "ach_error": str(ach_layer.get("error") or "") if isinstance(ach_layer, dict) else "",
        "blue": b,
        "red": r,
        "achievements_path": ach_layer.get("path"),
        "achievements_enabled": bool(ach_layer.get("enabled")),
        "has_ach_weights": bool(ach_layer.get("weights_points")),
        "artifact_version": getattr(art, "version", ""),
        "ml_pricing_source": pricing_source,
        "calibration_error": cal_error,
    }


def _seg_key(league: str, split: str, side: str) -> str:
    return f"{str(league or '').strip()}|{str(split or '').strip()}|{str(side or '').strip().lower()}"


@st.cache_data(show_spinner=False)
def _build_ml_microseg_table(
    csv_sig: Tuple[str, int, int],
    art_sig: Tuple[str, int, int],
    lookback_games: int = 1200,
    shrink_k: float = 90.0,
) -> Dict[str, Dict[str, float]]:
    """Build micro-segment calibration residuals with hierarchical fallback buckets.

    Segment hierarchy (for Blue-side probability):
    1) league + split + side
    2) league + * + side
    3) league + * + *
    4) * + * + *
    """
    out: Dict[str, Dict[str, float]] = {}
    try:
        tg = _load_team_games(csv_sig)
        if tg is None or tg.empty:
            return out
        if not {"side", "team", "opponent", "win"}.issubset(set(tg.columns)):
            return out

        df = tg.copy()
        df["side"] = df["side"].astype(str).str.lower()
        df = df[df["side"] == "blue"].copy()
        if df.empty:
            return out
        if "date" in df.columns:
            df = df.sort_values("date")
        df = df.tail(int(max(200, int(lookback_games))))

        art = _load_artifact_cached(art_sig)
        model = getattr(art, "model", None)
        cal = getattr(art, "calibrator", None)
        if model is None:
            return out

        agg: Dict[str, Dict[str, float]] = {}

        for r in df.itertuples(index=False):
            b = str(getattr(r, "team", "") or "").strip()
            rr = str(getattr(r, "opponent", "") or "").strip()
            lg = str(getattr(r, "league", "") or "").strip()
            sp = str(getattr(r, "split", "") or "").strip()
            sd = "blue"
            if not b or not rr:
                continue
            try:
                y = float(getattr(r, "win", 0) or 0)
            except Exception:
                y = 0.0

            tier = _tier_from_league(lg) if lg else "OTHER"
            try:
                p_raw = float(model.p_blue(b, rr, blue_league=(lg or None), red_league=(lg or None), blue_tier=tier, red_tier=tier))
                p_raw = float(clip_prob(p_raw))
            except Exception:
                continue
            try:
                p_cal = float(clip_prob(float(cal.predict_proba(p_raw)))) if cal is not None else float(p_raw)
            except Exception:
                p_cal = float(p_raw)

            err = float(y - p_cal)  # positive => model underestimates blue
            keys = [
                _seg_key(lg, sp, sd),
                _seg_key(lg, "*", sd),
                _seg_key(lg, "*", "*"),
                _seg_key("*", "*", "*"),
            ]
            for k in keys:
                d = agg.setdefault(k, {"n": 0.0, "sum_err": 0.0})
                d["n"] += 1.0
                d["sum_err"] += float(err)

        sk = float(max(1.0, float(shrink_k)))
        for k, d in agg.items():
            n = float(d.get("n", 0.0))
            if n <= 0:
                continue
            raw_bias = float(d.get("sum_err", 0.0)) / n
            shrink = n / (n + sk)
            bias = float(raw_bias * shrink)
            out[k] = {"n": n, "raw_bias": raw_bias, "bias": bias}
    except Exception:
        return {}
    return out


def _apply_ml_microseg_calibration(
    p_map_cal: float,
    *,
    league: str,
    split: str,
    side: str,
    table: Dict[str, Dict[str, float]],
    min_n: int,
    strength: float,
    cap_pp: float,
) -> Tuple[float, Dict[str, Any]]:
    p0 = float(clip_prob(float(p_map_cal)))
    keys = [
        _seg_key(league, split, side),
        _seg_key(league, "*", side),
        _seg_key(league, "*", "*"),
        _seg_key("*", "*", "*"),
    ]
    info: Dict[str, Any] = {"applied": False, "key": "", "n": 0.0, "bias": 0.0, "strength": float(strength)}
    for k in keys:
        d = table.get(k) if isinstance(table, dict) else None
        if not isinstance(d, dict):
            continue
        n = float(d.get("n", 0.0))
        if n < float(max(1, int(min_n))):
            continue
        b = float(d.get("bias", 0.0))
        cap = abs(float(cap_pp)) / 100.0
        if cap > 0:
            b = max(-cap, min(cap, b))
        p1 = float(clip_prob(p0 + float(strength) * b))
        info.update({"applied": True, "key": k, "n": n, "bias": b, "p_before": p0, "p_after": p1})
        return p1, info
    info.update({"p_before": p0, "p_after": p0})
    return p0, info

# -----------------------------
# ML engine alternativo: Elo season (time) + Elo global (players)
# -----------------------------

def _discover_player_csv_paths(csv_path: str) -> list[str]:
    """Tenta achar CSVs de histÃ³rico (>=2024) alÃ©m do csv atual.
    Funciona com estrutura tÃ­pica: .../CSV_DIARIO/2026_*.csv e .../CSV_HISTORICO/2024_*.csv
    """
    out: list[str] = []
    if not csv_path:
        return out
    try:
        p = Path(str(csv_path))
        if p.exists():
            out.append(str(p))
        base_dir = p.parent
        root = base_dir.parent if base_dir.name.lower().startswith("csv_") else base_dir
        hist_dir = root / "CSV_HISTORICO"
        if hist_dir.exists():
            for fp in sorted(hist_dir.glob("*.csv")):
                # tenta filtrar por ano no nome (2024+)
                nm = fp.name
                m = re.search(r"(20\d{2})", nm)
                if m:
                    y = int(m.group(1))
                    if y >= 2024:
                        out.append(str(fp))
                else:
                    out.append(str(fp))
        # garante Ãºnicos mantendo ordem
        seen = set()
        out2 = []
        for x in out:
            if x not in seen:
                out2.append(x)
                seen.add(x)
        return out2
    except Exception:
        return out


@st.cache_data(show_spinner=False)
def _load_player_rows_multi_cached(sig_list: Tuple[Tuple[str, int, int], ...]) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for sig in sig_list:
        try:
            d = _load_player_rows(sig)
            if d is not None and not d.empty:
                dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    # garante year numÃ©rico
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
    return out


def _player_key_from_row(row: Any) -> str:
    try:
        pid = getattr(row, "playerid", "")
        if pid is None:
            pid = ""
        pid = str(pid).strip()
        if pid and pid.lower() not in ("nan", "none"):
            return pid
    except Exception:
        pass
    try:
        pn = getattr(row, "playername", "")
        return str(pn).strip()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def _compute_player_elo_global_cached(
    sig_list: Tuple[Tuple[str, int, int], ...],
    *,
    half_life_days: float,
    k_player: float,
    shrink_m: float,
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> Dict[str, Any]:
    """Elo global por player (2024+) com decay por tempo sem jogar.
    Atualiza players por jogo, usando rating mÃ©dio do time como forÃ§a.
    """
    df = _load_player_rows_multi_cached(sig_list)
    if df is None or df.empty:
        return {
            "rating": {},
            "games": {},
            "last": {},
            "id2name": {},
            "name2id": {},
        }

    # filtra year>=2024 se existir
    if "year" in df.columns:
        df = df[df["year"].fillna(0) >= 2024].copy()

    need = ["gameid", "date", "teamname", "result", "playername"]
    for c in need:
        if c not in df.columns:
            return {
                "rating": {},
                "games": {},
                "last": {},
                "id2name": {},
                "name2id": {},
            }

    # normaliza date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["gameid", "date", "teamname", "playername", "result"])
    if df.empty:
        return {
            "rating": {},
            "games": {},
            "last": {},
            "id2name": {},
            "name2id": {},
        }

    # cria player_key
    if "playerid" in df.columns:
        df["player_key"] = df["playerid"].fillna("").astype(str).str.strip()
        # fallback para nome
        mask = (df["player_key"] == "") | (df["player_key"].str.lower().isin(["nan", "none"]))
        df.loc[mask, "player_key"] = df.loc[mask, "playername"].astype(str).str.strip()
    else:
        df["player_key"] = df["playername"].astype(str).str.strip()

    df["teamname"] = df["teamname"].astype(str).str.strip()
    df["playername"] = df["playername"].astype(str).str.strip()

    # alias maps (para debug/lookup)
    id2name: Dict[str, str] = {}
    name2id: Dict[str, str] = {}
    if "playerid" in df.columns:
        tmp = df[["playerid", "playername"]].copy()
        tmp["playerid"] = tmp["playerid"].fillna("").astype(str).str.strip()
        tmp = tmp[tmp["playerid"] != ""]
        if not tmp.empty:
            # pega modo
            for pid, g in tmp.groupby("playerid"):
                try:
                    nm = g["playername"].mode().iloc[0]
                except Exception:
                    nm = g["playername"].iloc[0]
                id2name[str(pid)] = str(nm)
            # name2id (modo)
            for nm, g in tmp.groupby("playername"):
                try:
                    pid = g["playerid"].mode().iloc[0]
                except Exception:
                    pid = g["playerid"].iloc[0]
                name2id[str(nm)] = str(pid)

    # agrupa por game/team para obter lista de players
    df = df.sort_values("date", ascending=True)
    grouped = df.groupby(["gameid", "teamname"], sort=False)

    # estrutura por gameid: {date, teams:[(team, players, result)]}
    games_by_gid: Dict[str, Any] = {}
    for (gid, tname), g in grouped:
        # usa a data do jogo (max/first)
        dt = g["date"].max()
        res = float(g["result"].iloc[0])
        players = g["player_key"].dropna().astype(str).str.strip().tolist()
        players = [p for p in players if p and p.lower() not in ("nan", "none")]
        if not players:
            continue
        d = games_by_gid.setdefault(str(gid), {"date": dt, "teams": []})
        d["teams"].append((str(tname), players, 1.0 if res >= 1.0 else 0.0))

    # filtra sÃ³ jogos com 2 times
    items = [(gid, v["date"], v["teams"]) for gid, v in games_by_gid.items() if len(v["teams"]) >= 2]
    items.sort(key=lambda x: x[1])

    rating: Dict[str, float] = {}
    games_cnt: Dict[str, int] = {}
    last_play: Dict[str, Any] = {}

    def _decay_player(pk: str, now_dt) -> None:
        if pk not in rating:
            rating[pk] = float(base_rating)
            games_cnt[pk] = 0
            last_play[pk] = now_dt
            return
        try:
            prev = last_play.get(pk)
            if prev is None:
                last_play[pk] = now_dt
                return
            dt_days = (now_dt - prev).total_seconds() / 86400.0
            if dt_days <= 0:
                last_play[pk] = now_dt
                return
            factor = 0.5 ** (float(dt_days) / float(half_life_days)) if half_life_days and half_life_days > 0 else 1.0
            rating[pk] = float(base_rating) + (float(rating.get(pk, base_rating)) - float(base_rating)) * float(factor)
            last_play[pk] = now_dt
        except Exception:
            last_play[pk] = now_dt

    for gid, dt, teams in items:
        # pega duas entradas (se vierem mais por bug, pega as 2 maiores listas)
        teams2 = sorted(teams, key=lambda x: len(x[1]), reverse=True)[:2]
        (ta, pa, sa), (tb, pb, sb) = teams2[0], teams2[1]

        # decay dos players envolvidos atÃ© dt
        for pk in pa:
            _decay_player(pk, dt)
        for pk in pb:
            _decay_player(pk, dt)

        ra = float(np.mean([rating.get(pk, base_rating) for pk in pa])) if pa else float(base_rating)
        rb = float(np.mean([rating.get(pk, base_rating) for pk in pb])) if pb else float(base_rating)
        ea = _elo_expected(ra, rb, scale=scale)

        # shrink por amostra (conservador)
        gpa = float(np.mean([games_cnt.get(pk, 0) for pk in pa])) if pa else 0.0
        gpb = float(np.mean([games_cnt.get(pk, 0) for pk in pb])) if pb else 0.0
        shrink = min(gpa, gpb) / (min(gpa, gpb) + float(shrink_m)) if shrink_m and shrink_m > 0 else 1.0
        k_eff = float(k_player) * float(shrink)

        # distribui update pelos 5
        da = k_eff * (float(sa) - float(ea))
        # update individual (divide igualmente)
        if pa:
            per = da / float(len(pa))
            for pk in pa:
                rating[pk] = float(rating.get(pk, base_rating)) + per
                games_cnt[pk] = int(games_cnt.get(pk, 0) + 1)
        if pb:
            per = (-da) / float(len(pb))
            for pk in pb:
                rating[pk] = float(rating.get(pk, base_rating)) + per
                games_cnt[pk] = int(games_cnt.get(pk, 0) + 1)

    return {
        "rating": rating,
        "games": games_cnt,
        "last": {k: (v.isoformat() if hasattr(v, "isoformat") else str(v)) for k, v in last_play.items()},
        "id2name": id2name,
        "name2id": name2id,
    }


@st.cache_data(show_spinner=False)
def _compute_team_elo_season_cached(
    team_games_sig: Tuple[int, int],
    *,
    filters: Filters,
    half_life_days: float,
    k_team: float,
    shrink_m: float,
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> Dict[str, Any]:
    """Elo season por time usando apenas o recorte (filters)."""
    # team_games_sig Ã© sÃ³ um hash barato para cache; o DataFrame vem do escopo (nÃ£o entra no cache)
    # (AtenÃ§Ã£o: este cache Ã© invalidado externamente ao mudar o CSV)
    return {"__not_used__": True}


def _compute_team_elo_season(
    team_games: pd.DataFrame,
    *,
    filters: Filters,
    half_life_days: float,
    k_team: float,
    shrink_m: float,
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> Dict[str, Any]:
    if team_games is None or team_games.empty:
        return {"rating": {}, "games": {}, "last": {}}

    tg = apply_filters(team_games, filters)
    need = ["gameid", "date", "team", "opponent", "win"]
    for c in need:
        if c not in tg.columns:
            return {"rating": {}, "games": {}, "last": {}}

    dfm = tg.copy()
    dfm["team"] = dfm["team"].astype(str)
    dfm["opponent"] = dfm["opponent"].astype(str)
    dfm = dfm[dfm["team"] != dfm["opponent"]].copy()
    dfm = dfm[dfm["team"] < dfm["opponent"]].copy()
    dfm["date"] = pd.to_datetime(dfm["date"], errors="coerce")
    dfm = dfm.dropna(subset=["date"])
    if dfm.empty:
        return {"rating": {}, "games": {}, "last": {}}

    dfm = dfm.sort_values("date", ascending=True)

    rating: Dict[str, float] = {}
    games_cnt: Dict[str, int] = {}
    last_play: Dict[str, Any] = {}

    def _decay_team(t: str, now_dt) -> None:
        if t not in rating:
            rating[t] = float(base_rating)
            games_cnt[t] = 0
            last_play[t] = now_dt
            return
        prev = last_play.get(t)
        if prev is None:
            last_play[t] = now_dt
            return
        dt_days = (now_dt - prev).total_seconds() / 86400.0
        if dt_days <= 0:
            last_play[t] = now_dt
            return
        factor = 0.5 ** (float(dt_days) / float(half_life_days)) if half_life_days and half_life_days > 0 else 1.0
        rating[t] = float(base_rating) + (float(rating.get(t, base_rating)) - float(base_rating)) * float(factor)
        last_play[t] = now_dt

    for row in dfm.itertuples(index=False):
        dt = getattr(row, "date")
        ta = str(getattr(row, "team"))
        tb = str(getattr(row, "opponent"))
        sa = 1.0 if float(getattr(row, "win")) >= 1.0 else 0.0

        _decay_team(ta, dt)
        _decay_team(tb, dt)

        ra = float(rating.get(ta, base_rating))
        rb = float(rating.get(tb, base_rating))
        ea = _elo_expected(ra, rb, scale=scale)

        gpa = float(games_cnt.get(ta, 0))
        gpb = float(games_cnt.get(tb, 0))
        shrink = min(gpa, gpb) / (min(gpa, gpb) + float(shrink_m)) if shrink_m and shrink_m > 0 else 1.0
        k_eff = float(k_team) * float(shrink)

        delta = k_eff * (sa - ea)
        rating[ta] = ra + delta
        rating[tb] = rb - delta
        games_cnt[ta] = int(games_cnt.get(ta, 0) + 1)
        games_cnt[tb] = int(games_cnt.get(tb, 0) + 1)

    return {
        "rating": rating,
        "games": games_cnt,
        "last": {k: (v.isoformat() if hasattr(v, "isoformat") else str(v)) for k, v in last_play.items()},
    }


def _get_lineup_keys(side: str) -> Dict[str, str]:
    out = {}
    for pos in _POS_CANON_ML:
        k = f"lineup_ml_{side}_{_safe_key(pos)}"
        out[pos] = str(st.session_state.get(k, "") or "").strip()
    return out


def _elo_season_players_p_map(
    *,
    team_games: pd.DataFrame,
    csv_path: str,
    teamA: str,
    teamB: str,
    filters: Filters,
    team_half_life_days: float,
    player_half_life_days: float,
    k_team: float,
    k_player: float,
    shrink_m: float,
    w_players: float,
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> Dict[str, Any]:
    # team elo season
    tinfo = _compute_team_elo_season(
        team_games,
        filters=filters,
        half_life_days=float(team_half_life_days),
        k_team=float(k_team),
        shrink_m=float(shrink_m),
        base_rating=float(base_rating),
        scale=float(scale),
    )
    tr = tinfo.get("rating") or {}
    team_elo_A = float(tr.get(str(teamA), base_rating))
    team_elo_B = float(tr.get(str(teamB), base_rating))

    # players elo global
    paths = _discover_player_csv_paths(csv_path)
    sigs = tuple([_file_sig(p) for p in paths if p])
    pinfo = _compute_player_elo_global_cached(
        sigs,
        half_life_days=float(player_half_life_days),
        k_player=float(k_player),
        shrink_m=float(shrink_m),
        base_rating=float(base_rating),
        scale=float(scale),
    )
    pr = pinfo.get("rating") or {}
    name2id = pinfo.get("name2id") or {}
    id2name = pinfo.get("id2name") or {}

    # lineup atual (sempre usa)
    la = _get_lineup_keys("A")
    lb = _get_lineup_keys("B")

    def _lookup(pk: str) -> float:
        if not pk:
            return float("nan")
        if pk in pr:
            return float(pr[pk])
        # tenta mapear nome->id
        if pk in name2id and name2id[pk] in pr:
            return float(pr[name2id[pk]])
        # tenta id->nome
        if pk in id2name and id2name[pk] in pr:
            return float(pr[id2name[pk]])
        return float("nan")

    def _avg_lineup(lineup: Dict[str, str]) -> Dict[str, Any]:
        vals = []
        rows = []
        for pos, pk in lineup.items():
            r = _lookup(pk)
            rows.append({"pos": pos, "player_key": pk, "elo": r})
            if math.isfinite(r):
                vals.append(float(r))
        avg = float(np.mean(vals)) if vals else float("nan")
        return {"avg": avg, "rows": rows, "found": len(vals), "total": len(lineup)}

    a_info = _avg_lineup(la)
    b_info = _avg_lineup(lb)

    avgA = float(a_info["avg"]) if math.isfinite(float(a_info["avg"])) else float(base_rating)
    avgB = float(b_info["avg"]) if math.isfinite(float(b_info["avg"])) else float(base_rating)

    diff_team = team_elo_A - team_elo_B
    diff_players = avgA - avgB
    diff_total = float(diff_team) + float(w_players) * float(diff_players)

    p_raw = _p_from_diff_points(diff_total, scale_points=float(scale))
    return {
        "p_raw": float(p_raw),
        "p_cal": float(p_raw),
        "diff_points": float(diff_total),
        "team_elo_A": float(team_elo_A),
        "team_elo_B": float(team_elo_B),
        "avg_player_elo_A": float(avgA),
        "avg_player_elo_B": float(avgB),
        "diff_team": float(diff_team),
        "diff_players": float(diff_players),
        "lineup_A": a_info,
        "lineup_B": b_info,
        "team_games_A": int((tinfo.get("games") or {}).get(str(teamA), 0)),
        "team_games_B": int((tinfo.get("games") or {}).get(str(teamB), 0)),
    }



def _mlcore_rank_table_v2(artifact_path: str, as_of: date, top_n: int = 200) -> pd.DataFrame:
    p = Path(artifact_path)
    if not p.exists():
        return pd.DataFrame(columns=["rank", "team", "league", "tier", "residual_elo", "tier_base", "league_base", "ach_boost", "total_strength"])

    art = _load_artifact_cached(_artifact_sig(str(p)))
    team_meta = _team_meta_from_artifact(art)
    ach_layer = _ach_layer_for_artifact(art, str(p))

    teams = set()
    teams.update((getattr(art.model, "ratings", {}) or {}).keys())
    teams.update(team_meta.keys())

    rows = []
    for t in teams:
        bd = _strength_breakdown(art=art, team=str(t), as_of=as_of, team_meta=team_meta, ach_layer=ach_layer)
        rows.append(bd)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["rank", "team", "league", "tier", "residual_elo", "tier_base", "league_base", "ach_boost", "total_strength"])

    df = df.sort_values("total_strength", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    if top_n is not None and int(top_n) > 0:
        df = df.head(int(top_n)).copy()

    # formatos amigÃ¡veis
    for c in ["residual_elo", "tier_base", "league_base", "ach_boost", "total_strength"]:
        df[c] = df[c].apply(lambda x: _format_num(x, 1))

    return df



def _parse_lines(text: str) -> List[float]:
    return parse_lines_text(text)


def _parse_time_lines(text: str) -> List[float]:
    return parse_time_lines_text(text)



# -----------------------------
# Linhas por confronto (persistem enquanto os times nÃ£o mudam)
# -----------------------------

def _matchup_key(season_label: str, teamA: str, teamB: str) -> str:
    return build_matchup_key(season_label, teamA, teamB)


# PersistÃªncia das linhas (para nÃ£o perder ao fechar o app)
_LINES_STORE_DEFAULTS_KEY = "__defaults__"  # padrÃ£o global (fallback quando o confronto nÃ£o tem linhas salvas)
_LINES_STORE_FILE = Path(__file__).resolve().parent / "user_lines_store.json"


def _load_lines_store_disk() -> dict:
    return load_json_store(_LINES_STORE_FILE)


def _persist_lines_store_disk(store: dict) -> None:
    persist_json_store(_LINES_STORE_FILE, store)


def _lines_store() -> dict:
    return ensure_session_store(st.session_state, "_lines_store", _load_lines_store_disk())


def _load_lines_for_matchup(key: str) -> None:
    _load_lines_for_matchup_impl(st.session_state, _lines_store(), key, defaults_key=_LINES_STORE_DEFAULTS_KEY)


def _save_lines_for_matchup(key: str) -> None:
    store = _lines_store()
    _save_lines_for_matchup_impl(st.session_state, store, key)

    _persist_lines_store_disk(store)


# -----------------------------
# Lineup (Players) â€” store + scoring (ajuste do ML)
# -----------------------------

def _safe_key(s: str) -> str:
    return _safe_key_impl(s)


def _norm_pos(pos: str) -> str:
    return _norm_pos_impl(pos)


def _norm_pos_ml(pos: str) -> str:
    return _norm_pos_ml_impl(pos)


def _parse_pct(x) -> float:
    return _parse_pct_impl(x)


def _lineup_store() -> dict:
    return _lineup_store_impl(st.session_state)


def _lineup_key(season_label: str, teamA: str, teamB: str) -> str:
    return _lineup_key_impl(season_label, teamA, teamB)


def _load_lineup_for_matchup(key: str) -> None:
    _load_lineup_for_matchup_impl(st.session_state, key)


def _save_lineup_for_matchup(key: str) -> None:
    _save_lineup_for_matchup_impl(st.session_state, key)


def _get_lineup_selected(side: str, *, mode: str = "manual") -> dict:
    return _get_lineup_selected_impl(st.session_state, side, mode=mode)


@st.cache_data(show_spinner=False)
def _load_players_impact(players_sig: tuple) -> pd.DataFrame:
    # players_sig = _file_sig(path)
    csv_path = str(players_sig[0])
    df = pd.read_csv(csv_path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["team", "pos", "player", "gp", "score"])

    # normaliza
    df = df.copy()
    # tenta nomes de colunas comuns
    col_player = "Player" if "Player" in df.columns else ("player" if "player" in df.columns else None)
    col_team = "Team" if "Team" in df.columns else ("team" if "team" in df.columns else None)
    col_pos = "Pos" if "Pos" in df.columns else ("pos" if "pos" in df.columns else None)
    if not (col_player and col_team and col_pos):
        return pd.DataFrame(columns=["team", "pos", "player", "gp", "score"])

    df["player"] = df[col_player].astype(str).str.strip()
    df["team"] = df[col_team].astype(str).str.strip()
    df["pos"] = df[col_pos].astype(str).map(_norm_pos)

    # GP (amostra)
    col_gp = "GP" if "GP" in df.columns else ("gp" if "gp" in df.columns else None)
    if col_gp:
        df["gp"] = pd.to_numeric(df[col_gp], errors="coerce").fillna(0).astype(int)
    else:
        df["gp"] = 0

    # features (tenta capturar as mais estÃ¡veis do OE)
    # - lanes: GD10, XPD10, CSD10
    # - dano: DMG%
    # - mortes: DTH%
    feats = {
        "gd10": ("GD10", "gd10"),
        "xpd10": ("XPD10", "xpd10"),
        "csd10": ("CSD10", "csd10"),
        "dmgp": ("DMG%", "DMG_PCT", "dmg%", "dmgp"),
        "dthp": ("DTH%", "DTH_PCT", "dth%", "dthp"),
    }

    def _get_first_col(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    col_gd10 = _get_first_col(feats["gd10"])
    col_xpd10 = _get_first_col(feats["xpd10"])
    col_csd10 = _get_first_col(feats["csd10"])
    col_dmgp = _get_first_col(feats["dmgp"])
    col_dthp = _get_first_col(feats["dthp"])

    df["gd10"] = pd.to_numeric(df[col_gd10], errors="coerce") if col_gd10 else float("nan")
    df["xpd10"] = pd.to_numeric(df[col_xpd10], errors="coerce") if col_xpd10 else float("nan")
    df["csd10"] = pd.to_numeric(df[col_csd10], errors="coerce") if col_csd10 else float("nan")
    df["dmgp"] = df[col_dmgp].apply(_parse_pct) if col_dmgp else float("nan")
    df["dthp"] = df[col_dthp].apply(_parse_pct) if col_dthp else float("nan")

    # z-score por posiÃ§Ã£o (robusto)
    rows = []
    for pos, g in df.groupby("pos"):
        gg = g.copy()
        # fill NaNs com mediana por pos
        for c in ["gd10", "xpd10", "csd10", "dmgp", "dthp"]:
            med = gg[c].median(skipna=True)
            gg[c] = gg[c].fillna(med if pd.notna(med) else 0.0)
        # std
        def z(col):
            mu = float(gg[col].mean()) if gg.shape[0] else 0.0
            sd = float(gg[col].std(ddof=0)) if gg.shape[0] else 0.0
            if not (sd > 1e-9):
                return (gg[col] * 0.0)
            return (gg[col] - mu) / sd

        z_gd = z("gd10")
        z_xp = z("xpd10")
        z_cs = z("csd10")
        z_dmg = z("dmgp")
        z_dth = z("dthp")

        # score (simples, interpretÃ¡vel): lane + dano - mortes
        score_raw = (1.0 * z_gd) + (0.8 * z_xp) + (0.6 * z_cs) + (0.8 * z_dmg) + (-0.8 * z_dth)
        gg["score_raw"] = score_raw
        rows.append(gg)

    out = pd.concat(rows, ignore_index=True) if rows else df.copy()

    # shrink por amostra (evita explosÃ£o com GP pequeno)
    # score = score_raw * gp/(gp+m)
    # m serÃ¡ aplicado depois (no UI) via _apply_shrink_m
    out = out[["team", "pos", "player", "gp", "score_raw"]].copy()
    out = out.rename(columns={"score_raw": "score"})
    return out


def _apply_shrink_m(df_impact: pd.DataFrame, m: int) -> pd.DataFrame:
    if df_impact is None or df_impact.empty:
        return df_impact
    mm = max(1, int(m) if m is not None else 10)
    out = df_impact.copy()
    gp = pd.to_numeric(out.get("gp"), errors="coerce").fillna(0.0)
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce").fillna(0.0) * (gp / (gp + float(mm)))
    return out


def _best_match(name: str, choices: list, cutoff: float = 0.72) -> str:
    import difflib
    n = str(name or "").strip()
    if not n:
        return ""
    # exact
    if n in choices:
        return n
    # case-insensitive exact
    low = {str(c).lower(): str(c) for c in choices}
    if n.lower() in low:
        return low[n.lower()]
    # fuzzy
    m = difflib.get_close_matches(n, [str(c) for c in choices], n=1, cutoff=cutoff)
    return str(m[0]) if m else ""


def _latest_lineup_for_team(
    df_player_rows: pd.DataFrame,
    team: str,
    filters: Filters,
    league_fixed: str | None = None,
    *,
    pos_col: str = "pos",
    player_col: str = "player",
    pos_list: list[str] | None = None,
) -> dict:
    if df_player_rows is None or df_player_rows.empty:
        return {}
    df = df_player_rows.copy()

    pos_list = pos_list or list(_POS_CANON)

    # normaliza colunas esperadas de _load_player_rows
    if "team" not in df.columns or player_col not in df.columns or pos_col not in df.columns:
        return {}

    df = df[df["team"].astype(str) == str(team)].copy()
    if league_fixed:
        if "league" in df.columns:
            df = df[df["league"].astype(str) == str(league_fixed)].copy()
    if filters is not None:
        if getattr(filters, "year", None) is not None and "year" in df.columns:
            df = df[df["year"] == int(filters.year)].copy()

        # split: nÃ£o filtrar quando for All/Todos (caso contrÃ¡rio, some o lineup)
        if getattr(filters, "split", None) and "split" in df.columns:
            _sp = str(getattr(filters, "split", "") or "").strip()
            if _sp and _sp.lower() not in ("all", "todos", "todas", "any"):
                df = df[df["split"].astype(str).str.strip() == _sp].copy()

        # playoffs: robusto (0/1, bool, string)
        if getattr(filters, "playoffs", None) is not None and "playoffs" in df.columns:
            target = 1 if bool(getattr(filters, "playoffs", False)) else 0
            po = pd.to_numeric(df["playoffs"], errors="coerce").fillna(0).astype(int)
            df = df[po == int(target)].copy()

    if df.empty:
        return {}

    # pos canÃ´nico
    norm_func = _norm_pos_ml if pos_col == "pos_ml" else _norm_pos
    df["pos_can"] = df[pos_col].apply(norm_func)
    df = df[df["pos_can"].isin(pos_list)].copy()

    # precisa gameid + date
    if "gameid" not in df.columns or "date" not in df.columns:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["gameid", "date"])
    if df.empty:
        return {}

    # pega o game mais recente
    df = df.sort_values("date", ascending=True)
    last_gid = str(df.iloc[-1]["gameid"])
    g = df[df["gameid"].astype(str) == last_gid].copy()

    out = {}
    for pos in pos_list:
        cand = g[g["pos_can"] == pos][player_col].dropna().astype(str).tolist()
        if cand:
            out[pos] = cand[0].strip()

    return out


def _calibrate_p_from_artifact(artifact_path: str, p_raw: float) -> float:
    try:
        p = Path(artifact_path)
        if not p.exists():
            return float(clip_prob(float(p_raw)))
        art = _load_artifact_cached(_artifact_sig(str(p)))
        return float(clip_prob(float(art.calibrator.predict_proba(float(p_raw)))))
    except Exception:
        return float(clip_prob(float(p_raw)))




# -----------------------------
# Ranking (forÃ§a tipo ML) - Elo com recÃªncia (half-life)
# -----------------------------

def _elo_expected(ra: float, rb: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / float(scale)))

def _elo_rank_table(
    team_games: pd.DataFrame,
    *,
    filters: Filters,
    half_life_days: float,
    k_factor: float = 24.0,
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> pd.DataFrame:
    """Ranking por Elo com peso de recÃªncia (mesma ideia do ML: dar mais peso ao recente).

    - Usa team_games (build_team_games) -> colunas esperadas: gameid, date, team, opponent, win, league/split/playoffs...
    - Aplica filtros (season/league/split/playoffs) com apply_filters
    - Dedup por jogo: usa somente linhas onde team < opponent para contar 1x por gameid
    """
    if team_games is None or team_games.empty:
        return pd.DataFrame(columns=["team", "league", "games", "winrate", "elo"])

    tg = apply_filters(team_games, filters)

    # colunas mÃ­nimas
    need = ["gameid", "date", "team", "opponent", "win"]
    for c in need:
        if c not in tg.columns:
            return pd.DataFrame(columns=["team", "league", "games", "winrate", "elo"])

    # dedup (1 linha por jogo)
    dfm = tg.copy()
    dfm["team"] = dfm["team"].astype(str)
    dfm["opponent"] = dfm["opponent"].astype(str)
    dfm = dfm[dfm["team"] != dfm["opponent"]]

    # pega sÃ³ 1 direÃ§Ã£o (evita duplicar o mesmo jogo)
    dfm = dfm[dfm["team"] < dfm["opponent"]].copy()

    # ordena por data (processo Elo)
    dfm["date"] = pd.to_datetime(dfm["date"], errors="coerce")
    ref_date = dfm["date"].max()
    if pd.isna(ref_date):
        ref_date = pd.Timestamp.utcnow().tz_localize(None)

    dfm = dfm.sort_values("date", ascending=True)
    w = recency_weights(dfm["date"], half_life_days=float(half_life_days), ref_date=ref_date)

    # inicializa ratings
    teams = sorted(set(dfm["team"].unique().tolist()) | set(dfm["opponent"].unique().tolist()))
    rating = {t: float(base_rating) for t in teams}

    # jogos contados (para winrate depois)
    games = {t: 0 for t in teams}
    wins = {t: 0.0 for t in teams}
    leagues = {}
    if "league" in tg.columns:
        # tenta capturar a liga principal do time no recorte (moda)
        tmp = tg[["team", "league"]].copy()
        tmp["team"] = tmp["team"].astype(str)
        tmp["league"] = tmp["league"].fillna("").astype(str)
        for t, g in tmp.groupby("team"):
            if g.shape[0]:
                leagues[t] = g["league"].mode().iloc[0] if not g["league"].mode().empty else ""
    # processa Elo
    for i, row in enumerate(dfm.itertuples(index=False)):
        ta = getattr(row, "team")
        tb = getattr(row, "opponent")
        sa = float(getattr(row, "win"))  # 1 se ta venceu
        sa = 1.0 if sa >= 1.0 else 0.0

        ra = rating.get(ta, float(base_rating))
        rb = rating.get(tb, float(base_rating))
        ea = _elo_expected(ra, rb, scale=scale)
        k = float(k_factor) * float(w[i])

        delta = k * (sa - ea)
        rating[ta] = ra + delta
        rating[tb] = rb - delta

        games[ta] = games.get(ta, 0) + 1
        games[tb] = games.get(tb, 0) + 1
        wins[ta] = wins.get(ta, 0.0) + sa
        wins[tb] = wins.get(tb, 0.0) + (1.0 - sa)

    # monta tabela
    rows = []
    for t in teams:
        g = int(games.get(t, 0))
        if g <= 0:
            continue
        wr = float(wins.get(t, 0.0)) / float(g) if g else float("nan")
        rows.append({
            "team": t,
            "league": leagues.get(t, "") if leagues else "",
            "games": g,
            "winrate": wr,
            "elo": float(rating.get(t, base_rating)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["team", "league", "games", "winrate", "elo"])

    out = out.sort_values(["elo", "winrate", "games"], ascending=[False, False, False]).reset_index(drop=True)
    out["winrate"] = out["winrate"].apply(_format_pct)
    out["elo"] = out["elo"].apply(lambda x: _format_num(x, 0))
    return out
# -----------------------------
# Helpers de exibiÃ§Ã£o (tirar colunas irrelevantes)
# -----------------------------

def _display_ou_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela (visual) para Odds justas (linhas) no padrÃ£o PT-BR."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Linha", "Acima (%)", "Odd Acima", "Abaixo (%)", "Odd Abaixo"])

    return pd.DataFrame({
        "Linha": df.get("line"),
        "Acima (%)": df.get("p_over").apply(_format_pct),
        "Odd Acima": df.get("odd_over").apply(lambda x: _format_num(x, 2)),
        "Abaixo (%)": df.get("p_under").apply(_format_pct),
        "Odd Abaixo": df.get("odd_under").apply(lambda x: _format_num(x, 2)),
    })


def _display_ml_totals_table(df: pd.DataFrame, team_label: Optional[str] = None) -> pd.DataFrame:
    """Tabela (visual) para ML + Totais (usa colunas p_ML+OVER / p_ML+UNDER da _lines_table)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Linha", "ML & Acima (%)", "Odd ML & Acima", "ML & Abaixo (%)", "Odd ML & Abaixo"])

    out = pd.DataFrame({
        "Linha": df.get("line"),
        "ML & Acima (%)": df.get("p_ML+OVER(map/map)").apply(_format_pct),
        "Odd ML & Acima": df.get("odd_ML+OVER(map/map)").apply(lambda x: _format_num(x, 2)),
        "ML & Abaixo (%)": df.get("p_ML+UNDER(map/map)").apply(_format_pct),
        "Odd ML & Abaixo": df.get("odd_ML+UNDER(map/map)").apply(lambda x: _format_num(x, 2)),
    })
    if team_label:
        out.insert(0, "Time", str(team_label))
    return out


def _inject_odds_cards_css() -> None:
    if st.session_state.get('_b2t_odds_cards_css_done'):
        return
    st.session_state['_b2t_odds_cards_css_done'] = True

    st.markdown(
        """
<style>
  .b2t-row { margin: 0.25rem 0 0.65rem 0; }
  .b2t-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 14px 14px;
  }
  .b2t-card.fav {
    background: rgba(255,255,255,0.06);
    border: 2px solid rgba(255,255,255,0.32);
  }
  .b2t-title { font-size: 0.85rem; opacity: 0.85; margin-bottom: 6px; }
  .b2t-big { font-size: 1.55rem; font-weight: 750; line-height: 1.15; }
  .b2t-mid { font-size: 1.10rem; font-weight: 650; line-height: 1.15; }
  .b2t-sub { font-size: 0.92rem; opacity: 0.75; margin-top: 6px; }
  .b2t-mono { font-variant-numeric: tabular-nums; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _inject_global_ui_css() -> None:
    if st.session_state.get("_gb_ui_css_done"):
        return
    st.session_state["_gb_ui_css_done"] = True
    st.markdown(
        """
<style>
  :root {
    --gb-accent: #ff5a5f;
    --gb-panel: rgba(255,255,255,0.035);
    --gb-panel-border: rgba(255,255,255,0.12);
  }
  .main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
  }
  .gb-hero {
    border: 1px solid var(--gb-panel-border);
    background: linear-gradient(135deg, rgba(255,90,95,0.10), rgba(255,255,255,0.02));
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
  }
  .gb-hero .title {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0.01em;
    line-height: 1.15;
  }
  .gb-hero .sub {
    opacity: 0.80;
    margin-top: 2px;
    font-size: 0.92rem;
  }
  div[data-testid="stTabs"] button[role="tab"] {
    border-radius: 10px;
    padding: 0.25rem 0.65rem;
    margin-right: 0.2rem;
  }
  div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: rgba(255,90,95,0.14);
    border-bottom-color: var(--gb-accent) !important;
  }
  .stButton > button {
    border-radius: 10px;
  }
  div[data-testid="stExpander"] {
    background: var(--gb-panel);
    border: 1px solid var(--gb-panel-border);
    border-radius: 10px;
  }
  div[data-testid="stMetric"] {
    border: 1px solid var(--gb-panel-border);
    border-radius: 10px;
    padding: 8px 10px;
    background: rgba(255,255,255,0.02);
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_odds_cards(df_raw: pd.DataFrame, mode: str = 'ou', table_label: str = 'Ver tabela') -> None:
    """Renderiza odds em cards (neutro), destacando o lado com maior probabilidade.

    mode:
      - 'ou': Acima/Abaixo (p_over/p_under)
      - 'ml_totals': ML & Acima / ML & Abaixo (p_ML+OVER / p_ML+UNDER)
    """
    if df_raw is None or df_raw.empty:
        st.info('Nenhuma linha informada.')
        return

    _inject_odds_cards_css()

    if mode == 'ml_totals':
        label_a, label_b = 'ML & Acima', 'ML & Abaixo'
        p_a_col, odd_a_col = 'p_ML+OVER(map/map)', 'odd_ML+OVER(map/map)'
        p_b_col, odd_b_col = 'p_ML+UNDER(map/map)', 'odd_ML+UNDER(map/map)'
        table_df = _display_ml_totals_table(df_raw)
    else:
        label_a, label_b = 'Acima', 'Abaixo'
        p_a_col, odd_a_col = 'p_over', 'odd_over'
        p_b_col, odd_b_col = 'p_under', 'odd_under'
        table_df = _display_ou_table(df_raw)

    def _safe_f(x):
        try:
            xf = float(x)
        except Exception:
            return float('nan')
        return xf if math.isfinite(xf) else float('nan')

    for _, row in df_raw.iterrows():
        line = row.get('line')
        p_a = _safe_f(row.get(p_a_col))
        p_b = _safe_f(row.get(p_b_col))
        odd_a = row.get(odd_a_col)
        odd_b = row.get(odd_b_col)

        fav_a = None
        if math.isfinite(p_a) and math.isfinite(p_b):
            fav_a = bool(p_a >= p_b)

        c1, c2, c3 = st.columns([1.05, 1.55, 1.55], gap='small')

        with c1:
            st.markdown(
                f"""<div class='b2t-row'><div class='b2t-card b2t-mono'>
                    <div class='b2t-title'>Linha</div>
                    <div class='b2t-mid'>{line}</div>
                </div></div>""",
                unsafe_allow_html=True,
            )

        with c2:
            cls = 'b2t-card fav' if fav_a is True else 'b2t-card'
            st.markdown(
                f"""<div class='b2t-row'><div class='{cls} b2t-mono'>
                    <div class='b2t-title'>{label_a}</div>
                    <div class='b2t-big'>{_format_num(odd_a, 2)}</div>
                    <div class='b2t-sub'>{_format_pct(p_a)}</div>
                </div></div>""",
                unsafe_allow_html=True,
            )

        with c3:
            cls = 'b2t-card fav' if fav_a is False else 'b2t-card'
            st.markdown(
                f"""<div class='b2t-row'><div class='{cls} b2t-mono'>
                    <div class='b2t-title'>{label_b}</div>
                    <div class='b2t-big'>{_format_num(odd_b, 2)}</div>
                    <div class='b2t-sub'>{_format_pct(p_b)}</div>
                </div></div>""",
                unsafe_allow_html=True,
            )

    with st.expander(str(table_label)):
        st.dataframe(table_df, width='stretch')


def _lines_table(metric: str, lines_list: List[float], total_obj) -> pd.DataFrame:
    """Gera tabela de odds justas (Over/Under) e, se disponÃ­vel, combos ML+Over/Under.

    - Over/Under: usa total_over_prob(total_obj, line)
    - ML+Totais: baseline independÃªncia => P(ML & Over) = P(ML) * P(Over)
    """
    lines_list = list(lines_list or [])
    if not lines_list:
        return pd.DataFrame(columns=[
            "line", "p_over", "odd_over", "p_under", "odd_under",
            "p_ML+OVER(map/map)", "odd_ML+OVER(map/map)",
            "p_ML+UNDER(map/map)", "odd_ML+UNDER(map/map)",
        ])

    p_ml = st.session_state.get("_p_ml_scope", float("nan"))
    rows = []
    for ln in lines_list:
        try:
            line = float(ln)
        except Exception:
            continue

        p_over = float(total_over_prob(total_obj, line))
        p_under = float(1.0 - p_over) if math.isfinite(p_over) else float("nan")

        row = {
            "line": _min_to_mmss(line) if metric == "time" else line,
            "p_over": p_over,
            "odd_over": _odd_from_p(p_over),
            "p_under": p_under,
            "odd_under": _odd_from_p(p_under),
            "p_ML+OVER(map/map)": float("nan"),
            "odd_ML+OVER(map/map)": float("nan"),
            "p_ML+UNDER(map/map)": float("nan"),
            "odd_ML+UNDER(map/map)": float("nan"),
        }

        if math.isfinite(p_ml):
            p_ml_over = float(p_ml) * float(p_over) if math.isfinite(p_over) else float("nan")
            p_ml_under = float(p_ml) * float(p_under) if math.isfinite(p_under) else float("nan")
            row["p_ML+OVER(map/map)"] = p_ml_over
            row["odd_ML+OVER(map/map)"] = _odd_from_p(p_ml_over)
            row["p_ML+UNDER(map/map)"] = p_ml_under
            row["odd_ML+UNDER(map/map)"] = _odd_from_p(p_ml_under)

        rows.append(row)

    df = pd.DataFrame(rows)
    # ordena por linha se for numÃ©rico
    try:
        if metric != "time":
            df = df.sort_values("line", ascending=True).reset_index(drop=True)
    except Exception:
        pass
    return df


def _lines_table_with_p_ml(metric: str, lines_list: List[float], total_obj, p_ml_value: float) -> pd.DataFrame:
    """Wrapper seguro para gerar tabela de linhas usando um p(ML) especÃ­fico.

    O app guarda o p(ML) corrente em st.session_state['_p_ml_scope'].
    Para mostrar ML+Totais pros dois times (Time 1 e Time 2), a gente precisa
    calcular a mesma tabela com p_ml e com (1 - p_ml), sem mexer na lÃ³gica.
    """
    old = st.session_state.get('_p_ml_scope', float('nan'))
    try:
        st.session_state['_p_ml_scope'] = float(p_ml_value) if p_ml_value is not None else float('nan')
        return _lines_table(metric, lines_list, total_obj)
    finally:
        st.session_state['_p_ml_scope'] = old


def _parse_quick_lines_kv(text: str) -> Dict[str, str]:
    """Parseia uma linha Ãºnica de linhas por mercado.

    Exemplo:
        kills=29.5 towers=10.5 drg=4.5 bar=1.5 inib=1.5 time=33:00

    Retorna dict com chaves canÃ´nicas: kills,towers,dragons,barons,inhibitors,time
    (valores como strings, para alimentar os text_inputs).
    """
    text = (text or "").strip()
    if not text:
        return {}

    # Normaliza separadores
    text = text.replace(",", " ")
    tokens = [t for t in re.split(r"\s+", text) if t.strip()]
    out: Dict[str, str] = {}

    aliases = {
        "k": "kills",
        "kill": "kills",
        "kills": "kills",
        "t": "towers",
        "tower": "towers",
        "towers": "towers",
        "drg": "dragons",
        "drag": "dragons",
        "dragon": "dragons",
        "dragons": "dragons",
        "d": "dragons",
        "bar": "barons",
        "baron": "barons",
        "barons": "barons",
        "b": "barons",
        "inib": "inhibitors",
        "inibs": "inhibitors",
        "inhib": "inhibitors",
        "inhibitor": "inhibitors",
        "inhibitors": "inhibitors",
        "i": "inhibitors",
        "time": "time",
        "tempo": "time",
        "tm": "time",
    }

    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = aliases.get(k.strip().lower(), None)
        if not k:
            continue
        v = v.strip()
        if not v:
            continue
        out[k] = v

    return out



def _parse_single_float(text: str) -> Optional[float]:
    vals = _parse_lines(text)
    return vals[0] if vals else None


def _parse_single_time_min(text: str) -> Optional[float]:
    vals = _parse_time_lines(text)
    return vals[0] if vals else None

def _odd_from_p(p: float) -> Optional[float]:
    return market_odd_from_prob(p)

def _fuse_probs_by_precision(
    *,
    p_team: float,
    p_players: float,
    n_team: float,
    n_players: float,
    team_scale: float = 1.0,
    players_scale: float = 1.0,
    coverage: Optional[float] = None,
    transfer_signal: float = 0.0,
    season_phase: str = "auto",
) -> Dict[str, Any]:
    nt = float(n_team) if math.isfinite(float(n_team)) else 0.0
    np_ = float(n_players) if math.isfinite(float(n_players)) else 0.0
    ts = float(team_scale) if math.isfinite(float(team_scale)) else 1.0
    ps = float(players_scale) if math.isfinite(float(players_scale)) else 1.0
    nt = max(0.0, nt * max(0.0, ts))
    np_ = max(0.0, np_ * max(0.0, ps))
    return core_fuse_with_coherence_guard(
        p_team=float(p_team),
        p_players=float(p_players),
        n_team=float(nt),
        n_players=float(np_),
        coverage=coverage,
        transfer_signal=float(transfer_signal),
        season_phase=str(season_phase or "auto"),
        team_weight_early=float(st.session_state.get("fusion_team_weight_early", 0.58) or 0.58),
        team_weight_mid=float(st.session_state.get("fusion_team_weight_mid", 0.75) or 0.75),
        team_weight_playoffs=float(st.session_state.get("fusion_team_weight_playoffs", 0.82) or 0.82),
        early_season_team_weight=float(st.session_state.get("fusion_early_season_team_weight", 0.55) or 0.55),
        season_knee_games=float(st.session_state.get("fusion_season_knee_games", 18.0) or 18.0),
        coverage_power=float(st.session_state.get("fusion_coverage_power", 1.5) or 1.5),
        transfer_boost=float(st.session_state.get("fusion_transfer_boost", 0.35) or 0.35),
        divergence_pp_cap=float(st.session_state.get("fusion_divergence_pp_cap", 22.0) or 22.0),
        divergence_low_coverage=float(st.session_state.get("fusion_divergence_low_coverage", 0.30) or 0.30),
        divergence_shrink=float(st.session_state.get("fusion_divergence_shrink", 0.35) or 0.35),
    )


def _delta_players_effect(delta_raw: float, *, mode: str = "saturado", cap: float = 1.2, slope: float = 1.6) -> float:
    """Transforms raw players delta before applying on logit/diff.

    - linear: effect = delta_raw
    - saturado: effect = cap * tanh(slope * delta_raw / cap)
    """
    try:
        d = float(delta_raw)
    except Exception:
        return 0.0
    if not math.isfinite(d):
        return 0.0
    m = str(mode or "saturado").strip().lower()
    if m == "linear":
        return float(d)
    c = float(cap) if math.isfinite(float(cap)) else 1.2
    s = float(slope) if math.isfinite(float(slope)) else 1.6
    c = max(0.05, abs(c))
    s = max(0.1, abs(s))
    return float(c * math.tanh((s * d) / c))


def _format_pct(p: Optional[float]) -> str:
    if p is None or not math.isfinite(p):
        return "-"
    return f"{100.0*p:.1f}%"


def _format_num(x: Optional[float], nd: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    return f"{x:.{nd}f}"





def _totals_table(tot) -> pd.DataFrame:
    """Tabela simples (mÃ©dia/DP) por mÃ©trica.

    `tot` Ã© um dict {metric: MatchupTotals}.
    ObservaÃ§Ã£o: `time` Ã© exibido como MM:SS.
    """
    metric_pt = {
        "kills": "Kills",
        "towers": "Torres",
        "dragons": "DragÃµes",
        "barons": "BarÃµes",
        "inhibitors": "Inibidores",
        "time": "Tempo",
    }

    rows = []
    for k in ["kills", "towers", "dragons", "barons", "inhibitors", "time"]:
        if k not in tot:
            continue
        t = tot[k]
        rows.append(
            {
                "MÃ©trica": metric_pt.get(k, k),
                "MÃ©dia": getattr(t, "mean", None),
                "DP": getattr(t, "sd", None),
                "Dist": getattr(t, "dist", None),
            }
        )

    df = pd.DataFrame(rows)

    def _clean(x):
        try:
            xf = float(x)
        except Exception:
            return None
        return None if (not math.isfinite(xf)) else xf

    if not df.empty:
        if "MÃ©dia" in df.columns:
            df["MÃ©dia"] = df["MÃ©dia"].apply(_clean)
        if "DP" in df.columns:
            df["DP"] = df["DP"].apply(_clean)

        # Formata tempo como MM:SS (inclusive o DP)
        if "MÃ©trica" in df.columns:
            mask = df["MÃ©trica"] == "Tempo"
            if mask.any():
                # Evita warning de dtype ao misturar float e string na mesma coluna.
                df["MÃ©dia"] = df["MÃ©dia"].astype(object)
                df["DP"] = df["DP"].astype(object)
                df.loc[mask, "MÃ©dia"] = df.loc[mask, "MÃ©dia"].apply(_min_to_mmss)
                df.loc[mask, "DP"] = df.loc[mask, "DP"].apply(_min_to_mmss)

        # Para as demais mÃ©tricas, deixa mais legÃ­vel (2 casas)
        mask_not_time = df.get("MÃ©trica", "") != "Tempo"
        if "MÃ©dia" in df.columns:
            df.loc[mask_not_time, "MÃ©dia"] = df.loc[mask_not_time, "MÃ©dia"].apply(lambda x: _format_num(x, 2) if x is not None else "-")
        if "DP" in df.columns:
            df.loc[mask_not_time, "DP"] = df.loc[mask_not_time, "DP"].apply(lambda x: _format_num(x, 2) if x is not None else "-")

    return df

def _series_totals_table(tot_avg) -> pd.DataFrame:
    """Tabela de totais da sÃ©rie (Bo3/Bo5) via Monte Carlo, por mÃ©trica.

    Usa `totals_mode`, `bo`, `p_map_cal`, `n_sims` e `_series_sims_cached` do escopo global.
    `tot_avg` deve ser o dict de totais por mapa (AVG maps).
    ObservaÃ§Ã£o: `time` Ã© exibido como mm:ss.
    """
    metric_pt = {
        "kills": "kills",
        "towers": "torres",
        "dragons": "dragÃµes",
        "barons": "barÃµes",
        "inhibitors": "inibidores",
        "time": "tempo",
    }

    rows = []
    use_series = (totals_mode == "series" and int(bo) > 1 and math.isfinite(float(p_map_cal)))

    for k in ["kills", "towers", "dragons", "barons", "inhibitors", "time"]:
        if k not in tot_avg:
            continue
        t = tot_avg[k]
        mean = getattr(t, "mean", None)
        sd = getattr(t, "sd", None)
        dist = getattr(t, "dist", None)

        if use_series:
            sims = _series_sims_cached(
                int(bo),
                float(p_map_cal),
                t,
                int(n_sims),
                int(st.session_state.get("_sim_seed_series", 1337) or 1337),
            )
            mean = float(np.mean(sims)) if len(sims) else float("nan")
            sd = float(np.std(sims, ddof=1)) if len(sims) > 1 else 0.0
            dist = "series_mc"

        # clean
        try:
            mean_f = float(mean)
        except Exception:
            mean_f = float("nan")
        try:
            sd_f = float(sd)
        except Exception:
            sd_f = float("nan")

        row = {"mÃ©trica": metric_pt.get(k, k), "mÃ©dia": mean_f, "dp": sd_f, "dist": dist}
        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty and "mÃ©trica" in df.columns:
        mask = df["mÃ©trica"] == "tempo"
        if mask.any():
            df.loc[mask, "mÃ©dia"] = df.loc[mask, "mÃ©dia"].apply(_min_to_mmss)
            df.loc[mask, "dp"] = df.loc[mask, "dp"].apply(_min_to_mmss)

    return df


def _min_to_mmss(x: Optional[float]) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    sec = int(round(float(x) * 60.0))
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"


# ---- Display helpers (PT labels / time formatting) ----
_METRIC_PT = {
    'kills': 'Kills',
    'towers': 'Torres',
    'dragons': 'DragÃµes',
    'barons': 'BarÃµes',
    'inhibitors': 'Inibidores',
    'time': 'Tempo',
}

def _metric_label_pt(metric_key: str) -> str:
    k = str(metric_key).strip().lower()
    return _METRIC_PT.get(k, str(metric_key).strip().title())

_SIDE_PT = {
    'over': 'Acima',
    'under': 'Abaixo',
}

def _side_label_pt(side: str) -> str:
    s = str(side).strip().lower()
    return _SIDE_PT.get(s, str(side).strip().title())

def _fmt_line_disp(metric_key: str, line_val) -> str:
    # Converte linha p/ string amigÃ¡vel. Tempo sempre em MM:SS.
    try:
        v = float(line_val)
    except Exception:
        return str(line_val)
    if str(metric_key).strip().lower() == 'time':
        return _min_to_mmss(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return (f'{v:.2f}').rstrip('0').rstrip('.')



def _fmt_recortes_disp(items) -> str:
    """Formata lista de recortes (ex.: ['Ano','5'] -> 'Ano, 5')."""
    if not items:
        return '-'
    out = []
    for it in items:
        s = str(it).strip()
        if s.lower() == 'ano':
            s = 'Ano'
        out.append(s)
    return ', '.join(out) if out else '-'
def _team_profile_table(p) -> pd.DataFrame:
    """Tabela estilo gol.gg: mÃ©tricas 'for' e 'against' do time."""
    rows = [
        {"metric": "kills", "for": p.kills_for, "against": p.kills_against, "sd_for": p.kills_for_sd, "sd_against": p.kills_against_sd},
        {"metric": "towers", "for": p.towers_for, "against": p.towers_against, "sd_for": p.towers_for_sd, "sd_against": p.towers_against_sd},
        {"metric": "dragons", "for": p.dragons_for, "against": p.dragons_against, "sd_for": p.dragons_for_sd, "sd_against": p.dragons_against_sd},
        {"metric": "barons", "for": p.barons_for, "against": p.barons_against, "sd_for": p.barons_for_sd, "sd_against": p.barons_against_sd},
        {"metric": "inhibitors", "for": p.inhib_for, "against": p.inhib_against, "sd_for": p.inhib_for_sd, "sd_against": p.inhib_against_sd},
        {"metric": "time", "for": p.time_min, "against": None, "sd_for": p.time_min_sd, "sd_against": None},
    ]
    df = pd.DataFrame(rows)
    def clean(x):
        try:
            return None if (x is None or not math.isfinite(float(x))) else float(x)
        except Exception:
            return None
    for c in ["for", "against", "sd_for", "sd_against"]:
        df[c] = df[c].apply(clean)

    # display-friendly columns (time em MM:SS)
    disp = df.copy()
    disp["for"] = disp.apply(lambda r: _min_to_mmss(r["for"]) if r["metric"] == "time" else r["for"], axis=1)
    disp["against"] = disp.apply(lambda r: _min_to_mmss(r["against"]) if r["metric"] == "time" else r["against"], axis=1)
    disp["sd_for"] = disp.apply(lambda r: _min_to_mmss(r["sd_for"]) if r["metric"] == "time" else r["sd_for"], axis=1)
    disp["sd_against"] = disp.apply(lambda r: _min_to_mmss(r["sd_against"]) if r["metric"] == "time" else r["sd_against"], axis=1)

    return disp[["metric", "for", "against", "sd_for", "sd_against"]]
# -----------------------------
# UI (fluxo: Times -> Linhas -> Resultados)
# -----------------------------
st.set_page_config(page_title="Plays (ML + Totais)", layout="wide")
_inject_global_ui_css()
st.markdown(
    """
<div class="gb-hero">
  <div class="title">GABINETE</div>
  <div class="sub">dev: Bolin</div>
</div>
    """,
    unsafe_allow_html=True,
)
st.caption("Fluxo: escolher Season/Times -> preencher linhas -> ver resultados (modelo + histórico/Laplace).")

APP_ROOT = Path(__file__).resolve().parent
_FIT_PARAMS_PATH = APP_ROOT / "params_fitted.json"


def _persist_combined_trace_file() -> None:
    """Persist unified trace (ml base + resumo markets + players) into ml_trace_last.json."""
    try:
        ml_base = st.session_state.get("_ml_trace_last")
        markets = st.session_state.get("_ml_trace_markets", [])
        players = st.session_state.get("_ml_trace_players", [])
        payload = core_build_combined_trace(
            ml_base_trace=ml_base,
            resumo_markets_trace=markets,
            players_trace=players,
        )
        # Metadata operacional para auditoria/reprodução.
        payload["meta"] = {
            "app_version": "gabinete-2026.02.12",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "season": str(st.session_state.get("season_label", "")),
            "teams": {
                "teamA": str(st.session_state.get("teamA", "")),
                "teamB": str(st.session_state.get("teamB", "")),
                "bo": int(st.session_state.get("bo", 1) or 1),
            },
            "filters": {
                "year": str(st.session_state.get("year_opt", "")),
                "split": str(st.session_state.get("split_opt", "")),
                "playoffs": str(st.session_state.get("playoffs_opt", "")),
                "map_mode": str(st.session_state.get("map_mode", "")),
                "league_mode": str(st.session_state.get("league_mode", "")),
            },
            "calibration": {
                "ml_engine": str(st.session_state.get("ml_engine", "")),
                "microseg_enabled": bool(st.session_state.get("ml_microseg_enabled", True)),
                "microseg_strength": float(st.session_state.get("ml_microseg_strength", 0.60) or 0.60),
                "wf_half_life_days": float(st.session_state.get("wf_half_life_days", 180.0) or 180.0),
                "series_gamma_on": bool(st.session_state.get("series_gamma_on", True)),
            },
        }
        st.session_state["_ml_trace_combined"] = payload
        (APP_ROOT / "ml_trace_last.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

_fit_cfg = {}
try:
    if _FIT_PARAMS_PATH.exists():
        _fit_cfg = _load_fitted_params_cached(_file_sig(str(_FIT_PARAMS_PATH))) or {}
except Exception:
    _fit_cfg = {}

_fit_team_scale = 1.0
_fit_players_scale = 1.0
try:
    _fit_team_scale = float(((_fit_cfg.get("best") or {}).get("team_scale")) or 1.0)
except Exception:
    _fit_team_scale = 1.0
try:
    _fit_players_scale = float(((_fit_cfg.get("best") or {}).get("players_scale")) or 1.0)
except Exception:
    _fit_players_scale = 1.0
_fit_team_scale = max(0.0, _fit_team_scale)
_fit_players_scale = max(0.0, _fit_players_scale)
st.session_state["_fit_team_scale"] = float(_fit_team_scale)
st.session_state["_fit_players_scale"] = float(_fit_players_scale)
st.session_state["_fit_team_scale_used"] = float(_fit_team_scale)
st.session_state["_fit_players_scale_used"] = float(_fit_players_scale)
st.session_state["_fit_scale_league_used"] = ""
_fit_form_cfg: Dict[str, Any] = {}
try:
    _fit_form_cfg = dict(((_fit_cfg.get("best_form") or {}).get("settings")) or {})
except Exception:
    _fit_form_cfg = {}
st.session_state["_fit_form_cfg"] = dict(_fit_form_cfg)
st.session_state["_fit_form_cfg_used"] = dict(_fit_form_cfg)
st.session_state["_fit_form_cfg_league_used"] = ""
st.session_state["_fit_series_gamma_cfg"] = {}
try:
    st.session_state["_fit_series_gamma_cfg"] = {
        "global": dict(_fit_cfg.get("series_gamma_global") or {}) if isinstance(_fit_cfg, dict) else {},
        "by_league": dict(_fit_cfg.get("series_gamma_by_league") or {}) if isinstance(_fit_cfg, dict) else {},
        "by_macro": dict(_fit_cfg.get("series_gamma_by_macro") or {}) if isinstance(_fit_cfg, dict) else {},
    }
except Exception:
    st.session_state["_fit_series_gamma_cfg"] = {"global": {}, "by_league": {}, "by_macro": {}}

def _fit_scales_for_league(league_hint: str) -> tuple[float, float, str]:
    lg = str(league_hint or "").strip()
    if not isinstance(_fit_cfg, dict):
        return float(_fit_team_scale), float(_fit_players_scale), ""
    by_lg = _fit_cfg.get("best_by_league") or {}
    if isinstance(by_lg, dict) and lg:
        row = by_lg.get(lg)
        if isinstance(row, dict):
            try:
                ts = float(row.get("team_scale", _fit_team_scale))
                ps = float(row.get("players_scale", _fit_players_scale))
                return max(0.0, ts), max(0.0, ps), lg
            except Exception:
                pass
    return float(_fit_team_scale), float(_fit_players_scale), ""


def _fit_form_for_league(league_hint: str) -> tuple[Dict[str, Any], str]:
    lg = str(league_hint or "").strip()
    base = dict(_fit_form_cfg) if isinstance(_fit_form_cfg, dict) else {}
    if not isinstance(_fit_cfg, dict):
        return base, ""
    def _macro_key(league: str) -> str:
        z = str(league or "").strip().upper()
        if not z:
            return "OTHER"
        if z.startswith("LPL") or z in {"LDL", "LPL CL"}:
            return "CN"
        if z.startswith("LCK") or z in {"LCK CL"}:
            return "KR"
        if z.startswith("LEC") or z.startswith("EMEA") or z in {"LFL", "LVP SL", "PRIME LEAGUE", "NLC", "UL"}:
            return "EMEA"
        if z.startswith("LCS") or z in {"NACL", "LTA N", "LTA NORTH"}:
            return "NA"
        if z.startswith("CBLOL") or z.startswith("LTA S") or z.startswith("LTA SOUTH"):
            return "BR"
        if z.startswith("PCS") or z.startswith("VCS") or z.startswith("LJL") or z.startswith("LCO"):
            return "APAC"
        return "OTHER"

    by_lg = _fit_cfg.get("best_form_by_league") or {}
    if isinstance(by_lg, dict) and lg:
        row = by_lg.get(lg)
        if isinstance(row, dict):
            s = row.get("settings")
            if isinstance(s, dict) and s:
                return dict(s), lg
    by_mk = _fit_cfg.get("best_form_by_macro") or {}
    if isinstance(by_mk, dict):
        mk = _macro_key(lg)
        row = by_mk.get(mk)
        if isinstance(row, dict):
            s = row.get("settings")
            if isinstance(s, dict) and s:
                return dict(s), f"macro:{mk}"
    return base, ""


def _series_gamma_for_league(league_hint: str, bo: int) -> tuple[float, str]:
    try:
        b = int(bo)
    except Exception:
        b = 1
    if b not in (3, 5):
        return 1.0, ""
    k = f"bo{b}"
    cfg = st.session_state.get("_fit_series_gamma_cfg") if isinstance(st.session_state.get("_fit_series_gamma_cfg"), dict) else {}
    lg = str(league_hint or "").strip()

    def _macro_key(league: str) -> str:
        z = str(league or "").strip().upper()
        if not z:
            return "OTHER"
        if z.startswith("LPL") or z in {"LDL", "LPL CL"}:
            return "CN"
        if z.startswith("LCK") or z in {"LCK CL", "LCKC"}:
            return "KR"
        if z.startswith("LEC") or z.startswith("EMEA") or z in {"LFL", "LVP SL", "PRIME LEAGUE", "NLC", "UL", "PRM"}:
            return "EMEA"
        if z.startswith("LCS") or z in {"NACL", "LTA N", "LTA NORTH"}:
            return "NA"
        if z.startswith("CBLOL") or z.startswith("LTA S") or z.startswith("LTA SOUTH"):
            return "BR"
        if z.startswith("PCS") or z.startswith("VCS") or z.startswith("LJL") or z.startswith("LCO") or z.startswith("LCP"):
            return "APAC"
        return "OTHER"

    by_lg = cfg.get("by_league") if isinstance(cfg, dict) else {}
    if isinstance(by_lg, dict) and lg:
        row = by_lg.get(lg)
        if isinstance(row, dict):
            rr = row.get(k)
            if isinstance(rr, dict) and rr.get("gamma") is not None:
                try:
                    return max(0.25, float(rr.get("gamma"))), f"league:{lg}"
                except Exception:
                    pass
    by_mk = cfg.get("by_macro") if isinstance(cfg, dict) else {}
    mk = _macro_key(lg)
    if isinstance(by_mk, dict):
        row = by_mk.get(mk)
        if isinstance(row, dict):
            rr = row.get(k)
            if isinstance(rr, dict) and rr.get("gamma") is not None:
                try:
                    return max(0.25, float(rr.get("gamma"))), f"macro:{mk}"
                except Exception:
                    pass
    gg = cfg.get("global") if isinstance(cfg, dict) else {}
    if isinstance(gg, dict):
        rr = gg.get(k)
        if isinstance(rr, dict) and rr.get("gamma") is not None:
            try:
                return max(0.25, float(rr.get("gamma"))), "global"
            except Exception:
                pass
    return 1.0, ""


def _apply_series_gamma(p_series: float, gamma: float) -> float:
    try:
        p0 = float(clip_prob(float(p_series)))
        g = max(0.25, float(gamma))
        z = math.log(p0 / (1.0 - p0))
        return float(clip_prob(1.0 / (1.0 + math.exp(-(z * g)))))
    except Exception:
        return float(clip_prob(float(p_series)))


def _list_csv_candidates(folder: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if not folder.exists():
        return out
    for p in sorted(folder.glob("*.csv")):
        try:
            st_ = p.stat()
            out.append({"name": p.name, "path": str(p), "mtime": int(st_.st_mtime), "size": int(st_.st_size)})
        except Exception:
            out.append({"name": p.name, "path": str(p), "mtime": 0, "size": 0})
    # mais novo primeiro
    out.sort(key=lambda x: (x.get("mtime", 0), x.get("size", 0)), reverse=True)
    return out


def _boot_select_csvs(root: Path) -> None:
    """Tela inicial para selecionar CSV diÃ¡rio + (opcional) CSVs histÃ³ricos para treino.

    - DiÃ¡rio: root/CSV_DIARIO/*.csv (default: mais novo)
    - HistÃ³rico: root/CSV_HISTORICO/*.csv (multi-select)
    """
    if st.session_state.get("boot_done", False):
        return

    # Fluxo padrão: não mostrar essa tela se já conseguimos inferir um CSV diário válido.
    # A tela só reaparece quando o usuário clicar em "Trocar CSVs".
    _force_boot_ui = bool(st.session_state.get("boot_show_selector", False))
    if not _force_boot_ui:
        _daily_dir = root / "CSV_DIARIO"
        _hist_dir = root / "CSV_HISTORICO"
        _daily_auto = _list_csv_candidates(_daily_dir)
        _hist_auto = _list_csv_candidates(_hist_dir)

        _cfg_csv = str(st.session_state.get("cfg_csv_override", "") or "").strip()
        _cfg_csv_ok = bool(_cfg_csv) and Path(_cfg_csv).exists()

        if _daily_auto:
            _daily_choice = str(_daily_auto[0]["path"])
            st.session_state["boot_daily_csv"] = _daily_choice
            st.session_state["boot_hist_csvs"] = [str(h.get("path")) for h in _hist_auto if str(h.get("path", "")).strip()]
            st.session_state["cfg_auto_paths"] = False
            st.session_state["cfg_csv_override"] = _daily_choice
            st.session_state["boot_done"] = True
            return
        if _cfg_csv_ok:
            st.session_state["boot_daily_csv"] = _cfg_csv
            st.session_state["boot_hist_csvs"] = [str(h.get("path")) for h in _hist_auto if str(h.get("path", "")).strip()]
            st.session_state["cfg_auto_paths"] = False
            st.session_state["boot_done"] = True
            return

    st.markdown("## Selecione os CSVs")
    st.caption("Antes de escolher times/linhas: selecione o CSV **diÃ¡rio** (para analisar) e, se quiser, os CSVs **histÃ³ricos** (para treinar o ML).")

    daily_dir = root / "CSV_DIARIO"
    hist_dir  = root / "CSV_HISTORICO"

    daily = _list_csv_candidates(daily_dir)
    hist  = _list_csv_candidates(hist_dir)

    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        st.markdown("### CSV diÃ¡rio (obrigatÃ³rio)")
        if not daily:
            st.warning(f"NÃ£o encontrei CSVs em {daily_dir}. Coloque seu CSV diÃ¡rio lÃ¡ ou informe um caminho abaixo.")
            manual_daily = st.text_input("Caminho do CSV diÃ¡rio", value=str(st.session_state.get("boot_daily_csv", "")))
            daily_choice = manual_daily.strip()
        else:
            opts = [d["name"] for d in daily]
            default = 0
            daily_name = st.selectbox("Arquivos em CSV_DIARIO", opts, index=default)
            daily_choice = next((d["path"] for d in daily if d["name"] == daily_name), daily[0]["path"])
            st.caption(f"Selecionado: `{Path(daily_choice).name}`")

    with c2:
        st.markdown("### CSVs histÃ³ricos (opcional â€” para treino)")
        if not hist:
            st.info(f"NÃ£o encontrei CSVs em {hist_dir}. Se vocÃª tiver (ex.: 2014/2015), coloque lÃ¡.")
            hist_choice = []
        else:
            opts_h = [h["name"] for h in hist]
            # default: todos (caso tÃ­pico 2014 + 2015)
            default_h = opts_h[:]
            sel = st.multiselect("Arquivos em CSV_HISTORICO", opts_h, default=default_h)
            hist_choice = [next(h["path"] for h in hist if h["name"] == nm) for nm in sel]

        if hist_choice:
            st.caption(f"HistÃ³rico selecionado: {len(hist_choice)} arquivo(s)")

    st.divider()
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        if st.button("Continuar", width='stretch'):
            if not str(daily_choice or "").strip():
                st.error("Selecione um CSV diÃ¡rio para continuar.")
            else:
                st.session_state["boot_daily_csv"] = str(daily_choice)
                st.session_state["boot_hist_csvs"] = list(hist_choice)
                # forÃ§a o app a usar o diÃ¡rio como fonte ativa (override)
                st.session_state["cfg_auto_paths"] = False
                st.session_state["cfg_csv_override"] = str(daily_choice)
                st.session_state["boot_done"] = True
                st.session_state["boot_show_selector"] = False
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
    with col_b:
        if st.button("Recarregar lista", width='stretch'):
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with col_c:
        st.caption("Dica: coloque o diÃ¡rio em `CSV_DIARIO/` e 2014/2015 em `CSV_HISTORICO/`.")




# -----------------------------
# Boot: carregar config salva + sidebar (antes do fluxo)
# -----------------------------
_init_app_settings_once()
_render_settings_sidebar()


# Tela inicial (antes de qualquer fluxo)
_boot_select_csvs(APP_ROOT)

# NavegaÃ§Ã£o (centraliza calibraÃ§Ã£o/retreino em um lugar sÃ³)
# --- hotfix: apply pending navigation before nav_main widget is instantiated ---
if st.session_state.get("_nav_pending"):
    st.session_state["nav_main"] = st.session_state["_nav_pending"]
    del st.session_state["_nav_pending"]

if str(st.session_state.get("nav_main", "") or "").strip() == "Calibrações":
    st.session_state["nav_main"] = "Técnico"
nav_page = st.radio("Página", ["Plays", "Técnico"], horizontal=True, key="nav_main")

def _season_from_year(year: int) -> str:
    # S15 ~ 2025 (2010 + 15 = 2025)
    return f"S{int(year) - 2010}"


def _discover_season_csvs(root: Path) -> List[Dict[str, object]]:
    """Procura CSVs OracleElixir (team rows) no projeto e monta catÃ¡logo de seasons."""
    candidates: List[Path] = []
    # root e subpastas comuns
    for pat in [
        "*LoL_esports_match_data_from_OraclesElixir.csv",
        "CSV_*/*LoL_esports_match_data_from_OraclesElixir.csv",
        "seasons/*/*LoL_esports_match_data_from_OraclesElixir.csv",
        "seasons/*LoL_esports_match_data_from_OraclesElixir.csv",
        "data/*LoL_esports_match_data_from_OraclesElixir.csv",
    ]:
        candidates.extend(root.glob(pat))

    out: Dict[str, Dict[str, object]] = {}
    for p in candidates:
        name = p.name
        m = re.match(r"(\d{4})_LoL_esports_match_data_from_OraclesElixir\.csv$", name)
        if not m:
            continue
        year = int(m.group(1))
        season = _season_from_year(year)
        key = season
        try:
            st_ = p.stat()
            mtime = int(st_.st_mtime)
            size = int(st_.st_size)
        except Exception:
            mtime, size = 0, 0

        cur = out.get(key)
        if cur is None or int(cur.get("mtime", 0)) < mtime:
            out[key] = {"season": season, "year": year, "csv_path": str(p), "mtime": mtime, "size": size}

    # Ordena por ano
    seasons = sorted(out.values(), key=lambda d: int(d.get("year", 0)))
    return seasons


def _default_artifact_for(root: Path, season: str, year: int) -> str:
    # procura em alguns lugares comuns
    cand = [
        root / "ml_artifact.json",
        root / f"ml_artifact_{year}.json",
        root / "artifacts" / "ml_artifact.json",
        root / "seasons" / season / "ml_artifact.json",
        root / "seasons" / f"{year}" / "ml_artifact.json",
    ]
    for p in cand:
        if p.exists():
            return str(p)
    return str(root / "ml_artifact.json")


def _reset_flow(keep_config: bool = True):
    # Reseta fluxo (mas preserva configs se quiser)
    st.session_state["flow_stage"] = "select"
    # linhas
    for k in [
        "lines_kills", "lines_towers", "lines_dragons", "lines_barons", "lines_inhib", "lines_time",
        "teamA_kills_line_text", "teamB_kills_line_text",
        "hc_kills_text", "hc_kills_team", "hc_towers_text", "hc_dragons_text",
        "combo_kills_text", "combo_time_text",
    ]:
        st.session_state.pop(k, None)
    # filtros/inputs de resultado
    for k in [
        "totals_mode", "ml_mode", "map_mode_ui", "window_opt", "window_short",
        "league_mode", "year_opt", "split_opt", "playoffs_opt", "fixed_league",
        "weight_mode", "half_life_days", "n_sims",
    ]:
        st.session_state.pop(k, None)

    if not keep_config:
        for k in ["cfg_auto_paths", "cfg_csv_override", "cfg_artifact_override"]:
            st.session_state.pop(k, None)



# Aplica carregamentos pendentes ANTES de instanciar widgets (evita erro de session_state em widgets como BO)
if "pending_load" in st.session_state:
    _pl = st.session_state.pop("pending_load") or {}
    try:
        if _pl.get("_reset_flow"):
            _reset_flow(keep_config=True)
    except Exception:
        pass
    for _k, _v in _pl.items():
        if str(_k).startswith("_"):
            continue
        st.session_state[_k] = _v
# -----------------------------
# Configurações (raramente muda)
# -----------------------------
seasons = _discover_season_csvs(APP_ROOT)

if nav_page == "Técnico":
    with st.expander("Configurações (raramente muda)", expanded=False):
        auto_paths = st.checkbox("Auto (por Season)", value=True, key="cfg_auto_paths")
        if not seasons:
            st.warning("NÃ£o encontrei CSVs no padrÃ£o 'YYYY_LoL_esports_match_data_from_OraclesElixir.csv'. Use o override abaixo.")
            auto_paths = False
            st.session_state["cfg_auto_paths"] = False

        if not auto_paths:
            st.text_input(
                "CSV OracleElixir (team rows)",
                value=st.session_state.get("cfg_csv_override", str(APP_ROOT / "2025_LoL_esports_match_data_from_OraclesElixir.csv")),
                key="cfg_csv_override",
            )
        st.text_input(
            "ml_artifact.json (opcional)",
            value=st.session_state.get("cfg_artifact_override", _default_artifact_for(APP_ROOT, seasons[-1]["season"], int(seasons[-1]["year"])) if seasons else str(APP_ROOT / "ml_artifact.json")),
            key="cfg_artifact_override",
        )
        st.caption("Se nÃ£o existir p_map para o confronto, o app continua (sÃ³ perde ML).")
        st.markdown("#### ML: Técnico / Retreino")
        st.info("Para evitar treinar toda hora e para você ter certeza quando treinou/retreinou, tudo de drift/retreino/update fica na aba Técnico (no topo).")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Limpar cache (CSV)", width='stretch'):
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        with c2:
            st.caption("Use isso se vocÃª trocou/atualizou o CSV mantendo o mesmo nome.")

        if st.button("Trocar CSVs (voltar para seleÃ§Ã£o inicial)"):
            for k in ["boot_done", "boot_daily_csv", "boot_hist_csvs"]:
                st.session_state.pop(k, None)
            st.session_state["cfg_auto_paths"] = False
            st.session_state["boot_show_selector"] = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()




def _paths_for_ml(root: Path) -> tuple[str, list[str], str]:
    daily = str(st.session_state.get("boot_daily_csv") or st.session_state.get("cfg_csv_override") or "").strip()
    hist = list(st.session_state.get("boot_hist_csvs") or [])
    art = str(st.session_state.get("cfg_artifact_override") or str(root / "ml_artifact.json")).strip()
    return daily, hist, art


def _safe_mtime(p: str) -> float | None:
    try:
        return float(Path(p).stat().st_mtime)
    except Exception:
        return None


def _read_artifact_summary(artifact_path: str) -> dict:
    p = Path(artifact_path)
    if not p.exists():
        return {"exists": False}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"exists": True, "error": "NÃ£o foi possÃ­vel ler o JSON (arquivo corrompido)"}
    meta = data.get("meta") or {}
    inc = meta.get("incremental") or {}
    return {
        "exists": True,
        "trained_at": meta.get("trained_at"),
        "data_max_date": meta.get("data_max_date"),
        "n_games_total": meta.get("n_games_total"),
        "added_games_last": inc.get("added_games") if isinstance(inc, dict) else None,
        "backfill_suspected": inc.get("backfill_suspected") if isinstance(inc, dict) else None,
    }


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _backup_file_if_exists(path_like: str | Path, *, tag: str = "manual") -> str:
    """Create timestamped backup under APP_ROOT/backups and return backup path or ''."""
    try:
        src = Path(path_like).expanduser()
        if not src.exists() or (not src.is_file()):
            return ""
        backup_dir = APP_ROOT / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(tag or "manual")).strip("_") or "manual"
        dst = backup_dir / f"{src.stem}.{safe_tag}.{ts}{src.suffix}"
        i = 1
        while dst.exists():
            dst = backup_dir / f"{src.stem}.{safe_tag}.{ts}.{i}{src.suffix}"
            i += 1
        shutil.copy2(src, dst)
        return str(dst)
    except Exception:
        return ""

@st.cache_data(show_spinner=False, ttl=1800)
def _run_walkforward_cached(csv_paths: tuple[str, ...], min_train_games: int, k: float, scale: float) -> dict:
    clean = [str(Path(p).expanduser()) for p in (csv_paths or ()) if str(p).strip()]
    return run_walkforward(
        clean,
        min_train_games=int(min_train_games),
        k=float(k),
        scale=float(scale),
    )


@st.cache_data(show_spinner=False, ttl=1800)
def _run_layer_validation_cached(
    csv_paths: tuple[str, ...],
    artifact_path: str,
    players_artifact_path: str,
    min_train_games: int,
    team_scale: float,
    players_scale: float,
    min_league_games: int,
) -> dict:
    clean = [str(Path(p).expanduser()) for p in (csv_paths or ()) if str(p).strip()]
    return run_layer_validation(
        artifact_path=str(Path(str(artifact_path)).expanduser()),
        players_artifact_path=str(Path(str(players_artifact_path)).expanduser()),
        csv_paths=clean,
        min_train_games=int(min_train_games),
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        min_league_games=int(min_league_games),
    )


@st.cache_data(show_spinner=False, ttl=1800)
def _run_calibration_diag_cached(
    csv_paths: tuple[str, ...],
    artifact_path: str,
    players_artifact_path: str,
    team_scale: float,
    players_scale: float,
    bins: int,
    min_league_games: int,
) -> dict:
    clean = [str(Path(p).expanduser()) for p in (csv_paths or ()) if str(p).strip()]
    return run_calibration_diagnostics(
        artifact_path=str(Path(str(artifact_path)).expanduser()),
        players_artifact_path=str(Path(str(players_artifact_path)).expanduser()),
        csv_paths=clean,
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        bins=int(bins),
        min_league_games=int(min_league_games),
    )


def _wf_gate_multiplier(report: Optional[Dict[str, Any]], league_hint: str = "") -> float:
    # Fator >1 deixa o gate mais exigente; <1 deixa mais permissivo.
    if not isinstance(report, dict):
        return 1.0
    factor = 1.0
    overall = report.get("overall") if isinstance(report.get("overall"), dict) else {}
    roi = overall.get("roi_proxy_mean")
    try:
        roi_f = float(roi)
        if math.isfinite(roi_f):
            if roi_f <= 0.0:
                factor *= 1.30
            elif roi_f < 0.01:
                factor *= 1.15
            elif roi_f > 0.03:
                factor *= 0.90
    except Exception:
        pass

    league_key = str(league_hint or "").strip()
    by_league = report.get("by_league") if isinstance(report.get("by_league"), dict) else {}
    if league_key and league_key in by_league and isinstance(by_league.get(league_key), dict):
        row = by_league.get(league_key) or {}
        try:
            n_lg = int(row.get("n_total") or 0)
        except Exception:
            n_lg = 0
        if n_lg >= 80:
            try:
                drift = float(row.get("drift_mean"))
                if math.isfinite(drift):
                    if drift > 0.02:
                        factor *= 1.10
                    elif drift < -0.02:
                        factor *= 0.95
            except Exception:
                pass
    return float(max(0.70, min(1.80, factor)))


def _wf_correct_p_map(
    report: Optional[Dict[str, Any]],
    p_in: float,
    league_hint: str = "",
    strength: float = 0.60,
) -> tuple[float, Dict[str, Any]]:
    info: Dict[str, Any] = {"applied": False}
    try:
        p0 = float(p_in)
    except Exception:
        return p_in, info
    if (not math.isfinite(p0)) or p0 <= 0.0 or p0 >= 1.0:
        return p_in, info
    if not isinstance(report, dict):
        return p0, info

    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0:
        return p0, info

    shrink = 1.0
    overall = report.get("overall") if isinstance(report.get("overall"), dict) else {}
    roi_g = overall.get("roi_proxy_mean")
    try:
        rg = float(roi_g)
        if math.isfinite(rg):
            if rg <= 0.0:
                shrink *= (1.0 - 0.10 * s)
            elif rg > 0.03:
                shrink *= (1.0 + 0.05 * s)
    except Exception:
        pass

    league_key = str(league_hint or "").strip()
    by_league = report.get("by_league") if isinstance(report.get("by_league"), dict) else {}
    if league_key and isinstance(by_league.get(league_key), dict):
        row = by_league.get(league_key) or {}
        try:
            n_lg = int(row.get("n_total") or 0)
        except Exception:
            n_lg = 0
        if n_lg >= 40:
            try:
                drift = float(row.get("drift_mean"))
                if math.isfinite(drift):
                    if drift > 0.02:
                        shrink *= (1.0 - 0.15 * s)
                    elif drift > 0.01:
                        shrink *= (1.0 - 0.08 * s)
                    elif drift < -0.02:
                        shrink *= (1.0 + 0.06 * s)
            except Exception:
                pass
            try:
                roi_l = float(row.get("roi_proxy_mean"))
                if math.isfinite(roi_l):
                    if roi_l < 0.0:
                        shrink *= (1.0 - 0.08 * s)
                    elif roi_l > 0.03:
                        shrink *= (1.0 + 0.04 * s)
            except Exception:
                pass
            info["league_n"] = n_lg

    shrink = float(max(0.80, min(1.10, shrink)))
    p_out = float(max(0.01, min(0.99, 0.5 + (p0 - 0.5) * shrink)))
    info.update({"applied": True, "p_in": p0, "p_out": p_out, "shrink": shrink, "league": league_key})
    return p_out, info


def _ml_confidence_from_layer_validation(
    report: Optional[Dict[str, Any]],
    *,
    league_hint: str = "",
    n_team_eff: float = float("nan"),
) -> Dict[str, Any]:
    score = 50.0
    source = "global"
    n_total = 0
    ll = float("nan")
    gain = float("nan")

    if isinstance(report, dict) and not report.get("error"):
        row = None
        by_lg = report.get("by_league") if isinstance(report.get("by_league"), dict) else {}
        lg = str(league_hint or "").strip()
        if lg and lg in by_lg and isinstance(by_lg.get(lg), dict):
            source = f"liga:{lg}"
            row = by_lg.get(lg)
        if row is None:
            source = "global"
            row = {"fused_active": report.get("models", {}).get("fused_active", {})}

        fused = row.get("fused_active") if isinstance(row, dict) else {}
        n_total = int((fused or {}).get("n_total") or 0)
        ll_raw = (fused or {}).get("logloss")
        gain_raw = (fused or {}).get("gain_vs_team_logloss")
        try:
            ll = float(ll_raw) if ll_raw is not None else float("nan")
        except Exception:
            ll = float("nan")
        try:
            gain = float(gain_raw) if gain_raw is not None else float("nan")
        except Exception:
            gain = float("nan")

        if math.isfinite(gain):
            if gain >= 0.015:
                score += 25
            elif gain >= 0.008:
                score += 15
            elif gain >= 0.0:
                score += 5
            else:
                score -= 20
        else:
            score -= 8

        if n_total >= 1500:
            score += 20
        elif n_total >= 700:
            score += 12
        elif n_total >= 300:
            score += 6
        else:
            score -= 8

        if math.isfinite(ll):
            if ll <= 0.62:
                score += 10
            elif ll <= 0.66:
                score += 5
            else:
                score -= 5
    else:
        score -= 15

    try:
        nte = float(n_team_eff)
    except Exception:
        nte = float("nan")
    if math.isfinite(nte):
        if nte >= 25:
            score += 10
        elif nte >= 12:
            score += 5
        else:
            score -= 5

    score = float(max(0.0, min(100.0, score)))
    if score >= 80.0:
        grade = "A (Alta)"
    elif score >= 65.0:
        grade = "B (Media)"
    else:
        grade = "C (Baixa)"

    return {
        "grade": grade,
        "score": score,
        "source": source,
        "n_total": n_total,
        "logloss": ll if math.isfinite(ll) else None,
        "gain_vs_team_logloss": gain if math.isfinite(gain) else None,
    }


def _render_calibracoes(root: Path) -> None:
    st.header("Técnico")
    st.caption("Aqui fica tudo de CSV/diagnóstico, calibração, drift, retreino e parâmetros.")

    daily_csv, hist_csvs, artifact_path = _paths_for_ml(root)
    _csv_diag_info = None
    try:
        if str(daily_csv or "").strip() and Path(str(daily_csv)).exists():
            _csv_diag_info = _csv_diag(_file_sig(str(daily_csv)))
    except Exception:
        _csv_diag_info = None

    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        st.markdown("### Fontes")
        st.text_input("CSV diÃ¡rio (usado no app)", value=daily_csv, disabled=True)
        st.text_area(
            "CSVs histÃ³ricos (treino completo)",
            value="\n".join([str(p) for p in hist_csvs]) if hist_csvs else "(nenhum selecionado)",
            height=120,
            disabled=True,
        )
        st.text_input("ml_artifact.json", value=artifact_path, disabled=True)

    with c2:
        st.markdown("### Status do artifact")
        summ = _read_artifact_summary(artifact_path)
        if not summ.get("exists"):
            st.warning("Ainda nÃ£o existe `ml_artifact.json`. FaÃ§a **Treino completo** abaixo para gerar.")
        else:
            if summ.get("error"):
                st.error(str(summ["error"]))
            st.write({
                "trained_at": summ.get("trained_at"),
                "data_max_date": summ.get("data_max_date"),
                "n_games_total": summ.get("n_games_total"),
                "added_games_last": summ.get("added_games_last"),
                "backfill_suspected": summ.get("backfill_suspected"),
            })

    if isinstance(_csv_diag_info, dict):
        st.caption(
            f"Diagnóstico CSV: {_csv_diag_info.get('rows',0)} linhas | {_csv_diag_info.get('unique_gameids',0)} gameids | "
            f"{_csv_diag_info.get('games_complete',0)} jogos completos | {_csv_diag_info.get('games_incomplete',0)} incompletos | "
            f"{_csv_diag_info.get('rows_missing_gameid',0)} linhas sem gameid"
        )

    st.markdown("### Parâmetros ativos (resumo)")
    try:
        _rows_params = []
        for _k in _PARAM_DEFAULTS.keys():
            _rows_params.append({"Parâmetro": str(_k), "Valor ativo": str(st.session_state.get(_k, _PARAM_DEFAULTS[_k]))})
        if _rows_params:
            st.dataframe(pd.DataFrame(_rows_params), width='stretch', hide_index=True)
    except Exception:
        pass

    with st.expander("Engine/Filtros do Plays", expanded=False):
        # Controles que antes apareciam no Plays.
        _cfg_profile = str(st.session_state.get("cfg_profile", "Padrão") or "Padrão")
        _is_adv_profile = (_cfg_profile in {"Avançado", "20/10"})
        if not _is_adv_profile:
            st.session_state["ml_engine"] = "mlcore_v2"
        st.selectbox(
            "ML engine",
            ["mlcore_v2", "elo_season_players"],
            index=0 if st.session_state.get("ml_engine", "mlcore_v2") == "mlcore_v2" else 1,
            key="ml_engine",
            disabled=(not _is_adv_profile),
            format_func=lambda x: "ML core v2 (ml_artifact.json)" if x == "mlcore_v2" else "Elo season + players",
        )
        st.checkbox(
            "Comparar/alerta: calcular também Elo season + players",
            key="ml_engine_compare",
            value=bool(st.session_state.get("ml_engine_compare", False)),
        )
        st.slider(
            "Alerta divergência (pp) [ML Mapa]",
            5, 30,
            int(st.session_state.get("ml_engine_alert_pp", 12) or 12),
            step=1,
            key="ml_engine_alert_pp",
        )

        _years = ["All"]
        _splits = ["All"]
        try:
            if str(daily_csv or "").strip() and Path(str(daily_csv)).exists():
                _tg = _load_team_games(_file_sig(str(daily_csv)))
                _years = ["All"] + sorted([int(y) for y in _tg["year"].dropna().unique().tolist()])
                _splits = ["All"] + sorted([str(x) for x in _tg["split"].dropna().unique().tolist()])
        except Exception:
            pass
        _cur_year = st.session_state.get("year_opt", "All")
        if _cur_year not in _years:
            _cur_year = _years[0]
        st.selectbox("Year", _years, index=_years.index(_cur_year), key="year_opt")
        st.selectbox("Split", _splits, index=0 if st.session_state.get("split_opt", "All") not in _splits else _splits.index(st.session_state.get("split_opt", "All")), key="split_opt")
        st.selectbox("Playoffs", ["all", "true", "false"], index=["all", "true", "false"].index(str(st.session_state.get("playoffs_opt", "all")) if str(st.session_state.get("playoffs_opt", "all")) in ["all", "true", "false"] else "all"), key="playoffs_opt")

    st.divider()

    # Auto-run incremental if requested from the banner
    if st.session_state.get("auto_run_incremental", False):
        st.session_state["auto_run_incremental"] = False
        if daily_csv and Path(daily_csv).exists() and Path(artifact_path).exists():
            cmd = [sys.executable, "-m", "mlcore.update_incremental", "--artifact", artifact_path, "--csv", daily_csv]
            with st.spinner("Atualizando (incremental) automaticamenteâ€¦"):
                res = _run_cmd(cmd)
            st.session_state["ml_last_log"] = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            if res.returncode == 0:
                st.success("AtualizaÃ§Ã£o incremental concluÃ­da.")
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
            else:
                st.error("Falha na atualizaÃ§Ã£o incremental automÃ¡tica.")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    st.markdown("### AÃ§Ãµes")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        holdout_days = st.number_input("Holdout (dias)", min_value=0, max_value=3650, value=180, step=30, key="cal_holdout_days")
    with colB:
        do_calib = st.selectbox("CalibraÃ§Ã£o (Platt)", ["auto", "yes", "no"], index=0, key="cal_do_calib")
    with colC:
        keep_playoffs = st.selectbox("Playoffs no treino/check", ["auto", "yes", "no"], index=0, key="cal_keep_playoffs")

    keep_playoffs_flag = "auto"
    if keep_playoffs == "yes":
        keep_playoffs_flag = "yes"
    elif keep_playoffs == "no":
        keep_playoffs_flag = "no"

    btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 1])
    with btn1:
        train_clicked = st.button("Treino completo", width='stretch')
    with btn2:
        incr_clicked = st.button("Atualizar rÃ¡pido (incremental)", width='stretch')
    with btn3:
        check_clicked = st.button("Checar drift / retreino", width='stretch')
    with btn4:
        back_clicked = st.button("Voltar para Plays", width='stretch')

    if back_clicked:
        st.session_state["_nav_pending"] = "Plays"

    if train_clicked:
        csvs = [p for p in (hist_csvs or []) if str(p).strip()]
        if daily_csv:
            csvs.append(daily_csv)
        if not csvs:
            st.error("Selecione pelo menos um CSV (diÃ¡rio ou histÃ³rico).")
        else:
            cmd = [sys.executable, "-m", "mlcore.train_offline"]
            for p in csvs:
                cmd += ["--csv", str(p)]
            cmd += ["--out", artifact_path, "--holdout-days", str(int(holdout_days)), "--calibrate", str(do_calib)]
            cmd += ["--keep-playoffs", keep_playoffs_flag]
            with st.spinner("Treinando (completo)â€¦ (primeira vez pode demorar)"):
                res = _run_cmd(cmd)
            st.session_state["ml_last_log"] = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            if res.returncode == 0:
                st.success("Treino completo concluÃ­do. Artifact atualizado.")
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            else:
                st.error("Falha no treino completo. Veja o log abaixo.")

    if incr_clicked:
        if not daily_csv or not Path(daily_csv).exists():
            st.error("CSV diÃ¡rio nÃ£o encontrado/selecionado.")
        elif not Path(artifact_path).exists():
            st.error("ml_artifact.json nÃ£o encontrado. Rode o **Treino completo** primeiro.")
        else:
            cmd = [sys.executable, "-m", "mlcore.update_incremental", "--artifact", artifact_path, "--csv", daily_csv]
            with st.spinner("Atualizando artifact (incremental)â€¦"):
                res = _run_cmd(cmd)
            st.session_state["ml_last_log"] = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            if res.returncode == 0:
                st.success("AtualizaÃ§Ã£o incremental concluÃ­da.")
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            else:
                st.error("Falha na atualizaÃ§Ã£o incremental. Veja o log abaixo.")

    if check_clicked:
        if not daily_csv or not Path(daily_csv).exists():
            st.error("CSV diÃ¡rio nÃ£o encontrado/selecionado.")
        elif not Path(artifact_path).exists():
            st.error("ml_artifact.json nÃ£o encontrado. Rode o **Treino completo** primeiro.")
        else:
            cmd = [
                sys.executable, "-m", "mlcore.retrain_check",
                "--artifact", artifact_path,
                "--csv", daily_csv,
                "--keep-playoffs", keep_playoffs_flag,
                "--exit-code",
            ]
            with st.spinner("Rodando retrain_checkâ€¦"):
                res = _run_cmd(cmd)
            out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            st.session_state["ml_last_log"] = out
            if res.returncode == 2:
                st.warning("O check recomendou **RETREINO COMPLETO**. (Veja o log abaixo.)")
            elif res.returncode == 0:
                st.success("Check OK: sem necessidade de retreino agora. (Veja o log abaixo.)")
            else:
                st.error("retrain_check falhou. Veja o log abaixo.")

    st.markdown("### Log")
    log = st.session_state.get("ml_last_log", "")
    if log:
        st.code(log)
    else:
        st.caption("Nada rodou ainda nesta sessÃ£o.")


def _maybe_show_update_banner(root: Path) -> None:
    if st.session_state.get("dismiss_update_banner", False):
        return
    daily_csv, _, artifact_path = _paths_for_ml(root)
    if not daily_csv or not Path(daily_csv).exists():
        return

    art_exists = Path(artifact_path).exists()
    csv_m = _safe_mtime(daily_csv)
    art_m = _safe_mtime(artifact_path) if art_exists else None

    if not art_exists:
        st.info("Você ainda não tem `ml_artifact.json`. Abra **Técnico** para fazer o primeiro treino.")
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Abrir Técnico", width='stretch', key="open_calib_no_art"):
                st.session_state["_nav_pending"] = "Técnico"
        with b2:
            if st.button("Dispensar", width='stretch', key="dismiss_no_art"):
                st.session_state["dismiss_update_banner"] = True
        return

    if csv_m is not None and art_m is not None and csv_m > art_m + 1.0:
        st.warning("Seu **CSV diÃ¡rio** Ã© mais novo que o `ml_artifact.json`. Quer fazer **Atualizar rÃ¡pido (incremental)** agora")
        c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
        with c1:
            if st.button("Atualizar agora (incremental)", width='stretch', key="banner_incr"):
                st.session_state["_nav_pending"] = "Técnico"
                st.session_state["auto_run_incremental"] = True
        with c2:
            if st.button("Abrir Técnico", width='stretch', key="banner_open_calib"):
                st.session_state["_nav_pending"] = "Técnico"
        with c3:
            if st.button("Dispensar", width='stretch', key="banner_dismiss"):
                st.session_state["dismiss_update_banner"] = True


# Página Técnico (só aqui tem drift/retreino/update)
if nav_page == "Técnico":
    _render_calibracoes(APP_ROOT)
    st.stop()

# Banner automÃ¡tico (Plays)
_maybe_show_update_banner(APP_ROOT)

# -----------------------------
# Etapa 1 â€” escolher Season + Times
# -----------------------------
# Season selector (sempre no topo)
if seasons:
    season_options = [str(s["season"]) for s in seasons]
    # default = Ãºltima (ano mais novo)
    default_idx = max(0, len(season_options) - 1)
    season_label = st.selectbox("Temporada", season_options, index=default_idx, key="season_label")
    season_info = next((s for s in seasons if s["season"] == season_label), seasons[-1])
    season_year = int(season_info["year"])
else:
    season_label = st.selectbox("Temporada", ["S15"], index=0, key="season_label")
    season_year = 2025

# Decide caminhos ativos
if st.session_state.get("cfg_auto_paths", True) and seasons:
    csv_path = str(season_info["csv_path"])
else:
    csv_path = str(st.session_state.get("cfg_csv_override", str(APP_ROOT / f"{season_year}_LoL_esports_match_data_from_OraclesElixir.csv")))
artifact_path = str(st.session_state.get("cfg_artifact_override", _default_artifact_for(APP_ROOT, season_label, season_year)))

# Achievements path default (para ML core v2 e aba Vencedores)
# - MantÃ©m em sessÃ£o para que o ML use o mesmo arquivo que vocÃª edita na aba "Vencedores"
if not str(st.session_state.get("ach_path_ui", "") or "").strip():
    try:
        _p_art = Path(artifact_path)
        if _p_art.exists():
            _art_tmp = _load_artifact_cached(_artifact_sig(str(_p_art)))
            _ach_meta_tmp = dict(_art_tmp.meta.get("achievements") or {})
            _ach_p = str(_ach_meta_tmp.get("path") or "").strip()
            if _ach_p:
                _ach_p = _resolve_relpath(artifact_path, _ach_p)
            else:
                _ach_p = str(_p_art.with_name("achievements.json"))
            st.session_state["ach_path_ui"] = _ach_p
        else:
            st.session_state["ach_path_ui"] = str(Path(artifact_path).with_name("achievements.json"))
    except Exception:
        try:
            st.session_state["ach_path_ui"] = str(Path(artifact_path).with_name("achievements.json"))
        except Exception:
            st.session_state["ach_path_ui"] = str(APP_ROOT / "achievements.json")


# Se trocou season, reseta fluxo (pra nÃ£o misturar times/linhas)
if st.session_state.get("_last_season_label") != season_label:
    st.session_state["_last_season_label"] = season_label
    _reset_flow(keep_config=True)
    # tambÃ©m zera o confronto (evita vir prÃ©-selecionado ao abrir/trocar temporada)
    st.session_state['teamA'] = ''
    st.session_state['teamB'] = ''

csv_sig = _file_sig(csv_path)

# Carrega times
try:
    teams = _list_teams(csv_sig)
except Exception as e:
    st.error(f"Falha ao ler CSV para listar times: {e}")
    st.stop()

# Normaliza grafias (ex.: "Dplus Kia" vs "Dplus KIA") para evitar cair na liga errada no modo auto.
try:
    _maps = _team_canon_maps(csv_sig)
    _key_to_best = _maps.get("variant_to_best_key") if isinstance(_maps, dict) else {}
    def _norm_team_value(x: Any) -> str:
        s = ("" if x is None else str(x)).strip()
        s = re.sub(r"\s+", " ", s)
        k = _canon_team_key(s)
        return str(_key_to_best.get(k, s)) if isinstance(_key_to_best, dict) else s

    for _k in ("teamA", "teamB"):
        if _k in st.session_state and str(st.session_state.get(_k) or "").strip():
            st.session_state[_k] = _norm_team_value(st.session_state.get(_k))
except Exception:
    pass


# Se pediu reset manual do confronto, aplica ANTES dos widgets (evita erro de session_state)
if st.session_state.pop('pending_reset_matchup', False):
    _reset_flow(keep_config=True)
    st.session_state['teamA'] = ''
    st.session_state['teamB'] = ''

flow_stage = st.session_state.get("flow_stage", "select")


# -----------------------------
# Jogos (agenda) - adicionar por colagem (na página Plays)
# -----------------------------
st.session_state.setdefault("games_list", [])
st.session_state.setdefault("team_aliases_user", None)
st.session_state.setdefault("agenda_has_unknown_aliases", False)

with st.expander("Jogos (agenda) - colar / adicionar", expanded=False):
    st.caption("Cole aqui um bloco de jogos (data, hora, time A, time B). O app tenta inferir a liga pelos cabeÃ§alhos e te pede para mapear times desconhecidos.")

    # ---- helpers locais ----
    def _sched_norm_key(s: str) -> str:
        s = (s or "").strip().lower()
        # remove acentos bÃ¡sicos (fallback simples)
        s = s.replace("Ã¡","a").replace("Ã ","a").replace("Ã£","a").replace("Ã¢","a")
        s = s.replace("Ã©","e").replace("Ãª","e")
        s = s.replace("Ã­","i")
        s = s.replace("Ã³","o").replace("Ã´","o").replace("Ãµ","o")
        s = s.replace("Ãº","u")
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    def _sched_dedup_repeat_token(s: str) -> str:
        s0 = html.unescape((s or "").strip())
        compact = re.sub(r"\s+", "", s0)
        n = len(compact)
        if n % 2 == 0 and n > 0:
            half = n // 2
            if compact[:half].lower() == compact[half:].lower():
                # tenta "cortar" mantendo espaÃ§os do original (se der)
                # fallback: devolve metade do compact
                cut = compact[:half]
                return cut.strip()
        return s0

    def _sched_extract_league_header(ln: str) -> str:
        s = html.unescape((ln or "").strip())
        if not s:
            return ""
        # remove tokens comuns
        s = s.replace("League Icon", " ").replace("Flag Icon", " ")
        s = re.sub(r"\s+", " ", s).strip()
        s = s.replace("  ", " ").strip()
        # alguns dumps vÃªm grudados: "LCKFlag Icon"
        s = s.replace("Flag Icon", " ").strip()
        # se ficar algo enorme/sem letras, ignora
        if len(s) < 2:
            return ""
        return s

    def _sched_parse(text_in: str) -> List[Dict[str, Any]]:
        lines = [html.unescape(ln).strip() for ln in (text_in or "").splitlines()]
        lines = [ln for ln in lines if ln]
        games: List[Dict[str, Any]] = []
        cur_date = ""
        cur_league = ""
        date_re = re.compile(r"(\b\d{2}\/\d{2}\b)")
        time_re = re.compile(r"(\b\d{2}\:\d{2}\b)")
        i = 0
        while i < len(lines):
            ln = lines[i]

            mdate = date_re.search(ln)
            if mdate:
                cur_date = mdate.group(1)
                i += 1
                continue

            mtime = time_re.search(ln)
            if mtime:
                cur_time = mtime.group(1)
                j = i + 1
                while j < len(lines) and (lines[j] in ["-", "â€“", "â€”"] or lines[j].strip("- ") == ""):
                    j += 1
                if j >= len(lines):
                    break
                team1_raw = _sched_dedup_repeat_token(lines[j])
                j += 1
                while j < len(lines) and (lines[j] in ["-", "â€“", "â€”"] or lines[j].strip("- ") == ""):
                    j += 1
                if j >= len(lines):
                    break
                team2_raw = _sched_dedup_repeat_token(lines[j])

                games.append({
                    "date": cur_date,
                    "time": cur_time,
                    "league": cur_league,
                    "teamA_raw": team1_raw,
                    "teamB_raw": team2_raw,
                })
                i = j + 1
                continue

            # possÃ­vel cabeÃ§alho de liga
            if not date_re.search(ln) and not time_re.search(ln):
                league_candidate = _sched_extract_league_header(ln)
                if league_candidate:
                    cur_league = league_candidate
            i += 1

        return games

    def _sched_load_aliases() -> Dict[str, str]:
        if isinstance(st.session_state.get("team_aliases_user"), dict):
            return st.session_state["team_aliases_user"]
        p = APP_ROOT / "user_team_aliases.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    st.session_state["team_aliases_user"] = {str(k): str(v) for k, v in data.items()}
                    return st.session_state["team_aliases_user"]
            except Exception:
                pass
        st.session_state["team_aliases_user"] = {}
        return st.session_state["team_aliases_user"]

    def _sched_save_aliases(aliases: Dict[str, str]) -> None:
        try:
            p = APP_ROOT / "user_team_aliases.json"
            p.write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _sched_resolve_team(raw: str, teams_all: List[str], aliases: Dict[str, str]) -> Tuple[str, bool]:
        raw = (raw or "").strip()
        if not raw:
            return "", False
        if raw in teams_all:
            return raw, True
        key = _sched_norm_key(raw)
        # alias salvo
        if key in aliases and str(aliases[key]) in teams_all:
            return str(aliases[key]), True
        # match normalizado (Ã s vezes sÃ³ muda pontuaÃ§Ã£o/maiÃºscula)
        for t in teams_all:
            if _sched_norm_key(t) == key:
                return t, True
        return raw, False

    def _sched_suggestions(raw: str, teams_all: List[str]) -> List[str]:
        raw = (raw or "").strip()
        if not raw:
            return teams_all[:50]
        key = _sched_norm_key(raw)
        pref = key[:2] if len(key) >= 2 else key
        # comeÃ§a com
        starts = [t for t in teams_all if _sched_norm_key(t).startswith(pref)] if pref else []
        contains = [t for t in teams_all if pref and pref in _sched_norm_key(t)] if pref else []
        fuzzy = difflib.get_close_matches(raw, teams_all, n=20, cutoff=0.25)
        out: List[str] = []
        for lst in (starts, fuzzy, contains):
            for t in lst:
                if t not in out:
                    out.append(t)
        return out[:60] if out else teams_all[:60]

    # ---- UI de importaÃ§Ã£o ----
    default_bo = st.selectbox("BO padrÃ£o (para os jogos importados)", [1, 3, 5], index=1, key="plays_schedule_default_bo")

    txt = st.text_area(
        "Cole os jogos aqui (formato livre)",
        value=st.session_state.get("plays_schedule_text", ""),
        height=220,
        key="plays_schedule_text",
        placeholder="Ex.:\\nLeague IconLCKFlag Icon\\n22/01\\n05:00\\nBRION\\nHanwha Life Esports\\n-\\n-\\n..."
    )

    cA, cB, cC = st.columns([1,1,2])
    with cA:
        parse_clicked = st.button("Processar texto", width='stretch', key="plays_schedule_parse_btn")
    with cB:
        clear_clicked = st.button("Limpar texto", width='stretch', key="plays_schedule_clear_btn")
    with cC:
        if st.button("Limpar agenda (lista atual)", width='stretch', key="plays_schedule_clear_list"):
            st.session_state["games_list"] = []
            st.success("Agenda limpa.")

    if clear_clicked:
        st.session_state["plays_schedule_text"] = ""
        st.rerun()


    with st.expander("Adicionar 1 jogo manual", expanded=False):
        cM1, cM2, cM3 = st.columns([1,1,1])
        with cM1:
            m_date = st.text_input("Data (DD/MM)", value="", key="plays_manual_date", placeholder="22/01")
        with cM2:
            m_time = st.text_input("Hora (HH:MM)", value="", key="plays_manual_time", placeholder="05:00")
        with cM3:
            m_league = st.text_input("Liga (opcional)", value="", key="plays_manual_league", placeholder="LCK")

        cM4, cM5, cM6 = st.columns([2,2,1])
        with cM4:
            m_teamA = st.text_input("Time A (como vocÃª tem)", value="", key="plays_manual_teamA", placeholder="BRION")
        with cM5:
            m_teamB = st.text_input("Time B (como vocÃª tem)", value="", key="plays_manual_teamB", placeholder="Hanwha Life Esports")
        with cM6:
            m_bo = st.selectbox("BO", [1,3,5], index=1, key="plays_manual_bo")

        if st.button("Adicionar (vai pedir mapeamento se precisar)", width='stretch', key="plays_manual_add"):
            # valida mÃ­nimo
            if not (m_date.strip() and m_time.strip() and m_teamA.strip() and m_teamB.strip()):
                st.warning("Preencha pelo menos Data, Hora, Time A e Time B.")
            else:
                pending_now = st.session_state.get("plays_pending_games_import") or []
                pending_now = list(pending_now) if isinstance(pending_now, list) else []
                pending_now.append({
                    "date": m_date.strip(),
                    "time": m_time.strip(),
                    "league": m_league.strip(),
                    "teamA_raw": m_teamA.strip(),
                    "teamB_raw": m_teamB.strip(),
                })
                st.session_state["plays_pending_games_import"] = pending_now
                st.success("Adicionado Ã  prÃ©via. Agora mapeie (se necessÃ¡rio) e clique em **Adicionar Ã  agenda**.")
                st.rerun()

    if parse_clicked:
        parsed = _sched_parse(txt)
        if not parsed:
            st.warning("NÃ£o consegui encontrar jogos nesse texto. Dica: precisa ter linhas com hora (HH:MM) e logo depois 2 linhas de times.")
        else:
            st.session_state["plays_pending_games_import"] = parsed
            st.success(f"Encontrados: {len(parsed)} jogo(s). Agora confira o mapeamento e clique em **Adicionar Ã  agenda**.")
            st.rerun()

    pending = st.session_state.get("plays_pending_games_import") or []
    st.session_state["agenda_has_unknown_aliases"] = False
    if isinstance(pending, list) and pending:
        st.markdown("#### PrÃ©via (antes de adicionar)")
        try:
            st.dataframe(pd.DataFrame(pending), width='stretch', hide_index=True)
        except Exception:
            pass

        aliases = _sched_load_aliases()

        # detecta times nÃ£o resolvidos
        unknowns: List[str] = []
        for g in pending:
            a_raw = str(g.get("teamA_raw", "") or "").strip()
            b_raw = str(g.get("teamB_raw", "") or "").strip()
            a_res, ok_a = _sched_resolve_team(a_raw, teams, aliases)
            b_res, ok_b = _sched_resolve_team(b_raw, teams, aliases)
            if not ok_a and a_raw:
                unknowns.append(a_raw)
            if not ok_b and b_raw:
                unknowns.append(b_raw)
        unknowns = sorted(list({u for u in unknowns}))
        st.session_state["agenda_has_unknown_aliases"] = bool(unknowns)

        mapping: Dict[str, str] = {}
        if unknowns:
            st.markdown("#### Mapear times desconhecidos")
            st.caption("Se o nome do site nÃ£o bater com o nome do CSV, escolha o time correto aqui. Isso fica salvo (apelido â†’ time do CSV).")
            for raw in unknowns:
                sug = _sched_suggestions(raw, teams)
                pick = st.selectbox(
                    f"Esse time '{raw}' seria qual",
                    options=[""] + sug,
                    index=0,
                    key=f"map_{_sched_norm_key(raw)}",
                )
                if pick:
                    mapping[raw] = pick

        c1, c2 = st.columns([1,1])
        with c1:
            add_clicked = st.button("Adicionar Ã  agenda", type="primary", width='stretch', key="plays_schedule_add_btn")
        with c2:
            cancel_clicked = st.button("Cancelar importaÃ§Ã£o", width='stretch', key="plays_schedule_cancel_btn")

        if cancel_clicked:
            st.session_state["plays_pending_games_import"] = []
            st.info("ImportaÃ§Ã£o cancelada.")
            st.rerun()

        if add_clicked:
            # aplica mapping + aliases existentes
            aliases = _sched_load_aliases()
            for raw, canon in mapping.items():
                aliases[_sched_norm_key(raw)] = canon
            _sched_save_aliases(aliases)

            out_games: List[Dict[str, Any]] = []
            for g in pending:
                a_raw = str(g.get("teamA_raw", "") or "").strip()
                b_raw = str(g.get("teamB_raw", "") or "").strip()
                a_res, ok_a = _sched_resolve_team(a_raw, teams, aliases)
                b_res, ok_b = _sched_resolve_team(b_raw, teams, aliases)

                # se ainda ficou desconhecido, mantÃ©m raw (nÃ£o bloqueia), mas nÃ£o vai carregar automÃ¡tico pro confronto
                out_games.append({
                    "date": str(g.get("date", "") or "").strip(),
                    "time": str(g.get("time", "") or "").strip(),
                    "league": str(g.get("league", "") or "").strip(),
                    "teamA": a_res,
                    "teamB": b_res,
                    "bo": int(default_bo),
                    "lines": "",
                })

            st.session_state["games_list"] = (st.session_state.get("games_list") or []) + out_games
            st.session_state["plays_pending_games_import"] = []
            st.success(f"Adicionados: {len(out_games)} jogo(s) na agenda.")
            st.rerun()

    st.caption(f"Agenda atual: {len(st.session_state.get('games_list') or [])} jogo(s).")


st.subheader("Time 1 - Time 2")
st.subheader("Confronto")

# Opcional: selecionar rapidamente um jogo salvo na aba Jogos
# (apenas preenche Time 1/2 e BO; vocÃª ainda clica em "Analisar Confronto")
if flow_stage == "select":
    _gl = st.session_state.get("games_list") or []
    if isinstance(_gl, list) and len(_gl) > 0:
        with st.expander("Selecionar jogo salvo (Agenda)", expanded=False):
            labels = []
            for i, g in enumerate(_gl):
                try:
                    d = str(g.get("date", "") or "").strip()
                    t = str(g.get("time", "") or "").strip()
                    dt = (d + (" " + t if t else "")).strip()
                    league = str(g.get("league", "") or "").strip()
                    ta = str(g.get("teamA", "") or "").strip()
                    tb = str(g.get("teamB", "") or "").strip()
                    bo_g = g.get("bo", 3)
                except Exception:
                    dt, league, ta, tb, bo_g = "", "", "", "", 3
                parts = []
                if dt:
                    parts.append(dt)
                if league:
                    parts.append(league)
                parts.append(f"{ta} x {tb}")
                parts.append(f"BO{bo_g}")
                labels.append(" | ".join(parts))

            pick = st.selectbox(
                "Jogo salvo",
                options=[-1] + list(range(len(labels))),
                index=0,
                key="main_saved_game_idx",
                format_func=lambda x: "Selecione ..." if int(x) == -1 else labels[int(x)],
            )
            bcols = st.columns([1, 2])
            with bcols[0]:
                do_pick = st.button("Usar este jogo", type="primary", width='stretch', disabled=(int(pick) == -1), key="main_saved_game_use")
            with bcols[1]:
                st.caption("Dica: se o nome do time nÃ£o bater com o CSV, use a aba **Jogos** para resolver/editar o texto.")


            # --- Linhas por jogo (opcional): salva junto na agenda e permite analisar direto ---
            if int(pick) >= 0:
                try:
                    _gsel = _gl[int(pick)]
                except Exception:
                    _gsel = None

                if isinstance(_gsel, dict):
                    _ta_sel = str(_gsel.get("teamA", "") or "").strip()
                    _tb_sel = str(_gsel.get("teamB", "") or "").strip()
                    _bo_sel = int(_gsel.get("bo", 3) or 3) if str(_gsel.get("bo", "")).strip() else 3

                    _lines_saved = _gsel.get("lines_saved")
                    if not isinstance(_lines_saved, dict):
                        _lines_saved = {}

                    with st.expander("Linhas deste jogo (opcional) â€” salvar e/ou analisar", expanded=False):
                        _pref = f"ag_lines_{int(pick)}_"
                        _FIELDS = [
                            "lines_kills", "lines_towers", "lines_dragons", "lines_barons", "lines_inhib", "lines_time",
                            "teamA_kills_line_text", "teamB_kills_line_text",
                            "hc_kills_team", "hc_kills_text", "hc_towers_text", "hc_dragons_text",
                            "combo_kills_text", "combo_time_text",
                        ]

                        # inicializa os campos (sem sobrescrever se o usuÃ¡rio jÃ¡ digitou nessa sessÃ£o)
                        for _f in _FIELDS:
                            _k = _pref + _f
                            if _k not in st.session_state:
                                st.session_state[_k] = str(_lines_saved.get(_f, "") or "")

                        st.markdown("**Mercados principais**")
                        g1, g2, g3 = st.columns(3)
                        with g1:
                            st.text_input("Kills (ex: 26.5, 27.5)", key=_pref + "lines_kills")
                            st.text_input("Torres (ex: 10.5)", key=_pref + "lines_towers")
                        with g2:
                            st.text_input("DragÃµes (ex: 4.5)", key=_pref + "lines_dragons")
                            st.text_input("BarÃµes (ex: 0.5)", key=_pref + "lines_barons")
                        with g3:
                            st.text_input("Inibidores (ex: 1.5)", key=_pref + "lines_inhib")
                            st.text_input("Tempo (ex: 32, 32.5)", key=_pref + "lines_time")

                        _extras_ctx = st.expander("Extras (opcional)", expanded=False)
                        with _extras_ctx:
                            st.text_input(f"Total Kills {_ta_sel or 'Time 1'} (ex: 16.5)", key=_pref + "teamA_kills_line_text")
                            st.text_input(f"Total Kills {_tb_sel or 'Time 2'} (ex: 15.5)", key=_pref + "teamB_kills_line_text")

                            st.caption("Handicap: interpreta como (mÃ©trica_for + handicap) > mÃ©trica_against (sem push; use .5).")
                            if _ta_sel and _tb_sel:
                                _cur = st.session_state.get(_pref + "hc_kills_team") or _ta_sel
                                if _cur not in [_ta_sel, _tb_sel]:
                                    _cur = _ta_sel
                                st.selectbox(
                                    "Handicap Kills â€” Time",
                                    options=[_ta_sel, _tb_sel],
                                    index=[_ta_sel, _tb_sel].index(_cur),
                                    key=_pref + "hc_kills_team",
                                )
                                _hc_team = st.session_state.get(_pref + "hc_kills_team") or _ta_sel
                            else:
                                _hc_team = _ta_sel or _tb_sel

                            st.text_input(f"HC Kills ({_hc_team or 'Time'}) (ex: -2.5)", key=_pref + "hc_kills_text")
                            # MantÃ©m Torres/DragÃµes como no app atual (perspectiva Time 1)
                            st.text_input(f"HC Torres ({_ta_sel or 'Time 1'}) (ex: -3.5)", key=_pref + "hc_towers_text")
                            st.text_input(f"HC DragÃµes ({_ta_sel or 'Time 1'}) (ex: -1.5)", key=_pref + "hc_dragons_text")

                            st.text_input("Linha Kills (para ML + Totais)", key=_pref + "combo_kills_text")
                            st.text_input("Linha Tempo (para ML + Totais)", key=_pref + "combo_time_text")

                        s1, s2 = st.columns(2)
                        with s1:
                            _btn_save_lines = st.button("Salvar linhas", width='stretch', key=f"ag_save_lines_{int(pick)}")
                        with s2:
                            _btn_analyze = st.button("Analisar confronto", type="primary", width='stretch', key=f"ag_analyze_{int(pick)}")

                        if _btn_save_lines or _btn_analyze:
                            _ld = {f: str(st.session_state.get(_pref + f, "") or "") for f in _FIELDS}
                            try:
                                _gl[int(pick)]["lines_saved"] = _ld
                                st.session_state["games_list"] = _gl
                            except Exception:
                                pass

                            if _btn_save_lines:
                                st.success("Linhas salvas para este jogo.")
                                try:
                                    st.rerun()
                                except Exception:
                                    st.experimental_rerun()

                            if _btn_analyze:
                                # vai salvar + analisar direto usando essas linhas
                                st.session_state["pending_load"] = {
                                    "_reset_flow": True,
                                    "teamA": _ta_sel if _ta_sel in teams else "",
                                    "teamB": _tb_sel if _tb_sel in teams else "",
                                    "bo": _bo_sel,
                                }
                                st.session_state["pending_auto_analyze"] = {"lines": _ld}
                                try:
                                    st.rerun()
                                except Exception:
                                    st.experimental_rerun()

            if do_pick and int(pick) >= 0:
                try:
                    g = _gl[int(pick)]
                    payload = {"_reset_flow": True}
                    # SÃ³ seta se existir no CSV (evita erro de selectbox com valor invÃ¡lido)
                    if str(g.get("teamA", "")) in teams:
                        payload["teamA"] = str(g.get("teamA"))
                    if str(g.get("teamB", "")) in teams:
                        payload["teamB"] = str(g.get("teamB"))
                    if str(g.get("bo")) in ["1", "3", "5"]:
                        payload["bo"] = int(g.get("bo"))
                    st.session_state["pending_load"] = payload
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass

# Quando jÃ¡ analisou, travamos seleÃ§Ã£o (fica limpo e consistente)
locked = flow_stage in ("lines", "results")

# PadrÃ£o: Times comeÃ§am em branco (sem prÃ©-seleÃ§Ã£o)
for _k in ["teamA", "teamB"]:
    if _k not in st.session_state:
        st.session_state[_k] = ""

_team_opts = [""] + teams

def _fmt_team_placeholder(x: str) -> str:
    return "" if str(x) == "" else str(x)

cA, cB, cC, cD = st.columns([3, 3, 1, 3])
with cA:
    teamA = st.selectbox("Time 1", options=_team_opts, index=0, key="teamA", disabled=locked, format_func=_fmt_team_placeholder)
with cB:
    teamB = st.selectbox("Time 2", options=_team_opts, index=0, key="teamB", disabled=locked, format_func=_fmt_team_placeholder)
with cC:
    bo = st.selectbox("BO", [1, 3, 5], index=0, key="bo", disabled=locked)
with cD:
    if flow_stage == "select":
        st.write("Ação")
        _can_analyze = bool(teamA) and bool(teamB) and (str(teamA) != str(teamB))

        # Auto (Agenda): se clicou "Analisar confronto" lÃ¡ em cima, jÃ¡ vem com linhas prontas
        if _can_analyze and "pending_auto_analyze" in st.session_state:
            _paa = st.session_state.pop("pending_auto_analyze") or {}
            _ld = _paa.get("lines") if isinstance(_paa, dict) else None
            if isinstance(_ld, dict):
                key = _matchup_key(season_label, teamA, teamB)
                st.session_state["matchup_key"] = key

                _FIELDS_MAIN = [
                    "lines_kills", "lines_towers", "lines_dragons", "lines_barons", "lines_inhib", "lines_time",
                    "teamA_kills_line_text", "teamB_kills_line_text",
                    "hc_kills_team", "hc_kills_text", "hc_towers_text", "hc_dragons_text",
                    "combo_kills_text", "combo_time_text",
                ]
                for _f in _FIELDS_MAIN:
                    st.session_state[_f] = str(_ld.get(_f, "") or "")

                # necessÃ¡rio: no modo results o app lÃª do store do confronto
                _save_lines_for_matchup(key)

                st.session_state["flow_stage"] = "results"
                st.session_state["_analysis_force_once"] = True
                st.session_state["_consistency_refresh_pending"] = True
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        if st.button("Analisar Confronto", type="primary", width='stretch', disabled=not _can_analyze, key="btn_go_lines_quick"):
            # trava o confronto e carrega linhas previamente salvas (por season+times)
            key = _matchup_key(season_label, teamA, teamB)
            st.session_state["matchup_key"] = key
            _load_lines_for_matchup(key)
            st.session_state["lines_loaded_key"] = key
            st.session_state["flow_stage"] = "lines"
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    elif flow_stage == "lines":
        if st.button("Trocar confronto", width='stretch'):
            # aplica no prÃ³ximo rerun, antes dos widgets (evita erro do Streamlit)
            st.session_state['pending_reset_matchup'] = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    else:
        if st.button("Editar linhas", width='stretch'):
            st.session_state["flow_stage"] = "lines"
            # forÃ§a recarregar as linhas salvas para este confronto
            st.session_state.pop("lines_loaded_key", None)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()


# Ajuda: mapear apelidos de time (só aparece quando houve mismatch na agenda/colagem)
if bool(st.session_state.get("agenda_has_unknown_aliases", False)):
 with st.expander("Time nÃ£o aparece na lista Mapear apelido", expanded=True):
    st.caption("Aparece quando algum nome da agenda/colagem não bateu com o CSV. Mapeie aqui e salve o alias.")

    def _alias_norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = s.replace("Ã¡","a").replace("Ã ","a").replace("Ã£","a").replace("Ã¢","a")
        s = s.replace("Ã©","e").replace("Ãª","e")
        s = s.replace("Ã­","i")
        s = s.replace("Ã³","o").replace("Ã´","o").replace("Ãµ","o")
        s = s.replace("Ãº","u")
        return re.sub(r"[^a-z0-9]+", "", s)

    # carrega aliases (mesmo arquivo usado pela agenda)
    alias_path = APP_ROOT / "user_team_aliases.json"
    aliases = {}
    try:
        if alias_path.exists():
            aliases = json.loads(alias_path.read_text(encoding="utf-8")) or {}
            if not isinstance(aliases, dict):
                aliases = {}
    except Exception:
        aliases = {}

    def _alias_suggestions(raw: str) -> List[str]:
        raw = (raw or "").strip()
        if not raw:
            return teams[:60]
        key = _alias_norm(raw)
        pref = key[:2] if len(key) >= 2 else key
        starts = [t for t in teams if _alias_norm(t).startswith(pref)] if pref else []
        fuzzy = difflib.get_close_matches(raw, teams, n=20, cutoff=0.25)
        contains = [t for t in teams if pref and pref in _alias_norm(t)] if pref else []
        out = []
        for lst in (starts, fuzzy, contains):
            for t in lst:
                if t not in out:
                    out.append(t)
        return out[:60] if out else teams[:60]

    cX, cY = st.columns(2)
    with cX:
        rawA = st.text_input("Apelido / nome do Time 1", value="", key="alias_rawA")
        pickA = st.selectbox("Mapear Time 1 paraâ€¦", options=[""] + _alias_suggestions(rawA), key="alias_pickA")
        if st.button("Aplicar no Time 1", width='stretch', key="alias_applyA"):
            if rawA.strip() and pickA:
                aliases[_alias_norm(rawA)] = pickA
                try:
                    alias_path.write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
                st.session_state["teamA"] = pickA
                st.success(f"Time 1 definido como: {pickA}")
                st.rerun()
    with cY:
        rawB = st.text_input("Apelido / nome do Time 2", value="", key="alias_rawB")
        pickB = st.selectbox("Mapear Time 2 paraâ€¦", options=[""] + _alias_suggestions(rawB), key="alias_pickB")
        if st.button("Aplicar no Time 2", width='stretch', key="alias_applyB"):
            if rawB.strip() and pickB:
                aliases[_alias_norm(rawB)] = pickB
                try:
                    alias_path.write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
                st.session_state["teamB"] = pickB
                st.success(f"Time 2 definido como: {pickB}")
                st.rerun()

def _render_retrain_alerts(csv_path_s: str, artifact_path_s: str) -> None:
    alerts: List[str] = []
    need_retrain_model = False
    need_retrain_params = False
    need_walkforward = False
    now = datetime.now()
    csv_p = Path(csv_path_s) if csv_path_s else None
    art_p = Path(artifact_path_s) if artifact_path_s else None
    params_p = APP_ROOT / "params_fitted.json"

    try:
        if csv_p and csv_p.exists() and art_p and art_p.exists():
            csv_m = datetime.fromtimestamp(csv_p.stat().st_mtime)
            art_m = datetime.fromtimestamp(art_p.stat().st_mtime)
            if csv_m > art_m + timedelta(minutes=2):
                alerts.append("Retreinar modelo: CSV diário está mais novo que o ml_artifact.json.")
                need_retrain_model = True
            elif now - art_m > timedelta(days=10):
                alerts.append("Retreinar modelo: artifact está antigo (mais de 10 dias).")
                need_retrain_model = True
        elif csv_p and csv_p.exists() and (not art_p or not art_p.exists()):
            alerts.append("Retreinar modelo: ml_artifact.json não encontrado.")
            need_retrain_model = True
    except Exception:
        pass

    try:
        if params_p.exists() and art_p and art_p.exists():
            prm_m = datetime.fromtimestamp(params_p.stat().st_mtime)
            art_m = datetime.fromtimestamp(art_p.stat().st_mtime)
            if prm_m + timedelta(minutes=2) < art_m:
                alerts.append("Retreinar parâmetros: params_fitted.json está desatualizado vs artifact.")
                need_retrain_params = True
        else:
            alerts.append("Retreinar parâmetros: params_fitted.json ausente.")
            need_retrain_params = True
    except Exception:
        pass

    wf_at = st.session_state.get("wf_last_report_at")
    if not wf_at:
        alerts.append("Rodar walk-forward: sem relatório nesta sessão.")
        need_walkforward = True
    else:
        try:
            wf_dt = datetime.fromisoformat(str(wf_at))
            if now - wf_dt > timedelta(days=7):
                alerts.append("Rodar walk-forward: último relatório tem mais de 7 dias.")
                need_walkforward = True
        except Exception:
            alerts.append("Rodar walk-forward: timestamp inválido do último relatório.")
            need_walkforward = True

    if alerts:
        st.warning("Manutenção recomendada:\n- " + "\n- ".join(alerts))
    else:
        st.success("Modelo em dia: sem alerta de retreino no momento.")

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        if st.button("Auto manutenção completa", key="quick_auto_maint", width='stretch'):
            daily_csv, hist_csvs, art_cfg = _paths_for_ml(APP_ROOT)
            art_target = str(art_cfg or artifact_path_s or (APP_ROOT / "ml_artifact.json"))
            csv_list = [str(p) for p in (hist_csvs or []) if str(p).strip() and Path(str(p)).exists()]
            if (not csv_list) and str(daily_csv or "").strip() and Path(str(daily_csv)).exists():
                csv_list = [str(daily_csv)]

            if not csv_list:
                st.error("Sem CSV válido para manutenção automática.")
            else:
                logs: List[str] = []
                all_ok = True

                if bool(need_retrain_model):
                    _bk_art = _backup_file_if_exists(art_target, tag="pre_retrain_model")
                    if _bk_art:
                        logs.append(f"[backup] artifact => {_bk_art}")
                    _k = 26.0
                    _scale = 400.0
                    _cal_mode = "auto"
                    try:
                        _cur_art = MLArtifact.load_json(art_target)
                        _k = float(getattr(_cur_art.model.params, "k", 26.0) or 26.0)
                        _scale = float(getattr(_cur_art.model.params, "scale_points", 400.0) or 400.0)
                        _ca = float(getattr(_cur_art.calibrator, "a", 1.0) or 1.0)
                        _cb = float(getattr(_cur_art.calibrator, "b", 0.0) or 0.0)
                        if abs(_ca - 1.0) < 0.05 and abs(_cb) < 0.05:
                            _cal_mode = "no"
                    except Exception:
                        pass
                    cmd_m = [sys.executable, "-m", "mlcore.train_offline"]
                    for p in csv_list:
                        cmd_m += ["--csv", p]
                    cmd_m += ["--out", art_target, "--k", str(_k), "--scale", str(_scale), "--calibrate", _cal_mode]
                    with st.spinner("Auto manutenção: retreinando modelo..."):
                        cp_m = _run_cmd(cmd_m)
                    logs.append("[modelo]\n" + (cp_m.stdout or "") + ("\n" + cp_m.stderr if cp_m.stderr else ""))
                    if int(cp_m.returncode) != 0:
                        all_ok = False

                if bool(need_retrain_params) and all_ok:
                    _bk_prm = _backup_file_if_exists(APP_ROOT / "params_fitted.json", tag="pre_retrain_params")
                    if _bk_prm:
                        logs.append(f"[backup] params => {_bk_prm}")
                    cmd_p = [
                        sys.executable, "-m", "mlcore.fit_ml_params",
                        "--artifact", art_target,
                        "--players-artifact", str(APP_ROOT / "players_artifact.json"),
                        "--out", str(APP_ROOT / "params_fitted.json"),
                    ]
                    for p in csv_list:
                        cmd_p += ["--csv", p]
                    with st.spinner("Auto manutenção: ajustando parâmetros..."):
                        cp_p = _run_cmd(cmd_p)
                    logs.append("[params]\n" + (cp_p.stdout or "") + ("\n" + cp_p.stderr if cp_p.stderr else ""))
                    if int(cp_p.returncode) != 0:
                        all_ok = False

                if bool(need_walkforward) and all_ok:
                    with st.spinner("Auto manutenção: rodando walk-forward..."):
                        rep = _run_walkforward_cached(
                            tuple(csv_list),
                            int(st.session_state.get("wf_min_train_games", 1500) or 1500),
                            float(st.session_state.get("wf_k", 26.0) or 26.0),
                            float(st.session_state.get("wf_scale", 400.0) or 400.0),
                        )
                    st.session_state["wf_last_report"] = rep
                    st.session_state["wf_last_report_at"] = datetime.now().isoformat(timespec="seconds")
                    if isinstance(rep, dict) and rep.get("error"):
                        all_ok = False
                        logs.append("[walk-forward]\n" + json.dumps(rep, ensure_ascii=False, indent=2))

                st.session_state["quick_maint_last_log"] = "\n\n".join(logs).strip()
                if all_ok:
                    st.success("Auto manutenção concluída.")
                    st.rerun()
                else:
                    st.error("Auto manutenção encontrou falhas. Veja o log abaixo.")

    with c1:
        if st.button("Retreinar modelo agora", key="quick_retrain_model", width='stretch'):
            daily_csv, hist_csvs, art_cfg = _paths_for_ml(APP_ROOT)
            art_target = str(art_cfg or artifact_path_s or (APP_ROOT / "ml_artifact.json"))
            csv_list = [str(p) for p in (hist_csvs or []) if str(p).strip() and Path(str(p)).exists()]
            if (not csv_list) and str(daily_csv or "").strip() and Path(str(daily_csv)).exists():
                csv_list = [str(daily_csv)]
            if not csv_list:
                st.error("Sem CSV válido para retreinar modelo.")
            else:
                _bk_art = _backup_file_if_exists(art_target, tag="pre_retrain_model")
                _k = 26.0
                _scale = 400.0
                _cal_mode = "auto"
                try:
                    _cur_art = MLArtifact.load_json(art_target)
                    _k = float(getattr(_cur_art.model.params, "k", 26.0) or 26.0)
                    _scale = float(getattr(_cur_art.model.params, "scale_points", 400.0) or 400.0)
                    _ca = float(getattr(_cur_art.calibrator, "a", 1.0) or 1.0)
                    _cb = float(getattr(_cur_art.calibrator, "b", 0.0) or 0.0)
                    if abs(_ca - 1.0) < 0.05 and abs(_cb) < 0.05:
                        _cal_mode = "no"
                except Exception:
                    pass
                cmd = [sys.executable, "-m", "mlcore.train_offline"]
                for p in csv_list:
                    cmd += ["--csv", p]
                cmd += [
                    "--out", art_target,
                    "--k", str(_k),
                    "--scale", str(_scale),
                    "--calibrate", _cal_mode,
                ]
                with st.spinner("Retreinando modelo..."):
                    cp = _run_cmd(cmd)
                st.session_state["quick_maint_last_log"] = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
                if _bk_art:
                    st.session_state["quick_maint_last_log"] = f"[backup] artifact => {_bk_art}\n" + st.session_state["quick_maint_last_log"]
                if int(cp.returncode) == 0:
                    st.success("Retreino do modelo concluído.")
                    st.rerun()
                else:
                    st.error("Retreino do modelo falhou. Veja o log abaixo.")
    with c2:
        if st.button("Retreinar parâmetros agora", key="quick_retrain_params", width='stretch'):
            daily_csv, hist_csvs, art_cfg = _paths_for_ml(APP_ROOT)
            art_target = str(art_cfg or artifact_path_s or (APP_ROOT / "ml_artifact.json"))
            csv_list = [str(p) for p in (hist_csvs or []) if str(p).strip() and Path(str(p)).exists()]
            if (not csv_list) and str(daily_csv or "").strip() and Path(str(daily_csv)).exists():
                csv_list = [str(daily_csv)]
            if not csv_list:
                st.error("Sem CSV válido para retreinar parâmetros.")
            else:
                _bk_prm = _backup_file_if_exists(APP_ROOT / "params_fitted.json", tag="pre_retrain_params")
                cmd = [
                    sys.executable, "-m", "mlcore.fit_ml_params",
                    "--artifact", art_target,
                    "--players-artifact", str(APP_ROOT / "players_artifact.json"),
                    "--out", str(APP_ROOT / "params_fitted.json"),
                ]
                for p in csv_list:
                    cmd += ["--csv", p]
                with st.spinner("Retreinando parâmetros..."):
                    cp = _run_cmd(cmd)
                st.session_state["quick_maint_last_log"] = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
                if _bk_prm:
                    st.session_state["quick_maint_last_log"] = f"[backup] params => {_bk_prm}\n" + st.session_state["quick_maint_last_log"]
                if int(cp.returncode) == 0:
                    st.success("Retreino de parâmetros concluído.")
                    st.rerun()
                else:
                    st.error("Retreino de parâmetros falhou. Veja o log abaixo.")
    with c3:
        if st.button("Rodar walk-forward agora", key="quick_run_wf", width='stretch'):
            daily_csv, hist_csvs, _ = _paths_for_ml(APP_ROOT)
            csv_list = [str(p) for p in (hist_csvs or []) if str(p).strip() and Path(str(p)).exists()]
            if (not csv_list) and str(daily_csv or "").strip() and Path(str(daily_csv)).exists():
                csv_list = [str(daily_csv)]
            if not csv_list:
                st.error("Sem CSV válido para walk-forward.")
            else:
                with st.spinner("Rodando walk-forward..."):
                    rep = _run_walkforward_cached(
                        tuple(csv_list),
                        int(st.session_state.get("wf_min_train_games", 1500) or 1500),
                        float(st.session_state.get("wf_k", 26.0) or 26.0),
                        float(st.session_state.get("wf_scale", 400.0) or 400.0),
                    )
                st.session_state["wf_last_report"] = rep
                st.session_state["wf_last_report_at"] = datetime.now().isoformat(timespec="seconds")
                if isinstance(rep, dict) and not rep.get("error"):
                    st.success("Walk-forward concluído.")
                    st.rerun()
                else:
                    st.error(f"Walk-forward retornou erro: {rep.get('error') if isinstance(rep, dict) else 'desconhecido'}")

    _ql = str(st.session_state.get("quick_maint_last_log", "") or "").strip()
    if _ql:
        with st.expander("Ver último log de manutenção", expanded=False):
            st.code(_ql[:12000], language="text")


def _run_pre_delivery_checks(app_root: Path, csv_path_s: str, artifact_path_s: str, run_smoke: bool = False) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    def _add(item: str, ok: bool, detail: str) -> None:
        rows.append({"Item": str(item), "Status": ("Aprovado" if ok else "Reprovado"), "Detalhe": str(detail), "_ok": bool(ok)})

    csv_p = Path(str(csv_path_s or "")).expanduser()
    art_p = Path(str(artifact_path_s or "")).expanduser()
    params_p = app_root / "params_fitted.json"
    players_p = app_root / "players_artifact.json"
    trace_p = app_root / "ml_trace_last.json"

    def _extract_p_map_used(trace_obj: Any) -> Optional[float]:
        if not isinstance(trace_obj, dict):
            return None
        v = trace_obj.get("p_map_used")
        if v is None and isinstance(trace_obj.get("probabilities"), dict):
            v = trace_obj.get("probabilities", {}).get("p_map_used")
        try:
            fv = float(v)
            if math.isfinite(fv):
                return float(fv)
        except Exception:
            return None
        return None

    _add("CSV ativo existe", bool(csv_p.exists()), str(csv_p))
    _add("Artifact ML existe", bool(art_p.exists()), str(art_p))
    _add("params_fitted.json existe", bool(params_p.exists()), str(params_p))
    _add("players_artifact.json existe", bool(players_p.exists()), str(players_p))

    # Patch/league calibration freshness
    try:
        if csv_p.exists() and params_p.exists():
            dfp = pd.read_csv(
                str(csv_p),
                usecols=["gameid", "league", "patch", "date", "participantid"],
                low_memory=False,
            )
            dfp = dfp[pd.to_numeric(dfp.get("participantid"), errors="coerce").isin([100, 200])].copy()
            # reduce to game-level one row
            dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
            dfp = dfp.dropna(subset=["date"]).sort_values(["date", "gameid"]).drop_duplicates(subset=["gameid"], keep="last")
            cur_snap = core_build_patch_snapshot(dfp)
            fit_payload = json.loads(params_p.read_text(encoding="utf-8"))
            fit_snap = fit_payload.get("patch_snapshot") if isinstance(fit_payload, dict) else {}
            fit_ts = (fit_payload or {}).get("generated_at") if isinstance(fit_payload, dict) else None
            rec = core_recommend_recalibration(
                current_snapshot=cur_snap if isinstance(cur_snap, dict) else {},
                fitted_snapshot=fit_snap if isinstance(fit_snap, dict) else {},
                fitted_generated_at=fit_ts,
                max_age_days=7,
                min_games_new_patch=20,
            )
            st_ok = str((rec or {}).get("status", "warn")) == "ok"
            changed = rec.get("changed_leagues") if isinstance(rec, dict) else []
            _add(
                "Patch/Liga calibração em dia",
                bool(st_ok),
                f"status={rec.get('status')} | changed={len(changed or [])} | age_days={rec.get('age_days')}",
            )
        else:
            _add("Patch/Liga calibração em dia", False, "CSV ou params_fitted.json ausente")
    except Exception as exc:
        _add("Patch/Liga calibração em dia", False, f"Erro: {exc}")

    # Trace file consistency
    if trace_p.exists():
        try:
            payload = json.loads(trace_p.read_text(encoding="utf-8"))
            _mlt = payload.get("ml_trace") if isinstance(payload, dict) else {}
            _p_trace = _extract_p_map_used(_mlt)
            ok = (
                isinstance(payload, dict)
                and isinstance(payload.get("ml_trace"), dict)
                and (_p_trace is not None)
                and isinstance(payload.get("resumo_markets_trace"), list)
                and isinstance(payload.get("players_trace"), list)
            )
            _add(
                "Trace combinado válido",
                bool(ok),
                f"p_map_used={_p_trace} | resumo={len(payload.get('resumo_markets_trace') or [])} | players={len(payload.get('players_trace') or [])}",
            )
        except Exception as exc:
            _add("Trace combinado válido", False, f"Erro ao ler trace: {exc}")
    else:
        _add("Trace combinado válido", False, f"Ausente: {trace_p}")

    # Session-side trace
    _ss_trace = st.session_state.get("_ml_trace_last")
    _ss_p = _extract_p_map_used(_ss_trace)
    # fallback para estado numérico direto
    if _ss_p is None:
        try:
            _pv = float(st.session_state.get("_p_map_used", float("nan")))
            if math.isfinite(_pv):
                _ss_p = float(_pv)
        except Exception:
            _ss_p = None
    _ss_ok = isinstance(_ss_trace, dict) and (_ss_p is not None)
    _add("Sessão com p_map_used", bool(_ss_ok), f"p_map_used={_ss_p}")

    # Navigation sanity
    try:
        tabs_full = list(_pick_tabs("Completo"))
        required = {"Visão Geral", "Consistência de Mercado (Laplace)", "Series", "Players", "Campeões", "Rankings", "Parâmetros"}
        _add("Navegação mínima íntegra", required.issubset(set(tabs_full)), f"abas={len(tabs_full)}")
    except Exception as exc:
        _add("Navegação mínima íntegra", False, str(exc))

    if run_smoke:
        try:
            _env = dict(os.environ)
            if str(csv_p or "").strip() and Path(str(csv_p)).exists():
                _env["MLCORE_CSV"] = str(csv_p)
            cp = subprocess.run(
                [sys.executable, "-m", "unittest", "mlcore.tests.test_smoke", "mlcore.tests.test_ui_smoke"],
                cwd=str(app_root),
                capture_output=True,
                text=True,
                timeout=180,
                env=_env,
            )
            ok = int(cp.returncode) == 0
            out = ((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")).strip()
            if out:
                _lines = []
                for _ln in out.splitlines():
                    _l = str(_ln)
                    if "missing ScriptRunContext" in _l:
                        continue
                    _lines.append(_l)
                out = "\n".join(_lines).strip()
            _add("Smoke test", ok, out[-800:] if out else "sem saída")
        except Exception as exc:
            _add("Smoke test", False, f"Erro ao executar: {exc}")

        # Smoke específico das abas Players/Campeões (render + navegação)
        try:
            cp_pc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "mlcore.tests.test_ui_smoke.UISmokeTest.test_e2e_tab_focus_players_and_champions",
                ],
                cwd=str(app_root),
                capture_output=True,
                text=True,
                timeout=180,
                env=_env,
            )
            ok_pc = int(cp_pc.returncode) == 0
            out_pc = ((cp_pc.stdout or "") + ("\n" + cp_pc.stderr if cp_pc.stderr else "")).strip()
            if out_pc:
                _lines_pc = []
                for _ln in out_pc.splitlines():
                    _l = str(_ln)
                    if "missing ScriptRunContext" in _l:
                        continue
                    _lines_pc.append(_l)
                out_pc = "\n".join(_lines_pc).strip()
            _add("Players/Campeões smoke", ok_pc, out_pc[-800:] if out_pc else "sem saída")
        except Exception as exc:
            _add("Players/Campeões smoke", False, f"Erro ao executar: {exc}")

        try:
            cp_series = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "mlcore.tests.test_ui_smoke.UISmokeTest.test_e2e_tab_focus_series",
                ],
                cwd=str(app_root),
                capture_output=True,
                text=True,
                timeout=180,
                env=_env,
            )
            ok_series = int(cp_series.returncode) == 0
            out_series = ((cp_series.stdout or "") + ("\n" + cp_series.stderr if cp_series.stderr else "")).strip()
            if out_series:
                _lines_series = []
                for _ln in out_series.splitlines():
                    _l = str(_ln)
                    if "missing ScriptRunContext" in _l:
                        continue
                    _lines_series.append(_l)
                out_series = "\n".join(_lines_series).strip()
            _add("Series smoke", ok_series, out_series[-800:] if out_series else "sem saída")
        except Exception as exc:
            _add("Series smoke", False, f"Erro ao executar: {exc}")

    ok_count = int(sum(1 for r in rows if bool(r.get("_ok"))))
    total_count = int(len(rows))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ok_count": ok_count,
        "total_count": total_count,
        "all_ok": bool(total_count > 0 and ok_count == total_count),
        "rows": rows,
    }

if flow_stage == "select":
    _can_continue_preview = bool(teamA) and bool(teamB) and (str(teamA) != str(teamB))
    if _can_continue_preview:
        st.info("Clique em **Analisar Confronto** para abrir o preenchimento das linhas.")
    else:
        st.info("Escolha os times e clique **Analisar Confronto**.")
    st.stop()

# -----------------------------
# Etapa 2 â€” preencher linhas
# -----------------------------
if flow_stage == "lines":
    st.subheader("Linhas (preencha e confirme)")

    # garante que as linhas exibidas correspondem ao confronto atual (season + times)
    key = _matchup_key(season_label, teamA, teamB)
    st.session_state["matchup_key"] = key
    if st.session_state.get("lines_loaded_key") != key:
        _load_lines_for_matchup(key)
        st.session_state["lines_loaded_key"] = key

    st.caption("Use vÃ­rgula para separar mÃºltiplas linhas. Decimal usa **ponto**. Tempo: x, x.0, x.5.")

    with st.expander("PadrÃ£o global de linhas (opcional)", expanded=False):
        st.caption("Se vocÃª costuma usar as mesmas linhas em vÃ¡rios confrontos, salve aqui para preencher automaticamente quando nÃ£o houver linhas salvas para o confronto.")
        cP1, cP2 = st.columns(2)
        with cP1:
            if st.button("Salvar como padrÃ£o global", width='stretch', key="lines_global_save"):
                _save_lines_for_matchup(_LINES_STORE_DEFAULTS_KEY)
                st.success("PadrÃ£o global salvo.")
        with cP2:
            if st.button("Limpar padrÃ£o global", width='stretch', key="lines_global_clear"):
                s = _lines_store()
                if isinstance(s, dict):
                    s.pop(_LINES_STORE_DEFAULTS_KEY, None)
                    _persist_lines_store_disk(s)
                st.success("PadrÃ£o global removido.")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.text_input("Kills (ex: 27.5, 30.5)", key="lines_kills", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input("Torres (ex: 11.5, 12.5)", key="lines_towers", on_change=_save_lines_for_matchup, args=(key,))
    with g2:
        st.text_input("DragÃµes (ex: 4.5, 5.5)", key="lines_dragons", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input("BarÃµes (ex: 1.5)", key="lines_barons", on_change=_save_lines_for_matchup, args=(key,))
    with g3:
        st.text_input("Inibidores (ex: 1.5)", key="lines_inhib", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input("Tempo (ex: 32, 32.5)", key="lines_time", on_change=_save_lines_for_matchup, args=(key,))

    _lines_extras_ctx = st.expander("Extras (opcional)", expanded=False)
    with _lines_extras_ctx:
        st.text_input(f"Total Kills {teamA} (ex: 16.5)", key="teamA_kills_line_text", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input(f"Total Kills {teamB} (ex: 15.5)", key="teamB_kills_line_text", on_change=_save_lines_for_matchup, args=(key,))

        st.caption("Handicap: interpreta como (mÃ©trica_for + handicap) > mÃ©trica_against (sem push; use .5).")
        hc_team_opts = [teamA, teamB]
        _hc_team_cur = st.session_state.get("hc_kills_team") or teamA
        if _hc_team_cur not in hc_team_opts:
            _hc_team_cur = teamA
        st.selectbox(
            "Handicap Kills â€” Time",
            options=hc_team_opts,
            index=hc_team_opts.index(_hc_team_cur),
            key="hc_kills_team",
            on_change=_save_lines_for_matchup,
            args=(key,),
        )
        _hc_team = st.session_state.get("hc_kills_team") or teamA
        st.text_input(f"HC Kills ({_hc_team}) (ex: -2.5)", key="hc_kills_text", on_change=_save_lines_for_matchup, args=(key,))
        _hc_val_preview = _parse_single_float(st.session_state.get("hc_kills_text", ""))
        if _hc_val_preview is not None:
            _hc_other = teamB if _hc_team == teamA else teamA
            st.caption(f"Outro lado (automÃ¡tico): **{_hc_other} {(-_hc_val_preview):+g}**")
        st.text_input(f"HC Torres {teamA} (ex: -3.5)", key="hc_towers_text", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input(f"HC DragÃµes {teamA} (ex: -1.5)", key="hc_dragons_text", on_change=_save_lines_for_matchup, args=(key,))

        st.text_input("Linha Kills (para ML + Totais)", key="combo_kills_text", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input("Linha Tempo (para ML + Totais)", key="combo_time_text", on_change=_save_lines_for_matchup, args=(key,))

    c1, c2 = st.columns([2, 1])
    with c1:
        if st.button("Salvar linhas e ver resultados", width='stretch'):
            # persiste linhas para esse confronto (para reaparecer ao editar, e ao voltar sem trocar times)
            _save_lines_for_matchup(_matchup_key(season_label, teamA, teamB))
            st.session_state["flow_stage"] = "results"
            st.session_state["_analysis_force_once"] = True
            st.session_state["_consistency_refresh_pending"] = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with c2:
        if st.button("Voltar", width='stretch'):
            _reset_flow(keep_config=True)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    st.stop()

# -----------------------------
# Etapa 3 â€” resultados
# -----------------------------
# Load base dataframe (team_games) uma vez (cacheado)
team_games = _load_team_games(csv_sig)

# Janela rÃ¡pida (layout estilo print: Ano Todo / 10 / 5)
win_map = {
    "Ano todo": "All",
    "Ãšltimos 10": "Last 10",
    "Ãšltimos 5": "Last 5",
}
window_short = st.radio("Janela (HistÃ³rico)", list(win_map.keys()), index=0, horizontal=True, key="window_short")
window_opt = win_map.get(window_short, "All")

# Defaults de modos (guardados no session_state)
if "totals_mode" not in st.session_state:
    st.session_state["totals_mode"] = "map"
if "ml_mode" not in st.session_state:
    st.session_state["ml_mode"] = "map"
if "map_mode_ui" not in st.session_state:
    st.session_state["map_mode_ui"] = "avg"
if "league_mode" not in st.session_state:
    st.session_state["league_mode"] = "auto"

# Defaults avanÃ§ados
n_sims = int(st.session_state.get("n_sims", 20000))

# Controles de engine/filtros foram movidos para a aba Técnico/Parâmetros.
# Aqui no Plays apenas consumimos os valores já definidos no session_state.
years = sorted([int(y) for y in team_games["year"].dropna().unique().tolist()])
default_year = season_year if season_year in years else (years[-1] if years else season_year)
if "year_opt" not in st.session_state:
    st.session_state["year_opt"] = default_year
if "split_opt" not in st.session_state:
    st.session_state["split_opt"] = "All"
if "playoffs_opt" not in st.session_state:
    st.session_state["playoffs_opt"] = "all"
if "fixed_league" not in st.session_state:
    st.session_state["fixed_league"] = "All"

# Resolve valores atuais
year_opt = st.session_state.get("year_opt", "All")
split_opt = st.session_state.get("split_opt", "All")
playoffs_opt = st.session_state.get("playoffs_opt", "all")
fixed_league = st.session_state.get("fixed_league", "All")

totals_mode = st.session_state.get("totals_mode", "map")
ml_mode = st.session_state.get("ml_mode", "map")
map_mode_ui = st.session_state.get("map_mode_ui", "avg")
league_mode = st.session_state.get("league_mode", "auto")
weight_mode = st.session_state.get("weight_mode", "RecÃªncia (half-life)")
half_life_days = None if weight_mode == "Uniforme" else st.session_state.get("half_life_days", 180)
n_sims = int(st.session_state.get("n_sims", 20000))

# Seed determinÃ­stica para simulaÃ§Ãµes (evita variaÃ§Ã£o por rerun sem mudanÃ§a real de inputs).
_seed_mode = str(st.session_state.get("sim_seed_mode", "Deterministico (confronto+config)") or "Deterministico (confronto+config)")
if _seed_mode == "Fixo manual":
    _sim_seed_series = int(st.session_state.get("sim_seed_manual", 1337) or 1337)
else:
    _sim_seed_series = _stable_seed_from_parts(
        season_label,
        teamA,
        teamB,
        int(bo),
        str(year_opt),
        str(split_opt),
        str(playoffs_opt),
        str(league_mode),
        str(map_mode_ui),
        str(weight_mode),
        int(half_life_days) if half_life_days is not None else "uniform",
        int(n_sims),
        str(totals_mode),
        str(ml_mode),
    )
st.session_state["_sim_seed_series"] = int(_sim_seed_series)

# Recorte -> max_games
max_games = None
if window_opt.startswith("Last"):
    try:
        max_games = int(window_opt.split()[-1])
    except Exception:
        max_games = None

# Linhas (texto): em "results", prefira o store (evita perder valores quando os widgets nÃ£o estÃ£o renderizados)
_active_match_key = _matchup_key(season_label, teamA, teamB)
_saved = (_lines_store().get(_active_match_key) or {}) if flow_stage == "results" else {}

def _pick_line(name: str, default: str = "") -> str:
    if flow_stage == "results":
        return str(_saved.get(name, default) or "")
    return str(st.session_state.get(name, default) or "")

kills_lines = _pick_line("lines_kills")
towers_lines = _pick_line("lines_towers")
dragons_lines = _pick_line("lines_dragons")
barons_lines = _pick_line("lines_barons")
inhib_lines = _pick_line("lines_inhib")
time_lines = _pick_line("lines_time")

teamA_kills_line_text = _pick_line("teamA_kills_line_text")
teamB_kills_line_text = _pick_line("teamB_kills_line_text")
hc_kills_text = _pick_line("hc_kills_text")
hc_towers_text = _pick_line("hc_towers_text")
hc_dragons_text = _pick_line("hc_dragons_text")
combo_kills_text = _pick_line("combo_kills_text")
combo_time_text = _pick_line("combo_time_text")

kills = _parse_lines(kills_lines)
towers = _parse_lines(towers_lines)
dragons = _parse_lines(dragons_lines)
barons = _parse_lines(barons_lines)
inhib = _parse_lines(inhib_lines)
time_m = _parse_time_lines(time_lines)

# mapeia as linhas preenchidas (por mercado) para reutilizar em ML+Totais
lines_for = {
    "kills": kills,
    "towers": towers,
    "dragons": dragons,
    "barons": barons,
    "inhibitors": inhib,
    "time": time_m,
}


def _analysis_signature() -> int:
    return _stable_seed_from_parts(
        season_label,
        str(teamA),
        str(teamB),
        int(bo),
        str(year_opt),
        str(split_opt),
        str(playoffs_opt),
        str(league_mode),
        str(fixed_league),
        str(window_opt),
        str(weight_mode),
        int(half_life_days) if half_life_days is not None else "uniform",
        int(n_sims),
        str(totals_mode),
        str(ml_mode),
        str(map_mode_ui),
        str(kills_lines),
        str(towers_lines),
        str(dragons_lines),
        str(barons_lines),
        str(inhib_lines),
        str(time_lines),
        str(teamA_kills_line_text),
        str(teamB_kills_line_text),
        str(hc_kills_text),
        str(hc_towers_text),
        str(hc_dragons_text),
        str(combo_kills_text),
        str(combo_time_text),
        bool(st.session_state.get("resumo_combo_use_mismatch", False)),
        float(st.session_state.get("fusion_team_weight_early", 0.58) or 0.58),
        float(st.session_state.get("fusion_team_weight_mid", 0.75) or 0.75),
        float(st.session_state.get("fusion_team_weight_playoffs", 0.82) or 0.82),
        float(st.session_state.get("fusion_early_season_team_weight", 0.55) or 0.55),
        float(st.session_state.get("fusion_season_knee_games", 18.0) or 18.0),
        float(st.session_state.get("fusion_coverage_power", 1.5) or 1.5),
        float(st.session_state.get("fusion_transfer_boost", 0.35) or 0.35),
        float(st.session_state.get("fusion_divergence_pp_cap", 22.0) or 22.0),
        float(st.session_state.get("fusion_divergence_low_coverage", 0.30) or 0.30),
        float(st.session_state.get("fusion_divergence_shrink", 0.35) or 0.35),
    )


def _consistency_analysis_signature() -> int:
    return _stable_seed_from_parts(
        season_label,
        str(teamA),
        str(teamB),
        int(bo),
        str(year_opt),
        str(split_opt),
        str(playoffs_opt),
        str(league_mode),
        str(fixed_league),
        str(window_opt),
        str(weight_mode),
        int(half_life_days) if half_life_days is not None else "uniform",
        int(n_sims),
        str(totals_mode),
        str(map_mode_ui),
        str(kills_lines),
        str(towers_lines),
        str(dragons_lines),
        str(barons_lines),
        str(inhib_lines),
        str(time_lines),
        str(teamA_kills_line_text),
        str(teamB_kills_line_text),
        str(hc_kills_text),
        str(hc_towers_text),
        str(hc_dragons_text),
        str(combo_kills_text),
        str(combo_time_text),
        bool(st.session_state.get("resumo_combo_use_mismatch", False)),
    )

if bool(st.session_state.pop("_consistency_refresh_pending", False)):
    try:
        st.session_state["_consistency_analysis_sig_last"] = int(_consistency_analysis_signature())
    except Exception:
        pass

# Build Filters object
filters = Filters(
    year=None if year_opt == "All" else int(year_opt),
    league=None,  # handled by league_mode
    split=None if split_opt == "All" else str(split_opt),
    playoffs=None if playoffs_opt == "all" else (True if playoffs_opt == "true" else False),
)

league_fixed_value: Optional[str] = None
if league_mode == "fixed" and fixed_league and fixed_league != "All":
    league_fixed_value = fixed_league
    filters = Filters(
        year=filters.year,
        league=league_fixed_value,
        split=filters.split,
        playoffs=filters.playoffs,
    )

# -----------------------------
# ML odds (engine)
# -----------------------------
ml_engine = st.session_state.get("ml_engine", "mlcore_v2")
if "ml_engine_compare" not in st.session_state:
    st.session_state["ml_engine_compare"] = False

as_of_date = st.session_state.get("ml_as_of_date", date.today())

p_map_raw = p_map_cal = float("nan")
_ml_info = None

# Calcula ML core v2 somente quando necessÃ¡rio (engine=mlcore ou comparaÃ§Ã£o no debug).
if ml_engine == "mlcore_v2" or bool(st.session_state.get("ml_engine_compare", False)):
    _lgA = _lgB = ""
    _tierA = _tierB = ""

    _art_sig = ("", 0, 0)
    try:
        if artifact_path:
            _art_sig = _file_sig(artifact_path)
    except Exception:
        _art_sig = ("", 0, 0)

    _mlcore_cache_key = (
        "mlcore_v2",
        csv_sig,
        _art_sig,
        str(teamA).strip().casefold(),
        str(teamB).strip().casefold(),
        str(as_of_date),
        league_mode,
        str(getattr(filters, "league", "")) if league_mode == "fixed" else "",
        getattr(filters, "year", None),
        st.session_state.get("ach_path_ui") or "",
        bool(st.session_state.get("ml_microseg_enabled", True)),
        int(st.session_state.get("ml_microseg_lookback_games", 1200) or 1200),
        float(st.session_state.get("ml_microseg_shrink_k", 90.0) or 90.0),
        int(st.session_state.get("ml_microseg_min_n", 30) or 30),
        float(st.session_state.get("ml_microseg_strength", 0.60) or 0.60),
        float(st.session_state.get("ml_microseg_cap_pp", 8.0) or 8.0),
    )

    if st.session_state.get("_mlcore_cache_key") == _mlcore_cache_key and isinstance(st.session_state.get("_mlcore_cache_val"), dict):
        _cache = st.session_state["_mlcore_cache_val"]
        _ml_info = _cache.get("ml_info")
        p_map_raw = float(_cache.get("p_raw", float("nan")))
        p_map_cal = float(_cache.get("p_cal", float("nan")))
        if isinstance(_cache.get("microseg_info"), dict):
            st.session_state["_ml_microseg_info"] = _cache.get("microseg_info")
        _lgA = _lgB = str(_cache.get("match_league", "") or "")
        _tierA = _tierB = str(_cache.get("match_tier", "") or "")
    else:
        try:
            if league_mode == "fixed" and getattr(filters, "league", None):
                _lgA = _lgB = str(filters.league)
            else:
                _lgA = _lgB = _infer_match_league_from_csv(csv_sig, teamA, teamB, getattr(filters, "year", None))
            if _lgA:
                _tierA = _tierB = _tier_from_league(_lgA)
        except Exception:
            _lgA = _lgB = ""
            _tierA = _tierB = ""

        _ml_info = _mlcore_p_map_v2(
            artifact_path,
            teamA,
            teamB,
            as_of_date,
            ach_override_path=st.session_state.get("ach_path_ui"),
            league_hint_blue=_lgA or None,
            league_hint_red=_lgB or None,
            tier_hint_blue=_tierA or None,
            tier_hint_red=_tierB or None,
        )

        if _ml_info is not None:
            p_map_raw = float(_ml_info.get("p_raw", float("nan")))
            p_map_cal = float(_ml_info.get("p_cal", float("nan")))

            # garante simetria perfeita: p(A,B) â‰ˆ 1 - p(B,A)
            _ml_info_rev = _mlcore_p_map_v2(
                artifact_path,
                teamB,
                teamA,
                as_of_date,
                ach_override_path=st.session_state.get("ach_path_ui"),
                league_hint_blue=_lgB or None,
                league_hint_red=_lgA or None,
                tier_hint_blue=_tierB or None,
                tier_hint_red=_tierA or None,
            )
            if _ml_info_rev is not None:
                _p_raw_rev = float(_ml_info_rev.get("p_raw", float("nan")))
                _p_cal_rev = float(_ml_info_rev.get("p_cal", float("nan")))
                if math.isfinite(p_map_cal) and math.isfinite(_p_cal_rev):
                    p_map_cal = 0.5 * (p_map_cal + (1.0 - _p_cal_rev))
                if math.isfinite(p_map_raw) and math.isfinite(_p_raw_rev):
                    p_map_raw = 0.5 * (p_map_raw + (1.0 - _p_raw_rev))

            # Microsegment calibration (safe): league/split/side with hierarchical fallback.
            st.session_state.pop("_ml_microseg_info", None)
            if math.isfinite(float(p_map_cal)) and bool(st.session_state.get("ml_microseg_enabled", True)):
                try:
                    _seg_tbl = _build_ml_microseg_table(
                        csv_sig=csv_sig,
                        art_sig=_art_sig,
                        lookback_games=int(st.session_state.get("ml_microseg_lookback_games", 1200) or 1200),
                        shrink_k=float(st.session_state.get("ml_microseg_shrink_k", 90.0) or 90.0),
                    )
                    _split_hint = str(getattr(filters, "split", "") or "")
                    p_map_cal, _seg_info = _apply_ml_microseg_calibration(
                        float(p_map_cal),
                        league=str(_lgA or ""),
                        split=_split_hint,
                        side="blue",
                        table=_seg_tbl,
                        min_n=int(st.session_state.get("ml_microseg_min_n", 30) or 30),
                        strength=float(st.session_state.get("ml_microseg_strength", 0.60) or 0.60),
                        cap_pp=float(st.session_state.get("ml_microseg_cap_pp", 8.0) or 8.0),
                    )
                    st.session_state["_ml_microseg_info"] = _seg_info
                except Exception:
                    pass

        st.session_state["_mlcore_cache_key"] = _mlcore_cache_key
        st.session_state["_mlcore_cache_val"] = {
            "ml_info": _ml_info,
            "p_raw": p_map_raw,
            "p_cal": p_map_cal,
            "microseg_info": st.session_state.get("_ml_microseg_info"),
            "match_league": _lgA,
            "match_tier": _tierA,
        }

# CorreÃ§Ã£o opcional por walk-forward (liga/qualidade) antes do cÃ¡lculo final de sÃ©rie.
st.session_state.pop("wf_ml_corr_info", None)
if math.isfinite(p_map_cal) and bool(st.session_state.get("wf_apply_ml_correction", False)):
    try:
        _league_hint = ""
        if _ml_info is not None:
            _lb = str(((_ml_info.get("blue") or {}).get("league")) or "").strip()
            _lr = str(((_ml_info.get("red") or {}).get("league")) or "").strip()
            if _lb and _lb == _lr:
                _league_hint = _lb
        _rep = st.session_state.get("wf_last_report")
        p_corr, corr_info = _wf_correct_p_map(
            _rep if isinstance(_rep, dict) else None,
            float(p_map_cal),
            league_hint=_league_hint,
            strength=float(st.session_state.get("wf_ml_corr_strength", 0.60) or 0.60),
        )
        if math.isfinite(float(p_corr)):
            p_map_cal = float(p_corr)
        if isinstance(corr_info, dict):
            st.session_state["wf_ml_corr_info"] = corr_info
    except Exception:
        pass

# prob de sÃ©rie (BO1/3/5) a partir do p_map calibrado (quando existir)
p_series = float("nan")
if math.isfinite(p_map_cal):
    try:
        p_series = float(prob_win_series(float(p_map_cal), int(bo)))
    except Exception:
        p_series = float("nan")

_fit_league_hint = ""
try:
    if _ml_info is not None and not (isinstance(_ml_info, dict) and _ml_info.get("__error__")):
        _lb = str(((_ml_info.get("blue") or {}).get("league")) or "").strip()
        _lr = str(((_ml_info.get("red") or {}).get("league")) or "").strip()
        if _lb and _lb == _lr:
            _fit_league_hint = _lb
except Exception:
    _fit_league_hint = ""
_ts_used, _ps_used, _lg_used = _fit_scales_for_league(_fit_league_hint)
st.session_state["_fit_team_scale_used"] = float(_ts_used)
st.session_state["_fit_players_scale_used"] = float(_ps_used)
st.session_state["_fit_scale_league_used"] = str(_lg_used or "")

if ml_engine == "mlcore_v2" and not math.isfinite(p_map_cal):
    st.warning("NÃ£o consegui calcular p_map via ML core v2 com esse ml_artifact.json. (ML fica indisponÃ­vel, mas Totais continuam.)")


map_mode = normalize_map_mode(map_mode_ui)
hl_txt = "Uniforme" if half_life_days is None else f"{half_life_days}d"

# -----------------------------
# Totals (map or series)
# -----------------------------
dfA_used = filter_team_games(team_games, team=teamA, filters=filters, league_mode=league_mode, map_mode=map_mode, max_games=max_games)
dfB_used = filter_team_games(team_games, team=teamB, filters=filters, league_mode=league_mode, map_mode=map_mode, max_games=max_games)

pA = build_profile(team_games, team=teamA, filters=filters, league_mode=league_mode, map_mode=map_mode, half_life_days=half_life_days, max_games=max_games)
pB = build_profile(team_games, team=teamB, filters=filters, league_mode=league_mode, map_mode=map_mode, half_life_days=half_life_days, max_games=max_games)

if pA.n_games == 0 or pB.n_games == 0:
    st.error("Sem jogos suficientes com esses filtros. Ajuste Split/Playoffs/League mode ou a janela.")
    st.caption("Aviso: os mercados do confronto podem ficar incompletos, mas abas independentes (ex.: Players/Campeões) continuam disponíveis.")

totals = matchup_expected_totals(pA, pB)

# AVG maps baseline (gol.gg style)
pA_avg = pA
pB_avg = pB
avg_totals = None
if map_mode != "avg":
    pA_avg = build_profile(team_games, team=teamA, filters=filters, league_mode=league_mode, map_mode="avg", half_life_days=half_life_days, max_games=max_games)
    pB_avg = build_profile(team_games, team=teamB, filters=filters, league_mode=league_mode, map_mode="avg", half_life_days=half_life_days, max_games=max_games)
    avg_totals = matchup_expected_totals(pA_avg, pB_avg)

totals_map = totals
totals_avg = avg_totals if avg_totals is not None else totals
totals_for_lines = totals_map if totals_mode == "map" else totals_avg

# -----------------------------
# Main result tabs (formato dos prints)
# -----------------------------
if "ui_page_mode" not in st.session_state:
    st.session_state["ui_page_mode"] = "Fluxo Rápido"
with st.sidebar:
    st.markdown("### NavegaÃ§Ã£o")
    _page_modes = ["Fluxo Rápido", "Mercados", "Análise", "Completo"]
    _page_mode_cur = _fix_mojibake_text(str(st.session_state.get("ui_page_mode", "Fluxo Rápido")) or "Fluxo Rápido")
    if _page_mode_cur not in _page_modes:
        _page_mode_cur = "Fluxo Rápido"
    st.session_state["ui_page_mode"] = st.radio(
        "Modo de pÃ¡ginas",
        options=_page_modes,
        index=_page_modes.index(_page_mode_cur),
        help="Reduz a quantidade de abas visÃ­veis sem remover funcionalidades.",
    )

# Gate de análise: em "results", só recalcula após clique explícito.
if flow_stage == "results":
    _sig_now = _analysis_signature()
    _force_once = bool(st.session_state.pop("_analysis_force_once", False))
    _sig_last = st.session_state.get("_analysis_sig_last")
    if _sig_last is None or _force_once:
        st.session_state["_analysis_sig_last"] = int(_sig_now)
    _needs_reanalyze = int(st.session_state.get("_analysis_sig_last", _sig_now)) != int(_sig_now)
    ga1, ga2 = st.columns([4, 1])
    with ga1:
        if _needs_reanalyze:
            st.warning("Parâmetros/linhas mudaram. Clique em **Analisar confronto** para atualizar as odds.")
        else:
            st.caption("Análise em dia para os parâmetros atuais.")
    with ga2:
        if st.button("Analisar confronto", type="primary", width='stretch', key="analyze_gate_btn"):
            st.session_state["_analysis_sig_last"] = int(_sig_now)
            try:
                st.session_state["_consistency_analysis_sig_last"] = int(_consistency_analysis_signature())
            except Exception:
                pass
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    if _needs_reanalyze:
        st.stop()

_active_tabs = _pick_tabs(str(st.session_state.get("ui_page_mode", "Fluxo Rápido")))
_page_mode_now = str(st.session_state.get("ui_page_mode", "Fluxo Rápido") or "Fluxo Rápido")
_tab_norm = lambda t: _fix_mojibake_text(str(t)).strip().casefold()
if _page_mode_now in ("Fluxo Rápido", "Mercados"):
    _cons_tab = "Consistência de Mercado (Laplace)"
    _active_norm = {_tab_norm(t) for t in _active_tabs}
    if _tab_norm(_cons_tab) not in _active_norm:
        _visao_label = next((t for t in _active_tabs if _tab_norm(t) == _tab_norm("Visão Geral")), None)
        if _visao_label is not None:
            _ix = _active_tabs.index(_visao_label) + 1
            _active_tabs = _active_tabs[:_ix] + [_cons_tab] + _active_tabs[_ix:]
        else:
            _active_tabs = [_cons_tab] + list(_active_tabs)

# Dedupe de abas por rÃ³tulo normalizado (evita duplicar quando hÃ¡ variante com encoding quebrado).
_dedup_tabs: List[str] = []
_seen_tabs: set[str] = set()
for _t in _active_tabs:
    _k = _tab_norm(_t)
    if _k in _seen_tabs:
        continue
    _seen_tabs.add(_k)
    _dedup_tabs.append(_t)
_active_tabs = _dedup_tabs

tab_refs = {n: t for n, t in zip(_active_tabs, st.tabs(_active_tabs))}

if "Parâmetros" in tab_refs:
 with tab_refs["Parâmetros"]:
    st.markdown("### ParÃ¢metros")
    st.caption("Central de ajustes. Use Aplicar para enviar mudanÃ§as e Voltar ao padrÃ£o para resetar tudo.")
    with st.expander("Checklist de validação final", expanded=False):
        st.caption("Use esta lista antes de fechar versão/entrega.")
        st.markdown(
            "- 1) Conferir confronto base: `WE x TES`, `LNG x TT`, `GEN.G x T1`\n"
            "- 2) Validar ML Mapa e Série (sem mudança ao salvar sem editar)\n"
            "- 3) Validar Consistência: Kills/Torres/Tempo com linha principal\n"
            "- 4) Validar ML + Totais: Modelo, Hist, Mix com linhas específicas\n"
            "- 5) Validar Players: auto-preencher, analisar, odds e médias\n"
            "- 6) Validar Campeões: colagem de draft + lane matchups\n"
            "- 7) Validar `ml_trace_last.json` com `p_map_used` e traces preenchidos\n"
            "- 8) Rodar manutenção: retreino/params/walk-forward se sinalizado\n"
            "- 9) Revisar strings visuais (sem caracteres quebrados)\n"
            "- 10) Executar suíte de testes local antes de versão final"
        )
    with st.expander("Pré-entrega (automático)", expanded=False):
        st.caption("Roda checagens automáticas e mostra Aprovado/Reprovado por item.")
        b1, b2 = st.columns(2)
        with b1:
            run_pre = st.button("Rodar pré-entrega", width='stretch', key="pre_delivery_run")
        with b2:
            run_pre_smoke = st.button("Rodar pré-entrega + smoke", width='stretch', key="pre_delivery_run_smoke")

        if run_pre or run_pre_smoke:
            with st.spinner("Executando validações de pré-entrega..."):
                rep = _run_pre_delivery_checks(APP_ROOT, csv_path, artifact_path, run_smoke=bool(run_pre_smoke))
            st.session_state["_pre_delivery_report"] = rep

        _rep_pre = st.session_state.get("_pre_delivery_report")
        if isinstance(_rep_pre, dict):
            ok_n = int(_rep_pre.get("ok_count", 0) or 0)
            total_n = int(_rep_pre.get("total_count", 0) or 0)
            all_ok = bool(_rep_pre.get("all_ok", False))
            st.caption(f"Última execução: {str(_rep_pre.get('generated_at', '-'))}")
            if all_ok:
                st.success(f"Pré-entrega aprovada ({ok_n}/{total_n}).")
            else:
                st.warning(f"Pré-entrega com pendências ({ok_n}/{total_n}).")
            _df_pre = pd.DataFrame(list(_rep_pre.get("rows", []) or []))
            if not _df_pre.empty:
                _show_cols = [c for c in ["Item", "Status", "Detalhe"] if c in _df_pre.columns]
                st.dataframe(_df_pre[_show_cols], width='stretch', hide_index=True)
            try:
                st.download_button(
                    "Baixar relatório pré-entrega (JSON)",
                    data=json.dumps(_rep_pre, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="pre_delivery_report.json",
                    mime="application/json",
                    key="pre_delivery_download_json",
                )
            except Exception:
                pass

    _lap_opts = [
        "Liga 15", "Liga 10", "Liga 5", "Liga Ano",
        "H2H 15", "H2H 10", "H2H 5", "H2H Ano",
        "MÃ©dia Liga+H2H 15", "MÃ©dia Liga+H2H 10", "MÃ©dia Liga+H2H 5", "MÃ©dia Liga+H2H Ano",
    ]

    def _ui_key(k: str) -> str:
        return f"ctl_{k}"

    for _k, _v in _PARAM_DEFAULTS.items():
        _uk = _ui_key(_k)
        if _uk not in st.session_state:
            st.session_state[_uk] = st.session_state.get(_k, _v)

    st.markdown("#### Resumo")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Odd mÃ¡xima (Resumo)", min_value=1.01, max_value=10.0, step=0.01, key=_ui_key("resumo_max_odd"))
        st.checkbox("Exigir Liga", key=_ui_key("resumo_req_liga"))
        st.checkbox("Exigir H2H", key=_ui_key("resumo_req_h2h"))
        st.checkbox("Exigir Modelo", key=_ui_key("resumo_req_model"))
        st.slider("Min amostra Liga", 0, 80, step=1, key=_ui_key("resumo_min_sample_liga"))
        st.slider("Min amostra H2H", 0, 30, step=1, key=_ui_key("resumo_min_sample_h2h"))
    with c2:
        st.checkbox("Mostrar colunas Laplace", key=_ui_key("resumo_show_laplace_cols"))
        st.checkbox("Usar StatsCamps", key=_ui_key("resumo_sc_use"))
        st.slider("Limiar StatsCamps", 50, 90, key=_ui_key("resumo_sc_thr"))
        st.slider("Shrink histÃ³rico (N)", 0, 80, step=1, key=_ui_key("resumo_hist_shrink_n"))
        _idx_lap = _lap_opts.index(str(st.session_state.get(_ui_key("resumo_laplace_pick"), "Liga 15"))) if str(st.session_state.get(_ui_key("resumo_laplace_pick"), "Liga 15")) in _lap_opts else 0
        st.selectbox("Fonte Laplace para Hist/Mix", options=_lap_opts, index=_idx_lap, key=_ui_key("resumo_laplace_pick"))
    with c3:
        st.slider("Peso Mix: Modelo", 0, 100, key=_ui_key("resumo_mix_w_model"))
        st.slider("Peso Mix: Laplace", 0, 100, key=_ui_key("resumo_mix_w_laplace"))
        st.slider("Peso Mix: StatsCamps", 0, 100, key=_ui_key("resumo_mix_w_sc"))

    st.markdown("#### ML + Totais")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.selectbox("Preset ML+Totais", options=["Conservador", "Neutro", "Agressivo"], key=_ui_key("resumo_combo_preset"))
        st.checkbox("Ativar mismatch no Modelo (experimental)", key=_ui_key("resumo_combo_use_mismatch"))
    with c5:
        st.slider("Peso histÃ³rico condicional", 0.0, 1.0, step=0.05, key=_ui_key("resumo_combo_hist_w"))
        st.slider("Peso estilo over/under", 0.0, 0.5, step=0.02, key=_ui_key("resumo_combo_style_w"))
    with c6:
        st.slider("Peso forma atual", 0.0, 0.40, step=0.02, key=_ui_key("resumo_combo_form_w"))

    st.markdown("#### Motor / CÃ¡lculo")
    c7, c8, c9 = st.columns(3)
    with c7:
        st.selectbox("ML engine", options=["mlcore_v2", "elo_season_players"], key=_ui_key("ml_engine"))
        st.checkbox("Comparar com engine alternativa", key=_ui_key("ml_engine_compare"))
        st.slider("Alerta divergÃªncia (pp)", 5, 30, key=_ui_key("ml_engine_alert_pp"))
        st.checkbox("Microsegmentacao de calibracao (liga/split/side)", key=_ui_key("ml_microseg_enabled"))
        st.slider("Microseg: forca", 0.0, 1.0, step=0.05, key=_ui_key("ml_microseg_strength"))
        st.slider("Microseg: min amostra", 10, 200, step=5, key=_ui_key("ml_microseg_min_n"))
        st.checkbox("Abrir 'Detalhar cÃ¡lculo' por padrÃ£o", key=_ui_key("ml_show_calc_details_default"))
        st.checkbox("Guardiao de consistencia ML", key=_ui_key("ml_consistency_guard"))
        st.number_input("Tolerancia drift (pp)", min_value=0.00, max_value=1.00, step=0.01, key=_ui_key("ml_consistency_tol_pp"))
    with c8:
        st.selectbox("Pesos", options=["RecÃªncia (half-life)", "Uniforme"], key=_ui_key("weight_mode"))
        st.slider("Half-life (dias)", 30, 365, step=10, key=_ui_key("half_life_days"))
        st.slider("SimulaÃ§Ãµes sÃ©rie", 3000, 60000, step=1000, key=_ui_key("n_sims"))
        st.slider("Microseg: shrink k", 10.0, 300.0, step=5.0, key=_ui_key("ml_microseg_shrink_k"))
        st.slider("Microseg: lookback games", 200, 5000, step=100, key=_ui_key("ml_microseg_lookback_games"))
    with c9:
        st.selectbox("Totais", options=["map", "series"], key=_ui_key("totals_mode"))
        st.selectbox("ML (combos)", options=["map", "series"], key=_ui_key("ml_mode"))
        st.selectbox("Stats map_mode", options=["avg", "map1", "map2", "map3", "map4", "map5"], key=_ui_key("map_mode_ui"))
        st.selectbox("League mode", options=["auto", "all", "fixed"], key=_ui_key("league_mode"))
        st.slider("Microseg: cap ajuste (pp)", 0.0, 20.0, step=0.5, key=_ui_key("ml_microseg_cap_pp"))
        st.selectbox("Seed simulacao", options=["Deterministico (confronto+config)", "Fixo manual"], key=_ui_key("sim_seed_mode"))
        st.number_input("Seed manual", min_value=1, max_value=2_147_483_647, step=1, key=_ui_key("sim_seed_manual"))
        st.caption(f"Seed atual (sessao): {int(st.session_state.get('_sim_seed_series', 1337) or 1337)}")

    st.markdown("#### Lineup (Players)")
    c10, c11 = st.columns(2)
    with c10:
        st.checkbox("Aplicar ajuste por lineup", key=_ui_key("use_lineup_adjust"))
        st.slider("Lineup peso (pontos)", 0, 200, step=5, key=_ui_key("lineup_lambda_points"))
        st.selectbox("Lineup delta mode", options=["saturado", "linear"], key=_ui_key("lineup_delta_mode"))
    with c11:
        st.slider("Lineup min GP", 0, 50, step=1, key=_ui_key("lineup_min_gp"))
        st.slider("Lineup shrink m", 1, 30, step=1, key=_ui_key("lineup_shrink_m"))
        st.slider("Lineup delta cap", 0.20, 3.00, step=0.05, key=_ui_key("lineup_delta_cap"))
        st.slider("Lineup delta slope", 0.20, 4.00, step=0.05, key=_ui_key("lineup_delta_slope"))

    with st.expander("Players: pesos por lane e liga", expanded=False):
        _pw1, _pw2 = st.columns(2)
        with _pw1:
            if st.button("Preset competitivo (balanceado)", key="players_weights_preset_comp", width='stretch'):
                _preset_vals = {
                    "ml_players_lane_w_top": 0.95,
                    "ml_players_lane_w_jungle": 1.10,
                    "ml_players_lane_w_mid": 1.15,
                    "ml_players_lane_w_adc": 1.10,
                    "ml_players_lane_w_support": 0.95,
                    "ml_players_lg_w_lpl": 1.10,
                    "ml_players_lg_w_lck": 1.10,
                    "ml_players_lg_w_emea": 1.00,
                    "ml_players_lg_w_na": 0.95,
                    "ml_players_lg_w_br": 0.90,
                    "ml_players_lg_w_apac": 0.92,
                    "ml_players_lg_w_other": 0.90,
                }
                for _k, _v in _preset_vals.items():
                    st.session_state[_k] = float(_v)
                _save_app_settings_to_disk()
                st.success("Preset competitivo aplicado.")
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        with _pw2:
            if st.button("Resetar pesos (1.00)", key="players_weights_reset", width='stretch'):
                for _k in [
                    "ml_players_lane_w_top",
                    "ml_players_lane_w_jungle",
                    "ml_players_lane_w_mid",
                    "ml_players_lane_w_adc",
                    "ml_players_lane_w_support",
                    "ml_players_lg_w_lpl",
                    "ml_players_lg_w_lck",
                    "ml_players_lg_w_emea",
                    "ml_players_lg_w_na",
                    "ml_players_lg_w_br",
                    "ml_players_lg_w_apac",
                    "ml_players_lg_w_other",
                ]:
                    st.session_state[_k] = 1.0
                _save_app_settings_to_disk()
                st.success("Pesos resetados para 1.00.")
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        lw1, lw2 = st.columns(2)
        with lw1:
            st.slider("Lane Top", 0.50, 1.80, step=0.05, key=_ui_key("ml_players_lane_w_top"))
            st.slider("Lane Jungle", 0.50, 1.80, step=0.05, key=_ui_key("ml_players_lane_w_jungle"))
            st.slider("Lane Mid", 0.50, 1.80, step=0.05, key=_ui_key("ml_players_lane_w_mid"))
            st.slider("Lane ADC/Bot", 0.50, 1.80, step=0.05, key=_ui_key("ml_players_lane_w_adc"))
            st.slider("Lane Support", 0.50, 1.80, step=0.05, key=_ui_key("ml_players_lane_w_support"))
        with lw2:
            st.slider("Liga LPL (CN)", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_lpl"))
            st.slider("Liga LCK (KR)", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_lck"))
            st.slider("Liga EMEA", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_emea"))
            st.slider("Liga NA", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_na"))
            st.slider("Liga BR", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_br"))
            st.slider("Liga APAC", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_apac"))
            st.slider("Liga OTHER", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_other"))

    with st.expander("Fusão Team x Players (coerência)", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.slider("Team weight (early)", 0.0, 1.0, step=0.01, key=_ui_key("fusion_team_weight_early"))
            st.slider("Team weight (mid)", 0.0, 1.0, step=0.01, key=_ui_key("fusion_team_weight_mid"))
            st.slider("Team weight (playoffs)", 0.0, 1.0, step=0.01, key=_ui_key("fusion_team_weight_playoffs"))
        with fc2:
            st.slider("Early season team weight", 0.0, 1.0, step=0.01, key=_ui_key("fusion_early_season_team_weight"))
            st.slider("Season knee games", 1.0, 80.0, step=1.0, key=_ui_key("fusion_season_knee_games"))
            st.slider("Coverage power", 0.0, 4.0, step=0.05, key=_ui_key("fusion_coverage_power"))
        with fc3:
            st.slider("Transfer boost", 0.0, 2.0, step=0.05, key=_ui_key("fusion_transfer_boost"))
            st.slider("Cap divergência (pp)", 5.0, 80.0, step=1.0, key=_ui_key("fusion_divergence_pp_cap"))
            st.slider("Low coverage gate", 0.0, 1.0, step=0.01, key=_ui_key("fusion_divergence_low_coverage"))
            st.slider("Pull-back divergência", 0.0, 1.0, step=0.01, key=_ui_key("fusion_divergence_shrink"))
    with st.expander("Ajuste por forma da temporada (data-driven)", expanded=False):
        sf1, sf2, sf3 = st.columns(3)
        with sf1:
            st.checkbox("Ativar blend de forma no ML", key=_ui_key("ml_form_blend_enabled"))
            st.checkbox("Usar parâmetros fitted (params_fitted.json)", key=_ui_key("ml_form_use_fitted"))
            st.slider("Peso máximo da forma", 0.0, 0.95, step=0.01, key=_ui_key("ml_form_blend_max_weight"))
        with sf2:
            st.slider("Knee de jogos (amostra)", 1.0, 40.0, step=1.0, key=_ui_key("ml_form_blend_knee_games"))
            st.slider("Beta prior", 0.5, 8.0, step=0.1, key=_ui_key("ml_form_blend_beta"))
            st.slider("Potência do sinal", 0.1, 3.0, step=0.1, key=_ui_key("ml_form_blend_signal_power"))
            st.slider("Paridade elite mesma liga", 0.0, 0.8, step=0.01, key=_ui_key("ml_form_blend_parity_strength"))
        with sf3:
            st.checkbox("Ativar blend híbrido (momentum/early/macro)", key=_ui_key("ml_hybrid_blend_enabled"))
            st.slider("Peso máx híbrido", 0.0, 0.95, step=0.01, key=_ui_key("ml_hybrid_blend_max_weight"))
            st.slider("Knee híbrido (jogos)", 1.0, 80.0, step=1.0, key=_ui_key("ml_hybrid_blend_knee_games"))
            st.slider("Janela momentum", 3, 30, step=1, key=_ui_key("ml_hybrid_momentum_games"))
            st.slider("Beta prior híbrido", 0.5, 8.0, step=0.1, key=_ui_key("ml_hybrid_beta"))
            st.checkbox("Guardrail de confiança (baixa amostra)", key=_ui_key("ml_conf_guard_enabled"))
            st.slider("Min jogos (guardrail)", 1.0, 60.0, step=1.0, key=_ui_key("ml_conf_guard_min_games"))
            st.slider("Máx shrink para 50%", 0.0, 0.9, step=0.01, key=_ui_key("ml_conf_guard_max_shrink"))
            st.checkbox("Usar calibracao série BO3/BO5 fitted", key=_ui_key("ml_series_gamma_use_fitted"))

    st.markdown("#### Walk-forward + Manutenção")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.checkbox("Aplicar correcao WF no ML", key=_ui_key("wf_apply_ml_correction"))
        st.caption("Edge gate foi removido da interface para simplificar o fluxo.")
    with w2:
        st.number_input("WF min train games", min_value=200, max_value=20000, step=100, key=_ui_key("wf_min_train_games"))
        st.number_input("WF k", min_value=4.0, max_value=80.0, step=1.0, key=_ui_key("wf_k"))
        st.slider("Forca correcao WF (ML)", 0.0, 1.0, step=0.05, key=_ui_key("wf_ml_corr_strength"))
    with w3:
        st.number_input("WF scale", min_value=100.0, max_value=800.0, step=10.0, key=_ui_key("wf_scale"))
        if st.button("Rodar walk-forward", width='stretch', key="ctl_run_walkforward"):
            _daily_csv, _hist_csvs, _ = _paths_for_ml(APP_ROOT)
            _paths = [str(p) for p in (_hist_csvs or []) if str(p).strip() and Path(str(p)).exists()]
            if (not _paths) and str(_daily_csv or "").strip() and Path(str(_daily_csv)).exists():
                _paths = [str(_daily_csv)]
            if not _paths:
                st.warning("Sem CSV valido para walk-forward. Configure os CSVs no bootstrap/calibracoes.")
            else:
                with st.spinner("Rodando walk-forward..."):
                    rep = _run_walkforward_cached(
                        tuple(_paths),
                        int(st.session_state.get(_ui_key("wf_min_train_games"), 1500) or 1500),
                        float(st.session_state.get(_ui_key("wf_k"), 26.0) or 26.0),
                        float(st.session_state.get(_ui_key("wf_scale"), 400.0) or 400.0),
                    )
                st.session_state["wf_last_report"] = rep
                st.session_state["wf_last_report_at"] = datetime.now().isoformat(timespec="seconds")
                if isinstance(rep, dict) and not rep.get("error"):
                    st.success("Walk-forward concluido.")
                else:
                    st.warning(f"Walk-forward retornou erro: {rep.get('error') if isinstance(rep, dict) else 'desconhecido'}")

    _wf_rep = st.session_state.get("wf_last_report")
    if isinstance(_wf_rep, dict) and not _wf_rep.get("error"):
        _ov = _wf_rep.get("overall") if isinstance(_wf_rep.get("overall"), dict) else {}
        _roi = _ov.get("roi_proxy_mean")
        _ll = _ov.get("logloss_mean")
        _nf = _ov.get("folds")
        _league_hint = ""
        try:
            _lb = str(((_ml_info or {}).get("blue") or {}).get("league") or "").strip()
            _lr = str(((_ml_info or {}).get("red") or {}).get("league") or "").strip()
            if _lb and _lb == _lr:
                _league_hint = _lb
        except Exception:
            _league_hint = ""
        _wf_mult = _wf_gate_multiplier(_wf_rep, _league_hint)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("WF folds", value=f"{int(_nf or 0)}")
        with s2:
            st.metric("WF logloss", value=_format_num(float(_ll), 4) if _ll is not None else "-")
        with s3:
            st.metric("WF roi_proxy", value=_format_num(100.0 * float(_roi), 2) + "%" if _roi is not None else "-")
        with s4:
            st.metric("Edge factor", value=_format_num(float(_wf_mult), 2))
        if _league_hint:
            st.caption(f"Liga do confronto para ajuste do edge gate: {_league_hint}")
        _corr_i = st.session_state.get("wf_ml_corr_info")
        if bool(st.session_state.get(_ui_key("wf_apply_ml_correction"), st.session_state.get("wf_apply_ml_correction", False))) and isinstance(_corr_i, dict) and _corr_i.get("applied"):
            _pin = _corr_i.get("p_in")
            _pout = _corr_i.get("p_out")
            _shr = _corr_i.get("shrink")
            try:
                st.caption(
                    f"Correcao WF no ML ativa: p_map {float(_pin):.4f} -> {float(_pout):.4f} "
                    f"(shrink={float(_shr):.3f})."
                )
            except Exception:
                pass

    st.markdown("#### CalibraÃ§Ã£o ML (Team vs Players)")
    _fit_live = {}
    try:
        if _FIT_PARAMS_PATH.exists():
            _fit_live = _load_fitted_params_cached(_file_sig(str(_FIT_PARAMS_PATH))) or {}
    except Exception:
        _fit_live = {}
    _fit_best = (_fit_live.get("best") or {}) if isinstance(_fit_live, dict) else {}
    _fit_best_comb = (_fit_live.get("best_combined") or {}) if isinstance(_fit_live, dict) else {}
    _fit_best_form = (_fit_live.get("best_form") or {}) if isinstance(_fit_live, dict) else {}
    _fit_base = (_fit_live.get("baseline") or {}) if isinstance(_fit_live, dict) else {}
    _ts_live = float(st.session_state.get("_fit_team_scale_used", st.session_state.get("_fit_team_scale", 1.0)) or 1.0)
    _ps_live = float(st.session_state.get("_fit_players_scale_used", st.session_state.get("_fit_players_scale", 1.0)) or 1.0)
    _lg_live = str(st.session_state.get("_fit_scale_league_used", "") or "").strip()

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.metric("team_scale (ativo)", value=_format_num(_ts_live, 3))
    with f2:
        st.metric("players_scale (ativo)", value=_format_num(_ps_live, 3))
    with f3:
        _ll_b = _fit_base.get("logloss")
        st.metric("Logloss baseline", value=_format_num(float(_ll_b), 5) if _ll_b is not None else "-")
    with f4:
        _ll_f = _fit_best_comb.get("logloss", _fit_best.get("logloss"))
        _imp = _fit_best_comb.get("improvement_logloss_vs_scale_only", _fit_best.get("improvement_logloss_vs_baseline"))
        st.metric(
            "Logloss fitted (final)",
            value=_format_num(float(_ll_f), 5) if _ll_f is not None else "-",
            delta=_format_num(float(_imp), 5) if _imp is not None else None,
        )
    ff1, ff2, ff3 = st.columns(3)
    with ff1:
        st.metric("Form fitted", value="ON" if bool((_fit_best_form or {}).get("enabled", False)) else "OFF")
    with ff2:
        _wf = (_fit_best_form or {}).get("max_weight")
        st.metric("Form max_weight", value=_format_num(float(_wf), 3) if _wf is not None else "-")
    with ff3:
        _ifm = (_fit_best_form or {}).get("improvement_logloss_vs_no_form")
        st.metric("Ganho vs sem forma", value=_format_num(float(_ifm), 5) if _ifm is not None else "-")

    st.caption(
        f"Arquivo: {_FIT_PARAMS_PATH.name} | atualizado: "
        f"{str(_fit_live.get('generated_at', '-')) if isinstance(_fit_live, dict) else '-'}"
    )
    if _lg_live:
        st.caption(f"Escala por liga aplicada neste confronto: {_lg_live}")
    else:
        st.caption("Escala global aplicada (sem override por liga).")
    _ff_lg_live = str(st.session_state.get("_fit_form_cfg_league_used", "") or "").strip()
    if _ff_lg_live:
        st.caption(f"Forma da temporada fitted por liga aplicada: {_ff_lg_live}")
    else:
        st.caption("Forma da temporada fitted global aplicada (sem override por liga).")

    rf1, rf2 = st.columns([1.2, 2.8])
    with rf1:
        if st.button("Auto-ajustar ML completo", width='stretch', key="ctl_refit_ml_params"):
            _daily_csv, _hist_csvs, _ = _paths_for_ml(APP_ROOT)
            _csv_fit = [str(p) for p in (_hist_csvs or []) if str(p).strip() and Path(str(p)).exists()]
            if str(_daily_csv or "").strip() and Path(str(_daily_csv)).exists():
                _csv_fit.append(str(_daily_csv))
            _csv_fit = list(dict.fromkeys(_csv_fit))

            _players_artifact_path = str(APP_ROOT / "players_artifact.json")
            try:
                _art_obj = _load_artifact_cached(_artifact_sig(str(artifact_path)))
                _pl_meta = dict(((_art_obj.meta or {}).get("players_layer")) or {})
                _meta_p = str(_pl_meta.get("players_artifact") or "").strip()
                if _meta_p:
                    _cand = Path(_meta_p)
                    if not _cand.is_absolute():
                        _cand = (Path(str(artifact_path)).parent / _cand).resolve()
                    if _cand.exists():
                        _players_artifact_path = str(_cand)
            except Exception:
                pass

            if not _csv_fit:
                st.warning("Sem CSV vÃ¡lido para recalibrar.")
            elif not Path(str(artifact_path)).exists():
                st.warning("ml_artifact.json nÃ£o encontrado para recalibraÃ§Ã£o.")
            elif not Path(_players_artifact_path).exists():
                st.warning("players_artifact.json nÃ£o encontrado para recalibraÃ§Ã£o.")
            else:
                _bk_prm = _backup_file_if_exists(_FIT_PARAMS_PATH, tag="pre_auto_fit")
                with st.spinner("Recalibrando parÃ¢metros Team/Players via walk-forward..."):
                    rep_fit = _fit_ml_params_run(
                        artifact_path=str(artifact_path),
                        players_artifact_path=str(_players_artifact_path),
                        csv_paths=_csv_fit,
                        min_train_games=int(st.session_state.get(_ui_key("wf_min_train_games"), 300) or 300),
                    )
                    Path(_FIT_PARAMS_PATH).write_text(json.dumps(rep_fit, ensure_ascii=False, indent=2), encoding="utf-8")

                if isinstance(rep_fit, dict) and rep_fit.get("best"):
                    _b = rep_fit.get("best") or {}
                    st.session_state["_fit_team_scale"] = float(_b.get("team_scale") or 1.0)
                    st.session_state["_fit_players_scale"] = float(_b.get("players_scale") or 1.0)
                    _bf = (rep_fit.get("best_form") or {})
                    _bfs = (_bf.get("settings") or {}) if isinstance(_bf, dict) else {}
                    if isinstance(_bfs, dict) and _bfs:
                        st.session_state["ml_form_blend_enabled"] = bool(_bfs.get("ml_form_blend_enabled", True))
                        st.session_state["ml_form_blend_max_weight"] = float(_bfs.get("ml_form_blend_max_weight", 0.75) or 0.75)
                        st.session_state["ml_form_blend_knee_games"] = float(_bfs.get("ml_form_blend_knee_games", 6.0) or 6.0)
                        st.session_state["ml_form_blend_beta"] = float(_bfs.get("ml_form_blend_beta", 2.0) or 2.0)
                        st.session_state["ml_form_blend_signal_power"] = float(_bfs.get("ml_form_blend_signal_power", 1.0) or 1.0)
                        st.session_state["ml_form_blend_parity_strength"] = float(
                            _bfs.get("ml_form_blend_parity_strength", 0.35) or 0.35
                        )
                        st.session_state["ml_form_use_fitted"] = bool(_bfs.get("ml_form_use_fitted", True))
                    if _bk_prm:
                        st.caption(f"Backup criado antes do auto-ajuste: `{_bk_prm}`")
                    st.success("Auto-ajuste concluído (Team/Players + Forma) e params_fitted.json atualizado.")
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                else:
                    st.warning(f"RecalibraÃ§Ã£o retornou erro: {rep_fit.get('error') if isinstance(rep_fit, dict) else 'desconhecido'}")
        if st.button("Voltar baseline (1.0 / 1.0)", width='stretch', key="ctl_refit_reset_baseline"):
            _cur = {}
            try:
                if _FIT_PARAMS_PATH.exists():
                    _cur = _load_fitted_params_cached(_file_sig(str(_FIT_PARAMS_PATH))) or {}
            except Exception:
                _cur = {}
            if not isinstance(_cur, dict):
                _cur = {}
            _base = (_cur.get("baseline") or {}) if isinstance(_cur.get("baseline"), dict) else {}
            _best = (_cur.get("best") or {}) if isinstance(_cur.get("best"), dict) else {}
            _best["team_scale"] = 1.0
            _best["players_scale"] = 1.0
            if "logloss" not in _best and "logloss" in _base:
                _best["logloss"] = _base.get("logloss")
            if "brier" not in _best and "brier" in _base:
                _best["brier"] = _base.get("brier")
            _cur["best"] = _best
            _cur["best_by_league"] = {}
            _cur["best_form_by_league"] = {}
            _cur["best_form_by_macro"] = {}
            _cur["best_form"] = {
                "enabled": False,
                "settings": {
                    "ml_form_blend_enabled": False,
                    "ml_form_blend_max_weight": 0.75,
                    "ml_form_blend_knee_games": 6.0,
                    "ml_form_blend_beta": 2.0,
                    "ml_form_blend_signal_power": 1.0,
                    "ml_form_blend_parity_strength": 0.35,
                    "ml_form_use_fitted": False,
                },
            }
            _cur["generated_at"] = datetime.now().isoformat(timespec="seconds")
            _bk_prm = _backup_file_if_exists(_FIT_PARAMS_PATH, tag="pre_reset_baseline")
            Path(_FIT_PARAMS_PATH).write_text(json.dumps(_cur, ensure_ascii=False, indent=2), encoding="utf-8")
            st.session_state["_fit_team_scale"] = 1.0
            st.session_state["_fit_players_scale"] = 1.0
            if _bk_prm:
                st.caption(f"Backup criado antes do reset: `{_bk_prm}`")
            st.session_state["ml_form_use_fitted"] = False
            st.success("Baseline restaurado: team_scale=1.0, players_scale=1.0.")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with rf2:
        st.caption("Usa walk-forward em CSVs histórico+diário para ajustar Team/Players e também os parâmetros de forma da temporada.")

    st.markdown("#### ValidaÃ§Ã£o por Camadas (Team / Players / Fused)")
    lv1, lv2, lv3 = st.columns(3)
    with lv1:
        _lv_min_train = st.number_input(
            "LV min train games",
            min_value=200,
            max_value=40000,
            value=int(st.session_state.get("lv_min_train_games", 1500) or 1500),
            step=100,
            key="lv_min_train_games",
        )
    with lv2:
        _lv_min_league = st.number_input(
            "LV min jogos por liga",
            min_value=100,
            max_value=5000,
            value=int(st.session_state.get("lv_min_league_games", 400) or 400),
            step=50,
            key="lv_min_league_games",
        )
    with lv3:
        _lv_run = st.button("Rodar validaÃ§Ã£o por camadas", width='stretch', key="ctl_run_layer_validation")

    if _lv_run:
        _daily_csv, _hist_csvs, _ = _paths_for_ml(APP_ROOT)
        _csv_lv = [str(p) for p in (_hist_csvs or []) if str(p).strip() and Path(str(p)).exists()]
        if str(_daily_csv or "").strip() and Path(str(_daily_csv)).exists():
            _csv_lv.append(str(_daily_csv))
        _csv_lv = list(dict.fromkeys(_csv_lv))
        _players_artifact_path = str(APP_ROOT / "players_artifact.json")
        try:
            _art_obj = _load_artifact_cached(_artifact_sig(str(artifact_path)))
            _pl_meta = dict(((_art_obj.meta or {}).get("players_layer")) or {})
            _meta_p = str(_pl_meta.get("players_artifact") or "").strip()
            if _meta_p:
                _cand = Path(_meta_p)
                if not _cand.is_absolute():
                    _cand = (Path(str(artifact_path)).parent / _cand).resolve()
                if _cand.exists():
                    _players_artifact_path = str(_cand)
        except Exception:
            pass

        if not _csv_lv:
            st.warning("Sem CSV vÃ¡lido para validar camadas.")
        elif not Path(str(artifact_path)).exists():
            st.warning("ml_artifact.json nÃ£o encontrado para validaÃ§Ã£o.")
        elif not Path(_players_artifact_path).exists():
            st.warning("players_artifact.json nÃ£o encontrado para validaÃ§Ã£o.")
        else:
            with st.spinner("Rodando validaÃ§Ã£o por camadas (walk-forward)..."):
                rep_lv = _run_layer_validation_cached(
                    tuple(_csv_lv),
                    str(artifact_path),
                    str(_players_artifact_path),
                    int(_lv_min_train),
                    float(st.session_state.get("_fit_team_scale_used", st.session_state.get("_fit_team_scale", 1.0)) or 1.0),
                    float(st.session_state.get("_fit_players_scale_used", st.session_state.get("_fit_players_scale", 1.0)) or 1.0),
                    int(_lv_min_league),
                )
            st.session_state["lv_last_report"] = rep_lv
            st.session_state["lv_last_report_at"] = datetime.now().isoformat(timespec="seconds")
            if isinstance(rep_lv, dict) and not rep_lv.get("error"):
                st.success("ValidaÃ§Ã£o por camadas concluÃ­da.")
            else:
                st.warning(f"ValidaÃ§Ã£o retornou erro: {rep_lv.get('error') if isinstance(rep_lv, dict) else 'desconhecido'}")

    _lv_rep = st.session_state.get("lv_last_report")
    if isinstance(_lv_rep, dict) and not _lv_rep.get("error"):
        _models = _lv_rep.get("models") if isinstance(_lv_rep.get("models"), dict) else {}
        _rows = []
        for _name in ["team_only", "players_only", "fused_active"]:
            _m = _models.get(_name) if isinstance(_models.get(_name), dict) else {}
            _rows.append(
                {
                    "modelo": _name,
                    "n_total": int(_m.get("n_total") or 0),
                    "logloss": float(_m.get("logloss")) if _m.get("logloss") is not None else float("nan"),
                    "brier": float(_m.get("brier")) if _m.get("brier") is not None else float("nan"),
                    "ganho_vs_team_logloss": float(_m.get("gain_vs_team_logloss")) if _m.get("gain_vs_team_logloss") is not None else float("nan"),
                }
            )
        st.dataframe(pd.DataFrame(_rows), width='stretch', hide_index=True)

        _by_lg = _lv_rep.get("by_league") if isinstance(_lv_rep.get("by_league"), dict) else {}
        _lg_rows = []
        for _lg, _pack in _by_lg.items():
            if not isinstance(_pack, dict):
                continue
            _m_team = _pack.get("team_only") if isinstance(_pack.get("team_only"), dict) else {}
            _m_fus = _pack.get("fused_active") if isinstance(_pack.get("fused_active"), dict) else {}
            _lg_rows.append(
                {
                    "liga": str(_lg),
                    "n_total": int(_m_team.get("n_total") or 0),
                    "logloss_team": float(_m_team.get("logloss")) if _m_team.get("logloss") is not None else float("nan"),
                    "logloss_fused": float(_m_fus.get("logloss")) if _m_fus.get("logloss") is not None else float("nan"),
                    "ganho_fused_vs_team": float(_m_fus.get("gain_vs_team_logloss")) if _m_fus.get("gain_vs_team_logloss") is not None else float("nan"),
                }
            )
        if _lg_rows:
            _df_lg = pd.DataFrame(_lg_rows).sort_values(["ganho_fused_vs_team", "n_total"], ascending=[False, False])
            st.dataframe(_df_lg, width='stretch', hide_index=True)

        _act = _lv_rep.get("active_scales") if isinstance(_lv_rep.get("active_scales"), dict) else {}
        st.caption(
            f"ValidaÃ§Ã£o gerada em {str(_lv_rep.get('generated_at', '-'))} | "
            f"amostras={int(_lv_rep.get('n_samples') or 0)} | "
            f"team_scale={_format_num(float(_act.get('team_scale', 1.0) or 1.0), 3)} | "
            f"players_scale={_format_num(float(_act.get('players_scale', 1.0) or 1.0), 3)}"
        )
    else:
        st.info("Sem relatório de validação por camadas nesta sessão. Rode a validação para habilitar o backtest rápido.")

    st.markdown("#### Backtest rápido (ML mapa)")
    _bt_rep = st.session_state.get("lv_last_report")
    if isinstance(_bt_rep, dict) and not _bt_rep.get("error"):
        _bt_by_lg = _bt_rep.get("by_league") if isinstance(_bt_rep.get("by_league"), dict) else {}
        _bt_scope_opts = ["Global"] + sorted([str(x) for x in _bt_by_lg.keys() if str(x).strip()])
        bt1, bt2, bt3 = st.columns(3)
        with bt1:
            _bt_scope = st.selectbox("Escopo", _bt_scope_opts, index=0, key="quick_bt_scope")
        with bt2:
            _bt_model = st.selectbox("Modelo", ["team_only", "players_only", "fused_active"], index=2, key="quick_bt_model")
        with bt3:
            _bt_run = st.button("Gerar backtest rápido", width='stretch', key="quick_bt_run")

        if _bt_run:
            _pack = _bt_rep.get("models") if str(_bt_scope) == "Global" else _bt_by_lg.get(str(_bt_scope), {})
            _row = _pack.get(str(_bt_model)) if isinstance(_pack, dict) else {}
            if isinstance(_row, dict) and _row:
                _ll = float(_row.get("logloss")) if _row.get("logloss") is not None else float("nan")
                _br = float(_row.get("brier")) if _row.get("brier") is not None else float("nan")
                _n = int(_row.get("n_total") or 0)
                _gain = float(_row.get("gain_vs_team_logloss")) if _row.get("gain_vs_team_logloss") is not None else float("nan")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "escopo": str(_bt_scope),
                                "modelo": str(_bt_model),
                                "n_total": _n,
                                "logloss": _format_num(_ll, 4),
                                "brier": _format_num(_br, 4),
                                "ganho_vs_team_logloss": _format_num(_gain, 4),
                            }
                        ]
                    ),
                    width='stretch',
                    hide_index=True,
                )
            else:
                st.warning("Não há dados para esse escopo/modelo no relatório atual.")

        st.markdown("##### Comparativo de perfis (A vs B)")
        cb1, cb2, cb3, cb4 = st.columns(4)
        with cb1:
            _cmp_scope = st.selectbox("Escopo (comparativo)", _bt_scope_opts, index=0, key="quick_bt_cmp_scope")
        with cb2:
            _cmp_a = st.selectbox("Perfil A", ["team_only", "players_only", "fused_active"], index=0, key="quick_bt_cmp_a")
        with cb3:
            _cmp_b = st.selectbox("Perfil B", ["team_only", "players_only", "fused_active"], index=2, key="quick_bt_cmp_b")
        with cb4:
            _cmp_run = st.button("Comparar perfis", width='stretch', key="quick_bt_cmp_run")

        if _cmp_run:
            _pack = _bt_rep.get("models") if str(_cmp_scope) == "Global" else _bt_by_lg.get(str(_cmp_scope), {})
            _a = _pack.get(str(_cmp_a)) if isinstance(_pack, dict) else {}
            _b = _pack.get(str(_cmp_b)) if isinstance(_pack, dict) else {}
            if isinstance(_a, dict) and isinstance(_b, dict) and _a and _b:
                _a_ll = float(_a.get("logloss")) if _a.get("logloss") is not None else float("nan")
                _b_ll = float(_b.get("logloss")) if _b.get("logloss") is not None else float("nan")
                _a_br = float(_a.get("brier")) if _a.get("brier") is not None else float("nan")
                _b_br = float(_b.get("brier")) if _b.get("brier") is not None else float("nan")
                _n_a = int(_a.get("n_total") or 0)
                _n_b = int(_b.get("n_total") or 0)
                _tbl_cmp = pd.DataFrame(
                    [
                        {
                            "escopo": str(_cmp_scope),
                            "perfil": str(_cmp_a),
                            "n_total": _n_a,
                            "logloss": _format_num(_a_ll, 4),
                            "brier": _format_num(_a_br, 4),
                        },
                        {
                            "escopo": str(_cmp_scope),
                            "perfil": str(_cmp_b),
                            "n_total": _n_b,
                            "logloss": _format_num(_b_ll, 4),
                            "brier": _format_num(_b_br, 4),
                        },
                    ]
                )
                st.dataframe(_tbl_cmp, width='stretch', hide_index=True)
                if math.isfinite(_a_ll) and math.isfinite(_b_ll):
                    _d_ll = float(_a_ll - _b_ll)
                    st.caption(f"Δ logloss (A-B): {_format_num(_d_ll, 4)} (negativo = A melhor)")
                if math.isfinite(_a_br) and math.isfinite(_b_br):
                    _d_br = float(_a_br - _b_br)
                    st.caption(f"Δ brier (A-B): {_format_num(_d_br, 4)} (negativo = A melhor)")
            else:
                st.warning("Comparativo indisponível para os perfis/escopo selecionados.")

    st.markdown("#### Diagnóstico de Calibração (Reliability/ECE)")
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        _cd_bins = st.slider("Bins (reliability)", 5, 20, int(st.session_state.get("cal_diag_bins", 10) or 10), step=1, key="cal_diag_bins")
    with cd2:
        _cd_min_lg = st.number_input(
            "Calibração min jogos por liga",
            min_value=100,
            max_value=5000,
            value=int(st.session_state.get("cal_diag_min_league_games", 250) or 250),
            step=50,
            key="cal_diag_min_league_games",
        )
    with cd3:
        _cd_run = st.button("Rodar diagnóstico de calibração", width='stretch', key="ctl_run_cal_diag")

    if _cd_run:
        _daily_csv, _hist_csvs, _ = _paths_for_ml(APP_ROOT)
        _csv_cd = [str(p) for p in (_hist_csvs or []) if str(p).strip() and Path(str(p)).exists()]
        if str(_daily_csv or "").strip() and Path(str(_daily_csv)).exists():
            _csv_cd.append(str(_daily_csv))
        _csv_cd = list(dict.fromkeys(_csv_cd))
        _players_artifact_path = str(APP_ROOT / "players_artifact.json")
        try:
            _art_obj = _load_artifact_cached(_artifact_sig(str(artifact_path)))
            _pl_meta = dict(((_art_obj.meta or {}).get("players_layer")) or {})
            _meta_p = str(_pl_meta.get("players_artifact") or "").strip()
            if _meta_p:
                _cand = Path(_meta_p)
                if not _cand.is_absolute():
                    _cand = (Path(str(artifact_path)).parent / _cand).resolve()
                if _cand.exists():
                    _players_artifact_path = str(_cand)
        except Exception:
            pass

        if not _csv_cd:
            st.warning("Sem CSV válido para diagnóstico de calibração.")
        elif not Path(str(artifact_path)).exists():
            st.warning("ml_artifact.json não encontrado para diagnóstico.")
        elif not Path(_players_artifact_path).exists():
            st.warning("players_artifact.json não encontrado para diagnóstico.")
        else:
            with st.spinner("Rodando diagnóstico de calibração..."):
                rep_cd = _run_calibration_diag_cached(
                    tuple(_csv_cd),
                    str(artifact_path),
                    str(_players_artifact_path),
                    float(st.session_state.get("_fit_team_scale_used", st.session_state.get("_fit_team_scale", 1.0)) or 1.0),
                    float(st.session_state.get("_fit_players_scale_used", st.session_state.get("_fit_players_scale", 1.0)) or 1.0),
                    int(_cd_bins),
                    int(_cd_min_lg),
                )
            st.session_state["cal_diag_last_report"] = rep_cd
            st.session_state["cal_diag_last_report_at"] = datetime.now().isoformat(timespec="seconds")
            if isinstance(rep_cd, dict) and not rep_cd.get("error"):
                st.success("Diagnóstico de calibração concluído.")
            else:
                st.warning(f"Diagnóstico retornou erro: {rep_cd.get('error') if isinstance(rep_cd, dict) else 'desconhecido'}")

    _cd_rep = st.session_state.get("cal_diag_last_report")
    if isinstance(_cd_rep, dict) and not _cd_rep.get("error"):
        _m = _cd_rep.get("models") if isinstance(_cd_rep.get("models"), dict) else {}
        _rows = []
        for _name in ["team_only", "players_only", "fused_active"]:
            _r = _m.get(_name) if isinstance(_m.get(_name), dict) else {}
            _rows.append(
                {
                    "modelo": _name,
                    "n": int(_r.get("n") or 0),
                    "logloss": float(_r.get("logloss")) if _r.get("logloss") is not None else float("nan"),
                    "brier": float(_r.get("brier")) if _r.get("brier") is not None else float("nan"),
                    "ece": float(_r.get("ece")) if _r.get("ece") is not None else float("nan"),
                }
            )
        st.dataframe(pd.DataFrame(_rows), width='stretch', hide_index=True)

        _monthly = _cd_rep.get("monthly_fused") if isinstance(_cd_rep.get("monthly_fused"), list) else []
        if _monthly:
            _dfm = pd.DataFrame(_monthly).sort_values("month")
            st.dataframe(_dfm, width='stretch', hide_index=True)

        _by_lg = _cd_rep.get("by_league") if isinstance(_cd_rep.get("by_league"), dict) else {}
        if _by_lg:
            _lg_keys = sorted(_by_lg.keys())
            _lg_sel = st.selectbox("Liga (reliability)", _lg_keys, index=0, key="cal_diag_league_sel")
            _pack = _by_lg.get(_lg_sel) if isinstance(_by_lg.get(_lg_sel), dict) else {}
            _fused = _pack.get("fused_active") if isinstance(_pack.get("fused_active"), dict) else {}
            _bins_rows = _fused.get("bins") if isinstance(_fused.get("bins"), list) else []
            if _bins_rows:
                st.dataframe(pd.DataFrame(_bins_rows), width='stretch', hide_index=True)

            st.markdown("#### Semáforo de Calibração por Liga")
            sg1, sg2, sg3 = st.columns(3)
            with sg1:
                _thr_n_ok = st.number_input(
                    "N mínimo (OK)",
                    min_value=50,
                    max_value=5000,
                    value=int(st.session_state.get("cal_gate_n_ok", 250) or 250),
                    step=25,
                    key="cal_gate_n_ok",
                )
            with sg2:
                _thr_ece_ok = st.slider(
                    "ECE máximo (OK)",
                    min_value=0.01,
                    max_value=0.20,
                    value=float(st.session_state.get("cal_gate_ece_ok", 0.05) or 0.05),
                    step=0.005,
                    key="cal_gate_ece_ok",
                )
            with sg3:
                _thr_ll_warn = st.slider(
                    "Logloss (alerta)",
                    min_value=0.45,
                    max_value=1.00,
                    value=float(st.session_state.get("cal_gate_logloss_warn", 0.70) or 0.70),
                    step=0.01,
                    key="cal_gate_logloss_warn",
                )

            _rows_sg: List[Dict[str, Any]] = []
            _n_ok = int(_thr_n_ok)
            _ece_ok = float(_thr_ece_ok)
            _ll_warn = float(_thr_ll_warn)

            for _lg in _lg_keys:
                _pk = _by_lg.get(_lg) if isinstance(_by_lg.get(_lg), dict) else {}
                _fz = _pk.get("fused_active") if isinstance(_pk.get("fused_active"), dict) else {}
                _n = int(_fz.get("n") or 0)
                _ece = float(_fz.get("ece")) if _fz.get("ece") is not None else float("nan")
                _ll = float(_fz.get("logloss")) if _fz.get("logloss") is not None else float("nan")
                _br = float(_fz.get("brier")) if _fz.get("brier") is not None else float("nan")

                _status = "OK"
                _acao = "Manter"
                _motivos: List[str] = []
                if _n < _n_ok:
                    _status = "REFIT"
                    _acao = "Aumentar base e refazer fit"
                    _motivos.append(f"N<{_n_ok}")
                if math.isfinite(_ece) and _ece > (_ece_ok * 1.6):
                    _status = "REFIT"
                    _acao = "Refazer fit + calibração"
                    _motivos.append(f"ECE>{_format_num(_ece_ok*1.6,3)}")
                elif math.isfinite(_ece) and _ece > _ece_ok and _status != "REFIT":
                    _status = "ATENÇÃO"
                    _acao = "Monitorar / considerar refit"
                    _motivos.append(f"ECE>{_format_num(_ece_ok,3)}")
                if math.isfinite(_ll) and _ll > (_ll_warn + 0.08):
                    _status = "REFIT"
                    _acao = "Refazer fit (degradação de logloss)"
                    _motivos.append(f"logloss>{_format_num(_ll_warn+0.08,3)}")
                elif math.isfinite(_ll) and _ll > _ll_warn and _status == "OK":
                    _status = "ATENÇÃO"
                    _acao = "Monitorar drift"
                    _motivos.append(f"logloss>{_format_num(_ll_warn,3)}")

                _rows_sg.append(
                    {
                        "Liga": str(_lg),
                        "Status": str(_status),
                        "N": int(_n),
                        "ECE": (float(_ece) if math.isfinite(_ece) else None),
                        "Logloss": (float(_ll) if math.isfinite(_ll) else None),
                        "Brier": (float(_br) if math.isfinite(_br) else None),
                        "Motivos": ", ".join(_motivos) if _motivos else "-",
                        "Ação": str(_acao),
                    }
                )

            _df_sg = pd.DataFrame(_rows_sg)
            if not _df_sg.empty:
                _order = {"REFIT": 0, "ATENÇÃO": 1, "OK": 2}
                _df_sg["_ord"] = _df_sg["Status"].map(lambda x: _order.get(str(x), 9))
                _df_sg = _df_sg.sort_values(["_ord", "Liga"]).drop(columns=["_ord"])
                st.dataframe(_df_sg, width='stretch', hide_index=True)
                _n_refit = int((_df_sg["Status"] == "REFIT").sum())
                _n_warn = int((_df_sg["Status"] == "ATENÇÃO").sum())
                _n_ok_rows = int((_df_sg["Status"] == "OK").sum())
                if _n_refit > 0:
                    st.warning(f"Semáforo: {_n_refit} liga(s) em REFIT, {_n_warn} em ATENÇÃO, {_n_ok_rows} em OK.")
                elif _n_warn > 0:
                    st.info(f"Semáforo: {_n_warn} liga(s) em ATENÇÃO, {_n_ok_rows} em OK.")
                else:
                    st.success(f"Semáforo: todas as ligas em OK ({_n_ok_rows}).")
    elif isinstance(_cd_rep, dict) and _cd_rep.get("error"):
        st.warning("Sem semáforo de calibração: diagnóstico com erro. Rode novamente o diagnóstico.")
    else:
        st.info("Sem semáforo de calibração ainda. Rode 'Diagnóstico de calibração' para gerar o painel por liga.")

    st.markdown("#### Drift por Patch/Liga (auto)")
    try:
        _cur_snap: Dict[str, Any] = {}
        if str(csv_path or "").strip() and Path(str(csv_path)).exists():
            _dfp = pd.read_csv(
                str(csv_path),
                usecols=["gameid", "league", "patch", "date", "participantid"],
                low_memory=False,
            )
            _dfp = _dfp[pd.to_numeric(_dfp.get("participantid"), errors="coerce").isin([100, 200])].copy()
            _dfp["date"] = pd.to_datetime(_dfp["date"], errors="coerce")
            _dfp = _dfp.dropna(subset=["date"]).sort_values(["date", "gameid"]).drop_duplicates(subset=["gameid"], keep="last")
            _cur_snap = core_build_patch_snapshot(_dfp)

        _fit_live2 = {}
        if _FIT_PARAMS_PATH.exists():
            _fit_live2 = _load_fitted_params_cached(_file_sig(str(_FIT_PARAMS_PATH))) or {}
        _fit_snap = _fit_live2.get("patch_snapshot") if isinstance(_fit_live2, dict) else {}
        _fit_ts = (_fit_live2 or {}).get("generated_at") if isinstance(_fit_live2, dict) else None

        _recp = core_recommend_recalibration(
            current_snapshot=_cur_snap if isinstance(_cur_snap, dict) else {},
            fitted_snapshot=_fit_snap if isinstance(_fit_snap, dict) else {},
            fitted_generated_at=_fit_ts,
            max_age_days=7,
            min_games_new_patch=20,
        )
        _stp = str((_recp or {}).get("status", "warn"))
        _agep = (_recp or {}).get("age_days")
        _chg = (_recp or {}).get("changed_leagues") or []
        if _stp == "ok":
            st.success(f"Patch/Liga: OK | age={_format_num(float(_agep),1) if _agep is not None else '-'}d | mudanças=0")
        elif _stp == "refit":
            st.warning(f"Patch/Liga: REFIT recomendado | age={_format_num(float(_agep),1) if _agep is not None else '-'}d | mudanças={len(_chg)}")
        else:
            st.info(f"Patch/Liga: ATENÇÃO | age={_format_num(float(_agep),1) if _agep is not None else '-'}d | mudanças={len(_chg)}")
        if _chg:
            st.dataframe(pd.DataFrame(_chg), width='stretch', hide_index=True)
    except Exception as _exc_patch:
        st.info(f"Drift por patch/liga indisponível: {_exc_patch}")

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Aplicar alteraÃ§Ãµes", type="primary", width='stretch', key="ctl_apply_all"):
            _vals = {}
            for _k in _PARAM_DEFAULTS.keys():
                _vals[_k] = st.session_state.get(_ui_key(_k), _PARAM_DEFAULTS[_k])
            st.session_state["_pending_apply_param_values"] = _vals
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with b2:
        if st.button("Preset Excelente (valor)", width='stretch', key="ctl_preset_excellent"):
            st.session_state["_pending_apply_excellent"] = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        if st.button("Preset 20/10 (agressivo)", width='stretch', key="ctl_preset_twenty_ten"):
            st.session_state["_pending_apply_twenty_ten"] = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with b3:
        if st.button("Voltar ao padrÃ£o", width='stretch', key="ctl_reset_all"):
            st.session_state["_pending_reset_all_tuning"] = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with b4:
        if st.button("Salvar configuraÃ§Ã£o", width='stretch', key="ctl_save_all"):
            _save_app_settings_to_disk()
            st.success("ConfiguraÃ§Ã£o salva em disco.")

if "Visão Geral" in tab_refs:
 with tab_refs["Visão Geral"]:
    # Card de odds único: exibimos apenas o painel principal abaixo,
    # já com todas as calibrações/fusões aplicadas.
    _msi = st.session_state.get("_ml_microseg_info")
    if isinstance(_msi, dict) and _msi.get("applied"):
        try:
            _bpp = float(_msi.get("bias", 0.0)) * 100.0
        except Exception:
            _bpp = 0.0
        st.caption(
            f"Microseg calibração: aplicada ({_msi.get('key')}) | n={int(_msi.get('n', 0) or 0)} | ajuste={_bpp:+.1f} pp"
        )

    _lg_hint = ""
    try:
        _lb = str(((_ml_info or {}).get("blue") or {}).get("league") or "").strip()
        _lr = str(((_ml_info or {}).get("red") or {}).get("league") or "").strip()
        if _lb and _lb == _lr:
            _lg_hint = _lb
    except Exception:
        _lg_hint = ""
    _conf = _ml_confidence_from_layer_validation(
        st.session_state.get("lv_last_report"),
        league_hint=_lg_hint,
        n_team_eff=float(st.session_state.get("_ml_team_games_min", float("nan"))),
    )
    ccf1, ccf2, ccf3 = st.columns(3)
    with ccf1:
        st.metric("ConfianÃ§a ML", value=str(_conf.get("grade", "C (Baixa)")))
    with ccf2:
        st.metric("Score", value=f"{_format_num(float(_conf.get('score', 0.0) or 0.0), 0)}/100")
    with ccf3:
        st.metric("Base", value=str(_conf.get("source", "global")))
    st.caption(
        f"SemÃ¡foro: n={int(_conf.get('n_total') or 0)} | "
        f"logloss={_format_num(float(_conf.get('logloss')) if _conf.get('logloss') is not None else float('nan'), 4)} | "
        f"ganho_vs_team={_format_num(float(_conf.get('gain_vs_team_logloss')) if _conf.get('gain_vs_team_logloss') is not None else float('nan'), 4)}."
    )
    if not isinstance(st.session_state.get("lv_last_report"), dict):
        st.caption("Para melhorar a confianÃ§a, rode 'ValidaÃ§Ã£o por Camadas' na aba ParÃ¢metros.")

    # -----------------------------
    # Lineup (Players) â€” ajuste opcional do ML
    # -----------------------------
    _lk = _lineup_key(season_label, teamA, teamB)
    if st.session_state.get('lineup_loaded_key') != _lk:
        _load_lineup_for_matchup(_lk)
        st.session_state['lineup_loaded_key'] = _lk

    # defaults para p usado (sem ajuste)
    ml_engine = st.session_state.get("ml_engine", "mlcore_v2")

    p_map_used = float(p_map_cal) if math.isfinite(float(p_map_cal)) else float("nan")
    p_series_used = float(p_series) if math.isfinite(float(p_series)) else float("nan")
    diff_used = float(_ml_info.get('diff_points', float('nan'))) if _ml_info else float('nan')
    delta_players = 0.0

    # MantÃ©m ajuste de lineup estritamente opt-in via checkbox para evitar
    # mudanÃ§as de odd em reruns (ex.: editar/salvar linhas sem alterar inputs de ML).

    with st.expander('Lineup (Players) â€” ajuste do ML', expanded=False):
        c0, c1, c2, c3 = st.columns([1.2, 2, 1.2, 1.2])
        with c0:
            use_lineup = st.checkbox('Aplicar ajuste por players (Lineup)', value=bool(st.session_state.get('use_lineup_adjust', False)), key='use_lineup_adjust')
        with c1:
            st.text_input('CSV players (OracleElixir) â€” porTournament', value=str(st.session_state.get('players_csv_path','data/oracleselixir_players.csv') or 'data/oracleselixir_players.csv'), key='players_csv_path', on_change=_save_lineup_for_matchup, args=(_lk,))
        with c2:
            st.slider('Peso (pontos)', 0, 200, int(st.session_state.get('lineup_lambda_points', 90) or 90), step=5, key='lineup_lambda_points', on_change=_save_lineup_for_matchup, args=(_lk,))
        with c3:
            st.slider('Min GP', 0, 50, int(st.session_state.get('lineup_min_gp', 5) or 5), step=1, key='lineup_min_gp', on_change=_save_lineup_for_matchup, args=(_lk,))

        st.slider('Shrink m (GP/(GP+m))', 1, 30, int(st.session_state.get('lineup_shrink_m', 10) or 10), step=1, key='lineup_shrink_m', on_change=_save_lineup_for_matchup, args=(_lk,))

        # tenta auto-preencher pelo Ãºltimo jogo do time no CSV principal
        df_pr = None
        try:
            df_pr = _load_player_rows(csv_sig)
        except Exception:
            df_pr = None

        auto_cols = st.columns([1.2, 1, 1])
        with auto_cols[0]:
            do_auto = st.button('Auto-preencher do Ãºltimo jogo', width='stretch', key='lineup_autofill_btn')
        with auto_cols[1]:
            st.caption('Dica: muda de confronto ele tenta preencher de novo.')
        with auto_cols[2]:
            pass

        if df_pr is not None and (do_auto or st.session_state.get('_lineup_autofilled_key') != _lk):
            # manual (playername) + ML (player_key) para funcionar em qualquer modo
            la = _latest_lineup_for_team(df_pr, teamA, filters, league_fixed_value, pos_col="pos", player_col="player", pos_list=_POS_CANON)
            lb = _latest_lineup_for_team(df_pr, teamB, filters, league_fixed_value, pos_col="pos", player_col="player", pos_list=_POS_CANON)

            la_ml = _latest_lineup_for_team(df_pr, teamA, filters, league_fixed_value, pos_col="pos_ml", player_col="player_key", pos_list=_POS_CANON_ML)
            lb_ml = _latest_lineup_for_team(df_pr, teamB, filters, league_fixed_value, pos_col="pos_ml", player_col="player_key", pos_list=_POS_CANON_ML)

            # aplica apenas se ainda estÃ¡ vazio ou se clicou no botÃ£o
            for pos in _POS_CANON:
                kA = f'lineup_A_{_safe_key(pos)}'
                kB = f'lineup_B_{_safe_key(pos)}'
                if do_auto or not str(st.session_state.get(kA,'') or '').strip():
                    st.session_state[kA] = str(la.get(pos, '') or '')
                if do_auto or not str(st.session_state.get(kB,'') or '').strip():
                    st.session_state[kB] = str(lb.get(pos, '') or '')

            for pos in _POS_CANON_ML:
                kA = f'lineup_ml_A_{_safe_key(pos)}'
                kB = f'lineup_ml_B_{_safe_key(pos)}'
                if do_auto or not str(st.session_state.get(kA,'') or '').strip():
                    st.session_state[kA] = str(la_ml.get(pos, '') or '')
                if do_auto or not str(st.session_state.get(kB,'') or '').strip():
                    st.session_state[kB] = str(lb_ml.get(pos, '') or '')

            st.session_state['_lineup_autofilled_key'] = _lk
            _save_lineup_for_matchup(_lk)

        # Preferir ajuste por PlayersArtifact (treinado) quando disponÃ­vel (auto-detect)
        pl_enabled = False
        pl_players_path = ""
        pl_w = 0.0
        pl_w_src = "meta"

        # tenta ler meta do ml_artifact.json (quando existir)
        try:
            _art_obj = _load_artifact_cached(_artifact_sig(str(artifact_path)))
            pl_meta = dict(((_art_obj.meta or {}).get("players_layer")) or {})
        except Exception:
            pl_meta = {}

        try:
            # 1) path do meta (pode ser relativo ao diretÃ³rio do artifact)
            _ap_dir = os.path.dirname(str(artifact_path)) if artifact_path else os.getcwd()
            _meta_p = str(pl_meta.get("players_artifact") or "").strip()
            _cand = ""
            if _meta_p:
                _cand = _meta_p
                if not os.path.isabs(_cand):
                    _cand2 = os.path.join(_ap_dir, _cand)
                    if os.path.exists(_cand2):
                        _cand = _cand2

            # 2) fallback: players_artifact.json ao lado do ml_artifact.json
            _fallback = os.path.join(_ap_dir, "players_artifact.json")
            if (not _cand) or (not os.path.exists(_cand)):
                if os.path.exists(_fallback):
                    _cand = _fallback

            if _cand and os.path.exists(_cand):
                pl_enabled = True
                pl_players_path = str(_cand)

            # peso (w_lineup): meta > UI (default 0.75)
            _w_meta = float(pl_meta.get("w_lineup", 0.0) or 0.0) if isinstance(pl_meta, dict) else 0.0
            if _w_meta and _w_meta > 0:
                pl_w = float(_w_meta)
                pl_w_src = "meta"
            else:
                pl_w_src = "ui"
                pl_w = float(st.session_state.get("ml_players_w_lineup", 1.15) or 1.15)
        except Exception:
            pl_enabled = False
            pl_players_path = ""
            pl_w_src = "ui"
            pl_w = float(st.session_state.get("ml_players_w_lineup", 1.15) or 1.15)

        # Se o artifact nÃ£o trouxe w_lineup, expÃµe slider (vocÃª pode ajustar)
        if pl_enabled and pl_w_src == "ui":
            st.slider(
                "Peso (w_lineup) â€” ML players layer",
                0.0,
                2.0,
                float(pl_w),
                step=0.05,
                key="ml_players_w_lineup",
                on_change=_save_lineup_for_matchup,
                args=(_lk,),
            )

        mode_opts = []
        if pl_enabled:
            mode_opts.append("ML players layer")
        mode_opts.append("CSV byTournament (manual)")
        lineup_mode = st.radio("Ajuste por players", options=mode_opts, horizontal=True, index=0 if pl_enabled else 0)

        use_ml_lineup = bool(pl_enabled) and (lineup_mode == "ML players layer")

        # carrega impacto (CSV) somente se for usar modo manual ou para sugestÃµes
        df_imp = pd.DataFrame(columns=['team','pos','player','gp','score'])
        try:
            psig = _file_sig(str(st.session_state.get('players_csv_path','data/oracleselixir_players.csv') or 'data/oracleselixir_players.csv'))
            df_imp = _load_players_impact(psig)
            df_imp = _apply_shrink_m(df_imp, int(st.session_state.get('lineup_shrink_m', 10) or 10))
        except Exception:
            df_imp = pd.DataFrame(columns=['team','pos','player','gp','score'])

        teams_imp = sorted(df_imp['team'].dropna().astype(str).unique().tolist()) if df_imp is not None and not df_imp.empty else []

        # tenta casar nomes de time do app com o CSV de players (modo manual)
        teamA_key = _best_match(teamA, teams_imp) if teams_imp else ''
        teamB_key = _best_match(teamB, teams_imp) if teams_imp else ''

        def _choices_for(team_ui: str, team_key: str) -> dict:
            # por posiÃ§Ã£o, junta: (a) players do df_imp, (b) fallback do CSV principal (quando disponÃ­vel)
            out = {p: [] for p in _POS_CANON}
            if df_imp is not None and not df_imp.empty and team_key:
                for pos in _POS_CANON:
                    sub = df_imp[(df_imp['team'].astype(str) == str(team_key)) & (df_imp['pos'].astype(str) == str(pos))].copy()
                    if not sub.empty:
                        sub = sub.sort_values(['score','gp'], ascending=[False, False])
                        out[pos] += sub['player'].astype(str).tolist()

            if df_pr is not None and not df_pr.empty:
                try:
                    d2 = df_pr[df_pr['team'].astype(str) == str(team_ui)].copy()
                    d2['pos_can'] = d2['pos'].apply(_norm_pos)
                    for pos in _POS_CANON:
                        out[pos] += d2[d2['pos_can'] == pos]['player'].dropna().astype(str).unique().tolist()
                except Exception:
                    pass

            for pos in _POS_CANON:
                # unique + ordena
                vals = [v.strip() for v in out[pos] if str(v).strip()]
                # mantÃ©m ordem aproximada (primeiro os mais fortes do df_imp)
                seen = set()
                uniq = []
                for v in vals:
                    if v not in seen:
                        uniq.append(v)
                        seen.add(v)
                out[pos] = uniq
            return out

        def _choices_for_ml(team_ui: str) -> tuple[dict, dict]:
            """Choices + display map (player_key/id -> playername) from match_data."""
            out = {p: [] for p in _POS_CANON_ML}
            disp: dict[str, str] = {}
            if df_pr is None or df_pr.empty:
                return out, disp
            d2 = df_pr[df_pr['team'].astype(str) == str(team_ui)].copy()
            if d2.empty:
                return out, disp
            # garante colunas
            if 'player_key' not in d2.columns:
                d2['player_key'] = d2.get('playerid', '').astype(str).str.strip()
                d2.loc[d2['player_key'].astype(str).str.strip() == '', 'player_key'] = d2.get('player', '').astype(str).str.strip()
            if 'pos_ml' not in d2.columns:
                d2['pos_ml'] = d2.get('pos', '').apply(_norm_pos_ml)

            # display: agrega possÃ­veis aliases por id
            try:
                grp = d2.groupby('player_key')['player'].agg(lambda s: [x for i, x in enumerate([str(v).strip() for v in s.tolist() if str(v).strip()]) if x and x not in [str(v).strip() for v in s.tolist()[:i]]])
                for pid, names in grp.items():
                    if not pid:
                        continue
                    if isinstance(names, list) and names:
                        main = names[0]
                        alts = [n for n in names[1:] if n and n != main]
                        if alts:
                            disp[str(pid)] = f"{main} ({' / '.join(alts[:2])})"
                        else:
                            disp[str(pid)] = str(main)
                    else:
                        disp[str(pid)] = str(pid)
            except Exception:
                pass

            for pos in _POS_CANON_ML:
                sub = d2[d2['pos_ml'].astype(str) == str(pos)].copy()
                if sub.empty:
                    continue
                # ordena por frequÃªncia (games) e recÃªncia
                sub['player_key'] = sub['player_key'].astype(str).str.strip()
                sub = sub[sub['player_key'] != '']
                if sub.empty:
                    continue
                try:
                    freq = sub.groupby('player_key').agg(gp=('gameid','nunique'), last=('date','max')).reset_index()
                    freq = freq.sort_values(['gp','last'], ascending=[False, False])
                    out[pos] = freq['player_key'].astype(str).tolist()
                except Exception:
                    out[pos] = sorted(sub['player_key'].astype(str).unique().tolist())

            return out, disp

        if use_ml_lineup:
            choicesA, dispA = _choices_for_ml(teamA)
            choicesB, dispB = _choices_for_ml(teamB)
        else:
            choicesA = _choices_for(teamA, teamA_key)
            choicesB = _choices_for(teamB, teamB_key)
            dispA, dispB = {}, {}

        # render selectors
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f'##### {teamA}')
            pos_list_ui = _POS_CANON_ML if use_ml_lineup else _POS_CANON
            prefix = 'lineup_ml' if use_ml_lineup else 'lineup'
            for pos in pos_list_ui:
                k = f'{prefix}_A_{_safe_key(pos)}'
                opts = [''] + (choicesA.get(pos, []) if isinstance(choicesA, dict) else [])
                cur = str(st.session_state.get(k, '') or '')
                idx = opts.index(cur) if cur in opts else 0
                if use_ml_lineup:
                    st.selectbox(
                        f"{pos} (ADC)" if pos == 'Bot' else pos,
                        options=opts,
                        index=idx,
                        key=k,
                        format_func=lambda pid: '' if not pid else str(dispA.get(str(pid), pid)),
                        on_change=_save_lineup_for_matchup,
                        args=(_lk,),
                    )
                else:
                    st.selectbox(pos, options=opts, index=idx, key=k, on_change=_save_lineup_for_matchup, args=(_lk,))
        with colB:
            st.markdown(f'##### {teamB}')
            pos_list_ui = _POS_CANON_ML if use_ml_lineup else _POS_CANON
            prefix = 'lineup_ml' if use_ml_lineup else 'lineup'
            for pos in pos_list_ui:
                k = f'{prefix}_B_{_safe_key(pos)}'
                opts = [''] + (choicesB.get(pos, []) if isinstance(choicesB, dict) else [])
                cur = str(st.session_state.get(k, '') or '')
                idx = opts.index(cur) if cur in opts else 0
                if use_ml_lineup:
                    st.selectbox(
                        f"{pos} (ADC)" if pos == 'Bot' else pos,
                        options=opts,
                        index=idx,
                        key=k,
                        format_func=lambda pid: '' if not pid else str(dispB.get(str(pid), pid)),
                        on_change=_save_lineup_for_matchup,
                        args=(_lk,),
                    )
                else:
                    st.selectbox(pos, options=opts, index=idx, key=k, on_change=_save_lineup_for_matchup, args=(_lk,))

        # scoring + debug
        def _score_team(team_key: str, lineup: dict) -> tuple[float, int, int, int]:
            if not team_key or df_imp is None or df_imp.empty:
                return 0.0, 0, 0, 0
            score = 0.0
            found = 0
            total = 0
            gp_sum = 0
            min_gp = int(st.session_state.get('lineup_min_gp', 5) or 5)
            for pos in _POS_CANON:
                total += 1
                pl = str(lineup.get(pos, '') or '').strip()
                if not pl:
                    continue
                sub = df_imp[(df_imp['team'].astype(str) == str(team_key)) & (df_imp['pos'].astype(str) == str(pos))]
                if sub.empty:
                    continue
                # tenta match do player
                pl_key = _best_match(pl, sub['player'].astype(str).tolist(), cutoff=0.68)
                if not pl_key:
                    continue
                row = sub[sub['player'].astype(str) == str(pl_key)].head(1)
                if row.empty:
                    continue
                gp = int(row.iloc[0].get('gp', 0) or 0)
                if gp < min_gp:
                    continue
                sc = float(row.iloc[0].get('score', 0.0) or 0.0)
                score += sc
                found += 1
                gp_sum += int(gp)
            return float(score), int(found), int(total), int(gp_sum)

        lineupA_sel = _get_lineup_selected('A', mode='ml' if use_ml_lineup else 'manual')
        lineupB_sel = _get_lineup_selected('B', mode='ml' if use_ml_lineup else 'manual')

        def _transfer_signal_from_details(details: list[dict]) -> float:
            """0..1 signal: stronger->weaker league moves increase reliance on players."""
            if not isinstance(details, list) or not details:
                return 0.0
            good = 0.0
            total = 0.0
            for row in details:
                if not isinstance(row, dict) or not bool(row.get("used")):
                    continue
                for side in ("blue_dbg", "red_dbg"):
                    d = row.get(side) or {}
                    if not isinstance(d, dict):
                        continue
                    lt = str(d.get("last_tier") or "").strip()
                    ct = str(d.get("cur_tier") or "").strip()
                    if not lt or not ct:
                        continue
                    total += 1.0
                    try:
                        if int(tier_priority(lt)) < int(tier_priority(ct)):
                            good += 1.0
                    except Exception:
                        continue
            if total <= 0:
                return 0.0
            return max(0.0, min(1.0, good / total))

        # defaults
        delta_players = 0.0
        delta_players_eff = 0.0
        diff_used = float(_ml_info.get('diff_points', float('nan'))) if _ml_info is not None else float('nan')
        p_map_used = float(p_map_cal)
        p_series_used = float(p_series)
        trace_p_map_team = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        trace_p_map_players = float("nan")
        trace_p_map_fused = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        trace_n_team_eff = float("nan")
        trace_n_players_eff = float("nan")
        trace_fusion_info: Dict[str, Any] = {}
        trace_form_info: Dict[str, Any] = {}
        trace_hybrid_info: Dict[str, Any] = {}
        trace_conf_guard_info: Dict[str, Any] = {}
        trace_series_gamma_info: Dict[str, Any] = {}
        trace_pipeline_steps: List[Dict[str, Any]] = []
        trace_pipeline_order: List[str] = [
            "engine_lineup",
            "season_form_blend",
            "hybrid_blend",
            "confidence_guard",
            "consistency_guard",
            "series_gamma",
        ]
        _p_pipe_prev = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        _ps_pipe_prev = float(p_series_used) if math.isfinite(float(p_series_used)) else float("nan")

        _delta_mode = str(st.session_state.get("lineup_delta_mode", "saturado") or "saturado")
        _delta_cap = float(st.session_state.get("lineup_delta_cap", 1.6) or 1.6)
        _delta_slope = float(st.session_state.get("lineup_delta_slope", 2.4) or 2.4)

        if lineup_mode == "ML players layer" and pl_enabled:
            try:
                p_art = _load_players_artifact_cached(_file_sig(str(pl_players_path)))
                match_league = str(((_ml_info.get("blue") or {}).get("league")) or "").strip()
                if not match_league:
                    match_league = str(((_ml_info.get("red") or {}).get("league")) or "").strip()
                _lane_w = {
                    "Top": float(st.session_state.get("ml_players_lane_w_top", 1.0) or 1.0),
                    "Jungle": float(st.session_state.get("ml_players_lane_w_jungle", 1.0) or 1.0),
                    "Mid": float(st.session_state.get("ml_players_lane_w_mid", 1.0) or 1.0),
                    "Bot": float(st.session_state.get("ml_players_lane_w_adc", 1.0) or 1.0),
                    "Support": float(st.session_state.get("ml_players_lane_w_support", 1.0) or 1.0),
                    "ADC": float(st.session_state.get("ml_players_lane_w_adc", 1.0) or 1.0),
                }
                _league_w = {
                    "LPL": float(st.session_state.get("ml_players_lg_w_lpl", 1.0) or 1.0),
                    "CN": float(st.session_state.get("ml_players_lg_w_lpl", 1.0) or 1.0),
                    "LCK": float(st.session_state.get("ml_players_lg_w_lck", 1.0) or 1.0),
                    "KR": float(st.session_state.get("ml_players_lg_w_lck", 1.0) or 1.0),
                    "EMEA": float(st.session_state.get("ml_players_lg_w_emea", 1.0) or 1.0),
                    "NA": float(st.session_state.get("ml_players_lg_w_na", 1.0) or 1.0),
                    "BR": float(st.session_state.get("ml_players_lg_w_br", 1.0) or 1.0),
                    "APAC": float(st.session_state.get("ml_players_lg_w_apac", 1.0) or 1.0),
                    "OTHER": float(st.session_state.get("ml_players_lg_w_other", 1.0) or 1.0),
                    "DEFAULT": 1.0,
                }
                info = compute_delta_and_confidence(
                    p_art,
                    blue_lineup=lineupA_sel,
                    red_lineup=lineupB_sel,
                    as_of=as_of_date,
                    match_league=match_league,
                    require_both_sides=False,
                    lane_weights=_lane_w,
                    league_weights=_league_w,
                )
                delta_players = float(info.get("delta", 0.0) or 0.0)
                delta_players_eff = _delta_players_effect(
                    delta_players,
                    mode=_delta_mode,
                    cap=_delta_cap,
                    slope=_delta_slope,
                )
                cov = float(info.get("coverage", 0.0) or 0.0)
                cov_ratio = float(cov / max(float(info.get("coverage_total", 5) or 5), 1.0))
                freshness_avg = float(info.get("freshness_avg", 0.0) or 0.0)
                transfer_sig = _transfer_signal_from_details(info.get("details") or [])
                st.caption(
                    f"Players layer: delta_raw={_format_num(delta_players,3)} | delta_eff={_format_num(delta_players_eff,3)} ({_delta_mode}) | "
                    f"coverage={_format_pct(cov_ratio)} | freshness={_format_pct(freshness_avg)} | transfer_sig={_format_num(transfer_sig,2)} | "
                    f"w_lineup={_format_num(pl_w,3)} | league_w={_format_num(float(info.get('league_weight', 1.0) or 1.0),2)}"
                )

                # aplica ajuste no LOGIT do p_raw e depois recalibra
                if ml_engine == "mlcore_v2" and use_lineup and _ml_info is not None:
                    base_p_raw = float(_ml_info.get("p_raw", float('nan')))
                    base_p_raw = min(max(base_p_raw, 1e-6), 1 - 1e-6)
                    logit_base = math.log(base_p_raw / (1.0 - base_p_raw))
                    logit_used = float(logit_base + float(pl_w) * float(delta_players_eff))
                    p_raw_used = 1.0 / (1.0 + math.exp(-logit_used))
                    p_map_players = float(_calibrate_p_from_artifact(artifact_path, float(p_raw_used)))
                    p_map_team = float(p_map_used)
                    n_team_eff = float(
                        min(
                            float(((_ml_info.get("blue") or {}).get("games_played", 0) or 0)),
                            float(((_ml_info.get("red") or {}).get("games_played", 0) or 0)),
                        )
                    )
                    n_players_eff = float(
                        max(0.0, float(info.get("avg_gp_min", 0.0) or 0.0) * float(cov_ratio) * float(freshness_avg))
                    )
                    fus = _fuse_probs_by_precision(
                        p_team=p_map_team,
                        p_players=p_map_players,
                        n_team=n_team_eff,
                        n_players=n_players_eff,
                        team_scale=float(st.session_state.get("_fit_team_scale_used", st.session_state.get("_fit_team_scale", 1.0)) or 1.0),
                        players_scale=float(st.session_state.get("_fit_players_scale_used", st.session_state.get("_fit_players_scale", 1.0)) or 1.0),
                        coverage=float(cov_ratio),
                        transfer_signal=float(transfer_sig),
                        season_phase=("playoffs" if str(playoffs_opt).lower() in {"playoffs", "true", "1"} else "auto"),
                    )
                    p_map_used = float(fus.get("p_fused", p_map_team))
                    trace_fusion_info = dict(fus)
                    trace_p_map_team = float(p_map_team)
                    trace_p_map_players = float(p_map_players)
                    trace_p_map_fused = float(p_map_used)
                    trace_n_team_eff = float(n_team_eff)
                    trace_n_players_eff = float(n_players_eff)
                    trace_fusion_info["freshness_avg"] = float(freshness_avg)
                    try:
                        p_series_used = float(prob_win_series(float(p_map_used), int(bo)))
                    except Exception:
                        p_series_used = float('nan')

                    # converte logit -> pontos (mesma escala do ML core v2)
                    if math.isfinite(float(_ml_info.get('scale_points', float('nan')))):
                        scale_pts = float(_ml_info.get('scale_points'))
                        diff_used = float((scale_pts / math.log(10.0)) * logit_used)
                    st.caption(
                        f"p_map team={_format_num(p_map_team,4)} | players={_format_num(p_map_players,4)} | "
                        f"fused={_format_num(p_map_used,4)} (w_team={_format_num(fus.get('w_team', float('nan')),2)}, "
                        f"w_players={_format_num(fus.get('w_players', float('nan')),2)}, n_team={_format_num(n_team_eff,1)}, n_players={_format_num(n_players_eff,1)}) | "
                        f"p_series={_format_num(p_series_used,4)}"
                    )
                    _cg = fus.get("coherence_guard") if isinstance(fus, dict) else None
                    if isinstance(_cg, dict) and bool(_cg.get("applied")):
                        st.caption(
                            f"Coerência Team/Players aplicada: Δ={_format_num(float(_cg.get('dpp', float('nan'))),1)} pp | "
                            f"coverage={_format_pct(float(_cg.get('coverage', float('nan'))))} | "
                            f"pull={_format_num(100.0 * float(_cg.get('pull', 0.0)),1)}%."
                        )
            except Exception as e:
                st.warning(f"Falha ao aplicar ML players layer: {type(e).__name__}: {e}")
                lineup_mode = "CSV byTournament (manual)"

        if lineup_mode == "CSV byTournament (manual)":
            sA, fA, tA, gpA = _score_team(teamA_key, lineupA_sel)
            sB, fB, tB, gpB = _score_team(teamB_key, lineupB_sel)
            delta_players = float(sA - sB)
            delta_players_eff = _delta_players_effect(
                delta_players,
                mode=_delta_mode,
                cap=_delta_cap,
                slope=_delta_slope,
            )
            st.caption(
                f"Score (players, manual): {teamA}={_format_num(sA,2)} (ok {fA}/{tA}) | {teamB}={_format_num(sB,2)} (ok {fB}/{tB}) | "
                f"delta_raw={_format_num(delta_players,2)} | delta_eff={_format_num(delta_players_eff,2)} ({_delta_mode})"
            )

            # aplica ajuste no diff do ML, se possÃ­vel
            if use_lineup and _ml_info is not None and math.isfinite(float(_ml_info.get('diff_points', float('nan')))) and math.isfinite(float(_ml_info.get('scale_points', float('nan')))):
                base_diff = float(_ml_info.get('diff_points'))
                scale_pts = float(_ml_info.get('scale_points'))
                lam = float(st.session_state.get('lineup_lambda_points', 90) or 90)
                diff_used = float(base_diff + lam * float(delta_players_eff))
                p_raw_used = _p_from_diff_points(diff_used, scale_points=scale_pts)
                p_map_players = float(_calibrate_p_from_artifact(artifact_path, p_raw_used))
                p_map_team = float(p_map_used)
                cov_manual = 0.0
                try:
                    cov_manual = min(float(fA / max(tA, 1)), float(fB / max(tB, 1)))
                except Exception:
                    cov_manual = 0.0
                n_team_eff = float(
                    min(
                        float(((_ml_info.get("blue") or {}).get("games_played", 0) or 0)),
                        float(((_ml_info.get("red") or {}).get("games_played", 0) or 0)),
                    )
                )
                avg_gp_min = min(
                    float(gpA) / max(float(fA), 1.0),
                    float(gpB) / max(float(fB), 1.0),
                ) if (fA > 0 and fB > 0) else 0.0
                n_players_eff = float(max(0.0, avg_gp_min * float(cov_manual)))
                fus = _fuse_probs_by_precision(
                    p_team=p_map_team,
                    p_players=p_map_players,
                    n_team=n_team_eff,
                    n_players=n_players_eff,
                    team_scale=float(st.session_state.get("_fit_team_scale_used", st.session_state.get("_fit_team_scale", 1.0)) or 1.0),
                    players_scale=float(st.session_state.get("_fit_players_scale_used", st.session_state.get("_fit_players_scale", 1.0)) or 1.0),
                    coverage=float(cov_manual),
                    transfer_signal=0.0,
                    season_phase=("playoffs" if str(playoffs_opt).lower() in {"playoffs", "true", "1"} else "auto"),
                )
                p_map_used = float(fus.get("p_fused", p_map_team))
                trace_fusion_info = dict(fus)
                trace_p_map_team = float(p_map_team)
                trace_p_map_players = float(p_map_players)
                trace_p_map_fused = float(p_map_used)
                trace_n_team_eff = float(n_team_eff)
                trace_n_players_eff = float(n_players_eff)
                # sÃ©rie tambÃ©m
                try:
                    p_series_used = float(prob_win_series(float(p_map_used), int(bo)))
                except Exception:
                    p_series_used = float('nan')

                st.caption(
                    f"Diff base={_format_num(base_diff,1)} | diff players={_format_num(diff_used,1)} | "
                    f"p_map team={_format_num(p_map_team,4)} | players={_format_num(p_map_players,4)} | "
                    f"fused={_format_num(p_map_used,4)} (cov={_format_pct(cov_manual)}, n_team={_format_num(n_team_eff,1)}, n_players={_format_num(n_players_eff,1)})"
                )
                _cg = fus.get("coherence_guard") if isinstance(fus, dict) else None
                if isinstance(_cg, dict) and bool(_cg.get("applied")):
                    st.caption(
                        f"Coerência Team/Players aplicada: Δ={_format_num(float(_cg.get('dpp', float('nan'))),1)} pp | "
                        f"coverage={_format_pct(float(_cg.get('coverage', float('nan'))))} | "
                        f"pull={_format_num(100.0 * float(_cg.get('pull', 0.0)),1)}%."
                    )

    
    # -----------------------------
    # ML engine switch: Elo season + players (usa lineup atual)
    # -----------------------------
    _elo_info = None
    if ml_engine == "elo_season_players" or bool(st.session_state.get("ml_engine_compare", False)):
        try:
            _elo_info = _elo_season_players_p_map(
                team_games=team_games,
                csv_path=str(csv_path),
                teamA=str(teamA),
                teamB=str(teamB),
                filters=filters,
                team_half_life_days=float(st.session_state.get("elo_team_hl", 45) or 45),
                player_half_life_days=float(st.session_state.get("elo_player_hl", 90) or 90),
                k_team=float(st.session_state.get("elo_k_team", 26) or 26),
                k_player=float(st.session_state.get("elo_k_player", 24) or 24),
                shrink_m=float(st.session_state.get("elo_shrink_m", 18) or 18),
                w_players=float(st.session_state.get("elo_w_players", 0.85) or 0.85),
            )
        except Exception as e:
            _elo_info = {"__error__": f"{type(e).__name__}: {e}"}

    if ml_engine == "elo_season_players":
        if _elo_info is None or _elo_info.get("__error__"):
            st.warning("Elo season + players: não consegui calcular (ver Detalhes do ML).")
        else:
            p_map_used = float(_elo_info.get("p_cal", float("nan")))
            diff_used = float(_elo_info.get("diff_points", float("nan")))
            try:
                p_series_used = float(prob_win_series(float(p_map_used), int(bo)))
            except Exception:
                p_series_used = float("nan")
            delta_players = 0.0

    try:
        _p_after = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        _ap = (
            math.isfinite(float(_p_pipe_prev))
            and math.isfinite(float(_p_after))
            and abs(float(_p_after) - float(_p_pipe_prev)) > 1e-12
        )
        trace_pipeline_steps.append(
            {
                "stage": "engine_lineup",
                "before": (float(_p_pipe_prev) if math.isfinite(float(_p_pipe_prev)) else None),
                "after": (float(_p_after) if math.isfinite(float(_p_after)) else None),
                "applied": bool(_ap),
                "engine": str(ml_engine),
                "lineup_mode": str(locals().get("lineup_mode", "")),
            }
        )
        _p_pipe_prev = float(_p_after)
    except Exception:
        pass

    # Ajuste data-driven por forma da temporada (wins/losses no recorte atual).
    # Atua apenas no Modelo (não altera Laplace/Histórico).
    try:
        _p_before_form = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        if math.isfinite(float(p_map_used)):
            _fit_form_settings = st.session_state.get("_fit_form_cfg") if isinstance(st.session_state.get("_fit_form_cfg"), dict) else {}
            _fit_form_lg = ""
            try:
                _lg_hint_form = ""
                if str(league_mode) == "fixed" and getattr(filters, "league", None):
                    _lg_hint_form = str(filters.league or "").strip()
                if not _lg_hint_form and isinstance(dfA_used, pd.DataFrame) and (not dfA_used.empty) and ("league" in dfA_used.columns):
                    _vfa = dfA_used["league"].dropna().astype(str).str.strip()
                    if not _vfa.empty:
                        _lg_hint_form = str(_vfa.value_counts().index[0]).strip()
                if not _lg_hint_form and isinstance(dfB_used, pd.DataFrame) and (not dfB_used.empty) and ("league" in dfB_used.columns):
                    _vfb = dfB_used["league"].dropna().astype(str).str.strip()
                    if not _vfb.empty:
                        _lg_hint_form = str(_vfb.value_counts().index[0]).strip()
                _fit_form_settings, _fit_form_lg = _fit_form_for_league(_lg_hint_form)
                st.session_state["_fit_form_cfg_used"] = dict(_fit_form_settings) if isinstance(_fit_form_settings, dict) else {}
                st.session_state["_fit_form_cfg_league_used"] = str(_fit_form_lg or "")
            except Exception:
                pass
            _use_fit_form = bool(st.session_state.get("ml_form_use_fitted", True))
            _sf_enabled = bool(st.session_state.get("ml_form_blend_enabled", True))
            _sf_max_weight = float(st.session_state.get("ml_form_blend_max_weight", 0.75) or 0.75)
            _sf_knee_games = float(st.session_state.get("ml_form_blend_knee_games", 6.0) or 6.0)
            _sf_beta = float(st.session_state.get("ml_form_blend_beta", 2.0) or 2.0)
            _sf_signal_power = float(st.session_state.get("ml_form_blend_signal_power", 1.0) or 1.0)
            _sf_parity = float(st.session_state.get("ml_form_blend_parity_strength", 0.35) or 0.35)
            if _use_fit_form and isinstance(_fit_form_settings, dict) and _fit_form_settings:
                _sf_enabled = bool(_fit_form_settings.get("ml_form_blend_enabled", _sf_enabled))
                _sf_max_weight = float(_fit_form_settings.get("ml_form_blend_max_weight", _sf_max_weight) or _sf_max_weight)
                _sf_knee_games = float(_fit_form_settings.get("ml_form_blend_knee_games", _sf_knee_games) or _sf_knee_games)
                _sf_beta = float(_fit_form_settings.get("ml_form_blend_beta", _sf_beta) or _sf_beta)
                _sf_signal_power = float(_fit_form_settings.get("ml_form_blend_signal_power", _sf_signal_power) or _sf_signal_power)
                _sf_parity = float(_fit_form_settings.get("ml_form_blend_parity_strength", _sf_parity) or _sf_parity)
            _form = core_season_form_blend(
                p_model=float(p_map_used),
                df_team_a=dfA_used,
                df_team_b=dfB_used,
                enabled=bool(_sf_enabled),
                max_weight=float(_sf_max_weight),
                knee_games=float(_sf_knee_games),
                beta_prior=float(_sf_beta),
                signal_power=float(_sf_signal_power),
                elite_same_league_parity_strength=float(_sf_parity),
            )
            if isinstance(_form, dict):
                trace_form_info = dict(_form)
                if bool(_use_fit_form and _fit_form_settings):
                    trace_form_info["source"] = f"fitted:{_fit_form_lg}" if _fit_form_lg else "fitted:global"
                else:
                    trace_form_info["source"] = "manual"
                _p_bl = _form.get("p_blended")
                if _form.get("applied") and _p_bl is not None and math.isfinite(float(_p_bl)):
                    p_map_used = float(_p_bl)
                    try:
                        p_series_used = float(prob_win_series(float(p_map_used), int(bo)))
                    except Exception:
                        p_series_used = float("nan")
                    st.caption(
                        f"Season form blend: p_model={_format_num(_form.get('p_model', float('nan')),4)} | "
                        f"p_form={_format_num(_form.get('p_form', float('nan')),4)} | "
                        f"w_form={_format_num(100.0*float(_form.get('w_form',0.0)),1)}% | "
                        f"p_final={_format_num(p_map_used,4)}"
                    )
        _p_after_form = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        _ap_form = (
            bool(trace_form_info.get("applied", False))
            or (
                math.isfinite(float(_p_before_form))
                and math.isfinite(float(_p_after_form))
                and abs(float(_p_after_form) - float(_p_before_form)) > 1e-12
            )
        )
        trace_pipeline_steps.append(
            {
                "stage": "season_form_blend",
                "before": (float(_p_before_form) if math.isfinite(float(_p_before_form)) else None),
                "after": (float(_p_after_form) if math.isfinite(float(_p_after_form)) else None),
                "applied": bool(_ap_form),
                "source": str(trace_form_info.get("source", "")) if isinstance(trace_form_info, dict) else "",
            }
        )
        _p_pipe_prev = float(_p_after_form)
    except Exception:
        pass

    # Camada híbrida data-driven (momentum + early + macro), sem alterar histórico/Laplace.
    try:
        _p_before_hybrid = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        if math.isfinite(float(p_map_used)):
            _hy_on = bool(st.session_state.get("ml_hybrid_blend_enabled", True))
            _hy_w = float(st.session_state.get("ml_hybrid_blend_max_weight", 0.70) or 0.70)
            _hy_knee = float(st.session_state.get("ml_hybrid_blend_knee_games", 12.0) or 12.0)
            _hy_mom = int(st.session_state.get("ml_hybrid_momentum_games", 10) or 10)
            _hy_beta = float(st.session_state.get("ml_hybrid_beta", 2.0) or 2.0)
            _hy = core_hybrid_momentum_blend(
                p_model=float(p_map_used),
                df_team_a=dfA_used,
                df_team_b=dfB_used,
                enabled=bool(_hy_on),
                max_weight=float(_hy_w),
                knee_games=float(_hy_knee),
                momentum_games=int(_hy_mom),
                beta_prior=float(_hy_beta),
            )
            if isinstance(_hy, dict):
                trace_hybrid_info = dict(_hy)
                _p_hy = _hy.get("p_blended")
                if _hy.get("applied") and _p_hy is not None and math.isfinite(float(_p_hy)):
                    p_map_used = float(_p_hy)
                    try:
                        p_series_used = float(prob_win_series(float(p_map_used), int(bo)))
                    except Exception:
                        p_series_used = float("nan")
                    st.caption(
                        f"Hybrid blend: p_model={_format_num(_hy.get('p_model', float('nan')),4)} | "
                        f"p_hybrid={_format_num(_hy.get('p_hybrid', float('nan')),4)} | "
                        f"w_hybrid={_format_num(100.0*float(_hy.get('w_hybrid',0.0)),1)}% | "
                        f"p_final={_format_num(p_map_used,4)}"
                    )
        _p_after_hybrid = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        _ap_hybrid = (
            bool(trace_hybrid_info.get("applied", False))
            or (
                math.isfinite(float(_p_before_hybrid))
                and math.isfinite(float(_p_after_hybrid))
                and abs(float(_p_after_hybrid) - float(_p_before_hybrid)) > 1e-12
            )
        )
        trace_pipeline_steps.append(
            {
                "stage": "hybrid_blend",
                "before": (float(_p_before_hybrid) if math.isfinite(float(_p_before_hybrid)) else None),
                "after": (float(_p_after_hybrid) if math.isfinite(float(_p_after_hybrid)) else None),
                "applied": bool(_ap_hybrid),
                "enabled": bool(trace_hybrid_info.get("enabled", False)) if isinstance(trace_hybrid_info, dict) else False,
            }
        )
        _p_pipe_prev = float(_p_after_hybrid)
    except Exception:
        pass

    # Guardrail de confiança (baixa amostra): reduz agressividade automática no ML final.
    try:
        _p_before_cg = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        _cg_on = bool(st.session_state.get("ml_conf_guard_enabled", True))
        _cg_min_games = float(st.session_state.get("ml_conf_guard_min_games", 12.0) or 12.0)
        _cg_max_shrink = float(st.session_state.get("ml_conf_guard_max_shrink", 0.35) or 0.35)
        _n_ref = float(trace_n_team_eff) if math.isfinite(float(trace_n_team_eff)) else float("nan")
        if (not math.isfinite(_n_ref)) or _n_ref < 0.0:
            _n_ref = float("nan")
        if _cg_on and math.isfinite(float(p_map_used)) and _cg_min_games > 0.0 and math.isfinite(_n_ref):
            _coverage = max(0.0, min(1.0, float(_n_ref) / float(_cg_min_games)))
            _shrink = max(0.0, min(1.0, float(_cg_max_shrink) * (1.0 - _coverage)))
            _p0 = float(p_map_used)
            _p1 = float(clip_prob(0.5 + (float(_p0) - 0.5) * (1.0 - _shrink)))
            trace_conf_guard_info = {
                "enabled": True,
                "n_ref": float(_n_ref),
                "min_games": float(_cg_min_games),
                "coverage": float(_coverage),
                "shrink": float(_shrink),
                "p_in": float(_p0),
                "p_out": float(_p1),
                "applied": bool(abs(_p1 - _p0) > 1e-12),
            }
            p_map_used = float(_p1)
            try:
                p_series_used = float(prob_win_series(float(p_map_used), int(bo)))
            except Exception:
                p_series_used = float("nan")
            if bool(trace_conf_guard_info.get("applied", False)):
                st.caption(
                    f"Confidence guard: n={_format_num(_n_ref,1)} | shrink={_format_num(100.0*_shrink,1)}% | "
                    f"p_final={_format_num(p_map_used,4)}"
                )
        else:
            trace_conf_guard_info = {
                "enabled": bool(_cg_on),
                "n_ref": (float(_n_ref) if math.isfinite(_n_ref) else None),
                "min_games": float(_cg_min_games),
                "coverage": None,
                "shrink": 0.0,
                "applied": False,
            }
        _p_after_cg = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        _ap_cg = (
            bool(trace_conf_guard_info.get("applied", False))
            or (
                math.isfinite(float(_p_before_cg))
                and math.isfinite(float(_p_after_cg))
                and abs(float(_p_after_cg) - float(_p_before_cg)) > 1e-12
            )
        )
        trace_pipeline_steps.append(
            {
                "stage": "confidence_guard",
                "before": (float(_p_before_cg) if math.isfinite(float(_p_before_cg)) else None),
                "after": (float(_p_after_cg) if math.isfinite(float(_p_after_cg)) else None),
                "applied": bool(_ap_cg),
                "enabled": bool(trace_conf_guard_info.get("enabled", False)) if isinstance(trace_conf_guard_info, dict) else False,
            }
        )
        _p_pipe_prev = float(_p_after_cg)
    except Exception:
        pass

    # Detalhes avançados do ML foram removidos da Visão Geral para manter a tela limpa.

    # Guardiao de consistencia: evita mudar odd de ML sem mudar inputs reais do confronto.
    _guard_triggered = False
    _guard_dpp = float("nan")
    try:
        _p_before_cons = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        _lineup_mode_sig = str(locals().get("lineup_mode", ""))
        _lineupA_sig = locals().get("lineupA_sel", {}) if isinstance(locals().get("lineupA_sel", {}), dict) else {}
        _lineupB_sig = locals().get("lineupB_sel", {}) if isinstance(locals().get("lineupB_sel", {}), dict) else {}
        _sig_ml = _ml_consistency_signature(
            season_label=str(season_label),
            teamA=str(teamA),
            teamB=str(teamB),
            bo=int(bo),
            ml_engine=str(ml_engine),
            calc_model_mode="Padrao",
            year_opt=year_opt,
            split_opt=split_opt,
            playoffs_opt=playoffs_opt,
            league_mode=str(league_mode),
            fixed_league=fixed_league,
            as_of_date=as_of_date,
            artifact_path=str(artifact_path),
            use_lineup_adjust=bool(st.session_state.get("use_lineup_adjust", False)),
            lineup_mode=_lineup_mode_sig,
            lineupA=_lineupA_sig,
            lineupB=_lineupB_sig,
        )
        _tol_pp = float(st.session_state.get("ml_consistency_tol_pp", 0.05) or 0.05)
        _guard_on = bool(st.session_state.get("ml_consistency_guard", True))
        _prev = st.session_state.get("_ml_consistency_prev")
        if isinstance(_prev, dict) and int(_prev.get("sig", -1)) == int(_sig_ml):
            _prev_p = float(_prev.get("p_map", float("nan")))
            _cur_p = float(p_map_used)
            if math.isfinite(_prev_p) and math.isfinite(_cur_p):
                _dpp = abs(_cur_p - _prev_p) * 100.0
                _guard_dpp = float(_dpp)
                if float(_dpp) > float(_tol_pp):
                    if _guard_on:
                        _guard_triggered = True
                        p_map_used = float(_prev_p)
                        p_series_used = float(_prev.get("p_series", p_series_used))
                        diff_used = float(_prev.get("diff_used", diff_used))
                        delta_players = float(_prev.get("delta_players", delta_players))
                        st.caption(
                            f"Guardiao ML ativo: drift {float(_dpp):.3f} pp com assinatura igual; mantendo odd anterior."
                        )
                    else:
                        st.caption(
                            f"Drift ML detectado com assinatura igual: {float(_dpp):.3f} pp (guardiao desligado)."
                        )
        st.session_state["_ml_consistency_prev"] = {
            "sig": int(_sig_ml),
            "p_map": float(p_map_used) if p_map_used is not None else float("nan"),
            "p_series": float(p_series_used) if p_series_used is not None else float("nan"),
            "diff_used": float(diff_used) if diff_used is not None else float("nan"),
            "delta_players": float(delta_players) if delta_players is not None else 0.0,
        }
        _p_after_cons = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        _ap_cons = bool(_guard_triggered) or (
            math.isfinite(float(_p_before_cons))
            and math.isfinite(float(_p_after_cons))
            and abs(float(_p_after_cons) - float(_p_before_cons)) > 1e-12
        )
        trace_pipeline_steps.append(
            {
                "stage": "consistency_guard",
                "before": (float(_p_before_cons) if math.isfinite(float(_p_before_cons)) else None),
                "after": (float(_p_after_cons) if math.isfinite(float(_p_after_cons)) else None),
                "applied": bool(_ap_cons),
                "enabled": bool(st.session_state.get("ml_consistency_guard", True)),
                "triggered": bool(_guard_triggered),
                "drift_pp": (float(_guard_dpp) if math.isfinite(float(_guard_dpp)) else None),
            }
        )
        _p_pipe_prev = float(_p_after_cons)
    except Exception:
        pass

    # Calibração data-driven de série (BO3/BO5) por liga -> macro -> global.
    try:
        _ps_before_sg = float(p_series_used) if math.isfinite(float(p_series_used)) else float("nan")
        _use_sg = bool(st.session_state.get("ml_series_gamma_use_fitted", True))
        _sg_info = {"enabled": bool(_use_sg), "applied": False, "gamma": 1.0, "source": ""}
        if _use_sg and int(bo) in (3, 5) and math.isfinite(float(p_map_used)):
            _lg_hint_sg = ""
            if str(league_mode) == "fixed" and getattr(filters, "league", None):
                _lg_hint_sg = str(filters.league or "").strip()
            if (not _lg_hint_sg) and isinstance(dfA_used, pd.DataFrame) and (not dfA_used.empty) and ("league" in dfA_used.columns):
                _vf = dfA_used["league"].dropna().astype(str).str.strip()
                if not _vf.empty:
                    _lg_hint_sg = str(_vf.value_counts().index[0]).strip()
            if (not _lg_hint_sg) and isinstance(dfB_used, pd.DataFrame) and (not dfB_used.empty) and ("league" in dfB_used.columns):
                _vf = dfB_used["league"].dropna().astype(str).str.strip()
                if not _vf.empty:
                    _lg_hint_sg = str(_vf.value_counts().index[0]).strip()

            _gamma, _src = _series_gamma_for_league(_lg_hint_sg, int(bo))
            _p_series_base = float(prob_win_series(float(p_map_used), int(bo)))
            _p_series_new = float(_apply_series_gamma(_p_series_base, float(_gamma)))
            p_series_used = float(_p_series_new)
            _sg_info.update(
                {
                    "applied": bool(abs(float(_gamma) - 1.0) > 1e-12),
                    "gamma": float(_gamma),
                    "source": str(_src or ""),
                    "bo": int(bo),
                    "p_base": float(_p_series_base),
                    "p_out": float(_p_series_new),
                    "league_hint": str(_lg_hint_sg or ""),
                }
            )
            if bool(_sg_info.get("applied")):
                st.caption(
                    f"Series gamma (BO{int(bo)}): gamma={_format_num(_gamma,3)} [{str(_src or 'global')}] | "
                    f"p_series={_format_num(_p_series_base,4)} -> {_format_num(_p_series_new,4)}"
                )
        trace_series_gamma_info = dict(_sg_info)
        _ps_after_sg = float(p_series_used) if math.isfinite(float(p_series_used)) else float("nan")
        _ap_sg = bool(trace_series_gamma_info.get("applied", False)) or (
            math.isfinite(float(_ps_before_sg))
            and math.isfinite(float(_ps_after_sg))
            and abs(float(_ps_after_sg) - float(_ps_before_sg)) > 1e-12
        )
        trace_pipeline_steps.append(
            {
                "stage": "series_gamma",
                "before": (float(_ps_before_sg) if math.isfinite(float(_ps_before_sg)) else None),
                "after": (float(_ps_after_sg) if math.isfinite(float(_ps_after_sg)) else None),
                "applied": bool(_ap_sg),
                "enabled": bool(st.session_state.get("ml_series_gamma_use_fitted", True)),
                "source": str(trace_series_gamma_info.get("source", "")) if isinstance(trace_series_gamma_info, dict) else "",
                "gamma": (float(trace_series_gamma_info.get("gamma")) if isinstance(trace_series_gamma_info, dict) and trace_series_gamma_info.get("gamma") is not None else None),
            }
        )
        _ps_pipe_prev = float(_ps_after_sg)
    except Exception:
        trace_series_gamma_info = {"enabled": bool(st.session_state.get("ml_series_gamma_use_fitted", True)), "applied": False}
# persistÃªncia para as outras abas
    _team_games_min_used = float('nan')
    try:
        if _ml_info is not None and not (isinstance(_ml_info, dict) and _ml_info.get("__error__")):
            _team_games_min_used = float(
                min(
                    float(((_ml_info.get("blue") or {}).get("games_played", 0) or 0)),
                    float(((_ml_info.get("red") or {}).get("games_played", 0) or 0)),
                )
            )
    except Exception:
        _team_games_min_used = float('nan')
    st.session_state['_p_map_used'] = float(p_map_used) if p_map_used is not None else float('nan')
    st.session_state['_p_series_used'] = float(p_series_used) if p_series_used is not None else float('nan')
    st.session_state['_ml_diff_used'] = float(diff_used) if diff_used is not None else float('nan')
    st.session_state['_ml_delta_players'] = float(delta_players) if delta_players is not None else 0.0
    st.session_state['_ml_delta_players_eff'] = float(delta_players_eff) if delta_players_eff is not None else 0.0
    st.session_state['_ml_team_games_min'] = float(_team_games_min_used) if _team_games_min_used is not None else float('nan')
    _wf_info = st.session_state.get("wf_ml_corr_info") if isinstance(st.session_state.get("wf_ml_corr_info"), dict) else {}
    _trace = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "matchup": {"season": str(season_label), "teamA": str(teamA), "teamB": str(teamB), "bo": int(bo)},
        "filters": {
            "year": str(year_opt),
            "split": str(split_opt),
            "playoffs": str(playoffs_opt),
            "league_mode": str(league_mode),
            "fixed_league": str(fixed_league or ""),
        },
        "engine": str(ml_engine),
        "model_mode": "PadrÃ£o",
        "probabilities": {
            "p_raw": float(p_map_raw) if math.isfinite(float(p_map_raw)) else None,
            "p_cal": float(p_map_cal) if math.isfinite(float(p_map_cal)) else None,
            "p_team": float(trace_p_map_team) if math.isfinite(float(trace_p_map_team)) else None,
            "p_players": float(trace_p_map_players) if math.isfinite(float(trace_p_map_players)) else None,
            "p_fused": float(trace_p_map_fused) if math.isfinite(float(trace_p_map_fused)) else None,
            "p_map_used": float(p_map_used) if math.isfinite(float(p_map_used)) else None,
            "p_series_used": float(p_series_used) if math.isfinite(float(p_series_used)) else None,
        },
        "players_adjustment": {
            "enabled": bool(use_lineup),
            "mode": str(locals().get("lineup_mode", "")),
            "delta_raw": float(delta_players),
            "delta_effect": float(delta_players_eff),
            "delta_mode": str(_delta_mode),
            "delta_cap": float(_delta_cap),
            "delta_slope": float(_delta_slope),
            "n_team_eff": float(trace_n_team_eff) if math.isfinite(float(trace_n_team_eff)) else None,
            "n_players_eff": float(trace_n_players_eff) if math.isfinite(float(trace_n_players_eff)) else None,
            "team_scale_used": float(st.session_state.get("_fit_team_scale_used", st.session_state.get("_fit_team_scale", 1.0)) or 1.0),
            "players_scale_used": float(st.session_state.get("_fit_players_scale_used", st.session_state.get("_fit_players_scale", 1.0)) or 1.0),
            "scale_league_used": str(st.session_state.get("_fit_scale_league_used", "") or ""),
        },
        "fusion": {
            "season_phase": str(trace_fusion_info.get("season_phase", "")) if isinstance(trace_fusion_info, dict) else "",
            "coverage_eff": float(trace_fusion_info.get("coverage_eff")) if isinstance(trace_fusion_info, dict) and trace_fusion_info.get("coverage_eff") is not None else None,
            "w_team": float(trace_fusion_info.get("w_team")) if isinstance(trace_fusion_info, dict) and trace_fusion_info.get("w_team") is not None else None,
            "w_players": float(trace_fusion_info.get("w_players")) if isinstance(trace_fusion_info, dict) and trace_fusion_info.get("w_players") is not None else None,
            "reason": str(trace_fusion_info.get("reason", "")) if isinstance(trace_fusion_info, dict) else "",
            "coherence_guard": (trace_fusion_info.get("coherence_guard") if isinstance(trace_fusion_info, dict) and isinstance(trace_fusion_info.get("coherence_guard"), dict) else {}),
        },
        "season_form_blend": {
            "applied": bool(trace_form_info.get("applied", False)) if isinstance(trace_form_info, dict) else False,
            "source": str(trace_form_info.get("source", "")) if isinstance(trace_form_info, dict) else "",
            "p_model": float(trace_form_info.get("p_model")) if isinstance(trace_form_info, dict) and trace_form_info.get("p_model") is not None else None,
            "p_form": float(trace_form_info.get("p_form")) if isinstance(trace_form_info, dict) and trace_form_info.get("p_form") is not None else None,
            "p_blended": float(trace_form_info.get("p_blended")) if isinstance(trace_form_info, dict) and trace_form_info.get("p_blended") is not None else None,
            "w_form": float(trace_form_info.get("w_form")) if isinstance(trace_form_info, dict) and trace_form_info.get("w_form") is not None else None,
            "n_a": int(trace_form_info.get("n_a")) if isinstance(trace_form_info, dict) and trace_form_info.get("n_a") is not None else None,
            "n_b": int(trace_form_info.get("n_b")) if isinstance(trace_form_info, dict) and trace_form_info.get("n_b") is not None else None,
            "wins_a": int(trace_form_info.get("wins_a")) if isinstance(trace_form_info, dict) and trace_form_info.get("wins_a") is not None else None,
            "wins_b": int(trace_form_info.get("wins_b")) if isinstance(trace_form_info, dict) and trace_form_info.get("wins_b") is not None else None,
            "reason": str(trace_form_info.get("reason", "")) if isinstance(trace_form_info, dict) else "",
        },
        "hybrid_blend": {
            "applied": bool(trace_hybrid_info.get("applied", False)) if isinstance(trace_hybrid_info, dict) else False,
            "enabled": bool(trace_hybrid_info.get("enabled", False)) if isinstance(trace_hybrid_info, dict) else False,
            "p_model": float(trace_hybrid_info.get("p_model")) if isinstance(trace_hybrid_info, dict) and trace_hybrid_info.get("p_model") is not None else None,
            "p_hybrid": float(trace_hybrid_info.get("p_hybrid")) if isinstance(trace_hybrid_info, dict) and trace_hybrid_info.get("p_hybrid") is not None else None,
            "p_blended": float(trace_hybrid_info.get("p_blended")) if isinstance(trace_hybrid_info, dict) and trace_hybrid_info.get("p_blended") is not None else None,
            "w_hybrid": float(trace_hybrid_info.get("w_hybrid")) if isinstance(trace_hybrid_info, dict) and trace_hybrid_info.get("w_hybrid") is not None else None,
            "n_eff": float(trace_hybrid_info.get("n_eff")) if isinstance(trace_hybrid_info, dict) and trace_hybrid_info.get("n_eff") is not None else None,
            "agree_to_model": float(trace_hybrid_info.get("agree_to_model")) if isinstance(trace_hybrid_info, dict) and trace_hybrid_info.get("agree_to_model") is not None else None,
            "reason": str(trace_hybrid_info.get("reason", "")) if isinstance(trace_hybrid_info, dict) else "",
        },
        "confidence_guard": {
            "enabled": bool(trace_conf_guard_info.get("enabled", False)) if isinstance(trace_conf_guard_info, dict) else False,
            "applied": bool(trace_conf_guard_info.get("applied", False)) if isinstance(trace_conf_guard_info, dict) else False,
            "n_ref": float(trace_conf_guard_info.get("n_ref")) if isinstance(trace_conf_guard_info, dict) and trace_conf_guard_info.get("n_ref") is not None else None,
            "min_games": float(trace_conf_guard_info.get("min_games")) if isinstance(trace_conf_guard_info, dict) and trace_conf_guard_info.get("min_games") is not None else None,
            "coverage": float(trace_conf_guard_info.get("coverage")) if isinstance(trace_conf_guard_info, dict) and trace_conf_guard_info.get("coverage") is not None else None,
            "shrink": float(trace_conf_guard_info.get("shrink")) if isinstance(trace_conf_guard_info, dict) and trace_conf_guard_info.get("shrink") is not None else None,
            "p_in": float(trace_conf_guard_info.get("p_in")) if isinstance(trace_conf_guard_info, dict) and trace_conf_guard_info.get("p_in") is not None else None,
            "p_out": float(trace_conf_guard_info.get("p_out")) if isinstance(trace_conf_guard_info, dict) and trace_conf_guard_info.get("p_out") is not None else None,
        },
        "series_gamma": {
            "enabled": bool(trace_series_gamma_info.get("enabled", False)) if isinstance(trace_series_gamma_info, dict) else False,
            "applied": bool(trace_series_gamma_info.get("applied", False)) if isinstance(trace_series_gamma_info, dict) else False,
            "gamma": float(trace_series_gamma_info.get("gamma")) if isinstance(trace_series_gamma_info, dict) and trace_series_gamma_info.get("gamma") is not None else None,
            "source": str(trace_series_gamma_info.get("source", "")) if isinstance(trace_series_gamma_info, dict) else "",
            "bo": int(trace_series_gamma_info.get("bo")) if isinstance(trace_series_gamma_info, dict) and trace_series_gamma_info.get("bo") is not None else None,
            "p_base": float(trace_series_gamma_info.get("p_base")) if isinstance(trace_series_gamma_info, dict) and trace_series_gamma_info.get("p_base") is not None else None,
            "p_out": float(trace_series_gamma_info.get("p_out")) if isinstance(trace_series_gamma_info, dict) and trace_series_gamma_info.get("p_out") is not None else None,
            "league_hint": str(trace_series_gamma_info.get("league_hint", "")) if isinstance(trace_series_gamma_info, dict) else "",
        },
        "pipeline": {
            "version": "v1",
            "order": list(trace_pipeline_order),
            "steps": list(trace_pipeline_steps),
            "final_p_map": float(p_map_used) if math.isfinite(float(p_map_used)) else None,
            "final_p_series": float(p_series_used) if math.isfinite(float(p_series_used)) else None,
        },
        "guards": {
            "consistency_guard_on": bool(st.session_state.get("ml_consistency_guard", True)),
            "consistency_guard_triggered": bool(_guard_triggered),
            "consistency_guard_drift_pp": float(_guard_dpp) if math.isfinite(float(_guard_dpp)) else None,
            "wf_correction_applied": bool(_wf_info.get("applied", False)),
            "wf_shrink": float(_wf_info.get("shrink")) if _wf_info.get("shrink") is not None else None,
            "wf_league": str(_wf_info.get("league") or ""),
        },
        "odds": {
            "teamA_map": float(_odd_from_p(float(p_map_used))) if math.isfinite(float(p_map_used)) else None,
            "teamB_map": float(_odd_from_p(float(1.0 - float(p_map_used)))) if math.isfinite(float(p_map_used)) else None,
        },
    }
    st.session_state["_ml_trace_last"] = _trace
    _persist_combined_trace_file()

    # Painel centralizado (times + liga + odds + chances)
    # IMPORTANT: a liga exibida deve refletir o recorte/league_mode (e nÃ£o "vazar" por grafia de time ou por artifact).
    _la = _lb = ""
    try:
        if league_mode == "fixed" and filters.league:
            _la = _lb = str(filters.league)
        elif league_mode == "auto":
            # Usa a liga efetivamente usada nas stats (apÃ³s normalizaÃ§Ã£o/auto-league)
            if isinstance(dfA_used, pd.DataFrame) and (not dfA_used.empty) and "league" in dfA_used.columns:
                _la = str(dfA_used["league"].dropna().astype(str).value_counts().index[0]).strip()
            if isinstance(dfB_used, pd.DataFrame) and (not dfB_used.empty) and "league" in dfB_used.columns:
                _lb = str(dfB_used["league"].dropna().astype(str).value_counts().index[0]).strip()
    except Exception:
        pass

    # Fallback: se nÃ£o conseguiu inferir pela base, usa o que veio do ML core (artifact)
    if (not _la or not _lb) and _ml_info is not None:
        _la = _la or str(((_ml_info.get("blue") or {}).get("league")) or "").strip()
        _lb = _lb or str(((_ml_info.get("red") or {}).get("league")) or "").strip()

    def _team_with_league(name: str, lg: str) -> str:
        return f"{name} ({lg})" if lg else name

    teamA_disp = _team_with_league(teamA, _la)
    teamB_disp = _team_with_league(teamB, _lb)

    pA_map = float(st.session_state.get('_p_map_used', p_map_cal))
    pB_map = float(1.0 - pA_map)
    pA_series = float(st.session_state.get('_p_series_used', p_series))
    pB_series = float(1.0 - pA_series)

    oddA_map = _odd_from_p(pA_map)
    oddB_map = _odd_from_p(pB_map)
    oddA_series = _odd_from_p(pA_series)
    oddB_series = _odd_from_p(pB_series)


    # Sempre mostrar ML de mapa, MD3 e MD5 (UI clean; BO selecionado sÃ³ decide a sÃ©rie principal)
    pA_bo3 = float("nan")
    pA_bo5 = float("nan")
    try:
        if math.isfinite(pA_map):
            pA_bo3 = float(prob_win_series(float(pA_map), 3))
            pA_bo5 = float(prob_win_series(float(pA_map), 5))
    except Exception:
        pA_bo3 = float("nan")
        pA_bo5 = float("nan")

    pB_bo3 = float(1.0 - pA_bo3) if math.isfinite(pA_bo3) else float("nan")
    pB_bo5 = float(1.0 - pA_bo5) if math.isfinite(pA_bo5) else float("nan")

    oddA_bo3 = _odd_from_p(pA_bo3)
    oddB_bo3 = _odd_from_p(pB_bo3)
    oddA_bo5 = _odd_from_p(pA_bo5)
    oddB_bo5 = _odd_from_p(pB_bo5)

    _l, _m, _r = st.columns([1, 2, 1])
    with _m:
        st.markdown(f"## {teamA_disp} vs {teamB_disp}")
        a_col, b_col = st.columns(2)
        with a_col:
            st.caption("Time 1")
            st.markdown(
                f"**{teamA_disp}**  \n"
                f"Mapa: **{_format_num(oddA_map, 2)}** ({_format_pct(pA_map)})  \n"
                f"MD3: **{_format_num(oddA_bo3, 2)}** ({_format_pct(pA_bo3)})  \n"
                f"MD5: **{_format_num(oddA_bo5, 2)}** ({_format_pct(pA_bo5)})"
            )
        with b_col:
            st.caption("Time 2")
            st.markdown(
                f"**{teamB_disp}**  \n"
                f"Mapa: **{_format_num(oddB_map, 2)}** ({_format_pct(pB_map)})  \n"
                f"MD3: **{_format_num(oddB_bo3, 2)}** ({_format_pct(pB_bo3)})  \n"
                f"MD5: **{_format_num(oddB_bo5, 2)}** ({_format_pct(pB_bo5)})"
            )

        # 2Âª opiniÃ£o (Elo season + players): alerta de divergÃªncia no ML de mapa
        try:
            if bool(st.session_state.get("ml_engine_compare", False)):
                thr_pp = float(st.session_state.get("ml_engine_alert_pp", 12) or 12)
                main_p = float(pA_map)
                other_p = float("nan")
                main_name = "ML core v2" if ml_engine == "mlcore_v2" else "Elo season + players"
                other_name = "Elo season + players" if ml_engine == "mlcore_v2" else "ML core v2"

                if ml_engine == "mlcore_v2":
                    if _elo_info and not (isinstance(_elo_info, dict) and _elo_info.get("__error__")):
                        other_p = float(_elo_info.get("p_cal", _elo_info.get("p_raw", float("nan"))))
                else:
                    if _ml_info and not (isinstance(_ml_info, dict) and _ml_info.get("__error__")):
                        other_p = float(_ml_info.get("p_cal", _ml_info.get("p_raw", float("nan"))))

                if math.isfinite(main_p) and math.isfinite(other_p):
                    dpp = abs(main_p - other_p) * 100.0
                    if dpp >= thr_pp:
                        st.warning(
                            f"⚠️ Divergência no ML (Mapa): {main_name}={_format_pct(main_p)} vs {other_name}={_format_pct(other_p)} "
                            f"(Δ={dpp:.1f} pp). Revise parâmetros, recorte e lineup."
                        )
        except Exception:
            pass

    # "Detalhar cálculo (ML core v2)" removido da Visão Geral.

    st.subheader("MÃ©dias esperadas (totais)")
    if totals_mode == "series" and int(bo) > 1:
        st.caption(f"Total da sÃ©rie (bo{bo}) via simulaÃ§Ã£o (soma de mapas jogados). Para distribuiÃ§Ã£o por mapa, usa AVG maps (gol.gg).")
        st.dataframe(_series_totals_table(totals_avg), width='stretch')
    else:
        tabs = st.tabs(["AVG (gol.gg)", "MAP (se aplicÃ¡vel)"])
        with tabs[0]:
            st.dataframe(_totals_table(totals_avg), width='stretch')
        with tabs[1]:
            if map_mode == "avg":
                st.info("VocÃª estÃ¡ em map_mode=avg. Troque para map1..map5 em Ajustes de cÃ¡lculo para ver MAP especÃ­fico.")
            else:
                st.dataframe(_totals_table(totals_map), width='stretch')

    st.subheader("Odds justas (linhas)")
    _pmap_u = float(st.session_state.get('_p_map_used', p_map_cal))
    _pser_u = float(st.session_state.get('_p_series_used', p_series))
    p_ml_scope = _pmap_u if ml_mode == "map" else _pser_u
    st.session_state["_p_ml_scope"] = float(p_ml_scope) if p_ml_scope is not None else float('nan')

    tabs2 = st.tabs(["Kills", "Torres", "DragÃµes", "BarÃµes", "Inibidores", "Tempo", "ML + Totais"])
    with tabs2[0]:
        _render_odds_cards(_lines_table("kills", kills, totals_for_lines["kills"]), mode="ou")
    with tabs2[1]:
        _render_odds_cards(_lines_table("towers", towers, totals_for_lines["towers"]), mode="ou")
    with tabs2[2]:
        _render_odds_cards(_lines_table("dragons", dragons, totals_for_lines["dragons"]), mode="ou")
    with tabs2[3]:
        _render_odds_cards(_lines_table("barons", barons, totals_for_lines["barons"]), mode="ou")
    with tabs2[4]:
        _render_odds_cards(_lines_table("inhibitors", inhib, totals_for_lines["inhibitors"]), mode="ou")
    with tabs2[5]:
        _render_odds_cards(_lines_table("time", time_m, totals_for_lines["time"]), mode="ou")

    with tabs2[6]:
        st.markdown("#### ML + Totais (modelo)")

        # ML + Totais usa APENAS as linhas "Extras" (Linha Kills/Tempo para ML + Totais)
        combo_kills_list = _parse_lines(combo_kills_text)
        combo_time_list = _parse_time_lines(combo_time_text)

        if not (combo_kills_list or combo_time_list):
            st.info("Digite uma linha em Extras: Linha Kills (para ML + Totais) e/ou Linha Tempo (para ML + Totais).")
        else:
            # ML+Totais: mostrar os dois lados (Time 1 e Time 2)
            _pA_ml = float(p_ml_scope) if p_ml_scope is not None else float('nan')
            _pB_ml = (1.0 - _pA_ml) if pd.notna(_pA_ml) else float('nan')

            tabs_ml = st.tabs(["Kills", "Tempo"])

            with tabs_ml[0]:
                if not combo_kills_list:
                    st.info("Digite uma linha em Extras: Linha Kills (para ML + Totais).")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"##### {teamA}")
                        _render_odds_cards(
                            _lines_table_with_p_ml("kills", combo_kills_list, totals_for_lines["kills"], _pA_ml),
                            mode="ml_totals",
                            table_label=f"Ver tabela ({teamA})",
                        )
                    with c2:
                        st.markdown(f"##### {teamB}")
                        _render_odds_cards(
                            _lines_table_with_p_ml("kills", combo_kills_list, totals_for_lines["kills"], _pB_ml),
                            mode="ml_totals",
                            table_label=f"Ver tabela ({teamB})",
                        )

            with tabs_ml[1]:
                if not combo_time_list:
                    st.info("Digite uma linha em Extras: Linha Tempo (para ML + Totais).")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"##### {teamA}")
                        _render_odds_cards(
                            _lines_table_with_p_ml("time", combo_time_list, totals_for_lines["time"], _pA_ml),
                            mode="ml_totals",
                            table_label=f"Ver tabela ({teamA})",
                        )
                    with c2:
                        st.markdown(f"##### {teamB}")
                        _render_odds_cards(
                            _lines_table_with_p_ml("time", combo_time_list, totals_for_lines["time"], _pB_ml),
                            mode="ml_totals",
                            table_label=f"Ver tabela ({teamB})",
                        )

if "Consistência de Mercado (Laplace)" in tab_refs:
 with tab_refs["Consistência de Mercado (Laplace)"]:
    st.markdown(
        """
<div class="gb-hero" style="margin-top:4px;">
  <div class="title" style="font-size:1.1rem;">Consistência de Mercado</div>
  <div class="sub">Histórico + Laplace (sanity-check) com comparação do Modelo no mesmo fluxo.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    _cons_sig_now = _consistency_analysis_signature()
    st.session_state["_consistency_analysis_sig_last"] = int(_cons_sig_now)
    st.caption("Consistência em dia para os parâmetros atuais.")

    st.markdown("### Resumo")

    # Bases jÃ¡ filtradas (mesmo filtros/league_mode/map_mode do motor)
    dfA_all = dfA_used.copy()
    dfB_all = dfB_used.copy()
    if True:
    
        # H2H: perspectiva do Team A
        df_h2h_A = dfA_all[dfA_all["opponent"] == teamB].copy()
        df_h2h_B = dfB_all[dfB_all["opponent"] == teamA].copy()
    
        _slices = core_recency_slices(df_a=dfA_all, df_b=dfB_all, df_h2h=df_h2h_A, id_col="gameid", date_col="date")
        df_league_year = _slices["league_year"]
        df_league_15 = _slices["league_15"]
        df_league_10 = _slices["league_10"]
        df_league_5 = _slices["league_5"]
        df_h2h_year = _slices["h2h_year"]
        df_h2h_15 = _slices["h2h_15"]
        df_h2h_10 = _slices["h2h_10"]
        df_h2h_5 = _slices["h2h_5"]
    
        # Painel extra (sem remover o que ja existe): resumo visual estilo print.
    
        def _num(v: Any, d: int = 1) -> str:
            try:
                x = float(v)
                if not math.isfinite(x):
                    return "-"
                return f"{x:.{int(d)}f}"
            except Exception:
                return "-"
    
        def _team_k_stats(df_in: pd.DataFrame) -> Dict[str, Any]:
            if df_in is None or df_in.empty or "total_kills" not in df_in.columns:
                return {"games": 0, "avg": float("nan"), "med": float("nan"), "min": float("nan"), "max": float("nan")}
            x = pd.to_numeric(df_in["total_kills"], errors="coerce").dropna()
            if x.empty:
                return {"games": 0, "avg": float("nan"), "med": float("nan"), "min": float("nan"), "max": float("nan")}
            return {
                "games": int(x.shape[0]),
                "avg": float(x.mean()),
                "med": float(x.median()),
                "min": float(x.min()),
                "max": float(x.max()),
            }
    
        def _last_rows(df_in: pd.DataFrame, n: int) -> list[str]:
            if df_in is None or df_in.empty:
                return []
            d = df_in.sort_values("date").tail(n)
            out = []
            for _, r in d.iterrows():
                opp = str(r.get("opponent", "") or "-")
                k = pd.to_numeric(pd.Series([r.get("total_kills")]), errors="coerce").iloc[0]
                out.append(f"{opp} - {_num(k, 0)}")
            return out
    
        # all-season simples: remove split/playoffs e mantem year/league quando definidos.
        _base_no_split = Filters(
            year=filters.year,
            split=None,
            playoffs=None,
            league=filters.league,
        )
        dfA_season = filter_team_games(team_games, team=teamA, filters=_base_no_split, league_mode=league_mode, map_mode=map_mode, max_games=max_games)
        dfB_season = filter_team_games(team_games, team=teamB, filters=_base_no_split, league_mode=league_mode, map_mode=map_mode, max_games=max_games)
    
        def _summary_block(title: str):
            st.markdown(
                "<div style='background:#2daac4;padding:8px 10px;border-radius:6px 6px 0 0;"
                "text-align:center;font-weight:700;letter-spacing:.03em;'>"
                f"{title}</div>",
                unsafe_allow_html=True,
            )
    
        def _render_market_context_panel(panel_title: str, col_name: str) -> None:
            _panel_title = core_panel_title_with_suffix(str(panel_title or "").strip() or "Mercado", "(combined)")
            st.markdown(f"### {_panel_title}")

            def _metric(col: str, value: Any, decimals: int = 1) -> str:
                return core_fmt_metric_value(col, value, decimals, time_formatter=_min_to_mmss)

            sA_cur = core_team_metric_stats(dfA_all, col_name)
            sB_cur = core_team_metric_stats(dfB_all, col_name)
            sA_all = core_team_metric_stats(dfA_season, col_name)
            sB_all = core_team_metric_stats(dfB_season, col_name)
            df_cur, df_all = core_build_overall_tables(
                teamA,
                teamB,
                col_name,
                sA_cur,
                sB_cur,
                sA_all,
                sB_all,
                num_formatter=_num,
                metric_formatter=_metric,
            )
    
            c_over, c_recent, c_h2h = st.columns(3)
            with c_over:
                _summary_block("OVERALL")
                with st.container(border=True):
                    st.markdown(f"**{teamA}**  |  **{teamB}**")
                    st.caption("Current Split")
                    st.dataframe(df_cur, width='stretch', hide_index=True)
                    st.caption("All Season")
                    st.dataframe(df_all, width='stretch', hide_index=True)
    
            with c_recent:
                _summary_block("RECENT FORM")
                a5 = core_team_metric_stats(dfA_all.tail(5), col_name)
                b5 = core_team_metric_stats(dfB_all.tail(5), col_name)
                a10 = core_team_metric_stats(dfA_all.tail(10), col_name)
                b10 = core_team_metric_stats(dfB_all.tail(10), col_name)
                with st.container(border=True):
                    st.markdown(f"**{teamA}**")
                    st.caption(
                        f"Avg (Last 5): {_metric(col_name, a5['avg'])} | "
                        f"Avg (Last 10): {_metric(col_name, a10['avg'])}"
                    )
                    _a_rows = core_last_rows_metric(dfA_all, col_name, 10, metric_formatter=_metric)
                    st.markdown("\n".join([f"- {x}" for x in _a_rows]) if _a_rows else "- sem jogos")
                    st.markdown("---")
                    st.markdown(f"**{teamB}**")
                    st.caption(
                        f"Avg (Last 5): {_metric(col_name, b5['avg'])} | "
                        f"Avg (Last 10): {_metric(col_name, b10['avg'])}"
                    )
                    _b_rows = core_last_rows_metric(dfB_all, col_name, 10, metric_formatter=_metric)
                    st.markdown("\n".join([f"- {x}" for x in _b_rows]) if _b_rows else "- sem jogos")
    
            with c_h2h:
                _summary_block("PAST FACEOFFS")
                _h = core_h2h_numeric_series(df_h2h_year, col_name)
                with st.container(border=True):
                    st.markdown(f"**{teamA} vs {teamB}**")
                    st.metric("H2H Games", int(_h.shape[0]))
                    st.metric("Avg", _metric(col_name, (_h.mean() if _h.shape[0] else float("nan"))))
                    if df_h2h_year is not None and not df_h2h_year.empty and col_name in df_h2h_year.columns:
                        _h2h_rows = core_last_rows_metric(df_h2h_year, col_name, 10, metric_formatter=_metric)
                        st.markdown("\n".join([f"- {x}" for x in _h2h_rows]) if _h2h_rows else "- sem H2H")
                    else:
                        st.markdown("- sem H2H")
    
        # Diagnostico rapido: evitar confusao quando H2H/Liga nao tem amostra
        # (nao depende de variaveis do Resumo; usa session_state se existir)
        _n_league = int(getattr(df_league_year, 'shape', [0])[0] or 0)
        _n_h2h = int(getattr(df_h2h_year, 'shape', [0])[0] or 0)
        req_h2h = state_get_bool(st.session_state, 'resumo_req_h2h', False)
        req_liga = state_get_bool(st.session_state, 'resumo_req_liga', False)
        min_sample_liga = state_get_int(st.session_state, 'resumo_min_sample_liga', 8)
        min_sample_h2h = state_get_int(st.session_state, 'resumo_min_sample_h2h', 3)
        hist_shrink_n = state_get_int(st.session_state, 'resumo_hist_shrink_n', 20)
        if req_h2h and _n_h2h < int(min_sample_h2h):
            st.warning(f'H2H: amostra insuficiente ({_n_h2h}/{int(min_sample_h2h)}).')
        if req_liga and _n_league < int(min_sample_liga):
            st.warning(f'Liga: amostra insuficiente ({_n_league}/{int(min_sample_liga)}).')
    
        def _fmt_count_odd(k: int, n: int) -> str:
            if n <= 0:
                return "-"
            odd = core_laplace_odds(k, n, floor_odd=1.10)
            if odd is None or not math.isfinite(float(odd)):
                return f"{k}/{n}"
            return f"{k}/{n} ({odd:.2f}x)"
    
        # Parse extra lines (multi-line)
        teamA_kills_lines = _parse_lines(teamA_kills_line_text)
        teamB_kills_lines = _parse_lines(teamB_kills_line_text)
        hc_kills_lines = _parse_lines(hc_kills_text)
        hc_towers_lines = _parse_lines(hc_towers_text)
        hc_dragons_lines = _parse_lines(hc_dragons_text)
        combo_kills_lines = _parse_lines(combo_kills_text)
        combo_time_lines = _parse_time_lines(combo_time_text)
    
        st.markdown("#### Mercados")
        # Tabs por mercado (igual seu app)
        _market_tab_names = [
            "Kills",
            "Torres",
            "Dragões",
            "Barões",
            "Inibidores",
            "Tempo",
            "Kills p/ Time",
            "Handicaps",
            "Primeiros Objetivos",
            "Vencedor",
            "ML + Totais",
            "Detalhes dos Jogos",
            "Seletor Rápido",
        ]
        tabs_c = st.tabs(_market_tab_names)
        tabs_c_map = dict(zip(_market_tab_names, tabs_c))
    
        def _render_metric_tab(metric_key: str, title: str, value_col: str, lines_list: list[float]):
            # Modelo (mÃ©dia/odds)
            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Modelo (média esperada + odds justas)**")
                try:
                    t = totals_for_lines[metric_key]
                    mean = t.mean
                    sd = t.sd
                    if metric_key == "time":
                        st.write(f"Média: {_min_to_mmss(mean)}  |  SD: {_min_to_mmss(sd)}")
                    else:
                        st.write(f"Média: {mean:.2f}  |  SD: {sd:.2f}")
                except Exception:
                    pass
                st.dataframe(_display_ou_table(_lines_table(metric_key if metric_key != "time" else "time", lines_list, totals_for_lines[metric_key])), width='stretch')
    
            with right:
                st.markdown("**Histórico (Laplace)**")
                if not lines_list:
                    st.info("Sem linhas para este mercado.")
                    return
                for ln in lines_list:
                    ln_disp = _fmt_line_disp(metric_key, ln)
                    st.markdown(f"**{title} — Linha: {ln_disp}**")
                    tbl = core_ou_block(
                        df_league_year, df_league_15, df_league_10, df_league_5,
                        df_h2h_year, df_h2h_15, df_h2h_10, df_h2h_5,
                        value_col=value_col, line=float(ln), floor_odd=1.10,
                    )
                    st.dataframe(tbl, width='stretch')
            st.divider()
            _render_market_context_panel(f"{title} (combined)", value_col)
    
        with tabs_c_map["Kills"]:
            _render_metric_tab("kills", "Kills (Total)", "total_kills", kills)
        with tabs_c_map["Torres"]:
            _render_metric_tab("towers", "Torres (Total)", "total_towers", towers)
        with tabs_c_map["Dragões"]:
            _render_metric_tab("dragons", "Dragões (Total)", "total_dragons", dragons)
        with tabs_c_map["Barões"]:
            _render_metric_tab("barons", "Barões (Total)", "total_nashors", barons)
        with tabs_c_map["Inibidores"]:
            _render_metric_tab("inhibitors", "Inibidores (Total)", "total_inhibitors", inhib)
        with tabs_c_map["Tempo"]:
            _render_metric_tab("time", "Tempo", "game_time_min", time_m)
    
        with tabs_c_map["Kills p/ Time"]:
            st.markdown(f"**Total Kills por time (linha por time)**")
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"### {teamA}")
                if not teamA_kills_lines:
                    st.info("Digite uma linha em Extras: Total Kills do Time 1.")
                else:
                    for ln in teamA_kills_lines:
                        st.markdown(f"**Linha {ln:g}**")
                        tbl = core_ou_block(
                            dfA_all, dfA_all.tail(15), dfA_all.tail(10), dfA_all.tail(5),
                            df_h2h_A, df_h2h_A.tail(15), df_h2h_A.tail(10), df_h2h_A.tail(5),
                            value_col="kills_for", line=float(ln), floor_odd=1.10,
                        )
                        st.dataframe(tbl, width='stretch')
            with colB:
                st.markdown(f"### {teamB}")
                if not teamB_kills_lines:
                    st.info("Digite uma linha em Extras: Total Kills do Time 2.")
                else:
                    for ln in teamB_kills_lines:
                        st.markdown(f"**Linha {ln:g}**")
                        tbl = core_ou_block(
                            dfB_all, dfB_all.tail(15), dfB_all.tail(10), dfB_all.tail(5),
                            df_h2h_B, df_h2h_B.tail(15), df_h2h_B.tail(10), df_h2h_B.tail(5),
                            value_col="kills_for", line=float(ln), floor_odd=1.10,
                        )
                        st.dataframe(tbl, width='stretch')
            st.divider()
            _render_market_context_panel("Kills por Time", "kills_for")
    
        with tabs_c_map["Handicaps"]:
            st.markdown("**Handicaps (cobrir vs nÃ£o cobrir)**")
            if (not hc_kills_lines) and (not hc_towers_lines) and (not hc_dragons_lines):
                st.info("Digite algum handicap em Extras.")
            else:
                def _render_hcap(
                    title: str,
                    diff_col_for: str,
                    diff_col_against: str,
                    line: Optional[float],
                    df_year: pd.DataFrame,
                    df_h2h_year: pd.DataFrame,
                ):
                    if line is None:
                        return
                    st.markdown(f"### {title} (linha {line:+g})")
    
                    def event_cover(df: pd.DataFrame):
                        diff = pd.to_numeric(df.get(diff_col_for), errors="coerce") - pd.to_numeric(df.get(diff_col_against), errors="coerce")
                        adj = diff + float(line)
                        return adj > 0
    
                    tbl = core_event_block(
                        df_year,
                        df_year.tail(15),
                        df_year.tail(10),
                        df_year.tail(5),
                        df_h2h_year,
                        df_h2h_year.tail(15),
                        df_h2h_year.tail(10),
                        df_h2h_year.tail(5),
                        event_fn=event_cover,
                        label_yes="Cover",
                        label_no="Fail",
                        floor_odd=1.10,
                    )
                    st.dataframe(tbl, width='stretch')
    
                # HC Kills: mostrar os dois lados automaticamente (Time escolhido + o oposto)
                if hc_kills_lines:
                    _hc_team = st.session_state.get("hc_kills_team") or teamA
                    if _hc_team not in [teamA, teamB]:
                        _hc_team = teamA
                    _other_team = teamB if _hc_team == teamA else teamA
    
                    _df_for = dfA_all if _hc_team == teamA else dfB_all
                    _df_h_for = df_h2h_A if _hc_team == teamA else df_h2h_B
                    _df_other = dfB_all if _hc_team == teamA else dfA_all
                    _df_h_other = df_h2h_B if _hc_team == teamA else df_h2h_A
    
                    for hln in hc_kills_lines:
                        st.markdown(f"### Linha HC Kills {hln:+g}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"#### {_hc_team}")
                            _render_hcap(
                                f"HC Kills ({_hc_team})",
                                "kills_for",
                                "kills_against",
                                float(hln),
                                _df_for,
                                _df_h_for,
                            )
                        with col2:
                            st.markdown(f"#### {_other_team}")
                            _render_hcap(
                                f"HC Kills ({_other_team})",
                                "kills_for",
                                "kills_against",
                                -float(hln),
                                _df_other,
                                _df_h_other,
                            )
    
                # MantÃ©m Torres/DragÃµes como antes (perspectiva Time 1) para evitar confusÃ£o
                for hln in hc_towers_lines:
                    _render_hcap("HC Torres (Time 1)", "towers_for", "towers_against", float(hln), dfA_all, df_h2h_A)
                for hln in hc_dragons_lines:
                    _render_hcap("HC DragÃµes (Time 1)", "dragons_for", "dragons_against", float(hln), dfA_all, df_h2h_A)
            st.divider()
            if hc_kills_lines:
                _render_market_context_panel("Handicap Kills (base de kills do time)", "kills_for")
            elif hc_towers_lines:
                _render_market_context_panel("Handicap Torres (base de torres do time)", "towers_for")
            elif hc_dragons_lines:
                _render_market_context_panel("Handicap Dragões (base de dragões do time)", "dragons_for")
            else:
                _render_market_context_panel("Handicaps (base de kills do time)", "kills_for")
    
        with tabs_c_map["Primeiros Objetivos"]:
            # Reaproveita bloco original do backup (primeiros objetivos)
            st.markdown("**Primeiros Objetivos (probabilidade do time conseguir)**")
            obj_cols = []
            for c in ["firstblood", "firsttower", "firstdragon", "firstherald", "firstbaron", "firstinhibitor"]:
                if c in dfA_all.columns:
                    obj_cols.append(c)
            if not obj_cols:
                st.info("CSV nÃ£o possui colunas de primeiros objetivos (firstblood/firsttower/...).")
            else:
                obj_tab_names = {
                    "firstblood": "FB",
                    "firsttower": "FT",
                    "firstdragon": "FD",
                    "firstherald": "FH",
                    "firstbaron": "FBaron",
                    "firstinhibitor": "FInib",
                }
                subtabs = st.tabs([obj_tab_names.get(c, c) for c in obj_cols])
                for i, col in enumerate(obj_cols):
                    with subtabs[i]:
                        st.markdown(f"### {obj_tab_names.get(col, col)}")
    
                        def ev_yes(df: pd.DataFrame):
                            return pd.to_numeric(df.get(col), errors="coerce") > 0.5
    
                        def _team_row(df_year, df_15, df_10, df_5):
                            y1 = no1 = n1 = 0
                            if df_year is not None and not df_year.empty:
                                ok = ev_yes(df_year).fillna(False).astype(bool)
                                n1 = int(ok.shape[0]); y1 = int(ok.sum()); no1 = n1 - y1
                            y15 = no15 = n15 = 0
                            if df_15 is not None and not df_15.empty:
                                ok = ev_yes(df_15).fillna(False).astype(bool)
                                n15 = int(ok.shape[0]); y15 = int(ok.sum()); no15 = n15 - y15
                            y10 = no10 = n10 = 0
                            if df_10 is not None and not df_10.empty:
                                ok = ev_yes(df_10).fillna(False).astype(bool)
                                n10 = int(ok.shape[0]); y10 = int(ok.sum()); no10 = n10 - y10
                            y5 = no5 = n5 = 0
                            if df_5 is not None and not df_5.empty:
                                ok = ev_yes(df_5).fillna(False).astype(bool)
                                n5 = int(ok.shape[0]); y5 = int(ok.sum()); no5 = n5 - y5
                            def fmt(y, n):
                                _odd = core_laplace_odds(y, n, floor_odd=1.10)
                                return f"{y}/{n} ({float(_odd):.2f}x)" if (n and _odd is not None and math.isfinite(float(_odd))) else (f"{y}/{n}" if n else "-")
                            return [fmt(y1, n1), fmt(y15, n15), fmt(y10, n10), fmt(y5, n5)]
    
                        rowA_l = _team_row(dfA_all, dfA_all.tail(15), dfA_all.tail(10), dfA_all.tail(5))
                        rowB_l = _team_row(dfB_all, dfB_all.tail(15), dfB_all.tail(10), dfB_all.tail(5))
                        rowA_h = _team_row(df_h2h_A, df_h2h_A.tail(15), df_h2h_A.tail(10), df_h2h_A.tail(5))
                        rowB_h = _team_row(df_h2h_B, df_h2h_B.tail(15), df_h2h_B.tail(10), df_h2h_B.tail(5))
    
                        tbl = pd.DataFrame(
                            [rowA_l, rowB_l, rowA_h, rowB_h],
                            columns=["Ano todo", "Ãšltimos 15", "Ãšltimos 10", "Ãšltimos 5"],
                            index=[f"Liga - {teamA}", f"Liga - {teamB}", f"H2H - {teamA}", f"H2H - {teamB}"],
                        )
                        st.dataframe(tbl, width='stretch')
                        st.divider()
                        _render_market_context_panel(f"{obj_tab_names.get(col, col)} (taxa por jogo)", col)

        with tabs_c_map["Vencedor"]:
            st.markdown("**Win Rate (vencedor)**")

            def _to_win_frame(df_in: pd.DataFrame) -> pd.DataFrame:
                if df_in is None or (not isinstance(df_in, pd.DataFrame)) or df_in.empty:
                    return pd.DataFrame(columns=["date", "opponent", "win", "side"])
                d = df_in.copy()
                if "date" not in d.columns:
                    d["date"] = pd.NaT
                if "opponent" not in d.columns:
                    d["opponent"] = "-"
                d["win"] = pd.to_numeric(d.get("win"), errors="coerce")
                if "side" not in d.columns:
                    d["side"] = ""
                return d[["date", "opponent", "win", "side"]].copy()

            def _wr_tuple(df_in: pd.DataFrame) -> tuple[int, int, int, float]:
                if df_in is None or df_in.empty:
                    return 0, 0, 0, float("nan")
                w = pd.to_numeric(df_in.get("win"), errors="coerce")
                m = w.notna()
                n = int(m.sum())
                if n <= 0:
                    return 0, 0, 0, float("nan")
                k = int((w[m] > 0.5).sum())
                l = int(n - k)
                return n, k, l, float(k / n)

            def _side_wr(df_in: pd.DataFrame, side_token: str) -> float:
                if df_in is None or df_in.empty or "side" not in df_in.columns:
                    return float("nan")
                side_ser = df_in["side"].astype(str).str.strip().str.lower()
                sub = df_in[side_ser == str(side_token).strip().lower()].copy()
                _, _, _, wr = _wr_tuple(sub)
                return float(wr)

            def _fmt_wr(wr: float) -> str:
                return f"{100.0 * float(wr):.1f}%" if math.isfinite(float(wr)) else "-"

            def _recent_rows(df_in: pd.DataFrame, n: int = 10) -> list[str]:
                d = _to_win_frame(df_in).sort_values("date").tail(int(n))
                out = []
                for _, r in d.iterrows():
                    opp = str(r.get("opponent", "") or "-")
                    wv = pd.to_numeric(pd.Series([r.get("win")]), errors="coerce").iloc[0]
                    mark = "✅" if pd.notna(wv) and float(wv) > 0.5 else "❌"
                    out.append(f"{opp} {mark}")
                return out

            def _vs_table(df_in: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
                d = _to_win_frame(df_in)
                if d.empty:
                    return pd.DataFrame(columns=["Vs", "Games", "W", "L", "Win Rate"])
                g = (
                    d.groupby("opponent", dropna=False)
                    .agg(Games=("win", "size"), W=("win", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.5).sum())))
                    .reset_index()
                )
                g["L"] = g["Games"] - g["W"]
                g["Win Rate"] = (100.0 * (g["W"] / g["Games"])).round(1).astype(str) + "%"
                g = g.rename(columns={"opponent": "Vs"}).sort_values(["Games", "W"], ascending=[False, False]).head(int(top_n))
                return g[["Vs", "Games", "W", "L", "Win Rate"]]

            dfA_w = _to_win_frame(dfA_all)
            dfB_w = _to_win_frame(dfB_all)
            dfA_w_season = _to_win_frame(dfA_season)
            dfB_w_season = _to_win_frame(dfB_season)
            df_h2h_w = _to_win_frame(df_h2h_A)

            c_over, c_recent, c_h2h = st.columns(3)
            with c_over:
                _summary_block("OVERALL")
                with st.container(border=True):
                    nA, wA, lA, wrA = _wr_tuple(dfA_w)
                    nB, wB, lB, wrB = _wr_tuple(dfB_w)
                    nA2, wA2, lA2, wrA2 = _wr_tuple(dfA_w_season)
                    nB2, wB2, lB2, wrB2 = _wr_tuple(dfB_w_season)
                    st.markdown(f"**{teamA}**  |  **{teamB}**")
                    st.caption("Current Split")
                    _tbl_current = pd.DataFrame(
                        [
                            {"Métrica": "Win Rate", teamA: _fmt_wr(wrA), teamB: _fmt_wr(wrB)},
                            {"Métrica": "# Games", teamA: nA, teamB: nB},
                            {"Métrica": "Record", teamA: f"{wA}W-{lA}L", teamB: f"{wB}W-{lB}L"},
                            {"Métrica": "Blue Side", teamA: _fmt_wr(_side_wr(dfA_w, "blue")), teamB: _fmt_wr(_side_wr(dfB_w, "blue"))},
                            {"Métrica": "Red Side", teamA: _fmt_wr(_side_wr(dfA_w, "red")), teamB: _fmt_wr(_side_wr(dfB_w, "red"))},
                        ]
                    ).astype(str)
                    st.dataframe(
                        _tbl_current,
                        width='stretch',
                        hide_index=True,
                    )
                    st.caption("All Season")
                    _tbl_all = pd.DataFrame(
                        [
                            {"Métrica": "Win Rate", teamA: _fmt_wr(wrA2), teamB: _fmt_wr(wrB2)},
                            {"Métrica": "# Games", teamA: nA2, teamB: nB2},
                            {"Métrica": "Record", teamA: f"{wA2}W-{lA2}L", teamB: f"{wB2}W-{lB2}L"},
                        ]
                    ).astype(str)
                    st.dataframe(
                        _tbl_all,
                        width='stretch',
                        hide_index=True,
                    )
                    st.markdown(f"**{teamA} — Vs (amostra)**")
                    st.dataframe(_vs_table(dfA_w_season), width='stretch', hide_index=True)
                    st.markdown(f"**{teamB} — Vs (amostra)**")
                    st.dataframe(_vs_table(dfB_w_season), width='stretch', hide_index=True)

            with c_recent:
                _summary_block("RECENT FORM")
                with st.container(border=True):
                    _, _, _, wrA5 = _wr_tuple(dfA_w.tail(5))
                    _, _, _, wrA10 = _wr_tuple(dfA_w.tail(10))
                    _, _, _, wrB5 = _wr_tuple(dfB_w.tail(5))
                    _, _, _, wrB10 = _wr_tuple(dfB_w.tail(10))
                    st.markdown(f"**{teamA}**")
                    st.caption(f"Last 5: {_fmt_wr(wrA5)} | Last 10: {_fmt_wr(wrA10)}")
                    _ra = _recent_rows(dfA_w, 10)
                    st.markdown("\n".join([f"- {x}" for x in _ra]) if _ra else "- sem jogos")
                    st.markdown("---")
                    st.markdown(f"**{teamB}**")
                    st.caption(f"Last 5: {_fmt_wr(wrB5)} | Last 10: {_fmt_wr(wrB10)}")
                    _rb = _recent_rows(dfB_w, 10)
                    st.markdown("\n".join([f"- {x}" for x in _rb]) if _rb else "- sem jogos")

            with c_h2h:
                _summary_block("PAST FACEOFFS")
                with st.container(border=True):
                    st.markdown(f"**{teamA} vs {teamB}**")
                    nH, wH, lH, wrH = _wr_tuple(df_h2h_w)
                    st.metric("H2H Games", int(nH))
                    st.metric(teamA, _fmt_wr(wrH))
                    st.metric(teamB, _fmt_wr(1.0 - wrH) if math.isfinite(float(wrH)) else "-")
                    if nH > 0:
                        d = df_h2h_w.sort_values("date").tail(10).copy()
                        rows = []
                        for _, r in d.iterrows():
                            dt = pd.to_datetime(r.get("date"), errors="coerce")
                            dts = dt.strftime("%Y-%m-%d") if pd.notna(dt) else "-"
                            wv = pd.to_numeric(pd.Series([r.get("win")]), errors="coerce").iloc[0]
                            winner = teamA if (pd.notna(wv) and float(wv) > 0.5) else teamB
                            rows.append(f"- {dts} — {winner}")
                        st.markdown("\n".join(rows) if rows else "- sem H2H")
                    else:
                        st.markdown("- sem H2H")
    
        with tabs_c_map["ML + Totais"]:
            st.markdown("**ML + Totais (frequÃªncia com Laplace)**")
            st.markdown('### Modelo (ML + Totais)')
            st.caption('Mostra as odds justas combinando ML e Total (ex.: ML&Over / ML&Under) usando apenas as linhas específicas de ML + Totais.')
            _ml_totals_payload = core_build_ml_totals_model_payload(
                team_a=teamA,
                team_b=teamB,
                totals_for_lines=totals_for_lines,
                combo_kills_lines=combo_kills_lines,
                combo_time_lines=combo_time_lines,
                p_ml_scope=st.session_state.get("_p_ml_scope"),
                lines_table_with_p_ml_fn=_lines_table_with_p_ml,
                display_ml_totals_table_fn=_display_ml_totals_table,
            )
            if not bool(_ml_totals_payload.get("has_any_lines", False)):
                st.info('Preencha em Extras: "Linha Kills (para ML + Totais)" e/ou "Linha Tempo (para ML + Totais)".')
            else:
                tabs_ml = st.tabs(["Kills", "Tempo"])
                _tabs_payload = list(_ml_totals_payload.get("tabs", []) or [])
                for _i, _tab in enumerate(_tabs_payload):
                    with tabs_ml[_i]:
                        if not bool(_tab.get("has_lines", False)):
                            st.info('Sem linhas para este mercado.')
                        else:
                            cA, cB = st.columns(2)
                            with cA:
                                st.markdown(f"**{_tab.get('team_a_label', f'Time 1: {teamA}')}**")
                                st.dataframe(_tab.get("table_a", pd.DataFrame()), width='stretch')
                            with cB:
                                st.markdown(f"**{_tab.get('team_b_label', f'Time 2: {teamB}')}**")
                                st.dataframe(_tab.get("table_b", pd.DataFrame()), width='stretch')
                        st.divider()
                        _render_market_context_panel(str(_tab.get("context_title", "ML + Totais")), str(_tab.get("context_col", "total_kills")))
            st.divider()
            _lap_payload = core_build_ml_totals_laplace_payload(
                team_a=teamA,
                team_b=teamB,
                combo_kills_lines=combo_kills_lines,
                combo_time_lines=combo_time_lines,
                df_a_all=dfA_all,
                df_b_all=dfB_all,
                df_h2h_a=df_h2h_A,
                df_h2h_b=df_h2h_B,
                event_block_fn=core_event_block,
                floor_odd=1.10,
            )
            if not bool(_lap_payload.get("has_any_lines", False)):
                st.info("Digite uma linha em Extras: Linha Kills e/ou Linha Tempo.")
            else:
                for _r in list(_lap_payload.get("rows", []) or []):
                    st.markdown(f"### {str(_r.get('title', 'ML + Totais'))} (linha {float(_r.get('line', 0.0))})")
                    c_l, c_r = st.columns(2)
                    with c_l:
                        st.markdown(f"**{str(_r.get('team_a_label', f'Time de referência (Time 1): {teamA}'))}**")
                        st.dataframe(_r.get("tbl_over_a", pd.DataFrame()), width='stretch')
                        st.dataframe(_r.get("tbl_under_a", pd.DataFrame()), width='stretch')
                    with c_r:
                        st.markdown(f"**{str(_r.get('team_b_label', f'Time de referência (Time 2): {teamB}'))}**")
                        st.dataframe(_r.get("tbl_over_b", pd.DataFrame()), width='stretch')
                        st.dataframe(_r.get("tbl_under_b", pd.DataFrame()), width='stretch')
    
        with tabs_c_map["Detalhes dos Jogos"]:
            st.markdown("**Detalhes dos jogos (auditoria)**")
            options = core_build_audit_base_options(
                teamA,
                teamB,
                df_league_year,
                df_league_15,
                df_league_10,
                df_league_5,
                df_h2h_year,
                df_h2h_15,
                df_h2h_10,
                df_h2h_5,
                dfA_all,
                dfB_all,
            )
            ds_name = st.selectbox("Base", [o[0] for o in options], index=0)
            df_base = dict(options).get(ds_name)

            market_opts = core_build_audit_market_options(
                teamA,
                teamB,
                kills,
                towers,
                dragons,
                barons,
                inhib,
                time_m,
                teamA_kills_lines,
                teamB_kills_lines,
            )
            market_name = st.selectbox("Mercado", list(market_opts.keys()), index=0)
            col, line_list = market_opts.get(market_name, (None, []))
            line_list = list(line_list or [])
            ln = st.selectbox("Linha para classificar (Acima/Abaixo)", options=line_list, index=0) if line_list else None
            if df_base is None or df_base.empty:
                st.info("Base vazia para este recorte.")
            else:
                view = core_build_audit_view(df_base, col, (float(ln) if ln is not None else None), _min_to_mmss)
                st.dataframe(view, width='stretch', height=420)
            st.divider()
            if col:
                _render_market_context_panel(f"{market_name} (contexto)", col)
    
        with tabs_c_map["Seletor Rápido"]:
            st.markdown("**Seletor rÃ¡pido de combinaÃ§Ã£o (Time + Lado + Resultado + Linha)**")
            st.caption("Escolha a combinaÃ§Ã£o e veja Stats + Odd (Modelo/Hist/Mix) para Acima e Abaixo na linha selecionada.")
    
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                team_sel = st.selectbox("Time", [teamA, teamB], index=0, key="quick_combo_team")
            with c2:
                side_sel_raw = st.selectbox("Lado", ["Blue", "Red", "Ambos"], index=0, key="quick_combo_side")
            with c3:
                result_sel = st.selectbox("Resultado", ["Vitória", "Derrota", "Ambos"], index=0, key="quick_combo_result")
            with c4:
                metric_opts = core_build_quick_metric_options(
                    teamA,
                    teamB,
                    kills,
                    towers,
                    dragons,
                    barons,
                    inhib,
                    time_m,
                    teamA_kills_lines,
                    teamB_kills_lines,
                    team_sel,
                )
                metric_name = st.selectbox("Mercado", list(metric_opts.keys()), index=0, key="quick_combo_metric")
            metric_key, value_col, line_values = metric_opts.get(metric_name, ("kills", "total_kills", []))
            line_values = list(line_values or [])
            with c5:
                if line_values:
                    line_sel_multi = st.multiselect(
                        "Linha (pré-definida)",
                        options=line_values,
                        default=[line_values[0]],
                        key="quick_combo_line_multi",
                    )
                else:
                    line_sel_multi = []
                    st.caption("Sem linhas pré-definidas")
            manual_lines_raw = st.text_input(
                "Linha manual (1 ou mais, separadas por vírgula)",
                value=str(st.session_state.get("quick_combo_line_manual", "") or ""),
                key="quick_combo_line_manual",
                placeholder="Ex.: 26.5, 27.5 (ou 31:30 para Tempo)",
            )

            selected_lines: list[float] = []
            try:
                for _lv in line_sel_multi:
                    _v = float(_lv)
                    if math.isfinite(_v):
                        selected_lines.append(_v)
            except Exception:
                pass
            manual_lines = _parse_time_lines(manual_lines_raw) if str(metric_key) == "time" else _parse_lines(manual_lines_raw)
            for _lv in list(manual_lines or []):
                try:
                    _v = float(_lv)
                    if math.isfinite(_v):
                        selected_lines.append(_v)
                except Exception:
                    continue
            selected_lines = sorted(set([float(x) for x in selected_lines if math.isfinite(float(x))]))
    
            if not selected_lines:
                st.info("Sem linhas para o mercado selecionado. Use linha pré-definida e/ou linha manual.")
            else:
                team_is_a = str(team_sel) == str(teamA)
                df_team, df_h2h_team = core_quick_team_frame(team_is_a, dfA_all, dfB_all, df_h2h_A, df_h2h_B)
                side_key_raw = str(side_sel_raw).strip().lower()
                side_key = None if side_key_raw.startswith("amb") else side_key_raw
                result_key_raw = str(result_sel).strip().lower()
                if result_key_raw.startswith("amb"):
                    want_win = None
                else:
                    want_win = bool(result_key_raw.startswith("vit"))
                _all_rows = []
                for line_f in selected_lines:
                    _pml_ss = st.session_state.get("_p_ml_scope", float("nan"))
                    p_ml_base = float(_pml_ss) if _pml_ss is not None else float("nan")
                    p_outcome, p_side_ctx = core_quick_outcome_probs(
                        p_ml_scope=p_ml_base,
                        team_is_a=team_is_a,
                        want_win=want_win,
                        df_team=df_team,
                        side_key=side_key,
                    )
    
                    p_over = None
                    if metric_key in totals_for_lines:
                        try:
                            p_over = float(total_over_prob(totals_for_lines[metric_key], line_f))
                        except Exception:
                            p_over = None
                    elif metric_key == "team_kills":
                        try:
                            if team_is_a:
                                mean_k = _combine_mean(pA.kills_for, pB.kills_against)
                                sd_k = _combine_sd(pA.kills_for_sd, pB.kills_against_sd)
                            else:
                                mean_k = _combine_mean(pB.kills_for, pA.kills_against)
                                sd_k = _combine_sd(pB.kills_for_sd, pA.kills_against_sd)
                            obj_k = MatchupTotals(mean=float(mean_k), sd=float(max(sd_k, 1.0)), dist="normal")
                            p_over = float(total_over_prob(obj_k, line_f))
                        except Exception:
                            p_over = None

                    def _line_fmt(metric_name_key: str, line_value: float) -> str:
                        return _min_to_mmss(line_value) if str(metric_name_key) == "time" else f"{float(line_value):g}"

                    result_rows = core_build_quick_selector_rows(
                        df_team=df_team,
                        df_h2h_team=df_h2h_team,
                        value_col=value_col,
                        line_f=float(line_f),
                        side_key=side_key,
                        want_win=want_win,
                        p_outcome=float(p_outcome),
                        p_side_ctx=p_side_ctx,
                        p_over=p_over,
                        laplace_pick=str(st.session_state.get("resumo_laplace_pick", "Liga 15") or "Liga 15"),
                        mix_w_model=float(st.session_state.get("resumo_mix_w_model", 40) or 40),
                        mix_w_laplace=float(st.session_state.get("resumo_mix_w_laplace", 40) or 40),
                        mix_w_sc=float(st.session_state.get("resumo_mix_w_sc", 20) or 20),
                        metric_key=str(metric_key),
                        format_num_fn=_format_num,
                        line_formatter_fn=_line_fmt,
                    )
                    _all_rows.extend(result_rows)

                st.markdown(f"**Configuração:** {team_sel} | {side_sel_raw} | {result_sel} | {metric_name}")
                st.dataframe(pd.DataFrame(_all_rows), width='stretch', hide_index=True)
            st.divider()
            if value_col:
                _render_market_context_panel(f"{metric_name} (contexto)", value_col)
    
    
    # -----------------------------
    # Ranking (Elo) - aba extra
    # -----------------------------
if "Rankings" in tab_refs:
 with tab_refs["Rankings"]:
    st.markdown("### Rankings (ML core v2) — total_strength")
    st.caption("total_strength = residual_elo + tier_base(tier) + league_base(league) + achievements_boost(decay)")


    # Filtros dependentes (Liga -> Ano -> Split -> Playoffs)
    leagues = []
    try:
        if team_games is not None and not team_games.empty and 'league' in team_games.columns:
            leagues = sorted([x for x in team_games['league'].fillna('').astype(str).unique().tolist() if x])
    except Exception:
        leagues = []

    # PadrÃ£o: tudo comeÃ§a "em branco" (sem prÃ©-seleÃ§Ã£o)
    for _k in ['rank_league_visual', 'rank_year_visual', 'rank_split_visual', 'rank_po_visual']:
        if _k not in st.session_state or st.session_state.get(_k) in ['All', None]:
            st.session_state[_k] = ''

    def _fmt_rank_placeholder(x: str) -> str:
        return 'Selecione ...' if str(x) == '' else str(x)

    def _on_rank_league_change():
        st.session_state['rank_year_visual'] = ''
        st.session_state['rank_split_visual'] = ''

    c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.0])
    with c1:
        rank_league = st.selectbox('Liga', options=[''] + leagues, index=0, key='rank_league_visual', format_func=_fmt_rank_placeholder, on_change=_on_rank_league_change)

    # base para opÃ§Ãµes dependentes
    df_opt = team_games
    if rank_league and df_opt is not None and not df_opt.empty and 'league' in df_opt.columns:
        df_opt = df_opt[df_opt['league'].astype(str) == str(rank_league)].copy()

    years = []
    try:
        if df_opt is not None and not df_opt.empty and 'year' in df_opt.columns:
            years = sorted([int(x) for x in pd.to_numeric(df_opt['year'], errors='coerce').dropna().unique().tolist()])
    except Exception:
        years = []

    with c2:
        if st.session_state.get('rank_year_visual','') not in [''] + [str(y) for y in years]:
            st.session_state['rank_year_visual'] = ''
        rank_year = st.selectbox('Ano', options=[''] + [str(y) for y in years], index=0, key='rank_year_visual', format_func=_fmt_rank_placeholder)

    df_opt2 = df_opt
    if rank_year and df_opt2 is not None and not df_opt2.empty and 'year' in df_opt2.columns:
        try:
            df_opt2 = df_opt2[pd.to_numeric(df_opt2['year'], errors='coerce') == int(rank_year)].copy()
        except Exception:
            pass

    splits = []
    try:
        if df_opt2 is not None and not df_opt2.empty and 'split' in df_opt2.columns:
            splits = sorted([str(x) for x in df_opt2['split'].fillna('').astype(str).unique().tolist() if str(x)])
    except Exception:
        splits = []

    with c3:
        if st.session_state.get('rank_split_visual','') not in [''] + splits:
            st.session_state['rank_split_visual'] = ''
        rank_split = st.selectbox('Split', options=[''] + splits, index=0, key='rank_split_visual', format_func=_fmt_rank_placeholder)

    with c4:
        if st.session_state.get('rank_po_visual','') not in ['', 'Regular', 'Playoffs']:
            st.session_state['rank_po_visual'] = ''
        rank_playoffs = st.selectbox('Playoffs', options=['', 'Regular', 'Playoffs'], index=0, key='rank_po_visual', format_func=_fmt_rank_placeholder)
    rank_filters = Filters(
        year=None if not rank_year else int(rank_year),
        league=None if not rank_league else rank_league,
        split=None if not rank_split else rank_split,
        playoffs=None if not rank_playoffs else (True if rank_playoffs == 'Playoffs' else False),
    )


    top_n = st.slider("Top N (ranking global)", min_value=50, max_value=400, value=200, step=50, key="rank_top_n")
    _rank_fetch_n = int(max(int(top_n), 600))
    df_rank_v2 = _mlcore_rank_table_v2(artifact_path, as_of=as_of_date, top_n=_rank_fetch_n)
    df_rank_v2_view = df_rank_v2.copy()
    if rank_league and (not df_rank_v2_view.empty) and ('league' in df_rank_v2_view.columns):
        df_rank_v2_view = df_rank_v2_view[df_rank_v2_view['league'].astype(str) == str(rank_league)].copy()

    # Fallback robusto: usa Elo do CSV quando v2 não vier OU vier sem cobertura global útil.
    df_fallback = pd.DataFrame()
    try:
        df_fallback = _elo_rank_table(
            team_games,
            filters=rank_filters,
            half_life_days=float(half_life_days),
            k_factor=24.0,
            base_rating=1500.0,
        )
        if not df_fallback.empty:
            df_fallback = df_fallback.head(int(top_n)).copy()
    except Exception:
        df_fallback = pd.DataFrame()

    df_rank_main = df_rank_v2_view.copy()
    _rank_main_source = "ML core v2"

    _no_filters = (not rank_league) and (not rank_year) and (not rank_split) and (not rank_playoffs)
    if df_rank_main.empty:
        if not df_fallback.empty:
            df_rank_main = df_fallback.copy()
            _rank_main_source = "Elo fallback (CSV)"
    elif _no_filters:
        try:
            _v2_lgs = set(df_rank_main["league"].astype(str).str.strip()) if "league" in df_rank_main.columns else set()
            _v2_teams = set(df_rank_main["team"].astype(str).str.strip().str.casefold()) if "team" in df_rank_main.columns else set()
            _need_fallback = False
            if "LCK" not in _v2_lgs:
                _need_fallback = True
            for _feat in ["gen.g", "t1"]:
                if _feat not in _v2_teams:
                    _need_fallback = True
                    break
            if _need_fallback and (not df_fallback.empty):
                df_rank_main = df_fallback.copy()
                _rank_main_source = "Elo fallback (CSV)"
        except Exception:
            pass

    if df_rank_main.empty:
        st.warning("Não consegui gerar ranking para esse recorte (v2 + fallback sem dados).")
    else:
        st.caption(f"Fonte do ranking principal: {_rank_main_source}")
        if rank_league and ('league' in df_rank_main.columns):
            df_rank_main = df_rank_main[df_rank_main['league'].astype(str) == str(rank_league)].copy()
        if rank_year and ('year' in df_rank_main.columns):
            try:
                df_rank_main = df_rank_main[pd.to_numeric(df_rank_main['year'], errors='coerce') == int(rank_year)].copy()
            except Exception:
                pass
        if rank_split and ('split' in df_rank_main.columns):
            df_rank_main = df_rank_main[df_rank_main['split'].astype(str) == str(rank_split)].copy()
        if rank_playoffs and ('playoffs' in df_rank_main.columns):
            try:
                _want_po = True if str(rank_playoffs) == "Playoffs" else False
                _po_ser = df_rank_main['playoffs']
                if _po_ser.dtype == bool:
                    df_rank_main = df_rank_main[_po_ser == _want_po].copy()
                else:
                    df_rank_main = df_rank_main[pd.to_numeric(_po_ser, errors='coerce').fillna(0).astype(int).astype(bool) == _want_po].copy()
            except Exception:
                pass
        # Sem filtros (ranking geral): garante presença mínima das ligas principais e times-chave.
        if (not rank_league) and (not rank_year) and (not rank_split) and (not rank_playoffs):
            try:
                if not df_rank_main.empty and ("league" in df_rank_main.columns):
                    _df_cov = df_rank_main.copy()
                    _score_col = None
                    for _c in ["total_strength", "elo", "rating", "score"]:
                        if _c in _df_cov.columns:
                            _score_col = _c
                            break
                    if _score_col is not None:
                        _df_cov["_score_num"] = pd.to_numeric(_df_cov[_score_col], errors="coerce")
                        _df_cov = _df_cov.sort_values("_score_num", ascending=False, na_position="last")
                        _top_n = int(top_n)
                        _top = _df_cov.head(_top_n).copy()
                        _majors = ["LCK", "LPL", "LEC", "LCS"]
                        _present = set(_top["league"].astype(str).tolist())
                        _extras = []
                        for _lg in _majors:
                            if _lg in _present:
                                continue
                            _cand = _df_cov[_df_cov["league"].astype(str) == _lg].head(1)
                            if not _cand.empty:
                                _extras.append(_cand)
                        if _extras:
                            _top = pd.concat([_top] + _extras, ignore_index=True)
                            _dedup_cols = [c for c in ["team", "league"] if c in _top.columns]
                            if _dedup_cols:
                                _top = _top.drop_duplicates(subset=_dedup_cols, keep="first")
                            if _top.shape[0] > _top_n:
                                _maj = _top[_top["league"].astype(str).isin(_majors)].copy()
                                _oth = _top[~_top["league"].astype(str).isin(_majors)].copy()
                                _keep = pd.concat([_maj, _oth], ignore_index=True).head(_top_n).copy()
                                _top = _keep
                            _featured = ["gen.g", "t1"]
                            _feat_rows = []
                            for _ft in _featured:
                                _cand = _df_cov[_df_cov["team"].astype(str).str.strip().str.casefold() == _ft].head(1)
                                if not _cand.empty:
                                    _feat_rows.append(_cand)
                            if _feat_rows:
                                _top = pd.concat([_top] + _feat_rows, ignore_index=True)
                                _dedup_cols = [c for c in ["team", "league"] if c in _top.columns]
                                if _dedup_cols:
                                    _top = _top.drop_duplicates(subset=_dedup_cols, keep="first")
                                _top = _top.head(_top_n).copy()
                            df_rank_main = _top.drop(columns=["_score_num"], errors="ignore")
            except Exception:
                pass
        if not df_rank_main.empty:
            df_rank_main = df_rank_main.head(int(top_n)).copy()
        st.dataframe(df_rank_main, width='stretch', height=520)

    st.divider()

    with st.expander("Ranking antigo (Elo / debug)", expanded=False):
        st.caption("Elo é atualizado jogo a jogo; jogos mais recentes valem mais (half-life).")
        cA, cB, cC = st.columns([1,1,2])
        with cA:
            k_factor = st.number_input("K (agressividade)", min_value=4.0, max_value=64.0, value=24.0, step=1.0)
        with cB:
            base_rating = st.number_input("Base Elo", min_value=1000.0, max_value=2000.0, value=1500.0, step=10.0)
        with cC:
            st.markdown("**Dica:** K menor = ranking mais estável | K maior = reage mais rápido.")

        elo_df = _elo_rank_table(
            team_games,
            filters=rank_filters,
            half_life_days=float(half_life_days),
            k_factor=float(k_factor),
            base_rating=float(base_rating),
        )
        if elo_df.empty:
            st.info("Sem jogos suficientes nesse recorte para calcular ranking.")
        else:
            st.dataframe(elo_df.head(50), width='stretch', height=520)

            if "league" in elo_df.columns and elo_df["league"].astype(str).str.len().sum() > 0:
                st.markdown("#### Força média por liga (no recorte)")
                try:
                    elo_num = pd.to_numeric(elo_df["elo"], errors="coerce")
                except Exception:
                    elo_num = pd.Series([None]*len(elo_df))
                tmp = elo_df.copy()
                tmp["elo_num"] = elo_num
                by = tmp.dropna(subset=["elo_num"]).groupby("league", dropna=False)["elo_num"].agg(["mean","count"]).reset_index()
                by = by.sort_values("mean", ascending=False)
                by["mean"] = by["mean"].apply(lambda x: _format_num(x, 0))
                st.dataframe(by, width='stretch', height=260)


# -----------------------------
# Vencedores (Achievements) - aba extra
# -----------------------------
if False and "Vencedores" in tab_refs:
 with tab_refs["Vencedores"]:
    st.markdown("### Vencedores (Achievements) â€” boost com decay")
    st.caption("Aqui vocÃª registra resultados (MSI/Worlds/Regional/Other) para adicionar pontos ao total_strength. O boost decai com o tempo (half-life).")

    # resolve caminho do achievements.json
    # 1) usa o caminho do artifact (se existir), senÃ£o usa fallback na mesma pasta do ml_artifact.json
    ach_path_default = ""
    try:
        art = _load_artifact_cached(_artifact_sig(str(Path(artifact_path))))
        ach_meta = dict(art.meta.get("achievements") or {})
        ach_path_default = str(ach_meta.get("path") or "").strip()
        if ach_path_default:
            ach_path_default = _resolve_relpath(artifact_path, ach_path_default)
    except Exception:
        ach_path_default = ""

    if not ach_path_default:
        try:
            ach_path_default = str(Path(artifact_path).with_name("achievements.json"))
        except Exception:
            ach_path_default = str(APP_ROOT / "achievements.json")

    ach_path_ui = st.text_input("Caminho do achievements.json", value=st.session_state.get("ach_path_ui", ach_path_default), key="ach_path_ui")
    st.caption("Dica: por padrÃ£o o app procura um achievements.json na mesma pasta do ml_artifact.json.")

    # Importar/Exportar (opcional)
    c_imp, c_exp = st.columns([2, 1])
    with c_imp:
        up = st.file_uploader("Importar achievements.json", type=["json"], key="ach_upload")
        if up is not None:
            try:
                up_data = json.loads(up.getvalue().decode("utf-8"))
                st.success(f"Arquivo carregado: {up.name} (preview abaixo).")
                st.json(up_data, expanded=False)
                if st.button("Usar este arquivo (substituir)", type="primary", key="ach_use_uploaded"):
                    try:
                        _save_achievements_json(ach_path_ui, up_data)
                        st.success(f"Salvo em: {ach_path_ui}")
                        try:
                            st.cache_data.clear()
                        except Exception:
                            pass
                        data = up_data
                    except Exception as e:
                        st.error(f"Falha ao salvar upload: {e}")
            except Exception as e:
                st.error(f"Falha ao ler JSON enviado: {e}")
    with c_exp:
        try:
            st.download_button(
                "Baixar achievements.json",
                data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="achievements.json",
                mime="application/json",
                width='stretch',
            )
        except Exception:
            pass


    # carrega (ou cria estrutura mÃ­nima)
    data: Dict[str, Any] = {}
    try:
        if Path(ach_path_ui).exists():
            try:
                data = load_achievements(ach_path_ui)
            except Exception as e:
                # JSON invÃ¡lido/corrompido ou outro erro: nÃ£o derruba a aba
                st.error(f"Falha ao ler achievements.json: {e}")
                # opÃ§Ã£o de reset (com backup)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Recriar achievements.json vazio (backup)", width='stretch'):
                        try:
                            src = Path(ach_path_ui)
                            if src.exists():
                                bak = src.with_suffix(src.suffix + ".broken")
                                shutil.copy2(src, bak)
                            _save_achievements_json(ach_path_ui, {
                                "config": {
                                    "half_life_days": {"worlds": 365, "msi": 240, "regional": 180, "other": 120},
                                    "fixed_weights_points": {}
                                },
                                "entries": []
                            })
                            st.success("Arquivo recriado. Recarregandoâ€¦")
                            st.rerun()
                        except Exception as e2:
                            st.error(f"NÃ£o foi possÃ­vel recriar: {e2}")
                with c2:
                    st.caption("Dica: se o JSON ficou corrompido, ele foi salvo com backup .broken.")
                data = {
                    "config": {
                        "half_life_days": {"worlds": 365, "msi": 240, "regional": 180, "other": 120},
                        "fixed_weights_points": {}
                    },
                    "entries": []
                }
        else:
            data = {
                "config": {
                    "half_life_days": {"worlds": 365, "msi": 240, "regional": 180, "other": 120},
                    "fixed_weights_points": {}
                },
                "entries": []
            }
    except Exception as e:
        st.error(f"Falha ao ler achievements.json: {e}")
        data = {"config": {"half_life_days": {"worlds": 365, "msi": 240, "regional": 180, "other": 120}, "fixed_weights_points": {}}, "entries": []}

    cfg = data.get("config") or {}
    hl = dict(cfg.get("half_life_days") or {"worlds": 365, "msi": 240, "regional": 180, "other": 120})
    fwp = dict(cfg.get("fixed_weights_points") or {})
    DEFAULT_W: Dict[str, float] = {
        "msi:top16": 6.0,
        "msi:top8": 12.0,
        "msi:top4": 20.0,
        "msi:runner_up": 32.0,
        "msi:champion": 45.0,
        "worlds:top16": 8.0,
        "worlds:top8": 16.0,
        "worlds:top4": 26.0,
        "worlds:runner_up": 42.0,
        "worlds:champion": 60.0,
        "regional:playoffs": 6.0,
        "regional:top8": 8.0,
        "regional:top4": 12.0,
        "regional:runner_up": 18.0,
        "regional:champion": 24.0,
        "other:champion": 12.0,
        "other:runner_up": 8.0,
        "other:top4": 6.0,
    }
    if not fwp:
        fwp = dict(DEFAULT_W)


    # validaÃ§Ãµes rÃ¡pidas
    entries = list(data.get("entries") or [])
    df_entries = pd.DataFrame(entries) if entries else pd.DataFrame(columns=["team", "tournament", "event", "placement", "end_date"])

    # DiagnÃ³stico do arquivo (erros comuns: time nÃ£o bate com CSV, data invÃ¡lida, duplicados)
    # times do CSV (para dropdown + validaÃ§Ãµes)
    # (no plays_app.py, a lista `teams` jÃ¡ vem do CSV ativo; evitamos depender de outras dfs aqui)
    teams_ui = teams if isinstance(teams, list) and teams else []

    teams_set = set(map(str, teams_ui or []))
    cfg_events = list((data.get("config") or {}).get("events") or [])
    cfg_places = list((data.get("config") or {}).get("placements") or [])
    valid_events = set(map(str, cfg_events)) if cfg_events else {"worlds", "msi", "regional", "other"}
    valid_places = set(map(str, cfg_places)) if cfg_places else {"champion", "runner_up", "top4", "top8", "top16", "playoffs"}
    issues: List[Dict[str, Any]] = []
    valid_mask = []
    for i_row, r in df_entries.iterrows():
        row_ok = True
        team = str(r.get("team", "") or "").strip()
        event = str(r.get("event", "") or "").strip()
        placement = str(r.get("placement", "") or "").strip()
        end_date_s = str(r.get("end_date", "") or "").strip()
        tournament = str(r.get("tournament", "") or "").strip()
        if not team or not event or not placement or not end_date_s:
            row_ok = False
            issues.append({"idx": int(i_row), "tipo": "campo_ausente", "team": team, "event": event, "placement": placement, "end_date": end_date_s, "detalhe": "Campos obrigatÃ³rios: team/event/placement/end_date"})
        if event and event not in valid_events:
            row_ok = False
            issues.append({"idx": int(i_row), "tipo": "event_invÃ¡lido", "team": team, "event": event, "placement": placement, "end_date": end_date_s, "detalhe": f"Eventos vÃ¡lidos: {sorted(valid_events)}"})
        if placement and placement not in valid_places:
            row_ok = False
            issues.append({"idx": int(i_row), "tipo": "placement_invÃ¡lido", "team": team, "event": event, "placement": placement, "end_date": end_date_s, "detalhe": f"Placements vÃ¡lidos: {sorted(valid_places)}"})
        # data ISO
        if end_date_s:
            try:
                _ = date.fromisoformat(end_date_s)
            except Exception:
                row_ok = False
                issues.append({"idx": int(i_row), "tipo": "data_invÃ¡lida", "team": team, "event": event, "placement": placement, "end_date": end_date_s, "detalhe": "Use YYYY-MM-DD"})
        # time existe no CSV
        if team and teams_set and team not in teams_set:
            row_ok = False
            sugg = difflib.get_close_matches(team, sorted(teams_set), n=3, cutoff=0.65)
            issues.append({"idx": int(i_row), "tipo": "time_nÃ£o_encontrado", "team": team, "event": event, "placement": placement, "end_date": end_date_s, "detalhe": ("SugestÃµes: " + ", ".join(sugg)) if sugg else "Time nÃ£o aparece no CSV atual"})
        if not tournament:
            # nÃ£o invalida, mas avisa
            issues.append({"idx": int(i_row), "tipo": "tournament_vazio", "team": team, "event": event, "placement": placement, "end_date": end_date_s, "detalhe": "Opcional, mas ajuda na organizaÃ§Ã£o"})
        valid_mask.append(bool(row_ok))

    # duplicados (mesmo team+event+placement+end_date)
    if not df_entries.empty:
        try:
            dup = df_entries.duplicated(subset=["team", "event", "placement", "end_date"], keep=False)
            if dup.any():
                for i_row in df_entries.index[dup]:
                    r = df_entries.loc[i_row]
                    issues.append({"idx": int(i_row), "tipo": "duplicado", "team": str(r.get("team","")), "event": str(r.get("event","")), "placement": str(r.get("placement","")), "end_date": str(r.get("end_date","")), "detalhe": "Entrada repetida"})
        except Exception:
            pass

    n_entries = int(len(df_entries))
    n_valid = int(sum(valid_mask)) if valid_mask else 0
    n_issues = int(len(issues))
    n_missing_team = int(sum(1 for x in issues if x.get("tipo") == "time_nÃ£o_encontrado"))
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Entradas", n_entries)
    c_m2.metric("VÃ¡lidas", n_valid)
    c_m3.metric("Alertas", n_issues)
    c_m4.metric("Times nÃ£o encontrados", n_missing_team)
    if n_issues:
        with st.expander("Ver alertas / problemas encontrados", expanded=False):
            df_issues = pd.DataFrame(issues)
            st.dataframe(df_issues, width='stretch', hide_index=True)
            st.caption("Dica: se um time nÃ£o aparece no CSV atual, normalmente Ã© diferenÃ§a de nome. Depois a gente liga isso na aba de aliases (renome).")


    st.markdown("#### Adicionar resultado")
    a1, a2, a3 = st.columns([3, 2, 2])
    with a1:
        team_sel = st.selectbox("Time", teams_ui, index=0 if teams_ui else None, key="ach_team_sel")
        tournament_txt = st.text_input("Campeonato", value=st.session_state.get("ach_tournament_txt", ""), key="ach_tournament_txt")
    with a2:
        event_sel = st.selectbox("Evento", ["worlds", "msi", "regional", "other"], index=1, key="ach_event_sel")
        placement_sel = st.selectbox("ColocaÃ§Ã£o", ["champion", "runner_up", "top4", "top8", "top16", "playoffs"], index=3, key="ach_placement_sel")
    with a3:
        end_date_sel = st.date_input("Data de tÃ©rmino", value=st.session_state.get("ach_end_date_sel", date.today()), key="ach_end_date_sel")
        st.write("")
        add_clicked = st.button("Adicionar", type="primary", width='stretch')

    if add_clicked:
        new_entry = {
            "team": str(team_sel),
            "tournament": str(tournament_txt).strip() or str(event_sel).upper(),
            "event": str(event_sel),
            "placement": str(placement_sel),
            "end_date": str(end_date_sel),
        }
        entries.append(new_entry)
        data["entries"] = entries
        st.success("Adicionado. Clique em **Salvar** para gravar no arquivo.")

    st.markdown("#### Pesos (pontos) e half-life")
    c_hl, c_w = st.columns([1, 2])
    with c_hl:
        hl_df = pd.DataFrame([{"evento": k, "half_life_days": int(v)} for k, v in hl.items()])
        hl_edit = st.data_editor(hl_df, width='stretch', hide_index=True, num_rows="fixed")
    with c_w:
        # tabela de pesos event:placement
        if fwp:
            rows = []
            for k, v in fwp.items():
                if ":" in str(k):
                    ev, pl = str(k).split(":", 1)
                else:
                    ev, pl = str(k), ""
                rows.append({"evento": ev, "colocaÃ§Ã£o": pl, "pontos": float(v)})
            w_df = pd.DataFrame(rows)
        else:
            w_df = pd.DataFrame(columns=["evento", "colocaÃ§Ã£o", "pontos"])
        w_edit = st.data_editor(w_df, width='stretch', hide_index=True, num_rows="dynamic")

    st.markdown("#### Entradas registradas")
    if df_entries.empty and (not entries):
        st.info("Nenhum resultado registrado ainda.")
    else:
        df_show = pd.DataFrame(entries)
        if "end_date" in df_show.columns:
            df_show["end_date"] = df_show["end_date"].astype(str)
        # coluna para deletar
        df_show["_apagar"] = False
        df_edit = st.data_editor(df_show, width='stretch', hide_index=True, num_rows="fixed")
        # botÃµes salvar/apagar
        b1, b2 = st.columns([1, 1])
        with b1:
            save_clicked = st.button("Salvar", type="primary", width='stretch')
        with b2:
            delete_clicked = st.button("Apagar marcados", width='stretch')

        if delete_clicked:
            keep = df_edit[~df_edit["_apagar"]].drop(columns=["_apagar"], errors="ignore")
            entries_new = keep.to_dict(orient="records")
            data["entries"] = entries_new
            st.success("Marcados removidos. Clique em **Salvar** para gravar.")

        if save_clicked:
            # atualiza config a partir dos editores
            try:
                hl_new = {str(r["evento"]): int(r["half_life_days"]) for _, r in hl_edit.iterrows() if str(r.get("evento","")).strip()}
            except Exception:
                hl_new = hl
            try:
                fwp_new: Dict[str, float] = {}
                for _, r in w_edit.iterrows():
                    ev = str(r.get("evento","")).strip()
                    pl = str(r.get("colocaÃ§Ã£o","")).strip()
                    pts = r.get("pontos", None)
                    if not ev or not pl:
                        continue
                    try:
                        pts_f = float(pts)
                    except Exception:
                        continue
                    fwp_new[f"{ev}:{pl}"] = pts_f
            except Exception:
                fwp_new = fwp

            data["config"] = {"half_life_days": hl_new, "fixed_weights_points": fwp_new}
            try:
                _save_achievements_json(ach_path_ui, data)
                st.success(f"Salvo em: {ach_path_ui}")
                # limpa cache para refletir imediato no ML/ranking
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Falha ao salvar: {e}")

    st.markdown("#### ConferÃªncia rÃ¡pida (boost)")
    sel_team = st.selectbox("Time para conferir", teams_ui, key="ach_check_team")
    as_of_check = st.date_input("Data base (para decay)", value=st.session_state.get("ml_as_of_date", date.today()), key="ach_check_asof")
    # calcula boost
    try:
        team_idx = build_team_index(data)
        # usa pesos fixos do arquivo (se vazio, cai no default via _ach_layer_for_artifact quando usado no ML)
        weights_for_check = (data.get("config") or {}).get("fixed_weights_points") or {}
        if not weights_for_check:
            weights_for_check = dict(DEFAULT_W)
        boost = float(team_boost_points(str(sel_team), as_of_check, team_idx, weights_for_check))
        st.metric("Boost (pontos)", value=_format_num(boost, 1))
        # detalhamento (por entrada) â€” ajuda a conferir se estÃ¡ tudo certo
        try:
            entries_team = [e for e in (data.get("entries") or []) if str(e.get("team","")).strip() == str(sel_team).strip()]
            rows = []
            for e in entries_team:
                ev = str(e.get("event","") or "").strip()
                pl = str(e.get("placement","") or "").strip()
                ed = str(e.get("end_date","") or "").strip()
                tour = str(e.get("tournament","") or "").strip()
                if not (ev and pl and ed):
                    continue
                try:
                    d_end = date.fromisoformat(ed)
                except Exception:
                    continue
                age_days = int((as_of_check - d_end).days)
                if age_days < 0:
                    age_days = 0
                half_life = float(hl.get(ev, 180) or 180)
                try:
                    decay = float(math.exp(-math.log(2) * age_days / half_life)) if half_life > 0 else 0.0
                except Exception:
                    decay = 0.0
                key = f"{ev}:{pl}"
                w = float(weights_for_check.get(key, 0.0) or 0.0)
                contrib = float(w * decay)
                rows.append({
                    "tournament": tour,
                    "event": ev,
                    "placement": pl,
                    "end_date": ed,
                    "age_days": age_days,
                    "half_life": half_life,
                    "weight_pts": w,
                    "decay": decay,
                    "contrib_pts": contrib,
                })
            if rows:
                with st.expander("Ver detalhamento do boost (por entrada)", expanded=False):
                    df_b = pd.DataFrame(rows)
                    # ordena por contribuiÃ§Ã£o
                    df_b = df_b.sort_values(["contrib_pts", "end_date"], ascending=[False, False])
                    st.dataframe(df_b, width='stretch', hide_index=True)
                    st.caption("contrib_pts = weight_pts Ã— decay (decay por half-life e idade em dias)")
        except Exception:
            pass
    except Exception as e:
        st.info(f"NÃ£o foi possÃ­vel calcular boost agora: {e}")

# -----------------------------
# Jogos (Agenda)
# -----------------------------
if "Jogos" in tab_refs:
 with tab_refs["Jogos"]:
    st.markdown('### Jogos (agenda / ediÃ§Ã£o rÃ¡pida)')
    st.caption('Cole jogos em formato livre (ex.: do site), adicione Ã  lista, edite e carregue para o painel principal.')

    def _dedup_repeat_token(s: str) -> str:
        s0 = (s or '').strip()
        if not s0:
            return ''
        compact = re.sub(r'\s+', '', s0)
        n = len(compact)
        if n % 2 == 0:
            half = compact[: n//2]
            if half == compact[n//2:]:
                return half
        return s0

    def _parse_schedule(text_in: str) -> List[Dict[str, Any]]:
        lines = [ln.strip() for ln in (text_in or '').splitlines()]
        lines = [ln for ln in lines if ln]
        games: List[Dict[str, Any]] = []
        cur_date = ''
        i = 0
        date_re = re.compile(r'(\b\d{2}\/\d{2}\b)')
        time_re = re.compile(r'(\b\d{2}\:\d{2}\b)')
        while i < len(lines):
            ln = lines[i]
            mdate = date_re.search(ln)
            if mdate:
                cur_date = mdate.group(1)
                i += 1
                continue
            mtime = time_re.search(ln)
            if mtime:
                cur_time = mtime.group(1)
                # prÃ³ximo nÃ£o-vazio = timeA, depois timeB
                j = i + 1
                # pula separadores
                while j < len(lines) and (lines[j] in ['-', 'â€“', 'â€”'] or lines[j].strip('- ') == ''):
                    j += 1
                if j >= len(lines):
                    break
                team1 = _dedup_repeat_token(lines[j])
                j += 1
                while j < len(lines) and (lines[j] in ['-', 'â€“', 'â€”'] or lines[j].strip('- ') == ''):
                    j += 1
                if j >= len(lines):
                    break
                team2 = _dedup_repeat_token(lines[j])

                games.append({
                    'date': cur_date,
                    'time': cur_time,
                    'league': '',
                    'teamA': team1,
                    'teamB': team2,
                    'bo': int(bo) if str(bo).isdigit() else 3,
                    'lines': '',
                })
                i = j + 1
                continue
            i += 1
        return games

    if 'games_list' not in st.session_state:
        st.session_state['games_list'] = []

    paste = st.text_area('Cole aqui a lista de jogos', height=180, key='schedule_paste')
    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        default_league = st.text_input('Liga (opcional)', value=st.session_state.get('schedule_default_league',''), key='schedule_default_league')
    with cB:
        default_bo = st.selectbox('Formato', options=[1,3,5], index=1, key='schedule_default_bo')
    with cC:
        if st.button('Adicionar jogos da colagem', type='primary', width='stretch', key='schedule_add_btn'):
            new_games = _parse_schedule(paste)
            for g in new_games:
                if default_league:
                    g['league'] = default_league
                g['bo'] = int(default_bo)
            st.session_state['games_list'] = (st.session_state['games_list'] or []) + new_games
            st.success(f'Adicionados: {len(new_games)} jogo(s).')

    if st.session_state.get('games_list'):
        df_games = pd.DataFrame(st.session_state['games_list'])
        if 'remove' not in df_games.columns:
            df_games['remove'] = False
        st.markdown('#### Lista (editÃ¡vel)')
        edited = st.data_editor(df_games, num_rows='dynamic', width='stretch', hide_index=True, key='schedule_editor')
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            if st.button('Salvar alteraÃ§Ãµes', width='stretch', key='schedule_save'):
                try:
                    out = edited.to_dict(orient='records')
                    st.session_state['games_list'] = out
                    st.success('Lista salva.')
                except Exception as e:
                    st.error(f'Falha ao salvar: {e}')
        with c2:
            if st.button('Aplicar remoÃ§Ãµes', width='stretch', key='schedule_remove'):
                try:
                    out = edited.to_dict(orient='records')
                    out2 = [r for r in out if not bool(r.get('remove'))]
                    st.session_state['games_list'] = out2
                    st.success(f'Removidos: {len(out)-len(out2)}')
                except Exception as e:
                    st.error(f'Falha ao remover: {e}')
        with c3:
            idx = 0
            if len(edited) > 1:
                idx = st.number_input('Carregar jogo #', min_value=1, max_value=int(len(edited)), value=1, step=1, key='schedule_load_idx') - 1
            if st.button('Carregar no painel principal (Time 1/2)', width='stretch', key='schedule_load'):
                try:
                    row = edited.iloc[int(idx)].to_dict()
                    payload = {'_reset_flow': True}
                    if row.get('teamA') in teams:
                        payload['teamA'] = row.get('teamA')
                    if row.get('teamB') in teams:
                        payload['teamB'] = row.get('teamB')
                    if str(row.get('bo')) in ['1','3','5']:
                        payload['bo'] = int(row.get('bo'))

                    # Aplica antes dos widgets via pending_load (evita erro: session_state.bo cannot be modified after widget)
                    st.session_state['pending_load'] = payload
                    st.rerun()
                except Exception as e:
                    st.error(f'Falha ao carregar: {e}')
    else:
        st.info('Sua lista estÃ¡ vazia. Cole alguns jogos e clique em Adicionar.')


# -----------------------------
# Resumo (filtro de odds em todos os recortes) - incorporado na aba Consistência
# -----------------------------
if "Consistência de Mercado (Laplace)" in tab_refs:
 with tab_refs["Consistência de Mercado (Laplace)"]:
    st.markdown("### Resumo de Odds")
    st.caption("Mostra as odds do Modelo que passam no filtro. As colunas Liga/H2H indicam em quais recortes (Ano/15/10/5) a odd do Histórico (Laplace) bate (<= filtro).")

    # linhas atuais (do painel principal)
    _lines_by_metric = {
        'kills': kills,
        'towers': towers,
        'dragons': dragons,
        'barons': barons,
        'inhibitors': inhib,
        'time': time_m,
    }

    # Defaults defensivos para evitar NameError em reruns parciais.
    ch_year = str(st.session_state.get("ch_year", "") or "")
    ch_league = str(st.session_state.get("ch_league", "") or "")
    ch_split = str(st.session_state.get("ch_split", "") or "")
    ch_playoffs = str(st.session_state.get("ch_playoffs", "") or "")
    ch_team = str(st.session_state.get("ch_team", "") or "")

    with st.container(border=True):
        max_odd = st.number_input('Filtro: odd máxima', min_value=1.01, max_value=5.0, value=state_get_float(st.session_state, 'resumo_max_odd', 1.72), step=0.01, key='resumo_max_odd')

        _profiles = dict(st.session_state.get("resumo_filter_profiles") or {})
        _preset_opts = ["Custom"] + sorted([str(k) for k in _profiles.keys() if str(k).strip()])
        _cur_preset = str(st.session_state.get("resumo_filter_preset", "Custom") or "Custom")
        _preset_idx = _preset_opts.index(_cur_preset) if _cur_preset in _preset_opts else 0
        _preset_sel = st.selectbox("Preset de filtro", _preset_opts, index=_preset_idx, key="resumo_filter_preset")
        _p1, _p2, _p3, _p4 = st.columns(4)
        with _p1:
            if st.button("Aplicar preset", key="resumo_apply_preset", width='stretch'):
                if str(_preset_sel) == "Custom":
                    _vals = dict(st.session_state.get("resumo_filter_custom") or {})
                else:
                    _vals = dict(_profiles.get(str(_preset_sel)) or {})
                for _k, _v in _vals.items():
                    st.session_state[_k] = _v
                if _vals:
                    st.session_state["resumo_filter_profile_active"] = "" if str(_preset_sel) == "Custom" else str(_preset_sel)
                    _save_app_settings_to_disk()
                    st.success(f"Preset aplicado: {_preset_sel}")
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
        with _p2:
            if st.button("Salvar como Custom", key="resumo_save_custom", width='stretch'):
                st.session_state["resumo_filter_custom"] = {
                    "resumo_max_odd": float(st.session_state.get("resumo_max_odd", 2.0) or 2.0),
                    "resumo_req_liga": bool(st.session_state.get("resumo_req_liga", True)),
                    "resumo_req_h2h": bool(st.session_state.get("resumo_req_h2h", True)),
                    "resumo_req_model": bool(st.session_state.get("resumo_req_model", True)),
                    "resumo_min_sample_liga": int(st.session_state.get("resumo_min_sample_liga", 8) or 8),
                    "resumo_min_sample_h2h": int(st.session_state.get("resumo_min_sample_h2h", 3) or 3),
                }
                st.session_state["resumo_filter_preset"] = "Custom"
                _save_app_settings_to_disk()
                st.success("Custom salvo em disco.")
        with _p3:
            if st.button("Carregar Custom", key="resumo_load_custom", width='stretch'):
                _vals = dict(st.session_state.get("resumo_filter_custom") or {})
                for _k, _v in _vals.items():
                    st.session_state[_k] = _v
                if _vals:
                    st.session_state["resumo_filter_preset"] = "Custom"
                    _save_app_settings_to_disk()
                    st.success("Custom carregado.")
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
        with _p4:
            _new_profile_name = st.text_input("Perfil (nome)", value="", key="resumo_profile_name_new", placeholder="ex: valor_lpl")
            _s1, _s2 = st.columns(2)
            with _s1:
                if st.button("Salvar perfil", key="resumo_save_profile", width='stretch'):
                    _nm = str(_new_profile_name or "").strip()
                    if not _nm:
                        st.warning("Informe um nome de perfil.")
                    else:
                        _profiles[_nm] = {
                            "resumo_max_odd": float(st.session_state.get("resumo_max_odd", 2.0) or 2.0),
                            "resumo_req_liga": bool(st.session_state.get("resumo_req_liga", True)),
                            "resumo_req_h2h": bool(st.session_state.get("resumo_req_h2h", True)),
                            "resumo_req_model": bool(st.session_state.get("resumo_req_model", True)),
                            "resumo_min_sample_liga": int(st.session_state.get("resumo_min_sample_liga", 8) or 8),
                            "resumo_min_sample_h2h": int(st.session_state.get("resumo_min_sample_h2h", 3) or 3),
                        }
                        st.session_state["resumo_filter_profiles"] = _profiles
                        st.session_state["resumo_filter_preset"] = _nm
                        st.session_state["resumo_filter_profile_active"] = _nm
                        _save_app_settings_to_disk()
                        st.success(f"Perfil salvo: {_nm}")
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
            with _s2:
                if st.button("Excluir perfil", key="resumo_delete_profile", width='stretch'):
                    _nm = str(st.session_state.get("resumo_filter_preset", "") or "").strip()
                    if _nm and _nm in _profiles:
                        _profiles.pop(_nm, None)
                        st.session_state["resumo_filter_profiles"] = _profiles
                        st.session_state["resumo_filter_preset"] = "Custom"
                        st.session_state["resumo_filter_profile_active"] = ""
                        _save_app_settings_to_disk()
                        st.success(f"Perfil removido: {_nm}")
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
                    else:
                        st.info("Selecione um perfil salvo para excluir.")

        cW1, cW2, cW3 = st.columns([1,1,2])

        with cW1:
            req_liga = st.checkbox('Exigir Liga', value=bool(st.session_state.get('resumo_req_liga', True)), key='resumo_req_liga')

        with cW2:
            req_h2h = st.checkbox('Exigir H2H', value=bool(st.session_state.get('resumo_req_h2h', True)), key='resumo_req_h2h')

        with cW3:
            req_model = st.checkbox('Exigir Modelo', value=bool(st.session_state.get('resumo_req_model', True)), key='resumo_req_model')

        st.caption('Recortes no Histórico (Laplace): Ano = todos os jogos; Liga 5/10/15 = últimos N de cada time (união); H2H 5/10/15 = últimos N do confronto.')

        # Opcional: também exibir as odds do Histórico (Laplace) no Resumo.
        show_laplace_cols = st.checkbox(
            'Mostrar odds do Histórico (Laplace) no Resumo',
            value=bool(st.session_state.get('resumo_show_laplace_cols', False)),
            key='resumo_show_laplace_cols',
        )

    # StatsCamps (indicador): mostra se a linha faz sentido pelo % forte dos dois times
    # (usa o recorte selecionado na aba StatsCamps; se não estiver selecionado, faz fallback nos filtros atuais)
    with st.expander('StatsCamps (indicador)', expanded=False):
        st.slider('Limiar % forte', 50, 90, int(st.session_state.get('resumo_sc_thr', 60)), step=1, key='resumo_sc_thr')
        st.checkbox('Mostrar indicador StatsCamps', value=bool(st.session_state.get('resumo_sc_use', True)), key='resumo_sc_use')
        st.caption('✅ = % forte (>= limiar) para o lado (Acima/Abaixo), usando jogos dos dois times no recorte do StatsCamps.')

    sc_use = state_get_bool(st.session_state, 'resumo_sc_use', True)
    sc_thr = state_get_float(st.session_state, 'resumo_sc_thr', 60.0)
    # --- Odd Mix (Modelo + Laplace + StatsCamps) ---
    _LAPLACE_PICK_OPTS = [
        'Liga 15', 'Liga 10', 'Liga 5', 'Liga Ano',
        'H2H 15', 'H2H 10', 'H2H 5', 'H2H Ano',
        'Média Liga+H2H 15', 'Média Liga+H2H 10', 'Média Liga+H2H 5', 'Média Liga+H2H Ano',
    ]
    with st.expander('Odd Mix (Modelo + Laplace + StatsCamps)', expanded=False):
        _def_pick = str(st.session_state.get('resumo_laplace_pick', 'Liga 15') or 'Liga 15')
        _idx_pick = _LAPLACE_PICK_OPTS.index(_def_pick) if _def_pick in _LAPLACE_PICK_OPTS else 0
        st.selectbox('Fonte do Histórico (Laplace) usada em Odd (Hist) / Mix', options=_LAPLACE_PICK_OPTS, index=_idx_pick, key='resumo_laplace_pick')
        cM1, cM2, cM3 = st.columns(3)
        with cM1:
            st.slider('Peso Modelo', 0, 100, int(st.session_state.get('resumo_mix_w_model', 40)), step=5, key='resumo_mix_w_model')
        with cM2:
            st.slider('Peso Laplace', 0, 100, int(st.session_state.get('resumo_mix_w_laplace', 40)), step=5, key='resumo_mix_w_laplace')
        with cM3:
            st.slider('Peso StatsCamps', 0, 100, int(st.session_state.get('resumo_mix_w_sc', 20)), step=5, key='resumo_mix_w_sc')
        st.caption('O Mix é calculado em probabilidade (1/odd). Se alguma fonte estiver ausente, os pesos são renormalizados automaticamente.')

    laplace_pick = state_get_str(st.session_state, 'resumo_laplace_pick', 'Liga 15') or 'Liga 15'
    w_mix_model = state_get_float(st.session_state, 'resumo_mix_w_model', 40.0)
    w_mix_lap = state_get_float(st.session_state, 'resumo_mix_w_laplace', 40.0)
    w_mix_sc = state_get_float(st.session_state, 'resumo_mix_w_sc', 20.0)

    def _apply_combo_preset():
        preset = str(st.session_state.get('resumo_combo_preset', 'Neutro') or 'Neutro')
        preset_map = {
            'Conservador': (0.30, 0.10),
            'Neutro': (0.45, 0.18),
            'Agressivo': (0.65, 0.30),
        }
        hw, sw = preset_map.get(preset, (0.45, 0.18))
        st.session_state['resumo_combo_hist_w'] = float(hw)
        st.session_state['resumo_combo_style_w'] = float(sw)

    with st.expander('Ajuste ML + Totais (favorito + estilo)', expanded=False):
        if 'resumo_combo_preset' not in st.session_state:
            st.session_state['resumo_combo_preset'] = 'Neutro'
        if 'resumo_combo_use_mismatch' not in st.session_state:
            st.session_state['resumo_combo_use_mismatch'] = False
        st.selectbox(
            'Preset de ajuste',
            options=['Conservador', 'Neutro', 'Agressivo'],
            index=1,
            key='resumo_combo_preset',
            on_change=_apply_combo_preset,
        )
        st.checkbox(
            'Aplicar mismatch no Modelo (favoritismo vs total, experimental)',
            value=bool(st.session_state.get('resumo_combo_use_mismatch', False)),
            key='resumo_combo_use_mismatch',
            help='Desligado (padrão): o Modelo de ML+Totais nao recebe ajuste de mismatch. Laplace/Hist permanece puro.',
        )
        st.slider(
            'Peso do histÃ³rico condicional (win/loss) no ML+Totais',
            0.0,
            1.0,
            float(st.session_state.get('resumo_combo_hist_w', 0.45) or 0.45),
            step=0.05,
            key='resumo_combo_hist_w',
        )
        st.slider(
            'ForÃ§a do viÃ©s por estilo (over/under)',
            0.0,
            0.5,
            float(st.session_state.get('resumo_combo_style_w', 0.18) or 0.18),
            step=0.02,
            key='resumo_combo_style_w',
        )
        st.slider(
            'Peso da forma atual (forÃ§a do time no recorte) no ML+Totais',
            0.0,
            0.40,
            float(st.session_state.get('resumo_combo_form_w', 0.16) or 0.16),
            step=0.02,
            key='resumo_combo_form_w',
        )
        st.caption('Ajustes de mismatch afetam apenas o Modelo. Laplace/Historico nao recebe mismatch.')
    combo_hist_w = state_get_float(st.session_state, 'resumo_combo_hist_w', 0.45)
    combo_style_w = state_get_float(st.session_state, 'resumo_combo_style_w', 0.18)
    combo_form_w = state_get_float(st.session_state, 'resumo_combo_form_w', 0.16)
    combo_use_mismatch = state_get_bool(st.session_state, 'resumo_combo_use_mismatch', False)
    use_edge_gate = state_get_bool(st.session_state, 'resumo_use_edge_gate', False)
    min_edge_pp_base = state_get_float(st.session_state, 'resumo_min_edge_pp', 2.5)
    min_sample_liga = state_get_int(st.session_state, 'resumo_min_sample_liga', 8)
    min_sample_h2h = state_get_int(st.session_state, 'resumo_min_sample_h2h', 3)
    hist_shrink_n = state_get_int(st.session_state, 'resumo_hist_shrink_n', 20)


    # mapping local (evita depender da ordem das defs dentro do Resumo)
    _SC_METRIC_COL = {
        'kills': 'total_kills',
        'towers': 'total_towers',
        'dragons': 'total_dragons',
        'barons': 'total_nashors',
        'inhibitors': 'total_inhibitors',
        'time': 'game_time_min',
    }

    _df_sc_ab = None
    if sc_use:
        try:
            sc_league = str(st.session_state.get('sc_league', '') or '').strip()
            sc_year = str(st.session_state.get('sc_year', '') or '').strip()
            sc_split = str(st.session_state.get('sc_split', '') or '').strip()
            sc_playoffs = str(st.session_state.get('sc_playoffs', '') or '').strip()
            sc_map = str(st.session_state.get('sc_map', 'All maps') or 'All maps').strip()

            df_sc_base = None
            # se StatsCamps estiver completo, usa exatamente o recorte de lÃ¡
            if sc_league and sc_year and sc_split:
                sc_filters = Filters(
                    year=int(sc_year),
                    league=sc_league,
                    split=sc_split,
                    playoffs=None if not sc_playoffs else (True if sc_playoffs == 'Playoffs' else False),
                )
                df_sc_base = apply_filters(team_games, sc_filters).copy()
                # mapa
                if sc_map != 'All maps' and df_sc_base is not None and 'map_number' in df_sc_base.columns:
                    try:
                        mn = int(sc_map.split()[-1])
                        df_sc_base = df_sc_base[df_sc_base['map_number'] == mn].copy()
                    except Exception:
                        pass
            else:
                # fallback: tenta usar a liga do confronto (se houver), senÃ£o usa filtros atuais
                _la = _lb = ''
                try:
                    if _ml_info is not None:
                        _la = str(((_ml_info.get('blue') or {}).get('league')) or '').strip()
                        _lb = str(((_ml_info.get('red') or {}).get('league')) or '').strip()
                except Exception:
                    _la = _lb = ''
                league_fb = getattr(filters, 'league', None)
                if not league_fb and _la and _lb and _la == _lb:
                    league_fb = _la
                sc_filters = Filters(
                    year=getattr(filters, 'year', None),
                    league=league_fb,
                    split=getattr(filters, 'split', None),
                    playoffs=getattr(filters, 'playoffs', None),
                )
                df_sc_base = apply_filters(team_games, sc_filters).copy()
                # mapa: segue map_mode_ui (se nÃ£o for avg)
                if 'map_mode_ui' in globals() and map_mode_ui and str(map_mode_ui).startswith('map') and df_sc_base is not None and 'map_number' in df_sc_base.columns:
                    try:
                        mn = int(str(map_mode_ui).replace('map', '').strip())
                        df_sc_base = df_sc_base[df_sc_base['map_number'] == mn].copy()
                    except Exception:
                        pass

            if df_sc_base is not None and not df_sc_base.empty and 'team' in df_sc_base.columns:
                df_sc_ab = df_sc_base[df_sc_base['team'].astype(str).isin([str(teamA), str(teamB)])].copy()
                # dedup por jogo (evita contar o mesmo jogo duas vezes quando Ã© A vs B)
                if 'gameid' in df_sc_ab.columns:
                    df_sc_ab = df_sc_ab.drop_duplicates(subset=['gameid'], keep='last')
                elif 'gameId' in df_sc_ab.columns:
                    df_sc_ab = df_sc_ab.drop_duplicates(subset=['gameId'], keep='last')
                _df_sc_ab = df_sc_ab
        except Exception:
            _df_sc_ab = None

    def _sc_pct(metric_key: str, side_key: str, line_val: float):
        """Retorna o % (0-100) do indicador StatsCamps para o lado (Over/Under)."""
        if (not sc_use) or (_df_sc_ab is None) or (not isinstance(_df_sc_ab, pd.DataFrame)) or _df_sc_ab.empty:
            return None
        col_sc = _SC_METRIC_COL.get(str(metric_key).strip().lower())
        if not col_sc:
            return None
        x = pd.to_numeric(_df_sc_ab.get(col_sc), errors='coerce').dropna()
        n = int(x.shape[0])
        if n <= 0:
            return None
        try:
            lf = float(line_val)
        except Exception:
            return None
        side_low = str(side_key).strip().lower()
        if side_low == 'over':
            pct = 100.0 * float((x > lf).mean())
        else:
            pct = 100.0 * float((x < lf).mean())
        return float(pct) if math.isfinite(float(pct)) else None

    def _sc_badge(metric_key: str, side_key: str, line_val: float) -> str:
        pct = _sc_pct(metric_key, side_key, line_val)
        if pct is None:
            return '-'
        icon = 'âœ…' if float(pct) >= float(sc_thr) else 'â€”'
        return f"{icon} {pct:.0f}%"
    # Bases iguais Ã  aba Laplace
    dfA_all = dfA_used.copy().sort_values('date')
    dfB_all = dfB_used.copy().sort_values('date')
    df_h2h_A = dfA_all[dfA_all['opponent'] == teamB].copy()
    df_h2h_B = dfB_all[dfB_all['opponent'] == teamA].copy()

    _slices = core_recency_slices(df_a=dfA_all, df_b=dfB_all, df_h2h=df_h2h_A, id_col="gameid", date_col="date")
    df_league_year = _slices["league_year"]
    df_h2h_year = _slices["h2h_year"]
    df_league_5 = _slices["league_5"]
    df_league_10 = _slices["league_10"]
    df_league_15 = _slices["league_15"]
    df_h2h_5 = _slices["h2h_5"]
    df_h2h_10 = _slices["h2h_10"]
    df_h2h_15 = _slices["h2h_15"]


    # Diagnostico rapido: evitar confusao quando H2H/Liga nao tem amostra
    _n_league = int(getattr(df_league_year, 'shape', [0])[0] or 0)
    _n_h2h = int(getattr(df_h2h_year, 'shape', [0])[0] or 0)
    if req_h2h and _n_h2h <= 0:
        st.warning('H2H: nao ha jogos no historico para esse confronto. Com "Exigir H2H" marcado, nenhuma odd vai passar.')
    if req_liga and _n_league <= 0:
        st.warning('Liga: nao ha jogos no historico para a liga/recorte atual. Ajuste os filtros ou escolha outro confronto.')

    _pricing = core_make_pricing_adapters(
        laplace_pick=str(laplace_pick or 'Liga 15'),
        w_model=float(w_mix_model),
        w_hist=float(w_mix_lap),
        w_stats=float(w_mix_sc),
        floor_odd=1.10,
    )

    def _ensure_lines(metric_key: str, ls) -> List[float]:
        """Garante que 'ls' vire lista de floats (evita erros quando vÃªm como string do Agenda)."""
        if ls is None:
            return []
        # se veio como texto (ex.: "26.5, 27.5")
        if isinstance(ls, str):
            return _parse_time_lines(ls) if str(metric_key).strip().lower() == 'time' else _parse_lines(ls)
        # iterÃ¡vel
        if isinstance(ls, (list, tuple, set)):
            out = []
            for v in ls:
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        out.append(fv)
                except Exception:
                    pass
            return out
        # fallback
        try:
            fv = float(ls)
            return [fv] if math.isfinite(fv) else []
        except Exception:
            return []

    metric_col = {
        'kills': 'total_kills',
        'towers': 'total_towers',
        'dragons': 'total_dragons',
        'barons': 'total_nashors',
        'inhibitors': 'total_inhibitors',
        'time': 'game_time_min',
    }

    p_ml_row = float(st.session_state.get('_p_ml_scope', float('nan')))
    rows = []
    rows.extend(
        core_build_ml_rows(
            p_ml_row=float(p_ml_row),
            team_a=teamA,
            team_b=teamB,
            req_liga=bool(req_liga),
            req_h2h=bool(req_h2h),
            req_model=bool(req_model),
            min_sample_liga=int(min_sample_liga),
            min_sample_h2h=int(min_sample_h2h),
            max_odd=float(max_odd),
            df_league_year=df_league_year,
            df_league_15=df_league_15,
            df_league_10=df_league_10,
            df_league_5=df_league_5,
            df_h2h_year=df_h2h_year,
            df_h2h_15=df_h2h_15,
            df_h2h_10=df_h2h_10,
            df_h2h_5=df_h2h_5,
            odd_from_prob_fn=_odd_from_p,
            laplace_odds_fn=_pricing["laplace_odds_fn"],
            laplace_pick_odd_fn=_pricing["laplace_pick_odd_fn"],
            odd_mix_fn=_pricing["odd_mix_fn"],
            fmt_laplace_filtered_fn=_pricing["fmt_laplace_filtered_fn"],
            fmt_recortes_disp_fn=_fmt_recortes_disp,
        )
    )

    rows.extend(
        core_build_metric_rows(
            lines_by_metric=_lines_by_metric,
            ensure_lines_fn=_ensure_lines,
            metric_col_map=metric_col,
            totals_for_lines=totals_for_lines,
            total_over_prob_fn=total_over_prob,
            odd_from_prob_fn=_odd_from_p,
            req_liga=bool(req_liga),
            req_h2h=bool(req_h2h),
            req_model=bool(req_model),
            min_sample_liga=int(min_sample_liga),
            min_sample_h2h=int(min_sample_h2h),
            max_odd=float(max_odd),
            df_league_year=df_league_year,
            df_league_15=df_league_15,
            df_league_10=df_league_10,
            df_league_5=df_league_5,
            df_h2h_year=df_h2h_year,
            df_h2h_15=df_h2h_15,
            df_h2h_10=df_h2h_10,
            df_h2h_5=df_h2h_5,
            laplace_odds_fn=_pricing["laplace_odds_fn"],
            laplace_pick_odd_fn=_pricing["laplace_pick_odd_fn"],
            odd_mix_fn=_pricing["odd_mix_fn"],
            fmt_laplace_filtered_fn=_pricing["fmt_laplace_filtered_fn"],
            fmt_recortes_disp_fn=_fmt_recortes_disp,
            fmt_line_disp_fn=_fmt_line_disp,
            metric_label_fn=_metric_label_pt,
            side_label_fn=_side_label_pt,
            sc_badge_fn=_sc_badge,
            sc_pct_fn=_sc_pct,
        )
    )

    # --- ML + Totais (combos) no Resumo ---
    # Usa as linhas do campo especÃ­fico "Linha (para ML + Totais)" (kills/tempo)
    # e aplica o mesmo filtro de (Modelo + HistÃ³rico) em Liga/H2H.
    combo_lines_by_metric = {
        'kills': _parse_lines(st.session_state.get('combo_kills_text', '')),
        'time': _parse_time_lines(st.session_state.get('combo_time_text', '')),
    }
    p_ml_combo = st.session_state.get('_p_ml_scope', float('nan'))

    def _combo_bases(side_key: str):
        if side_key == 'A':
            base_l_year = dfA_all.sort_values('date')
            base_h_year = df_h2h_A.sort_values('date')
        else:
            base_l_year = dfB_all.sort_values('date')
            base_h_year = df_h2h_B.sort_values('date')
        return {
            'l_year': base_l_year,
            'l_5': core_tail_df(base_l_year, 5).sort_values('date'),
            'l_10': core_tail_df(base_l_year, 10).sort_values('date'),
            'l_15': core_tail_df(base_l_year, 15).sort_values('date'),
            'h_year': base_h_year,
            'h_5': core_tail_df(base_h_year, 5).sort_values('date'),
            'h_10': core_tail_df(base_h_year, 10).sort_values('date'),
            'h_15': core_tail_df(base_h_year, 15).sort_values('date'),
        }

    rows.extend(
        core_build_combo_rows(
            combo_lines_by_metric=combo_lines_by_metric,
            metric_col_map=metric_col,
            p_ml_combo=float(p_ml_combo) if p_ml_combo is not None else float('nan'),
            team_a=teamA,
            team_b=teamB,
            combo_hist_w=float(combo_hist_w),
            combo_style_w=float(combo_style_w),
            combo_form_w=float(combo_form_w),
            combo_use_mismatch=bool(combo_use_mismatch),
            req_liga=bool(req_liga),
            req_h2h=bool(req_h2h),
            req_model=bool(req_model),
            min_sample_liga=int(min_sample_liga),
            min_sample_h2h=int(min_sample_h2h),
            max_odd=float(max_odd),
            get_combo_bases_fn=_combo_bases,
            wr_laplace_fn=core_wr_laplace,
            counts_ml_totals_fn=core_counts_ml_totals,
            totals_for_lines=totals_for_lines,
            total_over_prob_fn=total_over_prob,
            odd_from_prob_fn=_odd_from_p,
            laplace_odds_fn=_pricing["laplace_odds_fn"],
            laplace_pick_odd_fn=_pricing["laplace_pick_odd_fn"],
            odd_mix_fn=lambda om, oh, sc, mb: _pricing["odd_mix_fn"](om, oh, sc, model_boost=mb),
            fmt_line_disp_fn=_fmt_line_disp,
            metric_label_fn=_metric_label_pt,
            fmt_recortes_disp_fn=_fmt_recortes_disp,
            fmt_laplace_filtered_fn=_pricing["fmt_laplace_filtered_fn"],
            sc_badge_fn=_sc_badge,
            sc_pct_fn=_sc_pct,
        )
    )

    # trace padrao (mercados do resumo) para auditoria/fallback por pick
    try:
        st.session_state["_ml_trace_markets"] = [
            {
                "mercado": str(r.get("Mercado", "")),
                "lado": str(r.get("Lado", "")),
                "linha": str(r.get("Linha", "")),
                "odd_model": (float(r.get("Odd (Modelo)")) if r.get("Odd (Modelo)") is not None else None),
                "odd_hist": (float(r.get("Odd (Hist)")) if r.get("Odd (Hist)") is not None else None),
                "odd_mix": (float(r.get("Odd (Mix)")) if r.get("Odd (Mix)") is not None else None),
                "calib_reason": str(r.get("_calib_reason", "")),
                "calib_source": str(r.get("_calib_source", "")),
            }
            for r in rows
        ]
    except Exception:
        st.session_state["_ml_trace_markets"] = []
    _persist_combined_trace_file()

    df_res = pd.DataFrame(rows)
    if df_res.empty:
        if req_h2h and _n_h2h < int(min_sample_h2h):
            st.info(f'Nada passou porque H2H ficou abaixo do minimo ({_n_h2h}/{int(min_sample_h2h)}).')
        elif req_model:
            st.info('Nada passou no filtro com as linhas atuais. Dica: ajuste o filtro de odd maxima ou desmarque "Exigir Modelo"/"Exigir H2H" para diagnosticar.')
        else:
            st.info('Nada passou no filtro com as linhas atuais. Dica: ajuste o filtro ou informe mais linhas no painel principal.')
    else:
        df_res = core_finalize_resumo_dataframe(
            df_res,
            format_num_fn=lambda v, d: _format_num(v, d),
        )
        if df_res.empty:
            st.info('Nenhuma pick no filtro atual.')
        else:
            st.dataframe(df_res, width='stretch', hide_index=True)
            with st.expander("Trace de calibração (Resumo)"):
                _trows = st.session_state.get("_ml_trace_markets", []) or []
                if _trows:
                    st.dataframe(pd.DataFrame(_trows), width='stretch', hide_index=True)
                    st.download_button(
                        "Baixar trace (Resumo) JSON",
                        data=json.dumps(_trows, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="ml_trace_markets.json",
                        mime="application/json",
                        key="dl_trace_resumo_json",
                    )
                else:
                    st.caption("Sem trace nesta execução.")


# -----------------------------
# Series (linhas da sÃ©rie + handicaps)
# -----------------------------
if "Series" in tab_refs:
 with tab_refs["Series"]:
    st.markdown(
        """
<div class="gb-hero" style="margin-top:4px;">
  <div class="title" style="font-size:1.1rem;">Series</div>
  <div class="sub">Linhas de série (totais), ML da série e handicaps de mapas.</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # header
    _la = _lb = ''
    if _ml_info is not None:
        _la = str(((_ml_info.get('blue') or {}).get('league')) or '').strip()
        _lb = str(((_ml_info.get('red') or {}).get('league')) or '').strip()
    def _team_with_league(name: str, lg: str) -> str:
        return f"{name} ({lg})" if lg else name
    teamA_disp = _team_with_league(teamA, _la)
    teamB_disp = _team_with_league(teamB, _lb)

    # ML base (fonte única): usa p_map final da sessão e recalcula série pelo BO atual
    # para evitar divergência quando o usuário troca BO após uma análise.
    pA_map = state_get_float(st.session_state, '_p_map_used', float(p_map_cal))
    try:
        pA_series = float(prob_win_series(float(pA_map), int(bo))) if math.isfinite(float(pA_map)) else float("nan")
    except Exception:
        pA_series = state_get_float(st.session_state, '_p_series_used', float(p_series))
    oddA_series = _odd_from_p(pA_series)
    oddB_series = _odd_from_p(1.0 - pA_series)

    # Defaults defensivos para evitar NameError em reruns parciais.
    p_year = str(st.session_state.get("pl_year", "") or "")
    p_league = str(st.session_state.get("pl_league", "") or "")
    p_split = str(st.session_state.get("pl_split", "") or "")
    p_playoffs = str(st.session_state.get("pl_playoffs", "") or "")

    with st.container(border=True):
        st.markdown(f'#### {teamA_disp} vs {teamB_disp} (bo{bo})')
        oddA_map = _odd_from_p(pA_map)
        oddB_map = _odd_from_p(1.0 - pA_map) if math.isfinite(float(pA_map)) else None
        st.markdown(
            f'**ML mapa:** {teamA_disp} = **{_format_num(oddA_map,2)}** ({_format_pct(pA_map)})'
            f' | {teamB_disp} = **{_format_num(oddB_map,2)}** ({_format_pct(1.0 - pA_map)})'
        )
        st.markdown(f'**ML série:** {teamA_disp} = **{_format_num(oddA_series,2)}** | {teamB_disp} = **{_format_num(oddB_series,2)}**')

    # Handicaps: melhor maneira possÃ­vel (scorelines exatas sob p_map)
    def _score_probs(p: float, bo_: int) -> Dict[str, float]:
        q = 1.0 - float(p)
        if int(bo_) == 3:
            return {
                'A_2-0': p*p,
                'A_2-1': 2*p*p*q,
                'B_2-1': 2*p*q*q,
                'B_2-0': q*q,
            }
        if int(bo_) == 5:
            return {
                'A_3-0': p**3,
                'A_3-1': 3*(p**3)*q,
                'A_3-2': 6*(p**3)*(q**2),
                'B_3-2': 6*(q**3)*(p**2),
                'B_3-1': 3*(q**3)*p,
                'B_3-0': q**3,
            }
        return {}

    probs = _score_probs(pA_map, int(bo))
    fav_is_a = bool(oddA_series <= oddB_series)
    fav_team = teamA_disp if fav_is_a else teamB_disp
    dog_team = teamB_disp if fav_is_a else teamA_disp

    st.markdown('#### Handicaps (mapas)')
    if int(bo) == 3:
        if fav_is_a:
            p_fav_m15 = probs.get('A_2-0')
        else:
            p_fav_m15 = probs.get('B_2-0')
        p_dog_p15 = None if p_fav_m15 is None else max(0.0, 1.0 - float(p_fav_m15))
        odd_fav_m15 = _odd_from_p(p_fav_m15) if p_fav_m15 is not None else None
        odd_dog_p15 = _odd_from_p(p_dog_p15) if p_dog_p15 is not None else None
        df_h = pd.DataFrame([
            {'Linha': f'{fav_team} -1.5', 'Prob': p_fav_m15, 'Odd justa': odd_fav_m15},
            {'Linha': f'{dog_team} +1.5', 'Prob': p_dog_p15, 'Odd justa': odd_dog_p15},
        ])
    elif int(bo) == 5:
        if fav_is_a:
            p_sweep = probs.get('A_3-0')
            p_win31 = probs.get('A_3-1')
        else:
            p_sweep = probs.get('B_3-0')
            p_win31 = probs.get('B_3-1')
        p_fav_m25 = p_sweep
        p_fav_m15 = None if (p_sweep is None or p_win31 is None) else float(p_sweep) + float(p_win31)
        p_dog_p25 = None if p_fav_m25 is None else max(0.0, 1.0 - float(p_fav_m25))
        p_dog_p15 = None if p_fav_m15 is None else max(0.0, 1.0 - float(p_fav_m15))

        df_h = pd.DataFrame([
            {'Linha': f'{fav_team} -1.5', 'Prob': p_fav_m15, 'Odd justa': _odd_from_p(p_fav_m15) if p_fav_m15 is not None else None},
            {'Linha': f'{fav_team} -2.5', 'Prob': p_fav_m25, 'Odd justa': _odd_from_p(p_fav_m25) if p_fav_m25 is not None else None},
            {'Linha': f'{dog_team} +1.5', 'Prob': p_dog_p15, 'Odd justa': _odd_from_p(p_dog_p15) if p_dog_p15 is not None else None},
            {'Linha': f'{dog_team} +2.5', 'Prob': p_dog_p25, 'Odd justa': _odd_from_p(p_dog_p25) if p_dog_p25 is not None else None},
        ])
    else:
        df_h = pd.DataFrame([])

    with st.container(border=True):
        if df_h.empty:
            st.info('Selecione BO3 ou BO5 para ver handicaps.')
        else:
            df_h['Prob'] = df_h['Prob'].apply(lambda x: _format_pct(x) if x is not None and math.isfinite(float(x)) else '-')
            df_h['Odd justa'] = df_h['Odd justa'].apply(lambda x: _format_num(x, 2) if x is not None and math.isfinite(float(x)) else '-')
            st.dataframe(df_h, width='stretch', hide_index=True)

    st.divider()
    st.markdown('#### Linhas secundárias da série (totais)')

    # entradas
    cL1, cL2, cL3 = st.columns(3)
    with st.container(border=True):
        with cL1:
            kills_s_txt = st.text_input('Kills (série) — ex: 112.5, 115.5', value=state_get_str(st.session_state, 'series_kills_txt', ''), key='series_kills_txt')
            towers_s_txt = st.text_input('Torres (série) — ex: 22.5', value=state_get_str(st.session_state, 'series_towers_txt', ''), key='series_towers_txt')
        with cL2:
            dragons_s_txt = st.text_input('Dragões (série) — ex: 9.5', value=state_get_str(st.session_state, 'series_dragons_txt', ''), key='series_dragons_txt')
            barons_s_txt = st.text_input('Barões (série) — ex: 3.5', value=state_get_str(st.session_state, 'series_barons_txt', ''), key='series_barons_txt')
        with cL3:
            team1_k_txt = st.text_input(f'Total Kills {teamA} (série) — ex: 20.5', value=state_get_str(st.session_state, 'series_team1_k_txt', ''), key='series_team1_k_txt')
            team2_k_txt = st.text_input(f'Total Kills {teamB} (série) — ex: 20.5', value=state_get_str(st.session_state, 'series_team2_k_txt', ''), key='series_team2_k_txt')

    kills_s = _parse_lines(kills_s_txt)
    towers_s = _parse_lines(towers_s_txt)
    dragons_s = _parse_lines(dragons_s_txt)
    barons_s = _parse_lines(barons_s_txt)
    team1_k = _parse_lines(team1_k_txt)
    team2_k = _parse_lines(team2_k_txt)

    # sÃ©ries: usa MC dos totais por mapa (AVG) e p_map_cal
    if not (math.isfinite(float(pA_map)) and int(bo) in [3,5]):
        st.warning('Sem p_map válido ou BO inválido.')
    else:
        def _ou_from_sims(sims: np.ndarray, line: float) -> Dict[str, float]:
            if sims is None or len(sims) == 0:
                return {'Over': float('nan'), 'Under': float('nan')}
            line_f = float(line)
            p_over = float(np.mean(sims > line_f))
            p_under = float(np.mean(sims < line_f))
            # ignora push (=) por padrão (como na Laplace)
            return {'Over': _odd_from_p(p_over), 'Under': _odd_from_p(p_under)}

        # total metrics
        def _render_series_lines(metric: str, lines_list: List[float], total_obj) -> Optional[pd.DataFrame]:
            if not lines_list:
                return None
            sims = _series_sims_cached(
                int(bo),
                float(pA_map),
                total_obj,
                int(n_sims),
                state_get_int(st.session_state, "_sim_seed_series", 1337),
            )
            rows = []
            for ln in lines_list:
                ou = _ou_from_sims(sims, float(ln))
                rows.append({
                    'Linha': float(ln),
                    'Over': _format_num(ou['Over'], 2),
                    'Under': _format_num(ou['Under'], 2),
                })
            return pd.DataFrame(rows)

        blocks = []
        if kills_s:
            dfk = _render_series_lines('kills', kills_s, totals_avg['kills'])
            if dfk is not None:
                st.markdown('**Kills (série)**')
                st.dataframe(dfk, width='stretch', hide_index=True)
        if towers_s:
            dft = _render_series_lines('towers', towers_s, totals_avg['towers'])
            if dft is not None:
                st.markdown('**Torres (série)**')
                st.dataframe(dft, width='stretch', hide_index=True)
        if dragons_s and 'dragons' in totals_avg:
            dfd = _render_series_lines('dragons', dragons_s, totals_avg['dragons'])
            if dfd is not None:
                st.markdown('**Dragões (série)**')
                st.dataframe(dfd, width='stretch', hide_index=True)
        if barons_s and 'barons' in totals_avg:
            dfb = _render_series_lines('barons', barons_s, totals_avg['barons'])
            if dfb is not None:
                st.markdown('**Barões (série)**')
                st.dataframe(dfb, width='stretch', hide_index=True)

        # Team kills: simula somando mapas jogados (independente do resultado do mapa)
        def _team_kill_total_obj(p_for: float, p_for_sd: float, opp_against: float, opp_against_sd: float):
            mean = _combine_mean(p_for, opp_against)
            sd = _combine_sd(p_for_sd, opp_against_sd)
            return MatchupTotals(mean=float(mean), sd=float(max(sd, 1.0)), dist='normal')

        @st.cache_data(show_spinner=False)
        def _series_team_sims_cached(bo_: int, p_map_: float, mean: float, sd: float, n_sims_: int, seed_: int) -> np.ndarray:
            rng = np.random.default_rng(int(seed_))
            wins_needed = 2 if int(bo_) == 3 else 3
            out = []
            for _ in range(int(n_sims_)):
                wa = 0
                wb = 0
                tot = 0.0
                while wa < wins_needed and wb < wins_needed:
                    # sample team metric
                    x = rng.normal(float(mean), float(sd))
                    if x < 0:
                        x = 0.0
                    tot += float(x)
                    # decide winner for series length only
                    if rng.random() < float(p_map_):
                        wa += 1
                    else:
                        wb += 1
                out.append(tot)
            return np.asarray(out, dtype=float)

        # build team kill objects from profiles (pA/pB)
        try:
            objA = _team_kill_total_obj(pA.kills_for, pA.kills_for_sd, pB.kills_against, pB.kills_against_sd)
            objB = _team_kill_total_obj(pB.kills_for, pB.kills_for_sd, pA.kills_against, pA.kills_against_sd)
            _seed0 = state_get_int(st.session_state, "_sim_seed_series", 1337)
            simsA = _series_team_sims_cached(int(bo), float(pA_map), float(objA.mean), float(objA.sd), int(n_sims), _seed0 + 11)
            simsB = _series_team_sims_cached(int(bo), float(1.0 - pA_map), float(objB.mean), float(objB.sd), int(n_sims), _seed0 + 29)

            def _render_team_k(lines_list: List[float], sims: np.ndarray, label: str):
                if not lines_list:
                    return
                rows = []
                for ln in lines_list:
                    ou = _ou_from_sims(sims, float(ln))
                    rows.append({'Linha': float(ln), 'Over': _format_num(ou['Over'],2), 'Under': _format_num(ou['Under'],2)})
                st.markdown(f'**{label} (sÃ©rie)**')
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

            _render_team_k(team1_k, simsA, f'Total Kills {teamA}')
            _render_team_k(team2_k, simsB, f'Total Kills {teamB}')
        except Exception:
            pass


# -----------------------------
# StatsCamps (estilo op.gg)
# -----------------------------
if "StatsCamps" in tab_refs:
 with tab_refs["StatsCamps"]:
    st.markdown('### StatsCamps')
    st.caption('Tabela tipo op.gg por campeonato/liga + comparaÃ§Ã£o entre times. (Tudo calculado do CSV.)')

    # Base (editÃ¡vel nessa aba)
    base_cols = st.columns(4)

    # opÃ§Ãµes (lidas do CSV)
    years = []
    splits = []
    leagues = []
    try:
        if team_games is not None and not team_games.empty:
            if 'year' in team_games.columns:
                years = sorted([int(x) for x in pd.to_numeric(team_games['year'], errors='coerce').dropna().unique().tolist()])
            if 'split' in team_games.columns:
                splits = sorted([str(x) for x in team_games['split'].fillna('').astype(str).unique().tolist() if str(x)])
            if 'league' in team_games.columns:
                leagues = sorted([str(x) for x in team_games['league'].fillna('').astype(str).unique().tolist() if str(x)])
    except Exception:
        pass

    # PadrÃ£o: tudo comeÃ§a "em branco" (sem prÃ©-seleÃ§Ã£o)
    for _k in ['sc_year', 'sc_league', 'sc_split', 'sc_playoffs']:
        if _k not in st.session_state or state_get_str(st.session_state, _k, '') in ['All', 'None']:
            st.session_state[_k] = ''

    def _fmt_sc_placeholder(x: str) -> str:
        return 'Selecione ...' if str(x) == '' else str(x)

    def _on_sc_league_change():
        # ao mudar liga, resetar dependentes
        st.session_state['sc_split'] = ''
        st.session_state['sc_compare_sel'] = []
        st.session_state['sc_delta_a'] = ''
        st.session_state['sc_delta_b'] = ''

    # Liga vem primeiro para filtrar Split/Time (comparaÃ§Ã£o)
    with base_cols[0]:
        sc_league = st.selectbox('Liga', options=[''] + leagues, index=0, key='sc_league', format_func=_fmt_sc_placeholder, on_change=_on_sc_league_change)

    # base para opÃ§Ãµes dependentes
    df_opt = team_games
    if sc_league and df_opt is not None and not df_opt.empty and 'league' in df_opt.columns:
        df_opt = df_opt[df_opt['league'].astype(str) == str(sc_league)].copy()

    with base_cols[1]:
        years_opt = []
        try:
            if df_opt is not None and not df_opt.empty and 'year' in df_opt.columns:
                years_opt = sorted([int(x) for x in pd.to_numeric(df_opt['year'], errors='coerce').dropna().unique().tolist()])
        except Exception:
            years_opt = []
        # garante consistÃªncia do estado
        if state_get_str(st.session_state, 'sc_year', '') not in [''] + [str(y) for y in years_opt]:
            st.session_state['sc_year'] = ''
        sc_year = st.selectbox('Ano', options=[''] + [str(y) for y in years_opt], index=0, key='sc_year', format_func=_fmt_sc_placeholder)

    df_opt2 = df_opt
    if sc_year and df_opt2 is not None and not df_opt2.empty and 'year' in df_opt2.columns:
        try:
            df_opt2 = df_opt2[pd.to_numeric(df_opt2['year'], errors='coerce') == int(sc_year)].copy()
        except Exception:
            pass

    with base_cols[2]:
        splits_opt = []
        try:
            if df_opt2 is not None and not df_opt2.empty and 'split' in df_opt2.columns:
                splits_opt = sorted([str(x) for x in df_opt2['split'].fillna('').astype(str).unique().tolist() if str(x)])
        except Exception:
            splits_opt = []
        if state_get_str(st.session_state, 'sc_split', '') not in [''] + splits_opt:
            st.session_state['sc_split'] = ''
        sc_split = st.selectbox('Split', options=[''] + splits_opt, index=0, key='sc_split', format_func=_fmt_sc_placeholder)

    with base_cols[3]:
        if state_get_str(st.session_state, 'sc_playoffs', '') not in ['', 'Regular', 'Playoffs']:
            st.session_state['sc_playoffs'] = ''
        sc_playoffs = st.selectbox('Playoffs', options=['', 'Regular', 'Playoffs'], index=0, key='sc_playoffs', format_func=_fmt_sc_placeholder)

    # Bloquear execuÃ§Ã£o atÃ© selecionar o necessÃ¡rio (sem st.stop, pra nÃ£o quebrar as abas seguintes)
    sc_ready = True
    if not sc_league:
        st.info('Selecione uma liga para exibir StatsCamps.')
        sc_ready = False
    if not sc_year:
        st.info('Selecione um ano para exibir StatsCamps.')
        sc_ready = False
    if not sc_split:
        st.info('Selecione um split para exibir StatsCamps.')
        sc_ready = False

    if sc_ready:
        map_opts = ['All maps', 'Map 1', 'Map 2', 'Map 3', 'Map 4', 'Map 5']
        cM1, cM2 = st.columns([1,2])
        with cM1:
            sc_map = st.selectbox('Mapa', options=map_opts, index=0, key='sc_map')
        with cM2:
            st.caption('As mÃ©dias/percentuais sÃ£o por MAPA (cada linha do CSV Ã© um mapa).')
    
        # thresholds editÃ¡veis
        ct1, ct2, ct3 = st.columns(3)
        with ct1:
            th_kills = st.text_input('KILLS thresholds (C. Kills) â€” ex: 23.5,25.5,27.5,29.5', value=state_get_str(st.session_state, 'sc_th_kills', '23.5,25.5,27.5,29.5'), key='sc_th_kills')
            th_towers = st.text_input('TOWERS thresholds (C. Towers) â€” ex: 10.5,11.5,12.5', value=state_get_str(st.session_state, 'sc_th_towers', '10.5,11.5,12.5'), key='sc_th_towers')
        with ct2:
            th_dragons = st.text_input('DRAGONS thresholds (C. Dragons) â€” ex: 4.5,5.5', value=state_get_str(st.session_state, 'sc_th_dragons', '4.5,5.5'), key='sc_th_dragons')
            th_barons = st.text_input('NASHORS thresholds (C. Nashors) â€” ex: 1.5', value=state_get_str(st.session_state, 'sc_th_barons', '1.5'), key='sc_th_barons')
        with ct3:
            th_inhib = st.text_input('INHIB thresholds (C. Inhib) â€” ex: 1.5', value=state_get_str(st.session_state, 'sc_th_inhib', '1.5'), key='sc_th_inhib')
    
        th_k = _parse_lines(th_kills)
        th_t = _parse_lines(th_towers)
        th_d = _parse_lines(th_dragons)
        th_b = _parse_lines(th_barons)
        th_i = _parse_lines(th_inhib)
        # aplica base
        sc_filters = Filters(
            year=None if not sc_year else int(sc_year),
            league=None if not sc_league else sc_league,
            split=None if not sc_split else sc_split,
            playoffs=None if not sc_playoffs else (True if sc_playoffs == 'Playoffs' else False),
        )
    
        df_sc = apply_filters(team_games, sc_filters).copy()
        # map filter
        if sc_map != 'All maps' and 'map_number' in df_sc.columns:
            try:
                mn = int(sc_map.split()[-1])
                df_sc = df_sc[df_sc['map_number'] == mn].copy()
            except Exception:
                pass
    
        if not sc_league:
            st.info('Selecione uma liga para exibir StatsCamps.')
        elif df_sc.empty:
            st.info('Sem jogos nesse recorte.')
        else:
            # agregaÃ§Ã£o
            def _pct(series_bool):
                try:
                    return 100.0 * float(series_bool.mean())
                except Exception:
                    return float('nan')
    
            agg_rows = []
            for team, g in df_sc.groupby('team'):
                g = g.copy()
                games_n = int(g.shape[0])
                wr = float(g['win'].mean()) if games_n else float('nan')
                row = {
                    'Team': team,
                    '# Games': games_n,
                    'Win Rate': 100.0*wr if math.isfinite(wr) else float('nan'),
                    'Game Time': float(g['game_time_min'].mean()) if 'game_time_min' in g.columns else float('nan'),
                    'FB%': _pct(g.get('firstblood', pd.Series([0]*games_n)).astype(float) > 0.5),
                    'FRH%': _pct(g.get('firstherald', pd.Series([0]*games_n)).astype(float) > 0.5),
                    'FT%': _pct(g.get('firsttower', pd.Series([0]*games_n)).astype(float) > 0.5),
                    'FD%': _pct(g.get('firstdragon', pd.Series([0]*games_n)).astype(float) > 0.5),
                    'FNASH%': _pct(g.get('firstbaron', pd.Series([0]*games_n)).astype(float) > 0.5),
                    'Gold': float(g['gold_for'].mean()) if 'gold_for' in g.columns else float('nan'),
                    'Kills': float(g['kills_for'].mean()) if 'kills_for' in g.columns else float('nan'),
                    'Deaths': float(g['deaths_for'].mean()) if 'deaths_for' in g.columns else float('nan'),
                    'Towers': float(g['towers_for'].mean()) if 'towers_for' in g.columns else float('nan'),
                    'C. Gold': float(g['total_gold'].mean()) if 'total_gold' in g.columns else float('nan'),
                    'C. Kills': float(g['total_kills'].mean()) if 'total_kills' in g.columns else float('nan'),
                    'C. Towers': float(g['total_towers'].mean()) if 'total_towers' in g.columns else float('nan'),
                    'C. Dragons': float(g['total_dragons'].mean()) if 'total_dragons' in g.columns else float('nan'),
                    'C. Nashors': float(g['total_nashors'].mean()) if 'total_nashors' in g.columns else float('nan'),
                    'C. Inhib': float(g['total_inhibitors'].mean()) if 'total_inhibitors' in g.columns else float('nan'),
                }
                # thresholds
                for ln in th_k:
                    row[f'% Kills > {ln}'] = _pct(pd.to_numeric(g['total_kills'], errors='coerce') > float(ln))
                for ln in th_t:
                    row[f'% Towers > {ln}'] = _pct(pd.to_numeric(g['total_towers'], errors='coerce') > float(ln))
                for ln in th_d:
                    row[f'% Dragons > {ln}'] = _pct(pd.to_numeric(g['total_dragons'], errors='coerce') > float(ln))
                for ln in th_b:
                    row[f'% Nashors > {ln}'] = _pct(pd.to_numeric(g['total_nashors'], errors='coerce') > float(ln))
                for ln in th_i:
                    row[f'% Inhib > {ln}'] = _pct(pd.to_numeric(g['total_inhibitors'], errors='coerce') > float(ln))
    
                agg_rows.append(row)
    
            df_tab = pd.DataFrame(agg_rows)
            df_tab_raw = df_tab.copy()
    
            # formata colunas
            if not df_tab.empty:
                df_tab = df_tab.sort_values(['Win Rate', '# Games'], ascending=[False, False])
                # mm:ss
                if 'Game Time' in df_tab.columns:
                    df_tab['Game Time'] = df_tab['Game Time'].apply(_min_to_mmss)
                for c in df_tab.columns:
                    if c in ['Team','Game Time']:
                        continue
                    if c.startswith('%') or c.endswith('%') or c in ['Win Rate','FB%','FRH%','FT%','FD%','FNASH%']:
                        df_tab[c] = df_tab[c].apply(lambda x: _format_num(x, 1) if x is not None and math.isfinite(float(x)) else '-')
                    elif c == '# Games':
                        continue
                    else:
                        df_tab[c] = df_tab[c].apply(lambda x: _format_num(x, 1) if x is not None and math.isfinite(float(x)) else '-')
    
            st.dataframe(df_tab, width='stretch', height=600, hide_index=True)
    
            st.markdown('#### Comparar times')
            all_teams_sc = df_tab_raw['Team'].tolist() if 'Team' in df_tab_raw.columns else []
            sel = st.multiselect('Selecione 2+ times', options=all_teams_sc, default=[], key='sc_compare_sel')
    
            if sel and len(sel) >= 2:
                tab_cmp, tab_delta = st.tabs(['Tabela', 'Î” (DiferenÃ§a)'])
    
                with tab_cmp:
                    df_cmp = df_tab[df_tab['Team'].isin(sel)].copy()
                    st.dataframe(df_cmp, width='stretch', hide_index=True)
    
                with tab_delta:
                    # Se o usuÃ¡rio marcou 2+ times, ele escolhe o par para calcular Î”
                    cA, cB = st.columns(2)
    
                    # defaults (tenta manter estÃ¡vel)
                    _default_a = sel[0] if sel else ''
                    _default_b = sel[1] if len(sel) > 1 else ''
    
                    if st.session_state.get('sc_delta_a','') not in sel:
                        st.session_state['sc_delta_a'] = _default_a
                    if st.session_state.get('sc_delta_b','') not in sel:
                        st.session_state['sc_delta_b'] = _default_b
    
                    with cA:
                        team_a = st.selectbox('Time A', options=sel, index=max(0, sel.index(st.session_state.get('sc_delta_a', _default_a))) if st.session_state.get('sc_delta_a', _default_a) in sel else 0, key='sc_delta_a')
    
                    # B nÃ£o pode ser igual ao A
                    opts_b = [t for t in sel if t != team_a]
                    if not opts_b:
                        opts_b = sel
    
                    with cB:
                        # tenta manter o valor anterior se possÃ­vel
                        _b = st.session_state.get('sc_delta_b', _default_b)
                        if _b == team_a:
                            _b = opts_b[0] if opts_b else ''
                        if _b not in opts_b and opts_b:
                            _b = opts_b[0]
                        team_b = st.selectbox('Time B', options=opts_b, index=max(0, opts_b.index(_b)) if _b in opts_b else 0, key='sc_delta_b')
    
                    if (not team_a) or (not team_b) or (team_a == team_b):
                        st.info('Selecione dois times diferentes para ver Î”.')
                    else:
                        # pega linhas brutas
                        try:
                            ra = df_tab_raw.loc[df_tab_raw['Team'] == team_a].iloc[0]
                            rb = df_tab_raw.loc[df_tab_raw['Team'] == team_b].iloc[0]
                        except Exception:
                            ra, rb = None, None
    
                        def _is_num(x) -> bool:
                            try:
                                return x is not None and math.isfinite(float(x))
                            except Exception:
                                return False
    
                        def _fmt_val(col: str, x):
                            if col == 'Team':
                                return str(x)
                            if col == 'Game Time':
                                return _min_to_mmss(x) if _is_num(x) else '-'
                            if col == '# Games':
                                try:
                                    return str(int(x)) if _is_num(x) else '-'
                                except Exception:
                                    return '-'
                            # percentuais
                            if col.startswith('%') or col.endswith('%') or col in ['Win Rate','FB%','FRH%','FT%','FD%','FNASH%']:
                                return _format_num(x, 1) if _is_num(x) else '-'
                            return _format_num(x, 1) if _is_num(x) else '-'
    
                        def _fmt_delta_num(col: str, a, b):
                            if not _is_num(a) or not _is_num(b):
                                return '-'
                            d = float(a) - float(b)
                            # inteiro para # Games
                            if col == '# Games':
                                d_int = int(round(d))
                                return f"{d_int:+d}"
                            # percentual em p.p.
                            if col.startswith('%') or col.endswith('%') or col in ['Win Rate','FB%','FRH%','FT%','FD%','FNASH%']:
                                return f"{d:+.1f}"
                            return f"{d:+.1f}"
    
                        def _fmt_delta_time(a, b):
                            if not _is_num(a) or not _is_num(b):
                                return '-'
                            # a/b estÃ£o em minutos decimais
                            sec = int(round((float(a) - float(b)) * 60.0))
                            sign = '+' if sec > 0 else '-' if sec < 0 else ''
                            sec = abs(sec)
                            mm = sec // 60
                            ss = sec % 60
                            return f"{sign}{mm:02d}:{ss:02d}"
    
                        if ra is None or rb is None:
                            st.warning('NÃ£o consegui montar Î” para esse par (dados faltando).')
                        else:
                            rows = []
                            for col in df_tab_raw.columns:
                                if col == 'Team':
                                    continue
                                a_val = ra.get(col)
                                b_val = rb.get(col)
                                if col == 'Game Time':
                                    d_val = _fmt_delta_time(a_val, b_val)
                                else:
                                    d_val = _fmt_delta_num(col, a_val, b_val)
                                rows.append({
                                    'MÃ©trica': col,
                                    team_a: _fmt_val(col, a_val),
                                    team_b: _fmt_val(col, b_val),
                                    'Î” (A - B)': d_val,
                                })
    
                            df_delta = pd.DataFrame(rows)
                            st.dataframe(df_delta, width='stretch', hide_index=True)
    
    
    # -----------------------------
    # CampeÃµes
    # -----------------------------
if "Campeões" in tab_refs:
 with tab_refs["Campeões"]:
    st.markdown(
        """
<div class="gb-hero" style="margin-top:4px;">
  <div class="title" style="font-size:1.1rem;">Campeões</div>
  <div class="sub">Draft, winrate, matchups por posição e contexto por lane (100% data-driven).</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Base (editÃ¡vel aqui)
    years = []
    splits = []
    leagues = []
    teams = []
    try:
        if team_games is not None and not team_games.empty:
            if 'year' in team_games.columns:
                years = sorted([int(x) for x in pd.to_numeric(team_games['year'], errors='coerce').dropna().unique().tolist()])
            if 'split' in team_games.columns:
                splits = sorted([str(x) for x in team_games['split'].fillna('').astype(str).unique().tolist() if str(x)])
            if 'league' in team_games.columns:
                leagues = sorted([str(x) for x in team_games['league'].fillna('').astype(str).unique().tolist() if str(x)])
            if 'team' in team_games.columns:
                teams = sorted([str(x) for x in team_games['team'].fillna('').astype(str).unique().tolist() if str(x)])
    except Exception:
        pass

    def _fmt_blank(x: str) -> str:
        # Mostra em branco quando o valor Ã© '' (padrÃ£o).
        return ' ' if str(x) == '' else str(x)

    # PadrÃ£o: tudo comeÃ§a em branco (sem prÃ©-seleÃ§Ã£o)
    for _k in ['ch_year', 'ch_league', 'ch_split', 'ch_playoffs', 'ch_team']:
        if _k not in st.session_state or state_get_str(st.session_state, _k, '') in ['All', 'None']:
            st.session_state[_k] = ''

    def _on_ch_league_change():
        # ao mudar liga, resetar dependentes
        st.session_state['ch_split'] = ''
        st.session_state['ch_team'] = ''

    with st.container(border=True):
        st.markdown("#### Filtros")
        bc1, bc2, bc3, bc4 = st.columns(4)

        with bc1:
            ch_year = st.selectbox('Ano', options=[''] + [str(y) for y in years], index=0, key='ch_year', format_func=_fmt_blank)

        # Liga vem antes (no fluxo do script) para filtrar Split/Time
        with bc4:
            ch_league = st.selectbox('Liga', options=[''] + leagues, index=0, key='ch_league', format_func=_fmt_blank, on_change=_on_ch_league_change)

    # Listas dependentes da liga
    _splits_f = splits
    _teams_f = teams
    try:
        if ch_league and team_games is not None and not team_games.empty:
            if 'league' in team_games.columns and 'split' in team_games.columns:
                _splits_f = sorted([
                    str(x) for x in team_games.loc[team_games['league'].astype(str) == str(ch_league), 'split']
                    .fillna('').astype(str).unique().tolist() if str(x)
                ])
            if 'league' in team_games.columns and 'team' in team_games.columns:
                _teams_f = sorted([
                    str(x) for x in team_games.loc[team_games['league'].astype(str) == str(ch_league), 'team']
                    .fillna('').astype(str).unique().tolist() if str(x)
                ])
    except Exception:
        _splits_f = splits
        _teams_f = teams

    with st.container(border=True):
        bc2, bc3 = st.columns(2)
        with bc2:
            ch_split = st.selectbox('Split', options=[''] + _splits_f, index=0, key='ch_split', format_func=_fmt_blank)
        with bc3:
            ch_playoffs = st.selectbox('Playoffs', options=['', 'Regular', 'Playoffs'], index=0, key='ch_playoffs', format_func=_fmt_blank)

        tc1, tc2 = st.columns([2, 1])
        with tc1:
            ch_team = st.selectbox('Time (opcional)', options=[''] + _teams_f, index=0, key='ch_team', format_func=_fmt_blank)
        with tc2:
            min_games_mu = st.number_input('Min jogos p/ matchup', min_value=1, max_value=30, value=state_get_int(st.session_state, 'ch_min_games_mu', 3), step=1, key='ch_min_games_mu')

    # Carrega players sob demanda
    with st.spinner('Carregando players do CSV (1x por sessÃ£o / cache)â€¦'):
        df_pl = _load_player_rows(csv_sig)

    if df_pl is None or df_pl.empty:
        st.warning('NÃ£o achei rows de players nesse CSV.')
        df_pl = pd.DataFrame(columns=['gameid','pos','side','champ','result','team','player'])

    base_filters = Filters(
        year=None if (not ch_year) else int(ch_year),
        league=None if (not ch_league) else ch_league,
        split=None if (not ch_split) else ch_split,
        playoffs=None if (not ch_playoffs) else (True if ch_playoffs == 'Playoffs' else False),
    )
    dfb = _apply_filters_players(df_pl, base_filters).copy()

    # filtro por time (opcional) â€” mantÃ©m jogos onde o time aparece (qualquer side)
    if ch_team and 'team' in dfb.columns:
        dfb = dfb[dfb['team'].astype(str) == str(ch_team)].copy()

    # VocabulÃ¡rio de champs para matching
    vocab = _champion_vocab(dfb)
    if not vocab:
        vocab = _champion_vocab(df_pl)

    st.markdown('#### Colar campeões')
    st.caption('Formato esperado: 10 campeões em ordem Top/JG/Mid/ADC/Sup do Time A e depois Top/JG/Mid/ADC/Sup do Time B. Pode ser com TAB, espaço ou quebra de linha.')

    ch_blue_team = st.radio(
        'Quem está no Blue Side neste draft',
        options=[str(teamA), str(teamB)],
        horizontal=True,
        key='ch_blue_team',
    )

    champs_txt = st.text_area('Cole aqui', value=state_get_str(st.session_state, 'ch_paste', ''), height=90, key='ch_paste')

    # parse tokens
    tokens = [t for t in re.split(r"[\t\n,; ]+", champs_txt.strip()) if t]
    if len(tokens) < 2:
        st.info('Cole pelo menos 2 campeões para começar.')
        tokens = []

    # resolve fuzzy match para cada token
    resolved = []
    unmatched = []
    for t in tokens:
        m, sugg = _fuzzy_match_one(t, vocab, cutoff=0.6)
        if m is None:
            unmatched.append((t, sugg))
            resolved.append(None)
        else:
            resolved.append(m)

    if unmatched:
        st.warning('Alguns campeões não foram reconhecidos. Ajuste a escrita ou use uma das sugestões:')
        for t, sugg in unmatched[:8]:
            st.write({'entrada': t, 'sugestoes': sugg[:5]})

    # mantÃ©m apenas os resolvidos
    champs = [c for c in resolved if c]

    # Prepara join com team_games para total_kills do jogo
    tg_cols = ['gameid','team','total_kills']
    df_tg = team_games[tg_cols].copy() if (team_games is not None and not team_games.empty and all(c in team_games.columns for c in tg_cols)) else None

    def _champ_stats(champ_name: str) -> dict:
        d = dfb[dfb['champ'].astype(str) == str(champ_name)].copy()
        if d.empty:
            return {'Champion': champ_name, 'Games': 0, 'Win Rate': '-', 'Avg game kills': '-'}

        # winrate
        wins = float(pd.to_numeric(d.get('result'), errors='coerce').fillna(0).sum())
        n = int(d.shape[0])
        wr = wins / max(n, 1)

        # avg total kills no jogo
        avg_k = None
        if df_tg is not None and 'team' in d.columns:
            m = d[['gameid','team']].dropna().drop_duplicates().merge(df_tg, on=['gameid','team'], how='left')
            if 'total_kills' in m.columns:
                avg_k = float(pd.to_numeric(m['total_kills'], errors='coerce').dropna().mean()) if m['total_kills'].notna().any() else None
        if avg_k is None and 'ckpm' in d.columns and 'gamelength' in d.columns:
            try:
                gl_min = pd.to_numeric(d['gamelength'], errors='coerce') / 60.0
                ckpm = pd.to_numeric(d['ckpm'], errors='coerce')
                tot = (gl_min * ckpm)
                avg_k = float(tot.dropna().mean()) if tot.notna().any() else None
            except Exception:
                avg_k = None

        return {
            'Champion': champ_name,
            'Games': n,
            'Win Rate': _format_num(100.0*wr, 1) if math.isfinite(wr) else '-',
            'Avg game kills': _format_num(avg_k, 1) if (avg_k is not None and math.isfinite(float(avg_k))) else '-',
        }

    # Tabela geral dos campeões colados
    st.markdown('#### Resumo por campeão (na base selecionada)')
    rows = [_champ_stats(c) for c in champs]
    df_sum = pd.DataFrame(rows)
    if df_sum.empty:
        st.info('Nada para mostrar (provavelmente todos ficaram sem match).')
    else:
        st.dataframe(df_sum.sort_values(['Games','Champion'], ascending=[False, True]), width='stretch', hide_index=True)

    st.divider()
    st.markdown('#### Matchups por posição')
    st.caption('Calcula (por posição) contra quais campeões o pick costuma ir melhor/pior. Usa só jogos do recorte.')

    # monta pares (Blue vs Red) por gameid+pos
    d2 = dfb[['gameid','pos','side','champ','result']].copy()
    d2 = d2[(d2['pos'] != '') & (d2['champ'] != '')].copy()
    d2['side'] = d2['side'].astype(str).str.title()

    blue = d2[d2['side'] == 'Blue'].copy()
    red = d2[d2['side'] == 'Red'].copy()
    pairs = blue.merge(red, on=['gameid','pos'], suffixes=('_blue','_red'))
    # win flags
    pairs['blue_win'] = pd.to_numeric(pairs['result_blue'], errors='coerce').fillna(0).astype(int)
    pairs['red_win'] = pd.to_numeric(pairs['result_red'], errors='coerce').fillna(0).astype(int)

    def _mu_table(champ_name: str, pos: str, max_rows: int = 3):
        # champ como blue
        a = pairs[(pairs['pos'] == pos) & (pairs['champ_blue'] == champ_name)].copy()
        a['opp'] = a['champ_red']
        a['win'] = a['blue_win']
        # champ como red
        b = pairs[(pairs['pos'] == pos) & (pairs['champ_red'] == champ_name)].copy()
        b['opp'] = b['champ_blue']
        b['win'] = b['red_win']
        u = pd.concat([a[['opp','win']], b[['opp','win']]], ignore_index=True)
        if u.empty:
            return None, None
        g = u.groupby('opp').agg(games=('win','size'), wins=('win','sum')).reset_index()
        g['wr'] = (g['wins'] / g['games']).astype(float)
        g = g[g['games'] >= int(min_games_mu)].copy()
        if g.empty:
            return None, None
        best = g.sort_values(['wr','games'], ascending=[False, False]).head(max_rows).copy()
        worst = g.sort_values(['wr','games'], ascending=[True, False]).head(max_rows).copy()
        for z in (best, worst):
            z['Win Rate'] = z['wr'].apply(lambda x: _format_num(100.0*float(x), 1))
            z = z.drop(columns=['wr'])
        best = best[['opp','games','Win Rate']].rename(columns={'opp':'Vs','games':'Games'})
        worst = worst[['opp','games','Win Rate']].rename(columns={'opp':'Vs','games':'Games'})
        return best, worst

    # se tiver 10 picks, monta matchup do draft (Top..Sup)
    roles = ['top','jng','mid','bot','sup']
    # oracle costuma usar 'top','jng','mid','bot','sup' (Ã s vezes 'jungle','adc','support')
    # normaliza pela base
    pos_norm = {
        'top': 'top',
        'jng': 'jng', 'jungle': 'jng',
        'mid': 'mid',
        'bot': 'bot', 'adc': 'bot',
        'sup': 'sup', 'support': 'sup',
    }

    # tenta mapear os 10 campeÃµes nas 10 posiÃ§Ãµes
    picks = [c for c in resolved if c]
    if len(picks) >= 10:
        a5 = picks[:5]
        b5 = picks[5:10]
        if str(ch_blue_team) == str(teamB):
            a5, b5 = b5, a5

        st.markdown('##### Draft (head-to-head por lane)')
        drows = []
        for i, pos in enumerate(roles):
            ca = a5[i] if i < len(a5) else None
            cb = b5[i] if i < len(b5) else None
            if not ca or not cb:
                continue
            # calcula H2H direto ca vs cb na posiÃ§Ã£o
            # ca como blue vs cb como red
            x1 = pairs[(pairs['pos'] == pos) & (pairs['champ_blue'] == ca) & (pairs['champ_red'] == cb)].copy()
            w1 = int(x1['blue_win'].sum())
            n1 = int(x1.shape[0])
            # ca como red vs cb como blue
            x2 = pairs[(pairs['pos'] == pos) & (pairs['champ_red'] == ca) & (pairs['champ_blue'] == cb)].copy()
            w2 = int(x2['red_win'].sum())
            n2 = int(x2.shape[0])
            n = n1 + n2
            w = w1 + w2
            wr = (w / n) if n > 0 else None
            drows.append({
                'Pos': pos,
                'Time A': ca,
                'Time B': cb,
                'H2H Games': n,
                'Win Rate (A)': _format_num(100.0*wr, 1) if (wr is not None and math.isfinite(float(wr))) else '-',
            })
        if drows:
            st.dataframe(pd.DataFrame(drows), width='stretch', hide_index=True)

        st.markdown('##### Vantagem do draft (Liga + Player)')
        st.caption('Score por lane combina winrate do campeÃ£o no recorte da liga com winrate do player naquele campeÃ£o.')

        ch_pc1, ch_pc2 = st.columns(2)
        with ch_pc1:
            ch_pl_txtA = st.text_area(
                f'Players {teamA} (Top/JG/Mid/ADC/Sup)',
                value=state_get_str(st.session_state, 'pl_txtA', ''),
                height=110,
                key='ch_pl_txtA',
            )
        with ch_pc2:
            ch_pl_txtB = st.text_area(
                f'Players {teamB} (Top/JG/Mid/ADC/Sup)',
                value=state_get_str(st.session_state, 'pl_txtB', ''),
                height=110,
                key='ch_pl_txtB',
            )

        def _parse_names(txt: str) -> list[str]:
            return [t.strip() for t in re.split(r"[\n\t,;]+", str(txt or '').strip()) if t.strip()]

        def _wr_parts(champ_name: str, player_name: str | None) -> dict:
            d_ch = dfb[dfb['champ'].astype(str) == str(champ_name)].copy()
            n_ch = int(d_ch.shape[0])
            w_ch = float(pd.to_numeric(d_ch.get('result'), errors='coerce').fillna(0).sum()) if n_ch > 0 else 0.0
            p_ch = (w_ch + 2.0) / (n_ch + 4.0) if n_ch > 0 else 0.5

            n_pl = 0
            w_pl = 0.0
            p_pl = p_ch
            if player_name:
                d_pl = d_ch[d_ch['player'].astype(str) == str(player_name)].copy()
                n_pl = int(d_pl.shape[0])
                if n_pl > 0:
                    w_pl = float(pd.to_numeric(d_pl.get('result'), errors='coerce').fillna(0).sum())
                    p_pl = (w_pl + 1.0) / (n_pl + 2.0)

            # peso do player cresce com amostra, sem dominar cedo
            w_player = float(n_pl) / float(n_pl + 8.0) if n_pl > 0 else 0.0
            p_blend = (1.0 - w_player) * float(p_ch) + w_player * float(p_pl)
            return {
                'p_league': float(p_ch),
                'p_player': float(p_pl),
                'p_blend': float(p_blend),
                'n_league': int(n_ch),
                'n_player': int(n_pl),
            }

        namesA_raw = _parse_names(ch_pl_txtA)
        namesB_raw = _parse_names(ch_pl_txtB)
        player_vocab = sorted([x for x in dfb.get('player', pd.Series(dtype=str)).fillna('').astype(str).unique().tolist() if x])
        namesA = []
        namesB = []
        for nm in namesA_raw[:5]:
            m, _ = _fuzzy_match_one(nm, player_vocab, cutoff=0.6)
            namesA.append(m if m else nm)
        for nm in namesB_raw[:5]:
            m, _ = _fuzzy_match_one(nm, player_vocab, cutoff=0.6)
            namesB.append(m if m else nm)

        def _h2h_lane_wr(ca: str, cb: str, pos: str) -> tuple[Optional[float], int]:
            # ca como blue vs cb como red
            x1 = pairs[(pairs['pos'] == pos) & (pairs['champ_blue'] == ca) & (pairs['champ_red'] == cb)].copy()
            w1 = int(x1['blue_win'].sum())
            n1 = int(x1.shape[0])
            # ca como red vs cb como blue
            x2 = pairs[(pairs['pos'] == pos) & (pairs['champ_red'] == ca) & (pairs['champ_blue'] == cb)].copy()
            w2 = int(x2['red_win'].sum())
            n2 = int(x2.shape[0])
            n = int(n1 + n2)
            if n <= 0:
                return None, 0
            return float((w1 + w2) / float(n)), n

        # Tabela detalhada solicitada: WR do player no campeÃ£o + WR geral da liga + H2H lane.
        st.markdown('##### Liga x Player x H2H (por lane)')
        namesA_pad = list(namesA[:5]) + [''] * max(0, 5 - len(namesA))
        namesB_pad = list(namesB[:5]) + [''] * max(0, 5 - len(namesB))
        det_rows = []
        for i, pos in enumerate(roles):
            ca = a5[i] if i < len(a5) else ''
            cb = b5[i] if i < len(b5) else ''
            pa = namesA_pad[i] if i < len(namesA_pad) else ''
            pb = namesB_pad[i] if i < len(namesB_pad) else ''
            if not ca or not cb:
                continue

            wa = _wr_parts(ca, pa if pa else None)
            wb = _wr_parts(cb, pb if pb else None)
            h2h_a, h2h_n = _h2h_lane_wr(ca, cb, pos)
            h2h_b = (1.0 - float(h2h_a)) if h2h_a is not None else None

            det_rows.append({
                'Pos': pos,
                'Team': teamA,
                'Player': pa if pa else '-',
                'Champ': ca,
                'WR Liga Champ': _format_num(100.0 * wa['p_league'], 1),
                'N Liga': int(wa['n_league']),
                'WR Player Champ': _format_num(100.0 * wa['p_player'], 1),
                'N Player': int(wa['n_player']),
                'WR H2H Lane': (_format_num(100.0 * float(h2h_a), 1) if h2h_a is not None else '-'),
                'N H2H': int(h2h_n),
            })
            det_rows.append({
                'Pos': pos,
                'Team': teamB,
                'Player': pb if pb else '-',
                'Champ': cb,
                'WR Liga Champ': _format_num(100.0 * wb['p_league'], 1),
                'N Liga': int(wb['n_league']),
                'WR Player Champ': _format_num(100.0 * wb['p_player'], 1),
                'N Player': int(wb['n_player']),
                'WR H2H Lane': (_format_num(100.0 * float(h2h_b), 1) if h2h_b is not None else '-'),
                'N H2H': int(h2h_n),
            })
        if det_rows:
            st.dataframe(pd.DataFrame(det_rows), width='stretch', hide_index=True)

        if len(namesA) >= 5 and len(namesB) >= 5:
            adv_rows = []
            for i, pos in enumerate(roles):
                ca = a5[i] if i < len(a5) else None
                cb = b5[i] if i < len(b5) else None
                pa = namesA[i] if i < len(namesA) else ''
                pb = namesB[i] if i < len(namesB) else ''
                if not ca or not cb:
                    continue
                wa = _wr_parts(ca, pa)
                wb = _wr_parts(cb, pb)
                edge = float(wa['p_blend']) - float(wb['p_blend'])
                adv_rows.append(
                    {
                        'Pos': pos,
                        f'{teamA} player': pa,
                        f'{teamA} champ': ca,
                        f'WR blend {teamA}': _format_num(100.0 * wa['p_blend'], 1),
                        f'{teamB} player': pb,
                        f'{teamB} champ': cb,
                        f'WR blend {teamB}': _format_num(100.0 * wb['p_blend'], 1),
                        'Edge A-B (p.p.)': _format_num(100.0 * edge, 1),
                    }
                )
            if adv_rows:
                df_adv = pd.DataFrame(adv_rows)
                st.dataframe(df_adv, width='stretch', hide_index=True)
                try:
                    edge_total = float(pd.to_numeric(df_adv['Edge A-B (p.p.)'], errors='coerce').dropna().sum())
                except Exception:
                    edge_total = float('nan')
                if math.isfinite(edge_total):
                    fav = teamA if edge_total >= 0 else teamB
                    st.caption(f'Edge total do draft: {fav} ({_format_num(abs(edge_total), 1)} p.p.)')
        else:
            st.info('Para calcular a vantagem Liga+Player, informe 5 players de cada time (Top/JG/Mid/ADC/Sup).')

    st.markdown('##### Melhores / Piores matchups (por campeÃ£o e posiÃ§Ã£o)')
    csel1, csel2 = st.columns([2, 1])
    _champ_opts = sorted(set(champs))
    if not _champ_opts:
        _champ_opts = ['']
    with csel1:
        champ_sel = st.selectbox('CampeÃ£o', options=_champ_opts, index=0, key='ch_mu_champ')
    with csel2:
        pos_sel = st.selectbox('PosiÃ§Ã£o', options=['top','jng','mid','bot','sup'], index=0, key='ch_mu_pos')

    best, worst = _mu_table(champ_sel, pos_sel) if str(champ_sel).strip() else (None, None)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('**Costuma ganhar contra**')
        if best is None:
            st.info('Poucos jogos para esse recorte (aumente a base ou reduza o min jogos).')
        else:
            st.dataframe(best, width='stretch', hide_index=True)
    with cc2:
        st.markdown('**Costuma perder contra**')
        if worst is None:
            st.info('Poucos jogos para esse recorte (aumente a base ou reduza o min jogos).')
        else:
            st.dataframe(worst, width='stretch', hide_index=True)


# -----------------------------
# Players
# -----------------------------
if "Players" in tab_refs:
 with tab_refs["Players"]:
    st.markdown(
        """
<div class="gb-hero" style="margin-top:4px;">
  <div class="title" style="font-size:1.1rem;">Players</div>
  <div class="sub">ML por lineup com fusão Team + Players e mercados individuais por linha.</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Base
    years = []
    splits = []
    leagues = []
    try:
        if team_games is not None and not team_games.empty:
            if 'year' in team_games.columns:
                years = sorted([int(x) for x in pd.to_numeric(team_games['year'], errors='coerce').dropna().unique().tolist()])
            if 'split' in team_games.columns:
                splits = sorted([str(x) for x in team_games['split'].fillna('').astype(str).unique().tolist() if str(x)])
            if 'league' in team_games.columns:
                leagues = sorted([str(x) for x in team_games['league'].fillna('').astype(str).unique().tolist() if str(x)])
    except Exception:
        pass

    def _fmt_blank_pl(x: str) -> str:
        return ' ' if str(x) == '' else str(x)

    for _k in ['pl_year', 'pl_league', 'pl_split', 'pl_playoffs']:
        if _k not in st.session_state or st.session_state.get(_k) in ['All', None]:
            st.session_state[_k] = ''

    def _on_pl_league_change():
        st.session_state['pl_split'] = ''

    with st.container(border=True):
        st.markdown("#### Filtros")
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            p_year = st.selectbox('Ano', options=[''] + [str(y) for y in years], index=0, key='pl_year', format_func=_fmt_blank_pl)

        with pc4:
            p_league = st.selectbox('Liga', options=[''] + leagues, index=0, key='pl_league', format_func=_fmt_blank_pl, on_change=_on_pl_league_change)

    # Split dependente da liga
    _splits_pl = splits
    try:
        if p_league and team_games is not None and not team_games.empty and 'league' in team_games.columns and 'split' in team_games.columns:
            _splits_pl = sorted([
                str(x) for x in team_games.loc[team_games['league'].astype(str) == str(p_league), 'split']
                .fillna('').astype(str).unique().tolist() if str(x)
            ])
    except Exception:
        _splits_pl = splits

    with st.container(border=True):
        pc2, pc3 = st.columns(2)
        with pc2:
            p_split = st.selectbox('Split', options=[''] + _splits_pl, index=0, key='pl_split', format_func=_fmt_blank_pl)

        with pc3:
            p_playoffs = st.selectbox('Playoffs', options=['', 'Regular', 'Playoffs'], index=0, key='pl_playoffs', format_func=_fmt_blank_pl)

    base_filters = Filters(
        year=None if (not p_year) else int(p_year),
        league=None if (not p_league) else p_league,
        split=None if (not p_split) else p_split,
        playoffs=None if (not p_playoffs) else (True if p_playoffs == 'Playoffs' else False),
    )

    with st.spinner('Carregando players do CSV (1x por sessão / cache)...'):
        df_pl = _load_player_rows(csv_sig)

    if df_pl is None or df_pl.empty:
        st.warning('Não achei rows de players nesse CSV.')
        dfb = pd.DataFrame()
    else:
        dfb = _apply_filters_players(df_pl, base_filters).copy()
    if dfb.empty:
        st.info('Sem jogos nesse recorte.')

    else:
        # tabela agregada por player
        g = dfb.copy()
        g['win'] = (pd.to_numeric(g.get('result'), errors='coerce').fillna(0).astype(int) > 0).astype(int)
        byp = g.groupby('player').agg(games=('win','size'), wins=('win','sum')).reset_index()
        byp['p_smooth'] = (byp['wins'] + 2.0) / (byp['games'] + 4.0)  # Beta(2,2)
        # rating = logit(p)
        byp['rating'] = np.log(byp['p_smooth'] / (1.0 - byp['p_smooth']))
    
        vocab = sorted([x for x in byp['player'].fillna('').astype(str).unique().tolist() if x])
        vocab_A = sorted([x for x in dfb[dfb['team'].astype(str) == str(teamA)]['player'].fillna('').astype(str).unique().tolist() if x])
        vocab_B = sorted([x for x in dfb[dfb['team'].astype(str) == str(teamB)]['player'].fillna('').astype(str).unique().tolist() if x])
        if not vocab_A:
            vocab_A = vocab
        if not vocab_B:
            vocab_B = vocab
    
        st.markdown('#### Cole os players')
        st.caption('Um por linha. O matching é fuzzy (se errar um pouco o nome, ele sugere).')
    
        def _norm_pos_local_players(x: str) -> str:
            s = str(x or '').strip().lower()
            if s in ('top',):
                return 'top'
            if s in ('jng', 'jungle'):
                return 'jng'
            if s in ('mid',):
                return 'mid'
            if s in ('bot', 'adc'):
                return 'bot'
            if s in ('sup', 'support'):
                return 'sup'
            return ''
    
        def _latest_lineup_names(df_in: pd.DataFrame, team_name: str) -> list[str]:
            if df_in is None or df_in.empty or 'team' not in df_in.columns:
                return []
            d = df_in[df_in['team'].astype(str) == str(team_name)].copy()
            if d.empty:
                return []
            if 'date' in d.columns:
                d = d.sort_values('date')
            d['pos_norm'] = d.get('pos', '').apply(_norm_pos_local_players)
            out = []
            for pos in ['top', 'jng', 'mid', 'bot', 'sup']:
                sub = d[d['pos_norm'] == pos].copy()
                if sub.empty:
                    continue
                nm = str(sub.iloc[-1].get('player', '') or '').strip()
                if nm:
                    out.append(nm)
            return out
    
        a0, a1 = st.columns([1.4, 2.6])
        with a0:
            if st.button('Auto-preencher players dos times', key='pl_autofill_players'):
                autoA = _latest_lineup_names(dfb, teamA)
                autoB = _latest_lineup_names(dfb, teamB)
                if autoA:
                    _txtA_auto = "\n".join(autoA)
                    st.session_state['pl_txtA'] = _txtA_auto
                    st.session_state['pl_txtA_input'] = _txtA_auto
                    st.session_state['pl_txtA_applied'] = _txtA_auto
                if autoB:
                    _txtB_auto = "\n".join(autoB)
                    st.session_state['pl_txtB'] = _txtB_auto
                    st.session_state['pl_txtB_input'] = _txtB_auto
                    st.session_state['pl_txtB_applied'] = _txtB_auto
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        with a1:
            st.caption('Preenche Top/JG/Mid/ADC/Sup com o lineup mais recente encontrado no recorte atual.')
    
        core_ensure_players_text_state(st.session_state)
        pA = core_parse_players_text(st.session_state.get('pl_txtA_applied', ''))
        pB = core_parse_players_text(st.session_state.get('pl_txtB_applied', ''))
        rA, notesA = core_resolve_players(
            names=pA,
            vocab=vocab_A,
            fuzzy_match_fn=lambda nm, vv, c: _fuzzy_match_one(nm, vv, cutoff=c),
            cutoff=0.6,
        )
        rB, notesB = core_resolve_players(
            names=pB,
            vocab=vocab_B,
            fuzzy_match_fn=lambda nm, vv, c: _fuzzy_match_one(nm, vv, cutoff=c),
            cutoff=0.6,
        )
        rA_view = rA if rA else pA
        rB_view = rB if rB else pB
    
        if notesA or notesB:
            with st.expander('Ver matching / sugestões', expanded=False):
                if notesA:
                    st.markdown('**Time A**')
                    st.write(notesA)
                if notesB:
                    st.markdown('**Time B**')
                    st.write(notesB)
    
        def _team_rating(pl_list: list[str]) -> tuple[float, pd.DataFrame]:
            if not pl_list:
                return 0.0, pd.DataFrame([])
            sub = byp[byp['player'].isin(pl_list)].copy()
            if sub.empty:
                return 0.0, pd.DataFrame([])
            # peso: min(games, 20) para não deixar um cara com 300 jogos dominar tudo
            sub['w'] = sub['games'].clip(upper=20).astype(float)
            r = float((sub['rating'] * sub['w']).sum() / max(sub['w'].sum(), 1.0))
            view = sub[['player','games','wins','p_smooth','rating']].copy()
            view['Win Rate'] = (100.0 * (view['wins'] / view['games'])).apply(lambda x: _format_num(x,1))
            view['p_smooth'] = (100.0 * view['p_smooth']).apply(lambda x: _format_num(x,1))
            view['rating'] = view['rating'].apply(lambda x: _format_num(x,3))
            view = view.rename(columns={'player':'Player','games':'Games','wins':'Wins','p_smooth':'Smooth Win%','rating':'Rating'})
            return r, view
    
        trA, dfA = _team_rating(rA)
        trB, dfB = _team_rating(rB)
    
        # prob via diferenÃ§a de rating
        def _sigmoid(x: float) -> float:
            try:
                return 1.0 / (1.0 + math.exp(-float(x)))
            except Exception:
                return 0.5
    
        p_players_raw = _sigmoid(trA - trB)
        p_team_ref = state_get_float(st.session_state, '_p_map_used', float('nan'))
        if not math.isfinite(p_team_ref):
            p_team_ref = 0.5
        gamesA = int(pd.to_numeric(dfA.get('Games'), errors='coerce').fillna(0).sum()) if not dfA.empty else 0
        gamesB = int(pd.to_numeric(dfB.get('Games'), errors='coerce').fillna(0).sum()) if not dfB.empty else 0
        n_players_eff = float(min(gamesA, gamesB))
        n_team_eff = state_get_float(st.session_state, '_ml_team_games_min', float('nan'))
        fus_players = _fuse_probs_by_precision(
            p_team=p_team_ref,
            p_players=p_players_raw,
            n_team=n_team_eff,
            n_players=n_players_eff,
            team_scale=state_get_float(st.session_state, "_fit_team_scale_used", state_get_float(st.session_state, "_fit_team_scale", 1.0)),
            players_scale=state_get_float(st.session_state, "_fit_players_scale_used", state_get_float(st.session_state, "_fit_players_scale", 1.0)),
            coverage=float(max(0.0, min(1.0, n_players_eff / max(1.0, n_team_eff + n_players_eff)))),
            transfer_signal=0.0,
            season_phase=("playoffs" if str(playoffs_opt).lower() in {"playoffs", "true", "1"} else "auto"),
        )
        p_teamA = float(fus_players.get('p_fused', p_team_ref))
        w_team = float(fus_players.get('w_team', float('nan')))
        w_players = float(fus_players.get('w_players', float('nan')))
        p_teamA = float(max(0.02, min(0.98, p_teamA)))
        oddA = _odd_from_p(p_teamA)
        oddB = _odd_from_p(1.0 - p_teamA)
    
        st.markdown('#### Resultado (ML por players)')
        st.markdown(f"**Time A:** {_format_num(oddA,2)}  |  **Time B:** {_format_num(oddB,2)}")
        st.caption(
            f"Conexão com ML de times: p_players={_format_pct(p_players_raw)} | "
            f"p_time={_format_pct(p_team_ref)} | blend={_format_pct(p_teamA)} "
            f"(w_players={_format_num(100.0*w_players,0)}% | n_team={_format_num(n_team_eff,1)} | n_players={_format_num(n_players_eff,1)})"
        )
    
        cta, ctb = st.columns(2)
        with cta:
            st.markdown('**Detalhe Time A**')
            if dfA.empty:
                st.info('Nenhum player reconhecido na base.')
            else:
                st.dataframe(dfA, width='stretch', hide_index=True)
        with ctb:
            st.markdown('**Detalhe Time B**')
            if dfB.empty:
                st.info('Nenhum player reconhecido na base.')
            else:
                st.dataframe(dfB, width='stretch', hide_index=True)
    
        st.divider()
        st.markdown('#### Mercados de Players (linhas individuais)')
        st.caption('Defina linhas por lane/jogador. Atualiza apenas ao clicar em "Analisar players".')
    
        lane_rows = core_build_lane_rows(rA_view, rB_view, ['top', 'jng', 'mid', 'bot', 'sup'])
        core_ensure_player_line_input_state(st.session_state, lanes=len(lane_rows))
    
        with st.form('players_input_form', clear_on_submit=False):
            colA, colB = st.columns(2)
            with colA:
                st.text_area('Time A â€” players (5)', height=120, key='pl_txtA_input')
            with colB:
                st.text_area('Time B â€” players (5)', height=120, key='pl_txtB_input')
    
            submitted_players_top = st.form_submit_button('Analisar players', type='primary', key='pl_submit_top')
            st.caption('As lanes e odds abaixo atualizam quando clicar em "Analisar players".')
    
            st.markdown('##### Linhas individuais (10 jogadores)')
            st.caption('Para cada jogador, informe linhas separadas por vírgula (ex.: 2.5,3.5). O cálculo é individual: passou ou não passou da linha.')
            for i, rr in enumerate(lane_rows):
                c0, c1, c2 = st.columns([1.1, 1.45, 1.45])
                with c0:
                    st.markdown(f"**{rr['Lane']}**  \nTime A: {rr['A'] or '-'}  \nTime B: {rr['B'] or '-'}")
                with c1:
                    st.markdown(f"**{rr['A'] or '-'}**")
                    st.text_input(f'Kills A ({rr["Lane"]})', key=f'pl_in_{i}_kills_A')
                    st.text_input(f'Deaths A ({rr["Lane"]})', key=f'pl_in_{i}_deaths_A')
                    st.text_input(f'Assists A ({rr["Lane"]})', key=f'pl_in_{i}_assists_A')
                with c2:
                    st.markdown(f"**{rr['B'] or '-'}**")
                    st.text_input(f'Kills B ({rr["Lane"]})', key=f'pl_in_{i}_kills_B')
                    st.text_input(f'Deaths B ({rr["Lane"]})', key=f'pl_in_{i}_deaths_B')
                    st.text_input(f'Assists B ({rr["Lane"]})', key=f'pl_in_{i}_assists_B')
    
            submitted_players_bottom = st.form_submit_button('Analisar players', type='primary', key='pl_submit_bottom')
            submitted_players = bool(submitted_players_top or submitted_players_bottom)
    
        if submitted_players:
            core_apply_players_submission_state(st.session_state, lanes=len(lane_rows))
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    
        if not any(str(x.get('A') or '').strip() or str(x.get('B') or '').strip() for x in lane_rows):
            st.info('Informe players para abrir os mercados por lane.')
        else:
            wf_m = state_get_float(st.session_state, 'resumo_mix_w_model', 40.0)
            wf_h = state_get_float(st.session_state, 'resumo_mix_w_laplace', 40.0)
            wf_s = state_get_float(st.session_state, 'resumo_mix_w_sc', 20.0)
    
            med_rows = core_build_player_means_rows(
                dfb=dfb,
                lane_rows=lane_rows,
                team_a=teamA,
                team_b=teamB,
                format_num_fn=core_format_num,
            )
    
            if med_rows:
                st.markdown('#### Médias por jogador')
                st.caption('Base estatística usada no modelo de players por lane.')
                dfm = pd.DataFrame(med_rows)
                st.dataframe(
                    dfm[['Lane', 'Time', 'Player', 'Amostra', 'Kills (media)', 'Kills (sd)', 'Deaths (media)', 'Deaths (sd)', 'Assists (media)', 'Assists (sd)']],
                    width='stretch',
                    hide_index=True,
                )
                st.divider()
    
            per_player_lines_by_idx = core_build_per_player_lines_by_idx(
                session_state=st.session_state,
                lane_rows=lane_rows,
                parse_lines_fn=_parse_lines,
            )
    
            out_rows, fair_rows = core_build_player_market_rows(
                dfb=dfb,
                lane_rows=lane_rows,
                per_player_lines_by_idx=per_player_lines_by_idx,
                team_a=teamA,
                team_b=teamB,
                wf_m=float(wf_m),
                wf_h=float(wf_h),
                wf_s=float(wf_s),
                odd_from_prob_fn=_odd_from_p,
                format_num_fn=core_format_num,
                format_pct_fn=lambda p: core_format_pct(p, 1),
                max_odd=float(st.session_state.get("resumo_max_odd", 2.0) or 2.0),
                require_model=False,
            )
    
            if not out_rows:
                st.info('Sem dados suficientes para montar mercados de players no recorte atual.')
            else:
                if fair_rows:
                    st.markdown('#### Fair (Modelo) por jogador')
                    st.caption('Odds justas do modelo para cada jogador/linha em Kills, Deaths e Assists.')
                    dff = pd.DataFrame(fair_rows)
                    ff1, ff2, ff3 = st.columns(3)
                    with ff1:
                        ff_team = st.selectbox('Fair - Time', ['Todos', teamA, teamB], index=0, key='pl_fair_filter_team')
                    with ff2:
                        ff_lane = st.selectbox('Fair - Lane', ['Todos'] + sorted(dff['Lane'].unique().tolist()), index=0, key='pl_fair_filter_lane')
                    with ff3:
                        ff_player = st.selectbox('Fair - Player', ['Todos'] + sorted(dff['Player'].unique().tolist()), index=0, key='pl_fair_filter_player')
                    if ff_team != 'Todos':
                        dff = dff[dff['Time'].astype(str) == str(ff_team)].copy()
                    if ff_lane != 'Todos':
                        dff = dff[dff['Lane'].astype(str) == str(ff_lane)].copy()
                    if ff_player != 'Todos':
                        dff = dff[dff['Player'].astype(str) == str(ff_player)].copy()
                    dff = dff[['Lane', 'Time', 'Player', 'Mercado', 'Media', 'SD', 'Linha', 'Acima (%)', 'Odd Acima', 'Abaixo (%)', 'Odd Abaixo', 'Amostra']]
                    st.dataframe(dff, width='stretch', hide_index=True)
    
                st.divider()
                st.markdown('#### Mercados de Players (Modelo/Hist/Mix)')
                dfo = pd.DataFrame(out_rows)
                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    team_filter = st.selectbox('Filtro Time', ['Todos', teamA, teamB], index=0, key='pl_mkt_filter_team')
                with f2:
                    lane_filter = st.selectbox('Filtro Lane', ['Todos'] + sorted(dfo['Lane'].unique().tolist()), index=0, key='pl_mkt_filter_lane')
                with f3:
                    market_filter = st.selectbox('Filtro Mercado', ['Todos'] + sorted(dfo['Mercado'].unique().tolist()), index=0, key='pl_mkt_filter_market')
                with f4:
                    player_filter = st.selectbox('Filtro Player', ['Todos'] + sorted(dfo['Player'].unique().tolist()), index=0, key='pl_mkt_filter_player')
                if team_filter != 'Todos':
                    dfo = dfo[dfo['Team'].astype(str) == str(team_filter)].copy()
                if lane_filter != 'Todos':
                    dfo = dfo[dfo['Lane'].astype(str) == str(lane_filter)].copy()
                if market_filter != 'Todos':
                    dfo = dfo[dfo['Mercado'].astype(str) == str(market_filter)].copy()
                if player_filter != 'Todos':
                    dfo = dfo[dfo['Player'].astype(str) == str(player_filter)].copy()
                try:
                    st.session_state["_ml_trace_players"] = [
                        {
                            "lane": str(r.get("Lane", "")),
                            "team": str(r.get("Team", "")),
                            "player": str(r.get("Player", "")),
                            "mercado": str(r.get("Mercado", "")),
                            "lado": str(r.get("Lado", "")),
                            "linha": str(r.get("Linha", "")),
                            "odd_model": r.get("_odd_model_raw"),
                            "odd_hist": r.get("_odd_hist_raw"),
                            "odd_mix": r.get("_odd_mix_raw"),
                            "calib_reason": str(r.get("_calib_reason", "")),
                            "calib_source": str(r.get("_calib_source", "")),
                        }
                        for r in out_rows
                    ]
                except Exception:
                    st.session_state["_ml_trace_players"] = []
                _persist_combined_trace_file()
                dfo = dfo[['Lane', 'Team', 'Player', 'Mercado', 'Lado', 'Linha', 'Stats', 'Odd (Modelo)', 'Odd (Hist)', 'Odd (Mix)']]
                st.dataframe(dfo, width='stretch', hide_index=True)
                with st.expander("Trace de calibração (Players)"):
                    _trows_p = st.session_state.get("_ml_trace_players", []) or []
                    if _trows_p:
                        st.dataframe(pd.DataFrame(_trows_p), width='stretch', hide_index=True)
                        st.download_button(
                            "Baixar trace (Players) JSON",
                            data=json.dumps(_trows_p, ensure_ascii=False, indent=2).encode("utf-8"),
                            file_name="ml_trace_players.json",
                            mime="application/json",
                            key="dl_trace_players_json",
                        )
                    else:
                        st.caption("Sem trace nesta execução.")
    

