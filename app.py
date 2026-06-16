import re
import os
import math
from pathlib import Path
from io import BytesIO
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
import plotly.graph_objects as go

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="Generic Player — Dashboard")

# OPTIONAL DOCX IMPORT
DOCX_AVAILABLE = True
try:
    from docx import Document
except Exception:
    DOCX_AVAILABLE = False

# STYLE
st.markdown("""
""", unsafe_allow_html=True)

# CONSTANTS
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67
GOAL_X = 120.0
GOAL_Y = 40.0
FIG_W, FIG_H = 7.0, 4.7
FIG_DPI = 180
COLOR_SUCCESS = "#c8c8c8"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_FAIL = "#E07070"
ALPHA_SUCCESS = 0.07
C_BLUE = "#2F80ED"
C_BLUE_DARK = "#1a56db"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_PURPLE_LIGHT = "#a78bfa"
C_BLUE_PASTEL = "#5b9bd5"
C_GREEN_PASTEL = "#70ad47"
C_AMBER_PASTEL = "#d4a843"
# Card tone palettes by action type (lighter -> darker within each type)
# Passes = blue, Defensive Actions = green, Offensive Actions = red
PASS_TONES = ["#5b9bd5", "#3b82f6", "#1d4ed8"]
DEF_TONES = ["#70ad47", "#22c55e", "#15803d"]
OFF_TONES = ["#f08a8a", "#ef4444", "#b91c1c"]
# Toggled from the sidebar ("Caixinhas modernas")
MODERN_CARDS = False
CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a", "#f97316", "#b91c1c"])
NORM_TOP10 = Normalize(vmin=0.05, vmax=0.40)
NX_XT, NY_XT = 16, 12
D_REF, D_SCALE, BONUS_CAP = 10.0, 20.0, 0.60
LATERAL_MIN_DIST = 12.0
PENALTY_AREA_X = 18.0
FUNNEL_X_EXTEND = 33.0
PENALTY_AREA_Y_MIN = 18.0
PENALTY_AREA_Y_MAX = 62.0

def _hex_to_rgba(hex_color, alpha=1.0):
    if hex_color.startswith('#'):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color

def get_lane(y):
    if y >= LANE_LEFT_MIN:
        return "left"
    elif y < LANE_RIGHT_MAX:
        return "right"
    return "center"

def distance_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)

def is_progressive_pass(x_start, y_start, x_end, y_end):
    if x_start < 35:
        return False
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    if start_dist == 0:
        return False
    return ((start_dist - end_dist) / start_dist) >= 0.25

def classify_pass_direction(x_start, y_start, x_end, y_end):
    dx = x_end - x_start
    dy = y_end - y_start
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0:
        return "forward"
    if angle_deg >= 135.0:
        return "backward"
    if dist > LATERAL_MIN_DIST:
        return "lateral_right" if dy > 0 else "lateral_left"
    return "forward" if dx >= 0 else "backward"

def distance_bonus(distance):
    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)
    return np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))

@st.cache_data(show_spinner=False)
def compute_xt_grid(NX=16, NY=12, sub=24):
    ncols_hr = NX * sub
    nrows_hr = NY * sub
    xe = np.linspace(0, FIELD_X, ncols_hr + 1)
    ye = np.linspace(0, FIELD_Y, nrows_hr + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    yc_arr = (ye[:-1] + ye[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc_arr)
    xp = 0.01 + (Xc / FIELD_X) * 0.99
    yc = 1.0 - np.abs((Yc / FIELD_Y) - 0.5) * 2.0
    base = xp * (0.8 + 0.2 * yc)
    base = (base - base.min()) / (base.max() - base.min() + 1e-12)
    XT = base.copy()
    XT = (XT - XT.min()) / (XT.max() - XT.min() + 1e-12)
    XTc = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            XTc[iy, ix] = XT[iy * sub:(iy + 1) * sub, ix * sub:(ix + 1) * sub].mean()
    XTc = (XTc - XTc.min()) / (XTc.max() - XTc.min() + 1e-12)
    return XTc

XT_GRID = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x / FIELD_X) * NX_XT, 0, NX_XT - 1))
    iy = int(np.clip((y / FIELD_Y) * NY_XT, 0, NY_XT - 1))
    return float(XT_GRID[iy, ix])

def is_in_funnel_zone(x, y):
    """Check if a defensive action is in the penalty area + 15m extended zone."""
    return x <= FUNNEL_X_EXTEND and PENALTY_AREA_Y_MIN <= y <= PENALTY_AREA_Y_MAX

# GENERIC RANDOM DATA GENERATOR (follows the Hudson Cicala data pattern)
# Synthetic pass & defensive-action data for a generic player, 20 matches.
# Reproducible thanks to a fixed seed.
import random as _random

_GEN_SEED = 2026
_GEN_FIELD_X, _GEN_FIELD_Y = 120.0, 80.0


def _gen_clampx(v):
    return min(_GEN_FIELD_X, max(0.0, v))


def _gen_clampy(v):
    return min(_GEN_FIELD_Y, max(0.0, v))


def _gen_r2(v):
    return round(float(v), 2)


def _gen_lane_y(rng):
    """Bias start positions toward the wide corridors (like Hudson Cicala),
    with fewer touches through the central lane.
    Lanes: left (y >= 53.33), center (26.67 <= y < 53.33), right (y < 26.67)."""
    roll = rng.random()
    if roll < 0.43:            # left corridor
        return _gen_clampy(rng.gauss(67, 7))
    elif roll < 0.64:          # central lane
        return _gen_clampy(rng.gauss(40, 5))
    else:                      # right corridor
        return _gen_clampy(rng.gauss(13, 7))


def _gen_pass(rng, won):
    """One pass following the original pattern: starts around midfield and
    progresses forward; lost passes are more ambitious and deeper.
    Start positions favour the wide corridors over the central lane."""
    x_start = _gen_clampx(rng.gauss(55, 22))
    y_start = _gen_lane_y(rng)
    if won:
        dx = rng.gauss(9, 13)
        dy = rng.gauss(0, 16)
    else:
        x_start = _gen_clampx(x_start + rng.uniform(5, 20))
        dx = rng.gauss(18, 14)
        dy = rng.gauss(0, 18)
    x_end = _gen_clampx(x_start + dx)
    y_end = _gen_clampy(y_start + dy)
    return (_gen_r2(x_start), _gen_r2(y_start), _gen_r2(x_end), _gen_r2(y_end))


def _gen_match_passes(rng):
    n = rng.randint(16, 50)
    win_rate = rng.uniform(0.72, 0.88)
    n_won = int(round(n * win_rate))
    n_lost = max(1, n - n_won)
    rows = []
    for _ in range(n_won):
        x1, y1, x2, y2 = _gen_pass(rng, True)
        rows.append(("PASS WON", x1, y1, x2, y2, None))
    for _ in range(n_lost):
        x1, y1, x2, y2 = _gen_pass(rng, False)
        rows.append(("PASS LOST", x1, y1, x2, y2, None))
    return rows


def _gen_def_action(rng, kind):
    # Based on the Hudson Cicala defensive sample: actions concentrate in the
    # own/middle thirds and away from the touchlines and goal line.
    # Keep a safe margin from the back line (x >= 12) and sidelines (10 <= y <= 70).
    if kind == "INTERCEPTION":
        x = rng.gauss(50, 17)
    else:
        x = rng.gauss(46, 18)
    x = min(95.0, max(12.0, x))
    y = min(70.0, max(10.0, rng.gauss(41, 15)))
    return (kind, _gen_r2(x), _gen_r2(y))


def _gen_match_def(rng):
    n = rng.randint(2, 21)
    rows = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.45:
            kind = "DUEL_WON"
        elif roll < 0.70:
            kind = "DUEL_LOST"
        else:
            kind = "INTERCEPTION"
        rows.append(_gen_def_action(rng, kind))
    order = {"DUEL_WON": 0, "DUEL_LOST": 1, "INTERCEPTION": 2}
    rows.sort(key=lambda r: order[r[0]])
    return rows


def _generate_generic_dataset(seed=_GEN_SEED, n_matches=20):
    rng = _random.Random(seed)
    names = []
    minutes = {}
    for i in range(1, n_matches + 1):
        mm = 3 + (i // 11)
        dd = ((1 + i * 4 - 1) % 28) + 1
        name = f"Opponent {i:02d} ({mm:02d}-{dd:02d})"
        names.append(name)
        minutes[name] = float(rng.choice([90, 90, 90, 90, 75, 60, 63, 65, 45]))
    base = {name: _gen_match_passes(rng) for name in names}
    deff = {name: _gen_match_def(rng) for name in names}
    return base, deff, minutes


BASE_MATCHES_DATA, DEFENSIVE_MATCHES_DATA, MATCH_MINUTES = _generate_generic_dataset()


# OFFENSIVE RANDOM DATA GENERATOR (offensive duels, touches & shots)
# Modest volumes per match so the maps stay readable (not exaggerated).
def _gen_off_duel(rng, won):
    # offensive duels concentrate in the attacking half / final third,
    # kept away from the sidelines (margin of ~10 from each touchline)
    x = _gen_clampx(rng.gauss(80, 16))
    y = min(70.0, max(10.0, rng.gauss(40, 15)))
    return ("OFF_DUEL_WON" if won else "OFF_DUEL_LOST", _gen_r2(x), _gen_r2(y))


def _gen_match_off_duels(rng):
    n = rng.randint(3, 12)
    win_rate = rng.uniform(0.40, 0.65)
    rows = []
    for _ in range(n):
        rows.append(_gen_off_duel(rng, rng.random() < win_rate))
    order = {"OFF_DUEL_WON": 0, "OFF_DUEL_LOST": 1}
    rows.sort(key=lambda r: order[r[0]])
    return rows


def _gen_match_touches(rng):
    # touches: more numerous, biased toward attacking midfield / final third
    n = rng.randint(20, 45)
    rows = []
    for _ in range(n):
        x = _gen_clampx(rng.gauss(72, 23))
        y = _gen_clampy(rng.gauss(40, 22))
        rows.append((_gen_r2(x), _gen_r2(y)))
    return rows


def _gen_match_shots(rng):
    # shots: few per match (some matches with none), around the penalty area
    n = rng.randint(0, 5)
    rows = []
    for _ in range(n):
        x = _gen_clampx(rng.gauss(104, 7))
        y = _gen_clampy(rng.gauss(40, 9))
        roll = rng.random()
        if roll < 0.13:
            outcome = "GOAL"
        elif roll < 0.50:
            outcome = "ON_TARGET"
        else:
            outcome = "OFF_TARGET"
        rows.append((outcome, _gen_r2(x), _gen_r2(y)))
    order = {"GOAL": 0, "ON_TARGET": 1, "OFF_TARGET": 2}
    rows.sort(key=lambda r: order[r[0]])
    return rows


def _generate_offensive_dataset(match_names, seed=_GEN_SEED + 7):
    rng = _random.Random(seed)
    duels = {name: _gen_match_off_duels(rng) for name in match_names}
    touches = {name: _gen_match_touches(rng) for name in match_names}
    shots = {name: _gen_match_shots(rng) for name in match_names}
    return duels, touches, shots


OFFENSIVE_DUELS_DATA, TOUCHES_DATA, SHOTS_DATA = _generate_offensive_dataset(list(BASE_MATCHES_DATA.keys()))

# HELPERS
def apply_date_mapping(name: str) -> str:
    # Generic dataset: no name remapping needed.
    return name

def get_match_minutes(match_name: str) -> float:
    if match_name == "All Matches":
        total = 0.0
        for k in dfs_by_match:
            total += get_match_minutes(k)
        return total
    return float(MATCH_MINUTES.get(match_name, 90.0))

def read_docx_text(docx_path: Path) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not installed.")
    doc = Document(str(docx_path))
    return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())

def parse_docx_events(raw_text: str) -> dict:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    matches = {}
    current_match = None
    current_state = None
    re_match = re.compile(r"^Vs\s+(.+)$", re.IGNORECASE)
    re_success = re.compile(r"^Sucesso$", re.IGNORECASE)
    re_fail = re.compile(r"^Errado[s]?$", re.IGNORECASE)
    re_arrow = re.compile(
        r"^Seta\s+\d+:\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)\s*->\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)$",
        re.IGNORECASE,
    )
    for ln in lines:
        m_match = re_match.match(ln)
        if m_match:
            current_match = m_match.group(1).strip()
            matches.setdefault(current_match, [])
            current_state = None
            continue
        if re_success.match(ln):
            current_state = "PASS WON"
            continue
        if re_fail.match(ln):
            current_state = "PASS LOST"
            continue
        m_arrow = re_arrow.match(ln)
        if m_arrow and current_match and current_state:
            x1, y1, x2, y2 = map(float, m_arrow.groups())
            matches[current_match].append((current_state, x1, y1, x2, y2, None))
    return {k: v for k, v in matches.items() if len(v) > 0}

def load_docx_matches(docx_filename="Passes - Generic Player.docx") -> dict:
    p = Path(docx_filename)
    if not p.exists():
        return {}
    txt = read_docx_text(p)
    return parse_docx_events(txt)

# DATA LOADING
docx_matches_data = {}
try:
    docx_matches_data = load_docx_matches()
except Exception:
    pass

combined_matches_data = {}
for k, v in docx_matches_data.items():
    mapped_k = apply_date_mapping(k)
    name = mapped_k if mapped_k not in combined_matches_data else f"DOCX - {mapped_k}"
    combined_matches_data[name] = v
for k, v in BASE_MATCHES_DATA.items():
    combined_matches_data[k] = v

if len(combined_matches_data) == 0:
    st.error("Could not load data.")
    st.stop()

# BUILD DATAFRAMES & REORDER MATCHES
dfs_by_match = {}
for match_name, events in combined_matches_data.items():
    dfm = pd.DataFrame(events, columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
    dfm["match"] = match_name
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["is_won"] = dfm["type"].str.contains("WON", case=False)
    dfm["progressive"] = dfm.apply(
        lambda r: r["is_won"] and is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
        axis=1
    )
    dfm["direction"] = dfm.apply(
        lambda r: classify_pass_direction(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
        axis=1
    )
    dfm["is_forward"] = dfm["direction"] == "forward"
    dfm["is_backward"] = dfm["direction"] == "backward"
    dfm["is_lateral"] = dfm["direction"].isin(["lateral_left", "lateral_right"])
    dfm["pass_distance"] = np.sqrt((dfm["x_end"] - dfm["x_start"]) ** 2 + (dfm["y_end"] - dfm["y_start"]) ** 2)
    dfm["xt_start"] = dfm.apply(lambda r: xt_value(r["x_start"], r["y_start"]), axis=1)
    dfm["xt_end"] = dfm.apply(lambda r: xt_value(r["x_end"], r["y_end"]), axis=1)
    dfm["delta_xt"] = np.where(dfm["is_won"], dfm["xt_end"] - dfm["xt_start"], 0.0)
    dfm["dist_bonus"] = distance_bonus(dfm["pass_distance"].values)
    dfm["delta_xt_adj"] = np.where(dfm["is_won"], dfm["delta_xt"] * (1.0 + dfm["dist_bonus"]), 0.0)
    dfs_by_match[match_name] = dfm

# REORDER LOGIC
items = list(dfs_by_match.items())
if len(items) >= 18:
    part1 = items[:6]
    part2 = items[14:18]
    part3 = items[6:14]
    part4 = items[18:]
    dfs_by_match = dict(part1 + part2 + part3 + part4)

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)

# DEFENSIVE DATA LOADING
defensive_dfs_by_match = {}
for match_name, events in DEFENSIVE_MATCHES_DATA.items():
    df_def = pd.DataFrame(events, columns=["type", "x", "y"])
    df_def["match"] = match_name
    df_def["is_attacking_half"] = df_def["x"] >= FIELD_X / 2
    df_def["is_duel_won"] = df_def["type"] == "DUEL_WON"
    df_def["is_duel_lost"] = df_def["type"] == "DUEL_LOST"
    df_def["is_duel"] = df_def["is_duel_won"] | df_def["is_duel_lost"]
    df_def["is_interception"] = df_def["type"] == "INTERCEPTION"
    df_def["in_funnel"] = df_def.apply(lambda r: is_in_funnel_zone(r["x"], r["y"]), axis=1)
    defensive_dfs_by_match[match_name] = df_def

# OFFENSIVE DATA LOADING
offensive_duels_dfs_by_match = {}
for match_name, events in OFFENSIVE_DUELS_DATA.items():
    df_od = pd.DataFrame(events, columns=["type", "x", "y"])
    df_od["match"] = match_name
    df_od["is_won"] = df_od["type"] == "OFF_DUEL_WON"
    df_od["is_lost"] = df_od["type"] == "OFF_DUEL_LOST"
    offensive_duels_dfs_by_match[match_name] = df_od

touches_dfs_by_match = {}
for match_name, events in TOUCHES_DATA.items():
    df_to = pd.DataFrame(events, columns=["x", "y"])
    df_to["match"] = match_name
    df_to["is_final_third"] = df_to["x"] >= FINAL_THIRD_LINE_X
    touches_dfs_by_match[match_name] = df_to

shots_dfs_by_match = {}
for match_name, events in SHOTS_DATA.items():
    df_sh = pd.DataFrame(events, columns=["type", "x", "y"])
    df_sh["match"] = match_name
    df_sh["is_goal"] = df_sh["type"] == "GOAL"
    df_sh["is_on_target"] = df_sh["type"].isin(["GOAL", "ON_TARGET"])
    shots_dfs_by_match[match_name] = df_sh

# STATS & SCORES
def compute_stats(df: pd.DataFrame, match_name: str) -> dict:
    total = len(df)
    mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    if total == 0:
        return {
            "total_passes": 0,
            "successful_passes": 0,
            "unsuccessful_passes": 0,
            "accuracy_pct": 0.0,
            "progressive_attempted": 0,
            "progressive_successful": 0,
            "progressive_accuracy_pct": 0.0,
            "to_final_third_total": 0,
            "to_final_third_success": 0,
            "to_final_third_accuracy_pct": 0.0,
            "fwd": 0,
            "fwd_pct": 0.0,
            "bwd": 0,
            "bwd_pct": 0.0,
            "lat": 0,
            "lat_pct": 0.0,
            "pos_count": 0,
            "pos_pct": 0.0,
            "high_xt_pct": 0.0,
            "sum_dxt": 0.0,
            "total_p90": 0.0,
            "prog_p90": 0.0,
            "f3_p90": 0.0,
            "xt_p90": 0.0,
            "neg_xt_p90": 0.0,
            "minutes": mins,
            "long_acc_pct": 0.0,
            "high_xt_p90": 0.0,
            "dz_p90": 0.0,
            "adv_made": 0,
            "adv_att": 0,
            "adv_acc_pct": 0.0,
            "adv_p90": 0.0,
        }
    successful = int(df["is_won"].sum())
    unsuccessful = total - successful
    accuracy = successful / total * 100.0
    progressive_total = int(df["progressive"].sum())
    progressive_unsuccessful = int(
        (~df["is_won"] & df.apply(
            lambda r: is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
            axis=1
        )).sum()
    )
    progressive_attempted = progressive_total + progressive_unsuccessful
    progressive_accuracy = (progressive_total / progressive_attempted * 100.0) if progressive_attempted else 0.0
    to_final_third = (df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)
    to_final_third_total = int(to_final_third.sum())
    to_final_third_success = int((to_final_third & df["is_won"]).sum())
    to_final_third_accuracy = (to_final_third_success / to_final_third_total * 100.0) if to_final_third_total else 0.0
    long_passes = df[df["pass_distance"] > 25.0]
    long_total = len(long_passes)
    long_success = int(long_passes["is_won"].sum())
    long_acc_pct = (long_success / long_total * 100.0) if long_total > 0 else 0.0
    dz_mask = df["is_won"] & (
        (df["x_end"] >= 100.0) |
        ((df["x_end"] >= 80.0) & (df["x_end"] < 100.0) & (df["y_end"] >= LANE_RIGHT_MAX) & (df["y_end"] < LANE_LEFT_MIN))
    )
    dz_passes = int(dz_mask.sum())
    # Advanced passes = progressive + final-third passes (made / attempted)
    adv_made = progressive_total + to_final_third_success
    adv_att = progressive_attempted + to_final_third_total
    adv_acc_pct = (adv_made / adv_att * 100.0) if adv_att > 0 else 0.0
    fwd = int(df["is_forward"].sum())
    bwd = int(df["is_backward"].sum())
    lat = int(df["is_lateral"].sum())
    pos_count = int((df["is_won"] & (df["delta_xt_adj"] > 0)).sum())
    pos_pct = (pos_count / total * 100.0) if total > 0 else 0.0
    high_xt = int((df["delta_xt_adj"] > 0.1).sum())
    sum_dxt = float(df.loc[df["is_won"], "delta_xt_adj"].sum())
    neg_xt = float(df.loc[df["is_won"] & (df["delta_xt_adj"] < 0), "delta_xt_adj"].sum())
    return {
        "total_passes": total,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "progressive_attempted": progressive_attempted,
        "progressive_successful": progressive_total,
        "progressive_accuracy_pct": round(progressive_accuracy, 2),
        "to_final_third_total": to_final_third_total,
        "to_final_third_success": to_final_third_success,
        "to_final_third_accuracy_pct": round(to_final_third_accuracy, 2),
        "fwd": fwd,
        "fwd_pct": round(fwd / total * 100.0, 1),
        "bwd": bwd,
        "bwd_pct": round(bwd / total * 100.0, 1),
        "lat": lat,
        "lat_pct": round(lat / total * 100.0, 1),
        "pos_count": pos_count,
        "pos_pct": round(pos_pct, 1),
        "high_xt_pct": round(high_xt / total * 100.0, 1),
        "sum_dxt": round(sum_dxt, 3),
        "total_p90": round(total * p90_factor, 1),
        "prog_p90": round(progressive_total * p90_factor, 2),
        "f3_p90": round(to_final_third_success * p90_factor, 2),
        "xt_p90": round(sum_dxt * p90_factor, 3),
        "neg_xt_p90": round(neg_xt * p90_factor, 3),
        "minutes": mins,
        "long_acc_pct": round(long_acc_pct, 1),
        "high_xt_p90": round(high_xt * p90_factor, 2),
        "dz_p90": round(dz_passes * p90_factor, 2),
        "adv_made": adv_made,
        "adv_att": adv_att,
        "adv_acc_pct": round(adv_acc_pct, 1),
        "adv_p90": round(adv_made * p90_factor, 2),
    }

def compute_match_scores(dfs_dict, defensive_dfs_dict=None, offensive_dicts=None):
    records = []
    for m_name, df_m in dfs_dict.items():
        s = compute_stats(df_m, m_name)
        total_passes = s['total_passes']
        if total_passes == 0:
            continue
        records.append({
            'match': m_name,
            'xt_p90': s['xt_p90'],
            'prog_p90': s['prog_p90'],
            'f3_p90': s['f3_p90'],
            'pos_pct': s['pos_pct'],
            'total_p90': s['total_p90'],
            'neg_xt_p90': s['neg_xt_p90'],
            'accuracy_pct': s['accuracy_pct'],
            'long_acc_pct': s['long_acc_pct'],
            'high_xt_p90': s['high_xt_p90'],
            'dz_p90': s['dz_p90'],
            'prog_acc_pct': s['progressive_accuracy_pct'],
            'adv_p90': s['adv_p90'],
            'adv_acc_pct': s['adv_acc_pct'],
        })
    df_scores = pd.DataFrame(records)
    if df_scores.empty:
        return df_scores
    def normalize_fixed(series, val_min, val_max):
        clipped_series = series.clip(lower=val_min, upper=val_max)
        if val_max == val_min:
            return pd.Series([70.0] * len(series))
        return 40 + ((clipped_series - val_min) / (val_max - val_min)) * 60
    df_scores['xt_norm'] = normalize_fixed(df_scores['xt_p90'], val_min=0.05, val_max=0.45)
    df_scores['prog_norm'] = normalize_fixed(df_scores['prog_p90'], val_min=1.0, val_max=15.0)
    df_scores['f3_norm'] = normalize_fixed(df_scores['f3_p90'], val_min=1.0, val_max=15.0)
    df_scores['pos_pct_norm'] = normalize_fixed(df_scores['pos_pct'], val_min=25.0, val_max=75.0)
    df_scores['total_p90_norm'] = normalize_fixed(df_scores['total_p90'], val_min=10.0, val_max=85.0)
    df_scores['neg_xt_norm'] = normalize_fixed(df_scores['neg_xt_p90'], val_min=-0.15, val_max=0.00)
    df_scores['Grade'] = (
        df_scores['xt_norm'] * 0.30 +
        df_scores['prog_norm'] * 0.20 +
        df_scores['f3_norm'] * 0.20 +
        df_scores['pos_pct_norm'] * 0.10 +
        df_scores['total_p90_norm'] * 0.10 +
        df_scores['neg_xt_norm'] * 0.10
    )
    if defensive_dfs_dict is not None:
        def_bonus_list = []
        duels_p90_list = []
        duels_won_pct_list = []
        interceptions_p90_list = []
        duels_won_p90_list = []
        int_xt_avg_list = []
        funnel_p90_list = []
        for _, row in df_scores.iterrows():
            team = row['match'].split('(')[0].strip()
            bonus = 0
            dp = 0; dwp = 0; ip = 0
            dwp90 = 0; int_xt_avg_rounded = 0
            funnel_p90_val = 0.0
            for def_name, def_df in defensive_dfs_dict.items():
                def_team = def_name.split('(')[0].strip()
                if def_team == team:
                    mins = get_match_minutes(def_name)
                    p90 = 90.0 / mins if mins > 0 else 1.0
                    duels_won = int(def_df["is_duel_won"].sum())
                    duels_lost = int(def_df["is_duel_lost"].sum())
                    total_duels = duels_won + duels_lost
                    dwp = (duels_won / total_duels * 100.0) if total_duels > 0 else 0
                    dp = round(total_duels * p90, 1)
                    dwp90 = round(duels_won * p90, 1)
                    interceptions = int(def_df["is_interception"].sum())
                    ip = round(interceptions * p90, 1)
                    funnel_count = int(def_df["in_funnel"].sum())
                    funnel_p90_val = round(funnel_count * p90, 1)
                    int_df = def_df[def_df["is_interception"]]
                    if len(int_df) > 0:
                        int_xt_vals = [xt_value(float(r["x"]), float(r["y"])) for _, r in int_df.iterrows()]
                        int_xt_avg = float(np.mean(int_xt_vals)) if int_xt_vals else 0
                    else:
                        int_xt_avg = 0
                    int_xt_avg_rounded = round(int_xt_avg, 3)
                    duel_bonus_val = (dwp / 100.0) * min(dp / 20.0, 1.0) * 5
                    int_bonus_val = min(ip / 10.0, 1.0) * (0.5 + int_xt_avg * 2) * 3
                    bonus = duel_bonus_val + int_bonus_val
                    break
            def_bonus_list.append(round(bonus, 2))
            duels_p90_list.append(dp)
            duels_won_pct_list.append(round(dwp, 1))
            interceptions_p90_list.append(ip)
            duels_won_p90_list.append(dwp90)
            int_xt_avg_list.append(int_xt_avg_rounded)
            funnel_p90_list.append(funnel_p90_val)
        df_scores['def_bonus'] = def_bonus_list
        df_scores['duels_p90'] = duels_p90_list
        df_scores['duels_won_pct'] = duels_won_pct_list
        df_scores['interceptions_p90'] = interceptions_p90_list
        df_scores['duels_won_p90'] = duels_won_p90_list
        df_scores['int_xt_avg'] = int_xt_avg_list
        df_scores['funnel_p90'] = funnel_p90_list
    else:
        df_scores['def_bonus'] = 0.0
        df_scores['duels_p90'] = 0.0
        df_scores['duels_won_pct'] = 0.0
        df_scores['interceptions_p90'] = 0.0
        df_scores['duels_won_p90'] = 0.0
        df_scores['int_xt_avg'] = 0.0
        df_scores['funnel_p90'] = 0.0
    df_scores['pass_grade'] = df_scores['Grade'].round(1).copy()
    def _norm_def(s, lo, hi):
        clipped = s.clip(lower=lo, upper=hi)
        if hi <= lo:
            return pd.Series([70.0] * len(s))
        return 40 + ((clipped - lo) / (hi - lo)) * 60
    df_scores['duels_won_pct_norm'] = _norm_def(df_scores['duels_won_pct'], 0, 100)
    df_scores['int_xt_norm'] = _norm_def(df_scores['int_xt_avg'], 0, 0.30)
    df_scores['duels_won_p90_norm'] = _norm_def(df_scores['duels_won_p90'], 0, 15)
    df_scores['interceptions_p90_norm'] = _norm_def(df_scores['interceptions_p90'], 0, 15)
    df_scores['funnel_p90_norm'] = _norm_def(df_scores['funnel_p90'], 0, 15)
    df_scores['def_grade'] = (
        df_scores['duels_won_pct_norm'] * 0.30 +
        df_scores['funnel_p90_norm'] * 0.15 +
        df_scores['int_xt_norm'] * 0.25 +
        df_scores['duels_won_p90_norm'] * 0.15 +
        df_scores['interceptions_p90_norm'] * 0.15
    ).round(1)

    # OFFENSIVE GRADE (touches / final-third presence / duels / shots / finishing)
    if offensive_dicts is not None:
        touches_d, duels_d, shots_d = offensive_dicts
        touches_p90_l = []
        f3_touches_p90_l = []
        off_duels_p90_l = []
        off_duels_won_pct_l = []
        shots_p90_l = []
        goals_l = []
        shots_on_target_pct_l = []
        for _, row in df_scores.iterrows():
            team = row['match'].split('(')[0].strip()
            matched = None
            for nm in touches_d:
                if nm.split('(')[0].strip() == team:
                    matched = nm
                    break
            if matched is not None:
                os_ = compute_offensive_stats(touches_d[matched], duels_d[matched], shots_d[matched], matched)
            else:
                os_ = {"touches_p90": 0.0, "f3_touches_p90": 0.0, "off_duels_p90": 0.0,
                       "off_duels_won_pct": 0.0, "shots_p90": 0.0, "goals": 0, "shots_on_target_pct": 0.0}
            touches_p90_l.append(os_["touches_p90"])
            f3_touches_p90_l.append(os_["f3_touches_p90"])
            off_duels_p90_l.append(os_["off_duels_p90"])
            off_duels_won_pct_l.append(os_["off_duels_won_pct"])
            shots_p90_l.append(os_["shots_p90"])
            goals_l.append(os_["goals"])
            shots_on_target_pct_l.append(os_["shots_on_target_pct"])
        df_scores['touches_p90'] = touches_p90_l
        df_scores['f3_touches_p90'] = f3_touches_p90_l
        df_scores['off_duels_p90'] = off_duels_p90_l
        df_scores['off_duels_won_pct'] = off_duels_won_pct_l
        df_scores['shots_p90'] = shots_p90_l
        df_scores['goals'] = goals_l
        df_scores['shots_on_target_pct'] = shots_on_target_pct_l
        df_scores['touches_norm'] = _norm_def(df_scores['touches_p90'], 20, 70)
        df_scores['f3_touches_norm'] = _norm_def(df_scores['f3_touches_p90'], 4, 30)
        df_scores['off_duels_won_norm'] = _norm_def(df_scores['off_duels_won_pct'], 30, 70)
        df_scores['shots_norm'] = _norm_def(df_scores['shots_p90'], 0.5, 5.0)
        df_scores['finishing_norm'] = _norm_def(df_scores['shots_on_target_pct'], 20, 70)
        df_scores['off_grade'] = (
            df_scores['touches_norm'] * 0.20 +
            df_scores['f3_touches_norm'] * 0.25 +
            df_scores['off_duels_won_norm'] * 0.20 +
            df_scores['shots_norm'] * 0.20 +
            df_scores['finishing_norm'] * 0.15
        ).round(1)
        # Combined Grade: pass / defensive / offensive with equal weights
        df_scores['Grade'] = (
            (df_scores['pass_grade'] + df_scores['def_grade'] + df_scores['off_grade']) / 3.0
        ).round(1)
    else:
        df_scores['touches_p90'] = 0.0
        df_scores['f3_touches_p90'] = 0.0
        df_scores['off_duels_p90'] = 0.0
        df_scores['off_duels_won_pct'] = 0.0
        df_scores['shots_p90'] = 0.0
        df_scores['goals'] = 0
        df_scores['shots_on_target_pct'] = 0.0
        df_scores['off_grade'] = 0.0
        df_scores['Grade'] = (df_scores['pass_grade'] * 0.5 + df_scores['def_grade'] * 0.5).round(1)
    return df_scores

def compute_defensive_stats(df: pd.DataFrame, match_name: str) -> dict:
    total_actions = len(df)
    if match_name == "All Matches":
        mins = sum(get_match_minutes(k) for k in defensive_dfs_by_match)
    else:
        mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    duels_won = int(df["is_duel_won"].sum())
    duels_lost = int(df["is_duel_lost"].sum())
    total_duels = duels_won + duels_lost
    duels_won_pct = (duels_won / total_duels * 100.0) if total_duels > 0 else 0.0
    interceptions = int(df["is_interception"].sum())
    attacking_half = df[df["is_attacking_half"]]
    actions_attacking = len(attacking_half)
    interceptions_attacking = int(attacking_half["is_interception"].sum())
    funnel_actions = int(df["in_funnel"].sum())
    return {
        "total_actions": total_actions,
        "total_actions_p90": round(total_actions * p90_factor, 1),
        "actions_attacking": actions_attacking,
        "actions_attacking_p90": round(actions_attacking * p90_factor, 1),
        "total_duels": total_duels,
        "duels_p90": round(total_duels * p90_factor, 1),
        "duels_won_pct": round(duels_won_pct, 1),
        "duels_won": duels_won,
        "interceptions": interceptions,
        "interceptions_p90": round(interceptions * p90_factor, 1),
        "interceptions_attacking": interceptions_attacking,
        "interceptions_attacking_p90": round(interceptions_attacking * p90_factor, 1),
        "funnel_actions": funnel_actions,
        "funnel_actions_p90": round(funnel_actions * p90_factor, 1),
    }

def compute_defensive_match_scores(dfs_dict):
    records = []
    for m_name, df_m in dfs_dict.items():
        mins = get_match_minutes(m_name)
        p90 = 90.0 / mins if mins > 0 else 1.0
        duels_won = int(df_m["is_duel_won"].sum())
        duels_lost = int(df_m["is_duel_lost"].sum())
        total_duels = duels_won + duels_lost
        interceptions = int(df_m["is_interception"].sum())
        funnel_count = int(df_m["in_funnel"].sum())
        records.append({
            'match': m_name,
            'duels_p90': round(total_duels * p90, 1),
            'interceptions_p90': round(interceptions * p90, 1),
            'funnel_p90': round(funnel_count * p90, 1),
        })
    return pd.DataFrame(records)

def compute_defensive_evolution_df(dfs_dict, df_scores=None):
    records = []
    for m_name, df_m in dfs_dict.items():
        mins = get_match_minutes(m_name)
        p90 = 90.0 / mins if mins > 0 else 1.0
        duels_won = int(df_m["is_duel_won"].sum())
        duels_lost = int(df_m["is_duel_lost"].sum())
        total_duels = duels_won + duels_lost
        duels_won_pct = (duels_won / total_duels * 100.0) if total_duels > 0 else 0.0
        interceptions = int(df_m["is_interception"].sum())
        attacking_half = df_m[df_m["is_attacking_half"]]
        actions_attacking = len(attacking_half)
        int_df = df_m[df_m["is_interception"]]
        if len(int_df) > 0:
            int_xt_vals = [xt_value(float(r["x"]), float(r["y"])) for _, r in int_df.iterrows()]
            int_xt_avg = float(np.mean(int_xt_vals)) if int_xt_vals else 0
        else:
            int_xt_avg = 0
        funnel_count = int(df_m["in_funnel"].sum())
        grade = None
        if df_scores is not None and 'Grade' in df_scores.columns and 'match' in df_scores.columns:
            team = m_name.split('(')[0].strip()
            for _, row in df_scores.iterrows():
                if row['match'].split('(')[0].strip() == team:
                    grade = row['Grade']
                    break
        records.append({
            'match': m_name,
            'def_actions_p90': round(len(df_m) * p90, 1),
            'duels_won_p90': round(duels_won * p90, 1),
            'interceptions_p90': round(interceptions * p90, 1),
            'duels_won_pct': round(duels_won_pct, 1),
            'actions_attacking_p90': round(actions_attacking * p90, 1),
            'int_xt_avg': round(int_xt_avg, 3),
            'funnel_actions_p90': round(funnel_count * p90, 1),
            'grade': round(grade, 1) if grade is not None else None,
        })
    return pd.DataFrame(records)

def compute_offensive_stats(touches_df, duels_df, shots_df, match_name: str) -> dict:
    if match_name == "All Matches":
        mins = sum(get_match_minutes(k) for k in touches_dfs_by_match)
    else:
        mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    touches_total = len(touches_df)
    f3_touches = int(touches_df["is_final_third"].sum()) if touches_total > 0 else 0
    off_duels_total = len(duels_df)
    off_duels_won = int(duels_df["is_won"].sum()) if off_duels_total > 0 else 0
    off_duels_won_pct = (off_duels_won / off_duels_total * 100.0) if off_duels_total > 0 else 0.0
    shots_total = len(shots_df)
    goals = int(shots_df["is_goal"].sum()) if shots_total > 0 else 0
    shots_on_target = int(shots_df["is_on_target"].sum()) if shots_total > 0 else 0
    shots_on_target_pct = (shots_on_target / shots_total * 100.0) if shots_total > 0 else 0.0
    return {
        "touches": touches_total,
        "touches_p90": round(touches_total * p90_factor, 1),
        "f3_touches": f3_touches,
        "f3_touches_p90": round(f3_touches * p90_factor, 1),
        "off_duels": off_duels_total,
        "off_duels_p90": round(off_duels_total * p90_factor, 1),
        "off_duels_won": off_duels_won,
        "off_duels_won_pct": round(off_duels_won_pct, 1),
        "shots": shots_total,
        "shots_p90": round(shots_total * p90_factor, 2),
        "goals": goals,
        "shots_on_target": shots_on_target,
        "shots_on_target_pct": round(shots_on_target_pct, 1),
    }

# UI HELPERS
def _safe_pct_diff(a: float, b: float) -> float:
    base = max(abs(b), 1.0)
    pct = (abs(a - b) / base) * 100.0
    return min(pct, 999.0)

def _arrow_html(val_game: float, val_avg: float) -> str:
    if np.isclose(val_game, val_avg, atol=1e-9):
        return ""
    if abs(val_game) < 1 and abs(val_avg) < 1:
        return ""
    if val_game > val_avg:
        pct = _safe_pct_diff(val_game, val_avg)
        return f' <span style="display:inline-block;font-size:11px;font-weight:700;color:#fff;background:#059669;padding:1px 7px;border-radius:10px;vertical-align:middle;line-height:1.6">+{pct:.0f}%</span>'
    else:
        pct = _safe_pct_diff(val_avg, val_game)
        return f' <span style="display:inline-block;font-size:11px;font-weight:700;color:#fff;background:#dc2626;padding:1px 7px;border-radius:10px;vertical-align:middle;line-height:1.6">-{pct:.0f}%</span>'

def _accent_rgb(border_color):
    h = border_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def _modern_card(title, border_color, items, comparison=False):
    """Sleek, professional card style (glass + gradient + accent)."""
    r, g, b = _accent_rgb(border_color)
    accent = f"rgb({r},{g},{b})"
    grad = (f"linear-gradient(150deg, rgba({r},{g},{b},0.16) 0%, "
            f"rgba(24,24,38,0.55) 55%, rgba(16,16,26,0.80) 100%)")
    html = (f'<div style="position:relative;background:{grad};'
            f'border:1px solid rgba({r},{g},{b},0.30);border-radius:16px;'
            f'padding:16px 18px 12px 18px;margin-bottom:12px;'
            f'box-shadow:0 10px 26px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.05);'
            f'overflow:hidden">')
    html += (f'<div style="position:absolute;top:0;left:0;height:3px;width:100%;'
             f'background:linear-gradient(90deg, rgba({r},{g},{b},0.95), rgba({r},{g},{b},0.12))"></div>')
    html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">'
    html += (f'<span style="width:8px;height:8px;border-radius:50%;background:{accent};'
             f'box-shadow:0 0 8px rgba({r},{g},{b},0.85)"></span>')
    html += (f'<span style="font-size:12px;font-weight:700;letter-spacing:1.2px;'
             f'text-transform:uppercase;color:#eef1f7">{title}</span>')
    html += '</div>'
    for idx, item in enumerate(items):
        label = item[0]
        if comparison:
            val_game = item[1]
            val_avg = item[2]
            value_html = item[3] if len(item) > 3 else str(val_game)
            disp_avg = item[4] if len(item) > 4 else str(val_avg)
            tooltip = item[5] if len(item) > 5 else ""
            arrow = _arrow_html(float(val_game), float(val_avg))
            sub_html = f"AVG {disp_avg}"
        else:
            value_html = item[1]
            sub_html = item[2] if len(item) > 2 else ""
            tooltip = item[3] if len(item) > 3 else ""
            arrow = ""
        is_last = idx == len(items) - 1
        row_style = "" if is_last else ("margin-bottom:12px;padding-bottom:12px;"
                                        "border-bottom:1px solid rgba(255,255,255,0.06)")
        title_attr = f' title="{tooltip}"' if tooltip else ""
        cursor = "cursor:help;" if tooltip else ""
        html += f'<div style="{row_style}">'
        html += (f'<div style="font-size:10.5px;font-weight:600;letter-spacing:0.6px;'
                 f'text-transform:uppercase;color:#9aa3b5;{cursor}"{title_attr}>{label}</div>')
        html += '<div style="display:flex;align-items:baseline;justify-content:space-between;margin-top:3px">'
        html += f'<span style="font-size:24px;font-weight:800;color:#ffffff;line-height:1.1">{value_html}</span>'
        if arrow:
            html += f'<span>{arrow}</span>'
        html += '</div>'
        if sub_html:
            html += f'<div style="font-size:11px;color:#8b93a7;margin-top:3px">{sub_html}</div>'
        html += '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def section_card(title, border_color, items):
    if MODERN_CARDS:
        _modern_card(title, border_color, items, comparison=False)
        return
    bg = _hex_to_rgba(border_color, 0.55)
    bd = _hex_to_rgba(border_color, 0.30)
    html = f'<div style="background:{bg};border:1px solid {bd};border-radius:10px;padding:14px;margin-bottom:8px">'
    html += f'<div style="font-size:15px;font-weight:800;color:#ffffff;margin-bottom:10px;letter-spacing:0.3px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:8px">{title}</div>'
    html += f'<div style="opacity:0.88">'
    for idx, item in enumerate(items):
        label = item[0]
        value = item[1]
        sub = item[2] if len(item) > 2 else ""
        tooltip = item[3] if len(item) > 3 else ""
        is_last = idx == len(items) - 1
        sep = "" if is_last else 'style="border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:6px;margin-bottom:6px"'
        html += f'<div {sep}>'
        html += f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        if tooltip:
            label_html = f'<span style="font-size:15px;color:#ffffff;font-weight:600;cursor:help;border-bottom:1px dotted rgba(255,255,255,0.15)" title="{tooltip}">{label}</span>'
            label_html += f'<span style="display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;font-size:9px;font-weight:700;color:#888;background:rgba(0,0,0,0.25);margin-left:4px;cursor:help;vertical-align:middle" title="{tooltip}">?</span>'
            html += f'<div>{label_html}</div>'
        else:
            html += f'<div style="font-size:15px;color:#ffffff;font-weight:600">{label}</div>'
        html += f'<div style="text-align:right">'
        html += f'<div style="font-size:20px;font-weight:800;color:#ffffff;line-height:1.2">{value}</div>'
        if sub:
            html += f'<div style="font-size:12px;font-style:italic;color:#ffffff;opacity:0.55;margin-top:1px">{sub}</div>'
        html += '</div>'
        html += '</div>'
        html += '</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

def cmp_section_card(title, border_color, items):
    if MODERN_CARDS:
        _modern_card(title, border_color, items, comparison=True)
        return
    bg = _hex_to_rgba(border_color, 0.55)
    bd = _hex_to_rgba(border_color, 0.30)
    html = f'<div style="background:{bg};border:1px solid {bd};border-radius:10px;padding:14px;margin-bottom:8px">'
    html += f'<div style="font-size:15px;font-weight:800;color:#ffffff;margin-bottom:10px;letter-spacing:0.3px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:8px">{title}</div>'
    html += f'<div style="opacity:0.88">'
    for idx, item in enumerate(items):
        label = item[0]
        val_game = item[1]
        val_avg = item[2]
        disp_game = item[3] if len(item) > 3 else str(val_game)
        disp_avg = item[4] if len(item) > 4 else str(val_avg)
        tooltip = item[5] if len(item) > 5 else ""
        arrow = _arrow_html(float(val_game), float(val_avg))
        is_last = idx == len(items) - 1
        sep = "" if is_last else 'style="border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:6px;margin-bottom:6px"'
        html += f'<div {sep}>'
        html += f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        if tooltip:
            label_html = f'<span style="font-size:15px;color:#ffffff;font-weight:600;cursor:help;border-bottom:1px dotted rgba(255,255,255,0.15)" title="{tooltip}">{label}</span>'
            label_html += f'<span style="display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;font-size:9px;font-weight:700;color:#888;background:rgba(0,0,0,0.25);margin-left:4px;cursor:help;vertical-align:middle" title="{tooltip}">?</span>'
            html += f'<div>{label_html}</div>'
        else:
            html += f'<div style="font-size:15px;color:#ffffff;font-weight:600">{label}</div>'
        html += f'<div style="text-align:right">'
        html += f'<div style="font-size:20px;font-weight:800;color:#ffffff;line-height:1.2">{disp_game}{arrow}</div>'
        html += f'<div style="font-size:11px;color:#ffffff;opacity:0.55;margin-top:2px">AVG: {disp_avg}</div>'
        html += '</div>'
        html += '</div>'
        html += '</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

# DRAW HELPERS (PITCH)
def _base_pitch(bg="#1a1a2e"):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=bg, line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor(bg)
    fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#ffffff", lw=1.2, alpha=0.40, linestyle="--")
    ax.axvline(x=HALF_LINE_X, color="#ffffff", lw=0.7, alpha=0.12, linestyle="--")
    return fig, ax, pitch

def _attack_arrow(fig, has_cbar=False):
    ox = -0.04 if has_cbar else 0.0
    fig.patches.append(FancyArrowPatch(
        (0.44 + ox, 0.045), (0.56 + ox, 0.045),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=11,
        linewidth=1.6, color="#aaaaaa"
    ))
    fig.text(0.50 + ox, 0.012, "Attacking Direction", ha="center", va="bottom",
             transform=fig.transFigure, fontsize=7.5, color="#aaaaaa")

def _save_fig(fig):
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)

def draw_pass_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        is_lost = not row["is_won"]
        is_prog = bool(row["progressive"])
        if is_lost:
            color, alpha = COLOR_FAIL, 0.72
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, 0.88
        else:
            color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS
        pitch.arrows(row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                     color=color, width=1.3, headwidth=2.0, headlength=2.0,
                     ax=ax, zorder=3, alpha=alpha)
        pitch.scatter(row["x_start"], row["y_start"], s=32, marker="o",
                      color=color, edgecolors="white", linewidths=0.6,
                      ax=ax, zorder=6, alpha=alpha)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0, label="Completed", alpha=0.65),
            Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressive", alpha=0.90),
            Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Incomplete", alpha=0.90),
        ],
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize=6.5, labelspacing=0.35, borderpad=0.4
    )
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_corridor_heatmap(df):
    df_s = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {
        "left": (LANE_LEFT_MIN, FIELD_Y),
        "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "right": (0.0, LANE_RIGHT_MAX),
    }
    counts = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            arr[i] = int(((df_s["x_end"] >= x0_) & (df_s["x_end"] < x1_) &
                          (df_s["y_end"] >= y0) & (df_s["y_end"] < y1)).sum())
        counts[cname] = arr
    all_vals = np.concatenate([counts[c] for c in counts])
    vmax = max(1, int(all_vals.max()))
    cmap = LinearSegmentedColormap.from_list("wr", ["#ffffff", "#ffecec", "#ffbfbf", "#ff8080", "#ff3b3b", "#ff0000"])
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax * 0.35)
    fig, ax, pitch = _base_pitch()
    for cname, (y0, y1) in corridors.items():
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            value = counts[cname][i]
            ax.add_patch(Rectangle((x0_, y0), x1_ - x0_, y1 - y0,
                                   facecolor=cmap(norm(value)),
                                   edgecolor=(1, 1, 1, 0.12), lw=0.5, alpha=0.95, zorder=2))
            ax.text((x0_ + x1_) / 2, (y0 + y1) / 2, str(value),
                    ha="center", va="center",
                    color="#000000" if value <= threshold else "#ffffff",
                    fontsize=9, fontweight="700" if value >= vmax * 0.5 else "600", zorder=4)
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def _draw_comet_arrow(ax, x0, y0, x1, y1, color):
    segs = 12
    ts = np.linspace(0.0, 1.0, segs + 1)
    for i in range(segs):
        t0, t1 = ts[i], ts[i + 1]
        xa = x0 + (x1 - x0) * t0; ya = y0 + (y1 - y0) * t0
        xb = x0 + (x1 - x0) * t1; yb = y0 + (y1 - y0) * t1
        alpha = 0.85 * (0.15 + 0.85 * t1)
        lw = 2.5 * (0.80 + 0.20 * t1)
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, alpha=alpha, zorder=4, solid_capstyle="round")
    ax.scatter(x0, y0, s=20, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, zorder=5, alpha=0.85)
    ax.scatter(x1, y1, s=32, marker="o", facecolors=color, edgecolors="white", linewidths=0.9, zorder=6, alpha=0.85)

def draw_top_xt_map(df, top_n=5):
    fig, ax, pitch = _base_pitch()
    top_passes = (df[(df["is_won"]) & (df["delta_xt_adj"] > 0)]
            .sort_values("delta_xt_adj", ascending=False).head(top_n).copy().reset_index(drop=True))
    if not top_passes.empty:
        for _, row in top_passes.iterrows():
            val = float(row["delta_xt_adj"])
            color = CMAP_TOP10(NORM_TOP10(np.clip(val, 0.05, 0.40)))
            _draw_comet_arrow(ax, float(row["x_start"]), float(row["y_start"]),
                              float(row["x_end"]), float(row["y_end"]), color)
        sm = plt.cm.ScalarMappable(cmap=CMAP_TOP10, norm=NORM_TOP10)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
        cbar.set_label("Pass Impact", color="#ffffff", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig

# DEFENSIVE PITCH DRAW HELPERS
COLOR_DUEL_WON = "#10b981"
COLOR_DUEL_LOST = "#E07070"
COLOR_INTERCEPTION = "#2F80ED"

def draw_defensive_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        if row["is_duel_won"]:
            color, marker, s, alpha = COLOR_DUEL_WON, "o", 90, 0.85
        elif row["is_duel_lost"]:
            color, marker, s, alpha = COLOR_DUEL_LOST, "X", 100, 0.85
        else:
            color, marker, s, alpha = COLOR_INTERCEPTION, "^", 80, 0.85
        pitch.scatter(row["x"], row["y"], s=s, marker=marker, color=color,
                      edgecolors="white", linewidths=0.8, ax=ax, zorder=6, alpha=alpha)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DUEL_WON, markersize=7, label="Duel Won", alpha=0.90),
            Line2D([0], [0], marker="X", color="w", markerfacecolor=COLOR_DUEL_LOST, markersize=8, label="Duel Lost", alpha=0.90),
            Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_INTERCEPTION, markersize=7, label="Interception", alpha=0.90),
        ],
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize=6.5, labelspacing=0.35, borderpad=0.4
    )
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_funnel_protection_map(df):
    """Map showing defensive actions: golden stars inside funnel zone, faded outside."""
    fig, ax, pitch = _base_pitch()
    funnel_rect = Rectangle(
        (0, PENALTY_AREA_Y_MIN), FUNNEL_X_EXTEND, PENALTY_AREA_Y_MAX - PENALTY_AREA_Y_MIN,
        facecolor="#ffd700", edgecolor="#ffd700", lw=1.5, linestyle="--", alpha=0.12, zorder=2
    )
    ax.add_patch(funnel_rect)
    for _, row in df.iterrows():
        x, y = float(row["x"]), float(row["y"])
        in_funnel = bool(row.get("in_funnel", is_in_funnel_zone(x, y)))
        if in_funnel:
            marker, s, color, edge = "*", 120, "#ffd700", "#b8860b"
        else:
            marker, s, color, edge = "o", 60, "#888888", "#555555"
        pitch.scatter(x, y, s=s, marker=marker, color=color,
                      edgecolors=edge, linewidths=0.5, ax=ax, zorder=6, alpha=0.85)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffd700", markersize=9, label="Funnel Action", alpha=0.95),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markersize=6, label="Other Action", alpha=0.50),
        ],
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize=6.5, labelspacing=0.35, borderpad=0.4
    )
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_defensive_heatmap(df):
    corridors = {
        "Left": (LANE_LEFT_MIN, FIELD_Y),
        "Center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "Right": (0.0, LANE_RIGHT_MAX),
    }
    corridor_data = {}
    for cname, (y0, y1) in corridors.items():
        mask = (df["y"] >= y0) & (df["y"] < y1)
        corr_df = df[mask]
        total = len(corr_df)
        duels_total = int(corr_df["is_duel"].sum())
        duels_won = int(corr_df["is_duel_won"].sum())
        corridor_data[cname] = {"count": total, "duels_won": duels_won, "duels_total": duels_total}
    all_counts = [d["count"] for d in corridor_data.values()]
    vmax = max(1, max(all_counts))
    # Blue ramp: light blue for few actions -> very dark blue for many actions
    cmap_def = LinearSegmentedColormap.from_list(
        "def_corr", ["#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8", "#162e7a", "#0a1840"]
    )
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax * 0.45)
    fig, ax, pitch = _base_pitch()
    for cname, (y0, y1) in corridors.items():
        d = corridor_data[cname]
        value = d["count"]
        ax.add_patch(Rectangle((0, y0), FIELD_X, y1 - y0,
                               facecolor=cmap_def(norm(value)),
                               edgecolor=(1, 1, 1, 0.15), lw=0.5, alpha=0.95, zorder=2))
        duel_pct = (d["duels_won"] / d["duels_total"] * 100) if d["duels_total"] > 0 else None
        if duel_pct is not None:
            label = f"{cname}\nTotal: {value}\nWon: {d['duels_won']}/{d['duels_total']} ({duel_pct:.0f}%)"
        else:
            label = f"{cname}\nTotal: {value}"
        ax.text(FIELD_X / 2, (y0 + y1) / 2, label,
                ha="center", va="center",
                color="#000000" if value <= threshold else "#ffffff",
                fontsize=9, fontweight="600", zorder=4)
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", lw=0.5, alpha=0.20, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.20, linestyle="--", zorder=3)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_defensive_xt_map(df):
    fig, ax, pitch = _base_pitch()
    vals = [xt_value(float(row["x"]), float(row["y"])) for _, row in df.iterrows()]
    vals = np.array(vals)
    if len(vals) > 0:
        vmin, vmax_vals = float(vals.min()), float(vals.max())
        norm_def = Normalize(vmin=vmin, vmax=vmax_vals) if vmax_vals > vmin else Normalize(vmin=0, vmax=1)
        cmap_def_xt = LinearSegmentedColormap.from_list("def_xt", ["#2d1b69", "#4a148c", "#7b1fa2", "#ab47bc", "#ce93d8"])
        for i, (_, row) in enumerate(df.iterrows()):
            marker, s = ("o", 85) if row["is_duel_won"] else ("X", 95) if row["is_duel_lost"] else ("^", 75)
            pitch.scatter(row["x"], row["y"], s=s, marker=marker,
                          color=cmap_def_xt(norm_def(vals[i])),
                          edgecolors="white", linewidths=0.6, ax=ax, zorder=6, alpha=0.85)
        sm = plt.cm.ScalarMappable(cmap=cmap_def_xt, norm=norm_def)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
        cbar.set_label("xT Threat", color="#ffffff", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig

# OFFENSIVE PITCH DRAW HELPERS
COLOR_OFF_DUEL_WON = "#10b981"
COLOR_OFF_DUEL_LOST = "#E07070"
COLOR_GOAL = "#ffd700"
COLOR_SHOT_ON = "#10b981"
COLOR_SHOT_OFF = "#E07070"

def draw_offensive_duels_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        if row["is_won"]:
            color, marker, s = COLOR_OFF_DUEL_WON, "o", 90
        else:
            color, marker, s = COLOR_OFF_DUEL_LOST, "X", 100
        pitch.scatter(row["x"], row["y"], s=s, marker=marker, color=color,
                      edgecolors="white", linewidths=0.8, ax=ax, zorder=6, alpha=0.85)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_OFF_DUEL_WON, markersize=7, label="Duel Won", alpha=0.90),
            Line2D([0], [0], marker="X", color="w", markerfacecolor=COLOR_OFF_DUEL_LOST, markersize=8, label="Duel Lost", alpha=0.90),
        ],
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize=6.5, labelspacing=0.35, borderpad=0.4
    )
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_touches_heatmap(df):
    fig, ax, pitch = _base_pitch()
    if len(df) > 0:
        cmap_touch = LinearSegmentedColormap.from_list(
            "touch", ["#1a1a2e", "#27306b", "#3b82f6", "#22d3ee", "#fde047"]
        )
        bin_stat = pitch.bin_statistic(df["x"].values, df["y"].values, statistic="count", bins=(12, 8))
        mesh = pitch.heatmap(bin_stat, ax=ax, cmap=cmap_touch, edgecolors="#1a1a2e", alpha=0.85, zorder=2)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
        cbar.set_label("Touches", color="#ffffff", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig

def draw_shots_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        if row["is_goal"]:
            color, marker, s = COLOR_GOAL, "*", 230
        elif row["is_on_target"]:
            color, marker, s = COLOR_SHOT_ON, "o", 95
        else:
            color, marker, s = COLOR_SHOT_OFF, "X", 95
        pitch.scatter(row["x"], row["y"], s=s, marker=marker, color=color,
                      edgecolors="white", linewidths=0.8, ax=ax, zorder=6, alpha=0.90)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], marker="*", color="w", markerfacecolor=COLOR_GOAL, markersize=11, label="Goal", alpha=0.95),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SHOT_ON, markersize=7, label="On Target", alpha=0.90),
            Line2D([0], [0], marker="X", color="w", markerfacecolor=COLOR_SHOT_OFF, markersize=8, label="Off Target", alpha=0.90),
        ],
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize=6.5, labelspacing=0.35, borderpad=0.4
    )
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

# PLOTLY CHARTS
def _avg_reference_traces(fig, x_labels, series, avg_mode, value_fmt):
    """Add a reference line: flat average or rolling moving average (same layout)."""
    if avg_mode == "Moving Average":
        roll = series.rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=x_labels, y=roll,
            mode='lines', line=dict(color="rgba(255, 215, 0, 0.35)", width=1.8, dash='dash'),
            name="Moving Avg (5)", hoverinfo='skip'
        ))
    else:
        mean_val = series.mean()
        fig.add_trace(go.Scatter(
            x=x_labels, y=[mean_val] * len(x_labels),
            mode='lines', line=dict(color="rgba(255, 215, 0, 0.25)", width=1.5, dash='dash'),
            name=f"Avg: {mean_val:{value_fmt}}", hoverinfo='skip'
        ))


GRADE_OPTIONS = {
    "Combined Grade": ("Grade", "#1a56db"),
    "Pass Grade": ("pass_grade", "#10b981"),
    "Defensive Grade": ("def_grade", "#a78bfa"),
    "Offensive Grade": ("off_grade", "#f59e0b"),
}


def draw_grade_chart(df_scores, grade_label="Combined Grade", avg_mode="Average"):
    col, color = GRADE_OPTIONS.get(grade_label, ("Grade", "#1a56db"))
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores[col]
    rgb = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    fig.add_trace(go.Scatter(
        x=x_labels, y=y,
        customdata=df_scores["match"],
        mode='lines+markers',
        line=dict(color=color, width=3, shape='spline'),
        marker=dict(size=8, color=color),
        fill='tozeroy', fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.05)',
        name=grade_label,
        hovertemplate="<span style='color:#ffd700'>%{customdata}</span><br>" + grade_label + ": %{y:.1f}"
    ))
    _avg_reference_traces(fig, x_labels, y, avg_mode, ".1f")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=370, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[40, 100], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text=grade_label, font=dict(size=14, color="#ffffff"))
    )
    return fig


STATS_METRICS = {
    "Total Passes": ("total_p90", "#00d2ff", ".1f", ""),
    "Advanced Passes": ("adv_p90", "#22d3ee", ".2f", ""),
    "Progressive Passes": ("prog_p90", "#10b981", ".2f", ""),
    "Final Third Passes": ("f3_p90", "#8b5cf6", ".2f", ""),
    "Pass Impact Value": ("xt_p90", "#f59e0b", ".3f", ""),
    "% Positive Impact": ("pos_pct", "#f43f5e", ".1f", "%"),
    "Defensive Duels": ("duels_p90", "#f97316", ".1f", ""),
    "Interceptions": ("interceptions_p90", "#8b5cf6", ".1f", ""),
    "Touches": ("touches_p90", "#38bdf8", ".1f", ""),
    "Offensive Duels": ("off_duels_p90", "#34d399", ".1f", ""),
    "Shots": ("shots_p90", "#fbbf24", ".2f", ""),
}


def draw_metric_chart(df_scores, metric_label="Total Passes", avg_mode="Average"):
    col, color, value_fmt, suffix = STATS_METRICS.get(metric_label, ("total_p90", "#00d2ff", ".1f", ""))
    fig = go.Figure()
    x_labels = [f"Match {i+1}" for i in range(len(df_scores))]
    y = df_scores[col]
    rgb = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    fig.add_trace(go.Scatter(
        x=x_labels, y=y,
        customdata=df_scores["match"],
        mode='lines+markers',
        line=dict(color=color, width=3, shape='spline'),
        marker=dict(size=8, color=color),
        fill='tozeroy', fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.05)',
        name=metric_label,
        hovertemplate="%{customdata}<br>" + metric_label + ": %{y:" + value_fmt + "}" + suffix
    ))
    _avg_reference_traces(fig, x_labels, y, avg_mode, value_fmt)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=320, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text=metric_label, font=dict(size=14, color="#a0a0b5"))
    )
    return fig

def draw_comparison_bar(title, val_first, val_last, suffix=""):
    color_last = "#10b981" if val_last >= val_first else "#E07070"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["First 10 Matches", "Last 10 Matches"],
        y=[val_first, val_last],
        marker_color=["#444466", color_last],
        text=[f"{val_first:.2f}{suffix}", f"{val_last:.2f}{suffix}"],
        textposition='auto',
        width=[0.35, 0.35],
        hovertemplate="%{x}<br>" + title + ": %{y:.2f}" + suffix
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        height=250, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, max(val_first, val_last) * 1.2],
                   showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        title=dict(text=title, font=dict(size=14, color="#a0a0b5")),
        showlegend=False, bargap=0.2
    )
    return fig

# SIDEBAR
st.sidebar.markdown("""
<div style="text-align:center;padding:8px 0 4px 0">
    <div style="font-size:20px;font-weight:300;letter-spacing:2px;color:#a0a0b5;text-transform:uppercase">Pass Stats</div>
    <div style="font-size:13px;font-weight:600;color:#ffffff;margin-top:-2px">Dashboard</div>
</div>
<div style="border-bottom:1px solid #2a2a3e;margin:6px 0 12px 0"></div>
<div style="font-size:11px;font-weight:500;letter-spacing:1px;color:#6b6b80;text-transform:uppercase;margin:0 10px 6px 10px">2026 Season</div>
<div style="font-size:16px;font-weight:600;color:#e0e0f0;margin:0 10px 12px 10px">Generic Player</div>
""", unsafe_allow_html=True)

img_path = "player_photo.png"
if os.path.exists(img_path):
    st.sidebar.image(img_path, use_container_width=True)

st.sidebar.markdown("""
<div style="border-bottom:1px solid #2a2a3e;margin:16px 0 8px 0"></div>
""", unsafe_allow_html=True)

MODERN_CARDS = st.sidebar.toggle("Caixinhas modernas", value=False, key="modern_cards")

num_matches = len(dfs_by_match)
all_match_stats = [compute_stats(dfs_by_match[m], m) for m in dfs_by_match]
offensive_all_stats = [
    compute_offensive_stats(touches_dfs_by_match[m], offensive_duels_dfs_by_match[m], shots_dfs_by_match[m], m)
    for m in touches_dfs_by_match
]
OFF_DICTS = (touches_dfs_by_match, offensive_duels_dfs_by_match, shots_dfs_by_match)

# TABS & LAYOUT
tab_graf, tab_dash, tab_evo = st.tabs(["Charts & Analysis", "Detailed Dashboard", "Development"])

with tab_graf:
    st.markdown("### Overall Performance Summary")
    if num_matches > 0:
        total_passes_all = sum(s['total_passes'] for s in all_match_stats)
        total_succ_all = sum(s['successful_passes'] for s in all_match_stats)
        total_prog_all = sum(s['progressive_successful'] for s in all_match_stats)
        total_f3_all = sum(s['to_final_third_success'] for s in all_match_stats)
        total_pos_all = sum(s['pos_count'] for s in all_match_stats)
        total_xt_all = sum(s['sum_dxt'] for s in all_match_stats)
        avg_acc = sum(s['accuracy_pct'] for s in all_match_stats) / num_matches
        avg_prog_p90 = sum(s['prog_p90'] for s in all_match_stats) / num_matches
        avg_f3_p90 = sum(s['f3_p90'] for s in all_match_stats) / num_matches
        avg_pos_pct = sum(s['pos_pct'] for s in all_match_stats) / num_matches
        avg_xt_p90 = sum(s['xt_p90'] for s in all_match_stats) / num_matches
        avg_total_p90 = sum(s['total_p90'] for s in all_match_stats) / num_matches
        total_adv_made_all = sum(s['adv_made'] for s in all_match_stats)
        total_adv_att_all = sum(s['adv_att'] for s in all_match_stats)
        avg_adv_p90 = sum(s['adv_p90'] for s in all_match_stats) / num_matches
        avg_adv_acc = sum(s['adv_acc_pct'] for s in all_match_stats) / num_matches

        st.markdown("### Passes")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            section_card("📋 Overview", PASS_TONES[0], [
                ("Passes p90", f"{avg_total_p90:.1f}", f"Total: {total_passes_all}"),
                ("Successful %", f"{avg_acc:.1f}%", f"Total: {total_succ_all}"),
            ])
        with col_s2:
            section_card("📊 Advanced", PASS_TONES[1], [
                ("Advanced Passes p90", f"{avg_adv_p90:.1f}", f"Total: {total_adv_made_all}",
                 "Sum of progressive and final-third passes"),
                ("Advanced Acc %", f"{avg_adv_acc:.1f}%", f"({total_adv_made_all}/{total_adv_att_all})",
                 "Completion rate of progressive + final-third passes"),
            ])
        with col_s3:
            section_card("⚡ Impact", PASS_TONES[2], [
                ("% Positive Impact", f"{avg_pos_pct:.1f}%", f"Total: {total_pos_all}",
                 "Passes that generated a positive impact based on where they ended on the field"),
                ("Pass Impact Value", f"{avg_xt_p90:.3f}", f"Total: {total_xt_all:.3f}",
                 "Calculation used to evaluate the offensive value added by a pass."),
            ])

        st.markdown("", unsafe_allow_html=True)

        defensive_num_matches = len(defensive_dfs_by_match)
        defensive_all_stats = [compute_defensive_stats(defensive_dfs_by_match[m], m) for m in defensive_dfs_by_match]
        if defensive_num_matches > 0:
            total_def_actions_all = sum(s['total_actions'] for s in defensive_all_stats)
            total_def_att_all = sum(s['actions_attacking'] for s in defensive_all_stats)
            total_duels_all = sum(s['total_duels'] for s in defensive_all_stats)
            total_duels_won_all = sum(s['duels_won'] for s in defensive_all_stats)
            total_interceptions_all = sum(s['interceptions'] for s in defensive_all_stats)
            total_int_att_all = sum(s['interceptions_attacking'] for s in defensive_all_stats)
            avg_def_actions_p90 = sum(s['total_actions_p90'] for s in defensive_all_stats) / defensive_num_matches
            avg_def_att_p90 = sum(s['actions_attacking_p90'] for s in defensive_all_stats) / defensive_num_matches
            avg_duels_p90 = sum(s['duels_p90'] for s in defensive_all_stats) / defensive_num_matches
            avg_duels_won_pct = sum(s['duels_won_pct'] for s in defensive_all_stats) / defensive_num_matches
            avg_interceptions_p90 = sum(s['interceptions_p90'] for s in defensive_all_stats) / defensive_num_matches
            avg_int_att_p90 = sum(s['interceptions_attacking_p90'] for s in defensive_all_stats) / defensive_num_matches

            st.markdown("### Defensive Actions")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                section_card("🛡️ General", DEF_TONES[0], [
                    ("Defensive Actions p90", f"{avg_def_actions_p90:.1f}", f"Total: {total_def_actions_all}"),
                    ("Actions in Opp. Field p90", f"{avg_def_att_p90:.1f}", f"Total: {total_def_att_all}"),
                ])
            with col_d2:
                section_card("⚔️ Duels", DEF_TONES[1], [
                    ("Defensive Duels p90", f"{avg_duels_p90:.1f}", f"Total: {total_duels_all}"),
                    ("% Duels Won", f"{avg_duels_won_pct:.1f}%", f"({total_duels_won_all}/{total_duels_all})"),
                ])
            with col_d3:
                section_card("❌ Interceptions", DEF_TONES[2], [
                    ("Interceptions p90", f"{avg_interceptions_p90:.1f}", f"Total: {total_interceptions_all}"),
                    ("Interceptions in Opp Field p90", f"{avg_int_att_p90:.1f}", f"Total: {total_int_att_all}"),
                ])

        offensive_num_matches = len(touches_dfs_by_match)
        if offensive_num_matches > 0:
            total_touches_all = sum(s['touches'] for s in offensive_all_stats)
            total_f3_touches_all = sum(s['f3_touches'] for s in offensive_all_stats)
            total_off_duels_all = sum(s['off_duels'] for s in offensive_all_stats)
            total_off_duels_won_all = sum(s['off_duels_won'] for s in offensive_all_stats)
            total_shots_all = sum(s['shots'] for s in offensive_all_stats)
            total_goals_all = sum(s['goals'] for s in offensive_all_stats)
            total_on_target_all = sum(s['shots_on_target'] for s in offensive_all_stats)
            avg_touches_p90 = sum(s['touches_p90'] for s in offensive_all_stats) / offensive_num_matches
            avg_f3_touches_p90 = sum(s['f3_touches_p90'] for s in offensive_all_stats) / offensive_num_matches
            avg_off_duels_p90 = sum(s['off_duels_p90'] for s in offensive_all_stats) / offensive_num_matches
            avg_off_duels_won_pct = sum(s['off_duels_won_pct'] for s in offensive_all_stats) / offensive_num_matches
            avg_shots_p90 = sum(s['shots_p90'] for s in offensive_all_stats) / offensive_num_matches

            st.markdown("### Offensive Actions")
            col_o1, col_o2, col_o3 = st.columns(3)
            with col_o1:
                section_card("📋 Overview", OFF_TONES[0], [
                    ("Touches p90", f"{avg_touches_p90:.1f}", f"Total: {total_touches_all}"),
                    ("Final Third Touches p90", f"{avg_f3_touches_p90:.1f}", f"Total: {total_f3_touches_all}"),
                ])
            with col_o2:
                section_card("⚔️ Offensive Duels", OFF_TONES[1], [
                    ("Offensive Duels p90", f"{avg_off_duels_p90:.1f}", f"Total: {total_off_duels_all}"),
                    ("% Duels Won", f"{avg_off_duels_won_pct:.1f}%", f"({total_off_duels_won_all}/{total_off_duels_all})"),
                ])
            with col_o3:
                section_card("🥅 Shots", OFF_TONES[2], [
                    ("Shots p90", f"{avg_shots_p90:.2f}", f"Total: {total_shots_all}"),
                    ("Goals", f"{total_goals_all}", f"On Target: {total_on_target_all}"),
                ])

        st.markdown(f'<div style="text-align:center;font-size:12px;color:#666666;margin-top:12px">{num_matches} matches collected</div>', unsafe_allow_html=True)
        st.markdown("", unsafe_allow_html=True)

        df_scores = compute_match_scores(dfs_by_match, defensive_dfs_by_match, OFF_DICTS)
        if not df_scores.empty:
            st.markdown("### Grade per Match")
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                grade_choice = st.radio(
                    "Grade", list(GRADE_OPTIONS.keys()),
                    index=0, horizontal=True, key="grade_choice"
                )
            with gcol2:
                grade_avg_mode = st.radio(
                    "Reference Line", ["Average", "Moving Average"],
                    index=0, horizontal=True, key="grade_avg_mode"
                )
            fig_scores = draw_grade_chart(df_scores, grade_choice, grade_avg_mode)
            st.plotly_chart(fig_scores, use_container_width=True)

            with st.expander("How is the Grade calculated?"):
                st.markdown("""
**Combined Grade** = average (equal weights) of **Pass Grade**, **Defensive Grade** and **Offensive Grade**.

**Pass Grade**
- **Pass Impact:** Measures actual danger created by passes.
- **Progressive Passes:** Line-breaking ability.
- **Final Third Passes:** Attacking presence in dangerous zones.
- **% Positive Pass Impact:** Efficiency of threat generation.
- **Total Passes:** Overall involvement.
- **Negative Pass Impact:** Penalty for passes that lose threat.

**Defensive Grade**
- **Duels Won %:** Rewards efficiency in defensive duels.
- **Funnel Defensive Actions:** Rewards actions in the defensive funnel zone.
- **Interception xT:** Rewards interceptions in high-threat zones.
- **Duels Won Count:** Rewards volume of duels won.
- **Interceptions Count:** Rewards volume of interceptions.

**Offensive Grade**
- **Touches:** Overall offensive involvement.
- **Final Third Touches:** Presence in dangerous areas.
- **Offensive Duels Won %:** Efficiency in 1v1 attacking duels.
- **Shots:** Shot volume / threat generated.
- **Finishing (Shots on Target %):** Shooting accuracy.
""")

            st.markdown("", unsafe_allow_html=True)
            st.markdown("### Stats")
            scol1, scol2 = st.columns(2)
            with scol1:
                metric_choice = st.selectbox(
                    "Metric", list(STATS_METRICS.keys()), index=0, key="stats_metric"
                )
            with scol2:
                stats_avg_mode = st.radio(
                    "Reference Line", ["Average", "Moving Average"],
                    index=0, horizontal=True, key="stats_avg_mode"
                )
            fig_metric = draw_metric_chart(df_scores, metric_choice, stats_avg_mode)
            st.plotly_chart(fig_metric, use_container_width=True)

with tab_dash:
    sub_tab_passes, sub_tab_def, sub_tab_off = st.tabs(["Passes", "Defensive Actions", "Offensive Actions"])

    with sub_tab_passes:
        st.markdown("### Match Filters")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            pass_match_options = ["All Matches"] + list(dfs_by_match.keys())
            selected_match = st.selectbox("Select Match", options=pass_match_options, index=0, key="pass_match")
        with col_f2:
            pass_filter = st.radio(
                "Pass Type",
                ["All", "Successful", "Unsuccessful", "Progressive", "Final Third"],
                index=0, horizontal=True, key="pass_filter"
            )

        if selected_match == "All Matches":
            df_game_filtered = pd.concat(dfs_by_match.values(), ignore_index=True)
            match_name_for_stats = "All Matches"
        else:
            df_game_filtered = dfs_by_match[selected_match].copy()
            match_name_for_stats = selected_match

        def apply_filter(df):
            if pass_filter == "Successful":
                return df[df["is_won"]].copy()
            if pass_filter == "Unsuccessful":
                return df[~df["is_won"]].copy()
            if pass_filter == "Progressive":
                return df[df["progressive"]].copy()
            if pass_filter == "Final Third":
                return df[(df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)].copy()
            return df.copy()

        df_game = apply_filter(df_game_filtered)
        s_game = compute_stats(df_game, match_name_for_stats)
        s_avg = {}
        if num_matches > 0:
            for k in all_match_stats[0].keys():
                if isinstance(all_match_stats[0][k], (int, float)):
                    s_avg[k] = sum(s[k] for s in all_match_stats) / num_matches
                else:
                    s_avg[k] = 0
        else:
            s_avg = s_game.copy()

        force_avg = selected_match == "All Matches"
        if force_avg:
            s_game = s_avg.copy()

        st.markdown("---")
        img_pm_game, fig_pm_game = draw_pass_map(df_game); plt.close(fig_pm_game)
        img_ht_game, fig_ht_game = draw_corridor_heatmap(df_game); plt.close(fig_ht_game)
        top_n_xt = 10 if force_avg else 5
        img_xt_game, fig_xt_game = draw_top_xt_map(df_game, top_n=top_n_xt); plt.close(fig_xt_game)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Pass Map</div>', unsafe_allow_html=True)
            st.image(img_pm_game, use_container_width=True)
        with col_m2:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Zone Heatmap</div>', unsafe_allow_html=True)
            st.image(img_ht_game, use_container_width=True)
        with col_m3:
            label = "Top 10" if force_avg else "Top 5"
            st.markdown(f'<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">{label} Pass Impact</div>', unsafe_allow_html=True)
            st.image(img_xt_game, use_container_width=True)

        st.markdown("", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        if force_avg:
            with col_s1:
                section_card("📋 Pass Overview", PASS_TONES[0], [
                    ("Total Passes", f"{s_game['total_p90']:.2f}"),
                    ("Successful %", f"{s_game['accuracy_pct']:.2f}%"),
                ])
            with col_s2:
                section_card("📊 Advanced", PASS_TONES[1], [
                    ("Advanced Passes", f"{s_game['adv_p90']:.2f}"),
                    ("Advanced Acc %", f"{s_game['adv_acc_pct']:.2f}%"),
                ])
            with col_s3:
                section_card("⚡ Impact", PASS_TONES[2], [
                    ("% Positive Impact", f"{s_game['pos_pct']:.2f}%"),
                    ("Pass Impact Value", f"{s_game['xt_p90']:.3f}"),
                ])
        else:
            with col_s1:
                cmp_section_card("📋 Pass Overview", PASS_TONES[0], [
                    ("Total Passes", s_game["total_p90"], f"{s_avg['total_p90']:.1f}"),
                    ("Successful %", s_game["accuracy_pct"], s_avg["accuracy_pct"],
                     f"{s_game['accuracy_pct']:.1f}%", f"{s_avg['accuracy_pct']:.1f}%"),
                ])
            with col_s2:
                cmp_section_card("📊 Advanced", PASS_TONES[1], [
                    ("Advanced Passes", s_game["adv_p90"], f"{s_avg['adv_p90']:.1f}"),
                    ("Advanced Acc %", s_game["adv_acc_pct"], s_avg["adv_acc_pct"],
                     f"{s_game['adv_acc_pct']:.1f}%", f"{s_avg['adv_acc_pct']:.1f}%"),
                ])
            with col_s3:
                cmp_section_card("⚡ Impact", PASS_TONES[2], [
                    ("% Positive Impact", s_game["pos_pct"], s_avg["pos_pct"],
                     f"{s_game['pos_pct']:.1f}%", f"{s_avg['pos_pct']:.1f}%",
                     "Passes that generated a positive impact based on where they ended on the field"),
                    ("Pass Impact Value", s_game["xt_p90"], s_avg["xt_p90"],
                     f"{s_game['xt_p90']:.3f}", f"{s_avg['xt_p90']:.3f}",
                     "Calculation used to define the value of pass impact based on expected threat (xT) progression"),
                ])

    with sub_tab_def:
        st.markdown("### Match Filter")
        col_df1, col_df2 = st.columns(2)
        with col_df1:
            def_match_options = ["All Matches"] + list(defensive_dfs_by_match.keys())
            selected_def_match = st.selectbox("Select Match", options=def_match_options, index=0, key="def_match")
        with col_df2:
            def_type_filter = st.radio("Filter Type", ["All", "Duels Only", "Interceptions Only"],
                                       horizontal=True, key="def_type_filter")

        if selected_def_match == "All Matches":
            df_def_game_raw = pd.concat(defensive_dfs_by_match.values(), ignore_index=True)
            def_match_name_for_stats = "All Matches"
        else:
            df_def_game_raw = defensive_dfs_by_match[selected_def_match].copy()
            def_match_name_for_stats = selected_def_match

        if def_type_filter == "Duels Only":
            df_def_game = df_def_game_raw[df_def_game_raw["is_duel"]].copy()
        elif def_type_filter == "Interceptions Only":
            df_def_game = df_def_game_raw[df_def_game_raw["is_interception"]].copy()
        else:
            df_def_game = df_def_game_raw.copy()

        d_game = compute_defensive_stats(df_def_game, def_match_name_for_stats)
        def_all = [compute_defensive_stats(defensive_dfs_by_match[m], m) for m in defensive_dfs_by_match]
        d_avg = {}
        if len(def_all) > 0:
            for k in def_all[0].keys():
                if isinstance(def_all[0][k], (int, float)):
                    d_avg[k] = sum(s[k] for s in def_all) / len(def_all)
                else:
                    d_avg[k] = 0
        else:
            d_avg = d_game.copy()

        force_avg_def = selected_def_match == "All Matches"
        if force_avg_def:
            d_game = d_avg.copy()

        st.markdown("---")
        img_def_map, fig_def_map = draw_defensive_map(df_def_game); plt.close(fig_def_map)
        img_def_hm, fig_def_hm = draw_defensive_heatmap(df_def_game); plt.close(fig_def_hm)
        img_funnel, fig_funnel = draw_funnel_protection_map(df_def_game); plt.close(fig_funnel)

        col_dm1, col_dm2, col_dm3 = st.columns(3)
        with col_dm1:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Defensive Actions Map</div>', unsafe_allow_html=True)
            st.image(img_def_map, use_container_width=True)
        with col_dm2:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Defensive Heatmap</div>', unsafe_allow_html=True)
            st.image(img_def_hm, use_container_width=True)
        with col_dm3:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Funnel Protection Actions</div>', unsafe_allow_html=True)
            st.image(img_funnel, use_container_width=True)

        st.markdown("", unsafe_allow_html=True)
        col_ds1, col_ds2, col_ds3 = st.columns(3)
        if force_avg_def:
            with col_ds1:
                section_card("🛡️ General", DEF_TONES[0], [
                    ("Defensive Actions", f"{d_game['total_actions_p90']:.2f}"),
                    ("Actions in Opp. Field", f"{d_game['actions_attacking_p90']:.2f}"),
                ])
            with col_ds2:
                section_card("⚔️ Duels", DEF_TONES[1], [
                    ("Defensive Duels", f"{d_game['duels_p90']:.2f}"),
                    ("% Duels Won", f"{d_game['duels_won_pct']:.2f}%"),
                ])
            with col_ds3:
                section_card("👁️ Interceptions", DEF_TONES[2], [
                    ("Interceptions", f"{d_game['interceptions_p90']:.2f}"),
                    ("Interceptions in Opp Field", f"{d_game['interceptions_attacking_p90']:.2f}"),
                ])
        else:
            with col_ds1:
                cmp_section_card("🛡️ General", DEF_TONES[0], [
                    ("Defensive Actions", d_game["total_actions_p90"], f"{d_avg['total_actions_p90']:.1f}"),
                    ("Actions in Opp. Field", d_game["actions_attacking_p90"], f"{d_avg['actions_attacking_p90']:.1f}"),
                ])
            with col_ds2:
                cmp_section_card("⚔️ Duels", DEF_TONES[1], [
                    ("Defensive Duels", d_game["duels_p90"], f"{d_avg['duels_p90']:.1f}"),
                    ("% Duels Won", d_game["duels_won_pct"], d_avg["duels_won_pct"],
                     f"{d_game['duels_won_pct']:.1f}%", f"{d_avg['duels_won_pct']:.1f}%"),
                ])
            with col_ds3:
                cmp_section_card("👁️ Interceptions", DEF_TONES[2], [
                    ("Interceptions", d_game["interceptions_p90"], f"{d_avg['interceptions_p90']:.1f}"),
                    ("Interceptions in Opp Field", d_game["interceptions_attacking_p90"], f"{d_avg['interceptions_attacking_p90']:.1f}"),
                ])

    with sub_tab_off:
        st.markdown("### Match Filter")
        col_of1, col_of2 = st.columns(2)
        with col_of1:
            off_match_options = ["All Matches"] + list(touches_dfs_by_match.keys())
            selected_off_match = st.selectbox("Select Match", options=off_match_options, index=0, key="off_match")

        if selected_off_match == "All Matches":
            touches_game = pd.concat(touches_dfs_by_match.values(), ignore_index=True)
            duels_game = pd.concat(offensive_duels_dfs_by_match.values(), ignore_index=True)
            shots_game = pd.concat(shots_dfs_by_match.values(), ignore_index=True)
            off_match_name_for_stats = "All Matches"
        else:
            touches_game = touches_dfs_by_match[selected_off_match].copy()
            duels_game = offensive_duels_dfs_by_match[selected_off_match].copy()
            shots_game = shots_dfs_by_match[selected_off_match].copy()
            off_match_name_for_stats = selected_off_match

        o_game = compute_offensive_stats(touches_game, duels_game, shots_game, off_match_name_for_stats)
        o_avg = {}
        if len(offensive_all_stats) > 0:
            for k in offensive_all_stats[0].keys():
                if isinstance(offensive_all_stats[0][k], (int, float)):
                    o_avg[k] = sum(s[k] for s in offensive_all_stats) / len(offensive_all_stats)
                else:
                    o_avg[k] = 0
        else:
            o_avg = o_game.copy()

        force_avg_off = selected_off_match == "All Matches"
        if force_avg_off:
            o_game = o_avg.copy()

        st.markdown("---")
        img_od_game, fig_od_game = draw_offensive_duels_map(duels_game); plt.close(fig_od_game)
        img_th_game, fig_th_game = draw_touches_heatmap(touches_game); plt.close(fig_th_game)
        img_sh_game, fig_sh_game = draw_shots_map(shots_game); plt.close(fig_sh_game)

        col_om1, col_om2, col_om3 = st.columns(3)
        with col_om1:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Offensive Duels Map</div>', unsafe_allow_html=True)
            st.image(img_od_game, use_container_width=True)
        with col_om2:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Touches Heatmap</div>', unsafe_allow_html=True)
            st.image(img_th_game, use_container_width=True)
        with col_om3:
            st.markdown('<div style="text-align:center;font-weight:600;font-size:14px;margin-bottom:6px;color:#cccccc">Shots Map</div>', unsafe_allow_html=True)
            st.image(img_sh_game, use_container_width=True)

        st.markdown("", unsafe_allow_html=True)
        col_os1, col_os2, col_os3 = st.columns(3)
        if force_avg_off:
            with col_os1:
                section_card("📋 Overview", OFF_TONES[0], [
                    ("Touches", f"{o_game['touches_p90']:.2f}"),
                    ("Final Third Touches", f"{o_game['f3_touches_p90']:.2f}"),
                ])
            with col_os2:
                section_card("⚔️ Offensive Duels", OFF_TONES[1], [
                    ("Offensive Duels", f"{o_game['off_duels_p90']:.2f}"),
                    ("% Duels Won", f"{o_game['off_duels_won_pct']:.2f}%"),
                ])
            with col_os3:
                section_card("🥅 Shots", OFF_TONES[2], [
                    ("Shots", f"{o_game['shots_p90']:.2f}"),
                    ("Goals", f"{o_game['goals']:.2f}"),
                ])
        else:
            with col_os1:
                cmp_section_card("📋 Overview", OFF_TONES[0], [
                    ("Touches", o_game["touches_p90"], f"{o_avg['touches_p90']:.1f}"),
                    ("Final Third Touches", o_game["f3_touches_p90"], f"{o_avg['f3_touches_p90']:.1f}"),
                ])
            with col_os2:
                cmp_section_card("⚔️ Offensive Duels", OFF_TONES[1], [
                    ("Offensive Duels", o_game["off_duels_p90"], f"{o_avg['off_duels_p90']:.1f}"),
                    ("% Duels Won", o_game["off_duels_won_pct"], o_avg["off_duels_won_pct"],
                     f"{o_game['off_duels_won_pct']:.1f}%", f"{o_avg['off_duels_won_pct']:.1f}%"),
                ])
            with col_os3:
                cmp_section_card("🥅 Shots", OFF_TONES[2], [
                    ("Shots", o_game["shots_p90"], f"{o_avg['shots_p90']:.1f}"),
                    ("Goals", o_game["goals"], o_avg["goals"],
                     f"{o_game['goals']:.0f}", f"{o_avg['goals']:.2f}"),
                ])

with tab_evo:
    sub_tab_evo_passes, sub_tab_evo_def, sub_tab_evo_off = st.tabs(["Passes", "Defensive Actions", "Offensive Actions"])

    with sub_tab_evo_passes:
        st.markdown("### First 10 vs Last 10 Matches")
        st.markdown("Comparing the average passing performance between the first 10 and the last 10 matches to analyze player evolution.")
        df_scores = compute_match_scores(dfs_by_match, defensive_dfs_by_match, OFF_DICTS)
        if len(df_scores) > 0:
            if len(df_scores) < 20:
                st.info(f"Note: Only {len(df_scores)} matches available. The comparison will overlap or use available data.")
            first_10 = df_scores.head(10)
            last_10 = df_scores.tail(10)

            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                fig_grade = draw_comparison_bar("Pass Grade", first_10["Grade"].mean(), last_10["Grade"].mean())
                st.plotly_chart(fig_grade, use_container_width=True)
            with r1c2:
                fig_xt_evo = draw_comparison_bar("Σ Pass Impact", first_10["xt_p90"].mean(), last_10["xt_p90"].mean())
                st.plotly_chart(fig_xt_evo, use_container_width=True)
            with r1c3:
                fig_high_xt_evo = draw_comparison_bar("High Impact Passes", first_10["high_xt_p90"].mean(), last_10["high_xt_p90"].mean())
                st.plotly_chart(fig_high_xt_evo, use_container_width=True)

            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                fig_prog_evo = draw_comparison_bar("Progressive Passes", first_10["prog_p90"].mean(), last_10["prog_p90"].mean())
                st.plotly_chart(fig_prog_evo, use_container_width=True)
            with r2c2:
                fig_f3_evo = draw_comparison_bar("Final Third Passes", first_10["f3_p90"].mean(), last_10["f3_p90"].mean())
                st.plotly_chart(fig_f3_evo, use_container_width=True)
            with r2c3:
                fig_dz_evo = draw_comparison_bar("Dangerous Zone Passes", first_10["dz_p90"].mean(), last_10["dz_p90"].mean())
                st.plotly_chart(fig_dz_evo, use_container_width=True)

            r3c1, r3c2, r3c3 = st.columns(3)
            with r3c1:
                fig_acc_evo = draw_comparison_bar("Successful Passes %", first_10["accuracy_pct"].mean(), last_10["accuracy_pct"].mean(), suffix="%")
                st.plotly_chart(fig_acc_evo, use_container_width=True)
            with r3c2:
                fig_long_evo = draw_comparison_bar("Long Pass Accuracy %", first_10["long_acc_pct"].mean(), last_10["long_acc_pct"].mean(), suffix="%")
                st.plotly_chart(fig_long_evo, use_container_width=True)
            with r3c3:
                fig_prog_acc_evo = draw_comparison_bar("Progressive Accuracy %", first_10["prog_acc_pct"].mean(), last_10["prog_acc_pct"].mean(), suffix="%")
                st.plotly_chart(fig_prog_acc_evo, use_container_width=True)
        else:
            st.warning("Not enough data to generate evolution charts.")

    with sub_tab_evo_def:
        st.markdown("### First 10 vs Last 10 Matches")
        st.markdown("Comparing the average defensive performance between the first 10 and the last 10 matches.")
        df_scores = compute_match_scores(dfs_by_match, defensive_dfs_by_match, OFF_DICTS)
        df_def_evo = compute_defensive_evolution_df(defensive_dfs_by_match, df_scores)
        if len(df_def_evo) > 0:
            if len(df_def_evo) < 20:
                st.info(f"Note: Only {len(df_def_evo)} matches available. The comparison will overlap or use available data.")
            first_10_def = df_def_evo.head(10)
            last_10_def = df_def_evo.tail(10)

            rd1c1, rd1c2, rd1c3 = st.columns(3)
            with rd1c1:
                g1 = first_10_def["grade"].dropna()
                g2 = last_10_def["grade"].dropna()
                v1 = g1.mean() if len(g1) > 0 else 0
                v2 = g2.mean() if len(g2) > 0 else 0
                fig_grade_def = draw_comparison_bar("Defensive Actions Grade", v1, v2)
                st.plotly_chart(fig_grade_def, use_container_width=True)
            with rd1c2:
                fig_duels_won = draw_comparison_bar("Duels Won p90", first_10_def["duels_won_p90"].mean(), last_10_def["duels_won_p90"].mean())
                st.plotly_chart(fig_duels_won, use_container_width=True)
            with rd1c3:
                fig_duels_pct = draw_comparison_bar("% Duels Won", first_10_def["duels_won_pct"].mean(), last_10_def["duels_won_pct"].mean(), suffix="%")
                st.plotly_chart(fig_duels_pct, use_container_width=True)

            rd2c1, rd2c2, rd2c3 = st.columns(3)
            with rd2c1:
                fig_def_act = draw_comparison_bar("Defensive Actions p90", first_10_def["def_actions_p90"].mean(), last_10_def["def_actions_p90"].mean())
                st.plotly_chart(fig_def_act, use_container_width=True)
            with rd2c2:
                fig_int_evo = draw_comparison_bar("Interceptions p90", first_10_def["interceptions_p90"].mean(), last_10_def["interceptions_p90"].mean())
                st.plotly_chart(fig_int_evo, use_container_width=True)
            with rd2c3:
                fig_funnel_evo = draw_comparison_bar("Funnel Actions p90", first_10_def["funnel_actions_p90"].mean(), last_10_def["funnel_actions_p90"].mean())
                st.plotly_chart(fig_funnel_evo, use_container_width=True)
        else:
            st.warning("Not enough data to generate defensive evolution charts.")

    with sub_tab_evo_off:
        st.markdown("### First 10 vs Last 10 Matches")
        st.markdown("Comparing the average offensive performance between the first 10 and the last 10 matches.")
        df_scores = compute_match_scores(dfs_by_match, defensive_dfs_by_match, OFF_DICTS)
        if len(df_scores) > 0:
            if len(df_scores) < 20:
                st.info(f"Note: Only {len(df_scores)} matches available. The comparison will overlap or use available data.")
            first_10_off = df_scores.head(10)
            last_10_off = df_scores.tail(10)

            ro1c1, ro1c2, ro1c3 = st.columns(3)
            with ro1c1:
                fig_off_grade = draw_comparison_bar("Offensive Grade", first_10_off["off_grade"].mean(), last_10_off["off_grade"].mean())
                st.plotly_chart(fig_off_grade, use_container_width=True)
            with ro1c2:
                fig_touches_evo = draw_comparison_bar("Touches p90", first_10_off["touches_p90"].mean(), last_10_off["touches_p90"].mean())
                st.plotly_chart(fig_touches_evo, use_container_width=True)
            with ro1c3:
                fig_f3touch_evo = draw_comparison_bar("Final Third Touches p90", first_10_off["f3_touches_p90"].mean(), last_10_off["f3_touches_p90"].mean())
                st.plotly_chart(fig_f3touch_evo, use_container_width=True)

            ro2c1, ro2c2, ro2c3 = st.columns(3)
            with ro2c1:
                fig_offduels_evo = draw_comparison_bar("Offensive Duels p90", first_10_off["off_duels_p90"].mean(), last_10_off["off_duels_p90"].mean())
                st.plotly_chart(fig_offduels_evo, use_container_width=True)
            with ro2c2:
                fig_offduels_pct_evo = draw_comparison_bar("% Off. Duels Won", first_10_off["off_duels_won_pct"].mean(), last_10_off["off_duels_won_pct"].mean(), suffix="%")
                st.plotly_chart(fig_offduels_pct_evo, use_container_width=True)
            with ro2c3:
                fig_shots_evo = draw_comparison_bar("Shots p90", first_10_off["shots_p90"].mean(), last_10_off["shots_p90"].mean())
                st.plotly_chart(fig_shots_evo, use_container_width=True)
        else:
            st.warning("Not enough data to generate offensive evolution charts.")
