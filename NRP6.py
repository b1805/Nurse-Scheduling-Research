from __future__ import annotations

import copy
import itertools
import math
import os
import pickle
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import sys
from datetime import datetime
from contextlib import contextmanager

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False
import numpy as np
import pandas as pd
import pulp
import streamlit as st

EPS = 1e-6
CG_RC_TOL = 1e-5
DEFAULT_SURGE_FRACTION_LOW = 0.30
DEFAULT_SURGE_FRACTION_HIGH = 0.50
DEFAULT_SURGE_UNDER_PENALTY = 100.0

# CG/heuristic have their own short end-to-end cap, independent of the direct MIP.
HYBRID_MAX_SECONDS = 300.0
CG_MAX_SECONDS = 225.0
CG_TARGET_COLUMNS = 10
CG_MAX_COLUMNS_PER_NURSE = 30
CG_MIN_RMP_ROUNDS = 3
CG_STAGNATION_LIMIT = 100
HEURISTIC_STAGNATION_LIMIT = 50_000

DEFAULT_PRICING_TIME_CAP = 0.15
DEFAULT_PRICING_TIME_CAP_SMALL = 1.00
DEFAULT_PRICING_BEAM_WIDTH = 3500
DEFAULT_PRICING_BEAM_WIDTH_LARGE = 1200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
P_SCORE_TABLE_PATH = os.path.join(BASE_DIR, "P_Score_Table.pkl")
_ACTIVE_RUN_LOG = None
_NATIVE_RMP_LOGS = []

def require_gurobi():
    if not GUROBI_AVAILABLE:
        raise RuntimeError(
            "Gurobi is required for the Restricted Master Problem. "
            "Install gurobipy and provide a valid Gurobi license before running CG/ALNS."
        )

def weekend_indices(horizon: int) -> Dict[int, List[int]]:
    weekends: Dict[int, List[int]] = defaultdict(list)
    for d in range(horizon):
        if d % 7 in (5, 6):
            weekends[d // 7].append(d)
    return dict(weekends)

def infer_fatigue_type(shift_name: str) -> str:
    if shift_name == "OFF":
        return "OFF"

    u = shift_name.strip().upper()
    if u in {"N", "NIGHT", "NIGHTSHIFT", "NS", "L", "LATE", "LN"} or "NIGHT" in u or "LATE" in u:
        return "N"
    if u in {"E", "EVENING", "EVE", "ES"} or "EVENING" in u:
        return "E"
    return "D"

def fatigue_type(shift_name: str, shifts: Mapping[str, Mapping]) -> str:
    if shift_name == "OFF":
        return "OFF"
    info = shifts.get(shift_name, {})
    return info.get("fatigue_type", infer_fatigue_type(shift_name))

def find_night_shifts(shifts: Mapping[str, Mapping]) -> Tuple[str, ...]:
    return tuple(
        s for s, info in shifts.items()
        if info.get("fatigue_type", infer_fatigue_type(s)) == "N"
    )

def safe_copy_instance(inst: Mapping) -> dict:
    return copy.deepcopy(dict(inst))

@lru_cache(maxsize=4)
def load_fatigue_table(path: str = P_SCORE_TABLE_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required fatigue lookup table '{path}' was not found. "
            "Generate the Klyve-based P_Score_Table.pkl before enabling fatigue."
        )
    with open(path, "rb") as f:
        table = pickle.load(f)
    if not isinstance(table, dict) or not table:
        raise ValueError(f"Fatigue lookup table '{path}' is empty or invalid.")
    return table

def normalize_request_lists(inst: dict) -> dict:
    inst["shift_on_requests"] = [
        (e, int(d), s, float(w)) for e, d, s, w in inst.get("shift_on_requests", [])
    ]
    inst["shift_off_requests"] = [
        (e, int(d), s, float(w)) for e, d, s, w in inst.get("shift_off_requests", [])
    ]
    return inst

def parse_standard_nrp(text: str) -> dict:
    lines = [
        ln.replace("\ufeff", "").strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.startswith(("#", "//"))
    ]

    data: Dict[str, List[str]] = defaultdict(list)
    section = None
    for ln in lines:
        if ln.startswith("SECTION_"):
            section = ln[len("SECTION_") :].strip()
            continue
        if section is not None:
            data[section].append(ln)

    horizon = int(data["HORIZON"][0]) if data.get("HORIZON") else 14

    shifts: Dict[str, dict] = {}
    for ln in data.get("SHIFTS", []):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            raise ValueError(f"Invalid SHIFT line: {ln}")
        name = parts[0]
        forbidden = [x for x in parts[2].split("|") if x] if len(parts) > 2 else []
        ftype = parts[3].upper() if len(parts) > 3 and parts[3] else infer_fatigue_type(name)
        if ftype not in {"D", "E", "N"}:
            raise ValueError(f"Invalid fatigue type '{ftype}' in SHIFT line: {ln}")
        shifts[name] = {
            "length": int(parts[1]),
            "forbid_next": forbidden,
            "fatigue_type": ftype,
        }

    staff: Dict[str, dict] = {}
    for ln in data.get("STAFF", []):
        parts = [p.strip() for p in ln.split(",")]
        if not parts:
            continue
        eid = parts[0]
        maxshift_per_shift: Dict[str, int] = {}
        for token in (parts[1] if len(parts) > 1 else "").split("|"):
            if "=" in token:
                shiftid, val = token.split("=", 1)
                maxshift_per_shift[shiftid.strip()] = int(val.strip())

        staff[eid] = {
            "maxshift_per_shift": maxshift_per_shift,
            "max_total_min": int(parts[2]) if len(parts) > 2 and parts[2] != "" else 10000,
            "min_total_min": int(parts[3]) if len(parts) > 3 and parts[3] != "" else 0,
            "max_cons": int(parts[4]) if len(parts) > 4 and parts[4] != "" else 6,
            "min_cons": int(parts[5]) if len(parts) > 5 and parts[5] != "" else 2,
            "min_consec_days_off": int(parts[6]) if len(parts) > 6 and parts[6] != "" else 2,
            "max_weekends": int(parts[7]) if len(parts) > 7 and parts[7] != "" else 2,
            "biotype": 1,
        }

    days_off: Dict[str, List[int]] = defaultdict(list)
    for ln in data.get("DAYS_OFF", []):
        parts = [p.strip() for p in ln.split(",")]
        if not parts:
            continue
        days_off[parts[0]] = [int(x) for x in parts[1:] if x != ""]

    shift_on = []
    for ln in data.get("SHIFT_ON_REQUESTS", []):
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 4:
            shift_on.append((p[0], int(p[1]), p[2], float(p[3])))

    shift_off = []
    for ln in data.get("SHIFT_OFF_REQUESTS", []):
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 4:
            shift_off.append((p[0], int(p[1]), p[2], float(p[3])))

    cover: Dict[Tuple[int, str], dict] = {}
    for ln in data.get("COVER", []):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 5:
            cover[(int(parts[0]), parts[1])] = {
                "req": int(parts[2]),
                "w_under": float(parts[3]),
                "w_over": float(parts[4]),
            }

    return {
        "horizon": horizon,
        "shifts": shifts,
        "staff": staff,
        "days_off": dict(days_off),
        "shift_on_requests": shift_on,
        "shift_off_requests": shift_off,
        "cover": cover,
        "fatigue_limit": None,
    }

def build_custom_instance(weeks: int, num_employees: int, shifts: Sequence[str]) -> str:
    if not shifts:
        raise ValueError("Select at least one shift type.")

    days = weeks * 7
    shift_length = 480
    employees = [f"E{str(i).zfill(3)}" for i in range(num_employees)]

    base_schedule = {e: [] for e in employees}
    for i, e in enumerate(employees):
        offset = i % 6
        for d in range(days):
            if (d + offset) % 6 < 4:
                base_schedule[e].append(shifts[0])
            else:
                base_schedule[e].append("OFF")

    out: List[str] = []
    out.append("SECTION_HORIZON\n" + str(days) + "\n")
    out.append("SECTION_SHIFTS")
    for i, s in enumerate(shifts):
        cannot_follow = shifts[i - 1] if i > 0 else ""
        out.append(f"{s},{shift_length},{cannot_follow},{infer_fatigue_type(s)}")

    out.append("\nSECTION_STAFF")
    for e in employees:
        worked_days = sum(1 for d in base_schedule[e] if d != "OFF")
        shift_caps = [f"{shifts[0]}={worked_days + 2}"] + [
            f"{s}={worked_days}" for s in shifts[1:]
        ]
        max_minutes = (worked_days + 3) * shift_length
        min_minutes = max(0, (worked_days - 3) * shift_length)
        out.append(
            f"{e},{'|'.join(shift_caps)},{max_minutes},{min_minutes},6,2,2,{weeks}"
        )

    out.append("\nSECTION_DAYS_OFF")
    for e in employees:
        off_days = [d for d in range(days) if base_schedule[e][d] == "OFF"]
        if off_days:
            rng = random.Random(10_000 + int(e[1:]))
            chosen_off = sorted(rng.sample(off_days, min(2, len(off_days))))
            out.append(f"{e}," + ",".join(map(str, chosen_off)))

    out.append("\nSECTION_SHIFT_ON_REQUESTS")
    for e in employees:
        work_days = [d for d in range(days) if base_schedule[e][d] != "OFF"]
        if work_days:
            rng = random.Random(20_000 + int(e[1:]))
            chosen_on = sorted(rng.sample(work_days, min(2, len(work_days))))
            for d in chosen_on:
                out.append(
                    f"{e},{d},{base_schedule[e][d]},{rng.randint(1, 3)}"
                )

    out.append("\nSECTION_COVER")
    coverage = defaultdict(int)
    for d in range(days):
        for e in employees:
            s = base_schedule[e][d]
            if s != "OFF":
                coverage[(d, s)] += 1

    for d in range(days):
        for s in shifts:
            req = max(1, coverage[(d, s)])
            out.append(f"{d},{s},{req},100,10")

    return "\n".join(out)

def prepare_thesis_data(
    inst: Mapping,
    use_equity_proxy: bool = True,
    seed: int = 20261007,
    randomize_surges: bool = True,
) -> dict:
    inst = safe_copy_instance(inst)
    horizon = int(inst["horizon"])
    nurses = list(inst["staff"].keys())
    shifts = list(inst["shifts"].keys())

    for nurse in nurses:
        inst["staff"][nurse]["biotype"] = int(inst["staff"][nurse].get("biotype", 1))

    original_weights = defaultdict(float)
    for e, _d, _s, w in inst.get("shift_on_requests", []):
        original_weights[e] += float(w)
    for e, _d, _s, w in inst.get("shift_off_requests", []):
        original_weights[e] += float(w)

    W_TARGET = 0.0
    padding = defaultdict(float)
    padding_requests = []
    if use_equity_proxy:
        W_TARGET = max(original_weights.values()) if original_weights else 0.0
        for idx, e in enumerate(nurses):
            rng = np.random.default_rng(int(seed) + 300 + idx)
            deficit = W_TARGET - original_weights[e]
            attempts = 0
            while deficit > EPS and attempts < 100_000:
                attempts += 1
                req_type = str(rng.choice(["ON", "OFF"]))
                d = int(rng.integers(0, horizon))
                s = str(rng.choice(shifts))
                upper = min(4.0, deficit)
                if upper < 1.0:
                    w_add = float(deficit)
                else:
                    w_add = float(rng.integers(1, int(math.floor(upper)) + 1))
                request = (e, d, s, w_add)
                if req_type == "ON":
                    inst["shift_on_requests"].append(request)
                else:
                    inst["shift_off_requests"].append(request)
                padding_requests.append({
                    "Nurse": e,
                    "Type": req_type,
                    "Day": d,
                    "Shift": s,
                    "Added Weight": w_add,
                })
                padding[e] += w_add
                deficit -= w_add
            if deficit > EPS:
                raise RuntimeError(f"Equity padding failed for nurse {e}.")
    else:
        W_TARGET = 0.0

    final_weights = {
        e: float(original_weights[e] + padding[e]) for e in nurses
    }

    surges: Dict[Tuple[int, str], int] = {}
    surge_rng = np.random.default_rng(int(seed) + 700_000)

    for d in range(horizon):
        for s in shifts:
            req = int(inst["cover"].get((d, s), {}).get("req", 0))
            if req <= 0:
                surge = 0
            else:
                frac = (
                    float(surge_rng.uniform(DEFAULT_SURGE_FRACTION_LOW, DEFAULT_SURGE_FRACTION_HIGH))
                    if randomize_surges
                    else (DEFAULT_SURGE_FRACTION_LOW + DEFAULT_SURGE_FRACTION_HIGH) / 2.0
                )
                surge = int(math.ceil(req * frac))
            surges[(d, s)] = surge

    inst["synthetic"] = {
        "surges": surges,
        "W_TARGET": float(W_TARGET),
        "seed": int(seed),
        "randomize_surges": bool(randomize_surges),
        "original_preference_weights": dict(original_weights),
        "padding_added": dict(padding),
        "padding_requests": padding_requests,
        "final_preference_weights": final_weights,
        "surge_under_penalty": DEFAULT_SURGE_UNDER_PENALTY,
    }
    return normalize_request_lists(inst)

def map_shift_to_standard(shift_name: str, shifts: Mapping[str, Mapping]) -> str:
    return fatigue_type(shift_name, shifts)

def compute_fatigue_profile(
    inst: Mapping,
    nurse: str,
    shifts_dict: Mapping[int, str],
    P_Score: Mapping,
) -> Tuple[List[float], float]:
    horizon = int(inst["horizon"])
    shifts = inst["shifts"]
    b = int(inst["staff"][nurse].get("biotype", 1))
    night_shifts = set(find_night_shifts(shifts))

    profile = [0.0] * horizon
    history = ["OFF", "OFF", "OFF", "OFF"]
    max_f = 0.0

    for d in range(horizon):
        current = shifts_dict.get(d, "OFF")
        prev4 = history[-4:]
        prior_day = prev4[0]
        w = 1 if prior_day in night_shifts else 0
        s1 = map_shift_to_standard(prev4[1], shifts)
        s2 = map_shift_to_standard(prev4[2], shifts)
        s3 = map_shift_to_standard(prev4[3], shifts)
        s4 = map_shift_to_standard(current, shifts)
        key = (b, w, s1, s2, s3, s4)
        if key not in P_Score:
            raise KeyError(
                f"Missing fatigue table entry {key} for nurse {nurse}, day {d}. "
                "Do not silently replace missing fatigue scores with zero."
            )
        score = float(P_Score[key])
        profile[d] = score
        max_f = max(max_f, score)
        history = (history + [current])[-4:]

    fatigue_limit = inst.get("fatigue_limit")
    if fatigue_limit is not None and max_f > float(fatigue_limit) + EPS:
        return profile, max_f
    return profile, max_f

@dataclass
class ScheduleColumn:
    nurse_id: str
    shifts: Dict[int, str]
    base_cost: float
    max_fatigue: float
    fatigue_array: List[float]

    @property
    def cost(self) -> float:
        return self.base_cost

    def signature(self) -> Tuple[str, ...]:
        return tuple(self.shifts[d] for d in sorted(self.shifts))

def validate_column(inst: Mapping, nurse: str, col: ScheduleColumn) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    horizon = int(inst["horizon"])
    shifts = list(inst["shifts"].keys())
    staff = inst["staff"][nurse]
    days_off = set(inst.get("days_off", {}).get(nurse, []))

    if col.nurse_id != nurse:
        problems.append(f"Column belongs to nurse {col.nurse_id}, not {nurse}")
        return False, problems

    schedule = [col.shifts.get(d, "OFF") for d in range(horizon)]
    if any(s not in shifts and s != "OFF" for s in schedule):
        problems.append("Unknown shift in column")
        return False, problems

    for d, s in enumerate(schedule):
        if d in days_off and s != "OFF":
            problems.append(f"Unavailable on day {d}")

    for d in range(horizon - 1):
        s, t = schedule[d], schedule[d + 1]
        if s in inst["shifts"] and t in inst["shifts"]:
            if t in inst["shifts"][s].get("forbid_next", []):
                problems.append(f"Forbidden transition {s}->{t} on day {d}")

    for s, limit in staff.get("maxshift_per_shift", {}).items():
        if s in shifts and sum(schedule[d] == s for d in range(horizon)) > int(limit):
            problems.append(f"Shift-frequency limit violated for {s}")

    workload = sum(
        inst["shifts"][s]["length"] for s in schedule if s in inst["shifts"]
    )
    if workload > int(staff["max_total_min"]):
        problems.append("Maximum workload violated")
    if workload < int(staff["min_total_min"]):
        problems.append("Minimum workload violated")

    maxc = staff.get("max_cons")
    minc = staff.get("min_cons")
    minoff = staff.get("min_consec_days_off")

    work_run = 0
    off_run = 0
    work_started_after_boundary = False
    off_started_after_boundary = False

    for d, s in enumerate(schedule):
        working = s in inst["shifts"]

        if working:
            if (
                off_run > 0
                and off_started_after_boundary
                and minoff is not None
                and off_run < int(minoff)
            ):
                problems.append(f"Minimum consecutive off violated before day {d}")

            if work_run == 0:
                work_started_after_boundary = d > 0
            work_run += 1
            off_run = 0
            off_started_after_boundary = False

            if maxc is not None and work_run > int(maxc):
                problems.append(f"Maximum consecutive work violated at day {d}")

        else:
            if (
                work_run > 0
                and work_started_after_boundary
                and minc is not None
                and work_run < int(minc)
            ):
                problems.append(f"Minimum consecutive work violated before day {d}")

            if off_run == 0:
                off_started_after_boundary = d > 0
            off_run += 1
            work_run = 0
            work_started_after_boundary = False

    max_weekends = staff.get("max_weekends")
    if max_weekends is not None:
        worked_weekends = 0
        for _w, days in weekend_indices(horizon).items():
            if any(schedule[d] in inst["shifts"] for d in days):
                worked_weekends += 1
        if worked_weekends > int(max_weekends):
            problems.append("Maximum weekends violated")

    fatigue_limit = inst.get("fatigue_limit")
    if fatigue_limit is not None and col.max_fatigue > float(fatigue_limit) + EPS:
        problems.append("Fatigue threshold violated")

    return not problems, problems

def validate_solution(inst: Mapping, solution: Mapping[str, ScheduleColumn]) -> Tuple[bool, Dict[str, List[str]]]:
    all_problems: Dict[str, List[str]] = {}
    for nurse in inst["staff"]:
        if nurse not in solution:
            all_problems[nurse] = ["No column selected"]
            continue
        ok, probs = validate_column(inst, nurse, solution[nurse])
        if not ok:
            all_problems[nurse] = probs
    return not all_problems, all_problems

def uncertainty_surge_total(inst: Mapping, gamma: int) -> Tuple[float, Tuple[Tuple[int, str], ...]]:
    if gamma <= 0:
        return 0.0, tuple()

    surges = inst.get("synthetic", {}).get("surges", {})
    ranked = sorted(
        (
            (int(v), key)
            for key, v in surges.items()
            if int(v) > 0
        ),
        key=lambda x: (-x[0], x[1]),
    )
    selected = tuple(key for _v, key in ranked[: int(gamma)])
    total = float(sum(v for v, _key in ranked[: int(gamma)]))
    return total, selected

def pooled_recourse(
    inst: Mapping,
    over_total: float,
    gamma: int,
    agency_cost: float,
    surge_under_penalty: float = DEFAULT_SURGE_UNDER_PENALTY,
) -> dict:
    surge_total, selected = uncertainty_surge_total(inst, gamma)

    rearranged = min(float(over_total), surge_total)
    residual = max(0.0, surge_total - rearranged)

    if residual <= EPS:
        agency_total = 0.0
        unmet_total = 0.0
    elif float(agency_cost) < float(surge_under_penalty) - EPS:
        agency_total = residual
        unmet_total = 0.0
    else:
        agency_total = 0.0
        unmet_total = residual

    recourse_cost = (
        agency_total * float(agency_cost)
        + unmet_total * float(surge_under_penalty)
    )

    agency_alloc = {(d, s): 0.0 for d in range(inst["horizon"]) for s in inst["shifts"]}
    remaining_agency = agency_total
    for d, s in selected:
        if remaining_agency <= EPS:
            break
        cap = float(inst["synthetic"]["surges"].get((d, s), 0))
        hire = min(cap, remaining_agency)
        agency_alloc[(d, s)] = hire
        remaining_agency -= hire

    return {
        "surge_total": surge_total,
        "worst_case_shifts": selected,
        "rearranged_total": rearranged,
        "agency_total": agency_total,
        "unmet_total": unmet_total,
        "recourse_cost": recourse_cost,
        "agency_allocation": agency_alloc,
    }

def coverage_from_solution(inst: Mapping, solution: Mapping[str, "ScheduleColumn"]) -> Dict[Tuple[int, str], float]:
    coverage: Dict[Tuple[int, str], float] = {
        (d, s): 0.0
        for d in range(int(inst["horizon"]))
        for s in inst["shifts"]
    }

    valid_shifts = set(inst["shifts"])
    for col in solution.values():
        for d in range(int(inst["horizon"])):
            shift = col.shifts.get(d, "OFF")
            if shift in valid_shifts:
                coverage[(d, shift)] += 1.0

    return coverage

def preference_cost(inst: Mapping, solution: Mapping[str, "ScheduleColumn"]) -> float:
    total = 0.0
    for nurse, col in solution.items():
        for e, d, s, weight in inst.get("shift_on_requests", []):
            if e == nurse and d < inst["horizon"] and col.shifts.get(d, "OFF") != s:
                total += float(weight)
        for e, d, s, weight in inst.get("shift_off_requests", []):
            if e == nurse and d < inst["horizon"] and col.shifts.get(d, "OFF") == s:
                total += float(weight)
    return total

def nominal_coverage_penalty(
    inst: Mapping,
    coverage: Mapping[Tuple[int, str], float],
) -> Tuple[float, float, float]:
    penalty = 0.0
    under_total = 0.0
    over_total = 0.0

    for d in range(inst["horizon"]):
        for s in inst["shifts"]:
            req = float(inst["cover"].get((d, s), {}).get("req", 0))
            staff = float(coverage.get((d, s), 0.0))
            under = max(0.0, req - staff)
            over = max(0.0, staff - req)
            under_total += under
            over_total += over
            penalty += under * float(inst["cover"].get((d, s), {}).get("w_under", 0.0))
            penalty += over * float(inst["cover"].get((d, s), {}).get("w_over", 0.0))

    return penalty, under_total, over_total

def solution_signature(solution: Mapping[str, "ScheduleColumn"]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    return tuple(
        (n, solution[n].signature())
        for n in sorted(solution)
    )

def evaluate_solution(
    inst: Mapping,
    solution: Mapping[str, "ScheduleColumn"],
    gamma: int,
    agency_cost: float,
    fatigue_weight: float,
) -> dict:
    feasible, problems = validate_solution(inst, solution)
    coverage = coverage_from_solution(inst, solution)
    coverage_penalty, under_total, over_total = nominal_coverage_penalty(inst, coverage)
    pref = sum(float(col.base_cost) for col in solution.values())
    max_fatigue = max((float(col.max_fatigue) for col in solution.values()), default=0.0)

    surge_penalty = float(
        inst.get("synthetic", {}).get("surge_under_penalty", DEFAULT_SURGE_UNDER_PENALTY)
    )
    if gamma > 0:
        recourse = pooled_recourse(
            inst,
            over_total=over_total,
            gamma=gamma,
            agency_cost=agency_cost,
            surge_under_penalty=surge_penalty,
        )
    else:
        recourse = {
            "surge_total": 0.0,
            "worst_case_shifts": tuple(),
            "rearranged_total": 0.0,
            "agency_total": 0.0,
            "unmet_total": 0.0,
            "recourse_cost": 0.0,
            "agency_allocation": {},
        }

    objective = (
        pref
        + coverage_penalty
        + float(recourse["recourse_cost"])
        + float(fatigue_weight) * max_fatigue
    )

    return {
        "feasible": feasible,
        "problems": problems,
        "coverage": coverage,
        "preference_penalty": pref,
        "coverage_penalty": coverage_penalty,
        "under_total": under_total,
        "over_total": over_total,
        "max_fatigue": max_fatigue,
        "surge_total": float(recourse["surge_total"]),
        "rearranged_total": float(recourse["rearranged_total"]),
        "agency_total": float(recourse["agency_total"]),
        "surge_unmet": float(recourse["unmet_total"]),
        "surge_under_penalty": surge_penalty,
        "recourse_cost": float(recourse["recourse_cost"]),
        "worst_case_shifts": tuple(recourse["worst_case_shifts"]),
        "agency_allocation": recourse["agency_allocation"],
        "objective": float(objective),
    }

def _build_direct_mip(
    inst: Mapping,
    use_fatigue: bool,
    gamma: int,
    agency_cost: float,
    fatigue_weight: float,
    solver_choice: str,
    time_limit: float,
    warm_start_assign: Optional[Mapping[str, Mapping[int, str]]] = None,
    surge_under_penalty: float = DEFAULT_SURGE_UNDER_PENALTY,
):
    mdl = pulp.LpProblem("Thesis_Direct_MIP", pulp.LpMinimize)

    horizon = int(inst["horizon"])
    E = list(inst["staff"].keys())
    S = list(inst["shifts"].keys())

    x = pulp.LpVariable.dicts(
        "x",
        [(e, d, s) for e in E for d in range(horizon) for s in S],
        cat=pulp.LpBinary,
    )

    if warm_start_assign:
        for e in E:
            for d in range(horizon):
                chosen = warm_start_assign.get(e, {}).get(d, "OFF")
                for s in S:
                    x[(e, d, s)].setInitialValue(1.0 if chosen == s else 0.0)

    for e in E:
        for d in range(horizon):
            mdl += pulp.lpSum(x[(e, d, s)] for s in S) <= 1

    for e, days in inst.get("days_off", {}).items():
        for d in days:
            if 0 <= d < horizon:
                for s in S:
                    mdl += x[(e, d, s)] == 0

    for e in E:
        for s, info in inst["shifts"].items():
            for t in info.get("forbid_next", []):
                if t in S:
                    for d in range(horizon - 1):
                        mdl += x[(e, d, s)] + x[(e, d + 1, t)] <= 1

    for e, info in inst["staff"].items():
        for s, limit in info.get("maxshift_per_shift", {}).items():
            if s in S:
                mdl += pulp.lpSum(x[(e, d, s)] for d in range(horizon)) <= int(limit)

        total_mins = pulp.lpSum(
            x[(e, d, s)] * int(inst["shifts"][s]["length"])
            for d in range(horizon)
            for s in S
        )
        mdl += total_mins >= int(info["min_total_min"])
        mdl += total_mins <= int(info["max_total_min"])

        maxc = info.get("max_cons")
        if maxc is not None:
            maxc = int(maxc)
            for start_d in range(max(0, horizon - maxc)):
                mdl += (
                    pulp.lpSum(
                        x[(e, d, s)]
                        for d in range(start_d, start_d + maxc + 1)
                        for s in S
                    )
                    <= maxc
                )

        start = pulp.LpVariable.dicts(
            f"start_work_{e}", range(horizon), cat=pulp.LpBinary
        )
        mdl += start[0] == 0
        for d in range(1, horizon):
            prev_on = pulp.lpSum(x[(e, d - 1, s)] for s in S)
            curr_on = pulp.lpSum(x[(e, d, s)] for s in S)
            mdl += start[d] >= curr_on - prev_on
            mdl += start[d] <= curr_on
            mdl += start[d] <= 1 - prev_on

        minc = info.get("min_cons")
        if minc is not None and int(minc) > 1:
            minc = int(minc)
            for d in range(1, horizon):
                if d + minc - 1 < horizon:
                    mdl += (
                        pulp.lpSum(
                            x[(e, d + k, s)]
                            for k in range(minc)
                            for s in S
                        )
                        >= minc * start[d]
                    )

        start_off = pulp.LpVariable.dicts(
            f"start_off_{e}", range(horizon), cat=pulp.LpBinary
        )
        mdl += start_off[0] == 0
        for d in range(1, horizon):
            prev_on = pulp.lpSum(x[(e, d - 1, s)] for s in S)
            curr_on = pulp.lpSum(x[(e, d, s)] for s in S)
            curr_off = 1 - curr_on
            mdl += start_off[d] >= curr_off + prev_on - 1
            mdl += start_off[d] <= curr_off
            mdl += start_off[d] <= prev_on

        minoff = info.get("min_consec_days_off")
        if minoff is not None and int(minoff) > 0:
            minoff = int(minoff)
            for d in range(1, horizon):
                if d + minoff - 1 < horizon:
                    mdl += (
                        pulp.lpSum(
                            1 - pulp.lpSum(x[(e, d + k, s)] for s in S)
                            for k in range(minoff)
                        )
                        >= minoff * start_off[d]
                    )

        weekends = weekend_indices(horizon)
        y_weekend = pulp.LpVariable.dicts(
            f"worked_weekend_{e}",
            weekends.keys(),
            cat=pulp.LpBinary,
        )
        max_w = info.get("max_weekends")
        if max_w is not None:
            for w, days in weekends.items():
                for d in days:
                    mdl += y_weekend[w] >= pulp.lpSum(x[(e, d, s)] for s in S)
            mdl += pulp.lpSum(y_weekend[w] for w in weekends) <= int(max_w)

    under = pulp.LpVariable.dicts(
        "under",
        [(d, s) for d in range(horizon) for s in S],
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    over = pulp.LpVariable.dicts(
        "over",
        [(d, s) for d in range(horizon) for s in S],
        lowBound=0,
        cat=pulp.LpContinuous,
    )

    nominal_coverage = {}
    for d in range(horizon):
        for s in S:
            req = float(inst["cover"].get((d, s), {}).get("req", 0))
            nominal_coverage[d, s] = pulp.lpSum(x[(e, d, s)] for e in E)
            mdl += nominal_coverage[d, s] + under[d, s] - over[d, s] == req

    P_Score = load_fatigue_table() if use_fatigue else {}
    f_nt = {}
    f_GM = None

    if use_fatigue:
        f_nt = pulp.LpVariable.dicts(
            "fatigue_day",
            [(e, d) for e in E for d in range(horizon)],
            lowBound=0,
            cat=pulp.LpContinuous,
        )
        f_GM = pulp.LpVariable("fatigue_global_max", lowBound=0)

        all_states = list(S) + ["OFF"]

        def y_value(e: str, d: int, s: str):
            if d < 0:
                return 1.0 if s == "OFF" else 0.0
            if s == "OFF":
                return 1.0 - pulp.lpSum(x[(e, d, st)] for st in S)
            return x[(e, d, s)]

        for e in E:
            b = int(inst["staff"][e].get("biotype", 1))
            for d in range(horizon):
                mdl += f_GM >= f_nt[e, d]

                pattern_vars = []
                night_shifts = set(find_night_shifts(inst["shifts"]))
                prior_night_expr = (
                    pulp.lpSum(y_value(e, d - 4, ns) for ns in night_shifts)
                    if night_shifts else 0.0
                )

                for w in (0, 1):
                    if w == 1 and not night_shifts:
                        continue

                    for s1, s2, s3, s4 in itertools.product(all_states, repeat=4):
                        p = pulp.LpVariable(
                            f"pat_{e}_{d}_{w}_{s1}_{s2}_{s3}_{s4}",
                            cat=pulp.LpBinary,
                        )
                        pattern_vars.append((p, w, s1, s2, s3, s4))

                        y_prior = prior_night_expr if w == 1 else 1.0 - prior_night_expr
                        mdl += 5 * p <= (
                            y_prior
                            + y_value(e, d - 3, s1)
                            + y_value(e, d - 2, s2)
                            + y_value(e, d - 1, s3)
                            + y_value(e, d, s4)
                        )

                if not pattern_vars:
                    raise RuntimeError(f"No fatigue pattern states available for nurse {e}, day {d}.")

                mdl += pulp.lpSum(p for p, *_ in pattern_vars) == 1
                mdl += f_nt[e, d] == pulp.lpSum(
                    float(
                        P_Score[
                            (
                                b,
                                w,
                                fatigue_type(s1, inst["shifts"]),
                                fatigue_type(s2, inst["shifts"]),
                                fatigue_type(s3, inst["shifts"]),
                                fatigue_type(s4, inst["shifts"]),
                            )
                        ]
                    ) * p
                    for p, w, s1, s2, s3, s4 in pattern_vars
                )

        if inst.get("fatigue_limit") is not None:
            mdl += f_GM <= float(inst["fatigue_limit"])

    agency_total = None
    surge_unmet = None
    z_budget = None
    p_dual = None

    if int(gamma) > 0:
        surge = inst["synthetic"]["surges"]

        agency_total = pulp.LpVariable("agency_total", lowBound=0, cat=pulp.LpInteger)
        surge_unmet = pulp.LpVariable("surge_unmet", lowBound=0, cat=pulp.LpInteger)
        z_budget = pulp.LpVariable("z_budget", lowBound=0, cat=pulp.LpContinuous)
        p_dual = pulp.LpVariable.dicts(
            "p_dual",
            [(d, s) for d in range(horizon) for s in S],
            lowBound=0,
            cat=pulp.LpContinuous,
        )

        for d in range(horizon):
            for s in S:
                mdl += z_budget + p_dual[d, s] >= float(surge.get((d, s), 0))

        pooled_over = pulp.lpSum(over[d, s] for d in range(horizon) for s in S)
        pooled_surge = (
            float(gamma) * z_budget
            + pulp.lpSum(p_dual[d, s] for d in range(horizon) for s in S)
        )

        mdl += pooled_over + agency_total + surge_unmet >= pooled_surge

    obj_terms = []

    for d in range(horizon):
        for s in S:
            cover_info = inst["cover"].get((d, s), {})
            obj_terms.append(float(cover_info.get("w_under", 0.0)) * under[d, s])
            obj_terms.append(float(cover_info.get("w_over", 0.0)) * over[d, s])

    for e, d, s, weight in inst.get("shift_on_requests", []):
        if e in E and s in S and 0 <= d < horizon:
            obj_terms.append(float(weight) * (1 - x[e, d, s]))

    for e, d, s, weight in inst.get("shift_off_requests", []):
        if e in E and s in S and 0 <= d < horizon:
            obj_terms.append(float(weight) * x[e, d, s])

    if agency_total is not None:
        obj_terms.append(float(agency_cost) * agency_total)
        obj_terms.append(float(surge_under_penalty) * surge_unmet)

    if f_GM is not None:
        obj_terms.append(float(fatigue_weight) * f_GM)

    mdl += pulp.lpSum(obj_terms)
    return mdl, x, under, over, agency_total, surge_unmet, z_budget, p_dual, f_nt, f_GM

def solve_thesis_model(
    inst: Mapping,
    use_fatigue: bool,
    use_equity: bool,
    gamma_budget: int,
    agency_cost: float = 20.0,
    fatigue_weight: float = 1.0,
    time_limit: float = 300.0,
    solver_choice: str = "Gurobi",
    warm_start_assign: Optional[Mapping[str, Mapping[int, str]]] = None,
    surge_under_penalty: float = DEFAULT_SURGE_UNDER_PENALTY,
) -> dict:
    import time as _time

    start = _time.perf_counter()
    (
        mdl, x, under, over, agency_total, surge_unmet,
        z_budget, p_dual, f_nt, f_GM
    ) = _build_direct_mip(
        inst,
        use_fatigue=use_fatigue,
        gamma=int(gamma_budget),
        agency_cost=float(agency_cost),
        fatigue_weight=float(fatigue_weight),
        solver_choice=solver_choice,
        time_limit=float(time_limit),
        warm_start_assign=warm_start_assign,
        surge_under_penalty=float(surge_under_penalty),
    )

    if solver_choice == "Gurobi":
        require_gurobi()
        elapsed_build = _time.perf_counter() - start
        # The requested limit is an end-to-end wall-clock budget, not just a
        # Gurobi solve budget. Do not start another solve after the budget is gone.
        if elapsed_build >= max(0.10, float(time_limit) - 0.10):
            return {
                "status": "TimeLimit",
                "time": elapsed_build,
                "E": list(inst["staff"].keys()),
                "S": list(inst["shifts"].keys()),
                "horizon": int(inst["horizon"]),
            }

        remaining = max(0.10, float(time_limit) - elapsed_build)
        solver_margin = min(1.0, max(0.25, 0.01 * float(time_limit)))
        solver_limit = max(0.05, remaining - solver_margin)
        wall_deadline = _time.perf_counter() + solver_limit

        def _wall_clock_callback(model, where):
            if _time.perf_counter() >= wall_deadline:
                model.terminate()

        print("\n" + "=" * 80)
        print("DIRECT MIP - GUROBI")
        print("=" * 80)

        solver = pulp.GUROBI(
            msg=1,
            timeLimit=solver_limit,
            warmStart=True,
        )
        solver.actualSolve(mdl, callback=_wall_clock_callback)
        print("\n" + "=" * 80)
        print("DIRECT MIP - GUROBI")
        print("=" * 80)

        solver = pulp.GUROBI(
            msg=1,
            timeLimit=solver_limit,
            warmStart=True,
        )
    elif solver_choice == "HiGHS":
        remaining = max(0.05, float(time_limit) - (_time.perf_counter() - start))
        solver = pulp.getSolver("HiGHS", msg=0, timeLimit=remaining, warmStart=True)
        mdl.solve(solver)
    else:
        raise ValueError(f"Unknown direct solver {solver_choice}")

    if not any(v.varValue is not None for v in mdl.variables()):
        return {"status": "Infeasible", "time": _time.perf_counter() - start}

    E = list(inst["staff"].keys())
    S = list(inst["shifts"].keys())
    horizon = int(inst["horizon"])

    assign = defaultdict(dict)
    for e in E:
        for d in range(horizon):
            chosen = [
                s for s in S
                if x[e, d, s].varValue is not None and x[e, d, s].varValue > 0.5
            ]
            assign[e][d] = chosen[0] if chosen else "OFF"

    fatigue_table = load_fatigue_table() if use_fatigue else {}
    columns = {}
    for e in E:
        profile, max_f = (
            compute_fatigue_profile(inst, e, assign[e], fatigue_table)
            if use_fatigue
            else ([0.0] * horizon, 0.0)
        )
        base = 0.0
        for re_, d, s, w in inst.get("shift_on_requests", []):
            if re_ == e and assign[e].get(d, "OFF") != s:
                base += float(w)
        for re_, d, s, w in inst.get("shift_off_requests", []):
            if re_ == e and assign[e].get(d, "OFF") == s:
                base += float(w)
        columns[e] = ScheduleColumn(e, dict(assign[e]), base, max_f, profile)

    evaluation = evaluate_solution(
        inst,
        columns,
        gamma=int(gamma_budget),
        agency_cost=float(agency_cost),
        fatigue_weight=float(fatigue_weight),
    )

    best_bound = None
    try:
        if solver_choice == "Gurobi" and getattr(mdl, "solverModel", None) is not None:
            best_bound = float(mdl.solverModel.ObjBound)
    except Exception:
        best_bound = None

    model_obj = pulp.value(mdl.objective)
    consistency_error = (
        abs(float(model_obj) - evaluation["objective"])
        if model_obj is not None
        else None
    )

    if consistency_error is not None and consistency_error > 1e-4:
        status = "ModelEvaluatorMismatch"
    elif mdl.status == pulp.LpStatusOptimal:
        status = "Optimal"
    else:
        status = "TimeLimit"

    gap = None
    if best_bound is not None and abs(best_bound) > EPS:
        gap = max(0.0, (evaluation["objective"] - best_bound) / abs(best_bound))

    return {
        "status": status,
        "assign": {e: dict(days) for e, days in assign.items()},
        "columns": columns,
        "objective": evaluation["objective"],
        "model_objective": float(model_obj) if model_obj is not None else None,
        "objective_consistency_error": consistency_error,
        "evaluation": evaluation,
        "best_bound": best_bound,
        "gap": gap,
        "time": _time.perf_counter() - start,
        "E": E,
        "S": S,
        "horizon": horizon,
        "fatigue_max": evaluation["max_fatigue"],
        "agency_total": evaluation["agency_total"],
        "surge_unmet": evaluation["surge_unmet"],
        "rearranged_total": evaluation["rearranged_total"],
    }

@dataclass
class Label:
    day: int
    last_shift: str
    work_run: int
    off_run: int
    work_run_at_boundary: bool
    off_run_at_boundary: bool
    workload: int
    shift_counts: Tuple[int, ...]
    weekend_mask: int
    history: Tuple[str, str, str, str]
    reduced_cost: float
    max_fatigue: float
    fatigue_array: Tuple[float, ...]
    path: Tuple[str, ...]

class ColumnGenerationEngine:

    def __init__(
        self,
        inst: Mapping,
        agency_cost: float,
        use_fatigue: bool,
        fatigue_weight: float,
        gamma_budget: int,
        seed: int = 20261007,
    ):
        require_gurobi()
        self.inst = inst
        self.agency_cost = float(agency_cost)
        self.use_fatigue = bool(use_fatigue)
        self.fatigue_w = float(fatigue_weight)
        self.gamma = int(gamma_budget)
        self.seed = int(seed)

        self.D = tuple(range(int(inst["horizon"])))
        self.S = tuple(inst["shifts"].keys())
        self.E = tuple(inst["staff"].keys())
        self.shift_index = {s: j for j, s in enumerate(self.S)}
        self.night_shifts = set(find_night_shifts(inst["shifts"]))

        self.P_Score = load_fatigue_table() if self.use_fatigue else {}
        self.arc_cost = {
            (e, d, s): 0.0
            for e in self.E for d in self.D for s in ("OFF",) + self.S
        }
        for e in self.E:
            for d in self.D:
                for s in ("OFF",) + self.S:
                    cost = 0.0
                    for req_e, rd, rs, w in inst.get("shift_on_requests", []):
                        if req_e == e and rd == d and rs != s:
                            cost += float(w)
                    for req_e, rd, rs, w in inst.get("shift_off_requests", []):
                        if req_e == e and rd == d and rs == s:
                            cost += float(w)
                    self.arc_cost[e, d, s] = cost

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 1)
        self.env.start()

        self.stats = {
            "iterations": 0,
            "columns_added": 0,
            "enrichment_columns": 0,
            "pricing_calls": 0,
            "pricing_time": 0.0,
            "label_extensions": 0,
            "label_pruned": 0,
            "pricing_truncated": False,
            "gurobi_pricing_optimal": 0,
            "gurobi_pricing_timeout": 0,
            "nurse_mip_calls": 0,
            "converged": False,
            "time_limited": False,
        }

    def close(self):
        try:
            self.env.dispose()
        except Exception:
            pass

    def shift_type(self, s: str) -> str:
        return fatigue_type(s, self.inst["shifts"])

    def _preference_arc_cost(self, nurse: str, d: int, s: str) -> float:
        return float(self.arc_cost.get((nurse, d, s), 0.0))

    def _fatigue_score(
        self,
        nurse: str,
        history: Tuple[str, str, str, str],
        current_shift: str,
    ) -> float:
        if not self.use_fatigue:
            return 0.0

        b = int(self.inst["staff"][nurse].get("biotype", 1))
        w = 1 if history[0] in self.night_shifts else 0

        key = (
            b,
            w,
            self.shift_type(history[1]),
            self.shift_type(history[2]),
            self.shift_type(history[3]),
            self.shift_type(current_shift),
        )
        if key not in self.P_Score:
            raise KeyError(f"Missing fatigue lookup key {key}.")
        return float(self.P_Score[key])

    def _initial_label(self) -> Label:
        return Label(
            day=0,
            last_shift="OFF",
            work_run=0,
            off_run=0,
            work_run_at_boundary=False,
            off_run_at_boundary=False,
            workload=0,
            shift_counts=tuple(0 for _ in self.S),
            weekend_mask=0,
            history=("OFF", "OFF", "OFF", "OFF"),
            reduced_cost=0.0,
            max_fatigue=0.0,
            fatigue_array=tuple(),
            path=tuple(),
        )

    def _extend(
        self,
        nurse: str,
        label: Label,
        d: int,
        s: str,
        dual_alpha: Mapping[Tuple[int, str], float],
        dual_fatigue: float,
    ) -> Optional[Label]:
        staff = self.inst["staff"][nurse]
        working = s != "OFF"

        if working and d in set(self.inst.get("days_off", {}).get(nurse, [])):
            return None

        if working and label.last_shift in self.inst["shifts"]:
            if s in self.inst["shifts"][label.last_shift].get("forbid_next", []):
                return None

        if working and label.off_run > 0 and not label.off_run_at_boundary:
            minoff = int(staff.get("min_consec_days_off", 0) or 0)
            if minoff > 0 and label.off_run < minoff:
                return None

        if not working and label.work_run > 0 and not label.work_run_at_boundary:
            minc = int(staff.get("min_cons", 0) or 0)
            if minc > 0 and label.work_run < minc:
                return None

        if working:
            new_work_run = label.work_run + 1 if label.work_run > 0 else 1
            new_off_run = 0
            new_work_boundary = (
                label.work_run_at_boundary if label.work_run > 0 else (d == 0)
            )
            new_off_boundary = False
        else:
            new_work_run = 0
            new_off_run = label.off_run + 1 if label.off_run > 0 else 1
            new_off_boundary = (
                label.off_run_at_boundary if label.off_run > 0 else (d == 0)
            )
            new_work_boundary = False

        maxc = staff.get("max_cons")
        if working and maxc is not None and new_work_run > int(maxc):
            return None

        counts = list(label.shift_counts)
        if working:
            si = self.shift_index[s]
            counts[si] += 1
            limit = int(staff.get("maxshift_per_shift", {}).get(s, self.inst["horizon"]))
            if counts[si] > limit:
                return None

        new_workload = label.workload + (
            int(self.inst["shifts"][s]["length"]) if working else 0
        )
        if new_workload > int(staff["max_total_min"]):
            return None

        if new_workload + self._max_additional_work_minutes(
            nurse,
            d + 1,
            tuple(counts),
        ) < int(staff["min_total_min"]):
            return None

        weekend_mask = label.weekend_mask
        if working and d % 7 in (5, 6):
            weekend_mask |= 1 << (d // 7)
        max_w = staff.get("max_weekends")
        if max_w is not None and weekend_mask.bit_count() > int(max_w):
            return None

        today_fatigue = self._fatigue_score(nurse, label.history, s)
        new_max_f = max(label.max_fatigue, today_fatigue)
        fatigue_limit = self.inst.get("fatigue_limit")
        if fatigue_limit is not None and new_max_f > float(fatigue_limit) + EPS:
            return None

        path = label.path + (s,)
        history = (label.history + (s,))[-4:]

        rc = label.reduced_cost + self._preference_arc_cost(nurse, d, s)
        if working:
            rc -= float(dual_alpha.get((d, s), 0.0))

        return Label(
            day=d + 1,
            last_shift=s,
            work_run=new_work_run,
            off_run=new_off_run,
            work_run_at_boundary=new_work_boundary,
            off_run_at_boundary=new_off_boundary,
            workload=new_workload,
            shift_counts=tuple(counts),
            weekend_mask=weekend_mask,
            history=history,
            reduced_cost=rc,
            max_fatigue=new_max_f,
            fatigue_array=label.fatigue_array + (today_fatigue,),
            path=path,
        )

    def _max_additional_work_minutes(
        self,
        nurse: str,
        next_day: int,
        shift_counts: Tuple[int, ...],
    ) -> int:
        staff = self.inst["staff"][nurse]
        remaining_days = [d for d in self.D if d >= next_day]
        available_days = [
            d for d in remaining_days
            if d not in set(self.inst.get("days_off", {}).get(nurse, []))
        ]
        if not available_days:
            return 0

        capacities = []
        for j, s in enumerate(self.S):
            limit = int(staff.get("maxshift_per_shift", {}).get(s, len(self.D)))
            remaining_cap = max(0, limit - int(shift_counts[j]))
            capacities.extend([int(self.inst["shifts"][s]["length"])] * remaining_cap)

        capacities.sort(reverse=True)
        return int(sum(capacities[: len(available_days)]))

    @staticmethod
    def _dominance_signature(label: Label) -> tuple:
        return (
            label.last_shift,
            label.work_run,
            label.off_run,
            label.work_run_at_boundary,
            label.off_run_at_boundary,
            label.shift_counts,
            label.weekend_mask,
            label.history,
        )

    def _dominates(self, nurse: str, a: Label, b: Label) -> bool:
        if a.reduced_cost > b.reduced_cost + EPS:
            return False
        if a.max_fatigue > b.max_fatigue + EPS:
            return False
        if a.workload > b.workload:
            return False
        min_work = int(self.inst["staff"][nurse]["min_total_min"])
        future_cap = self._max_additional_work_minutes(
            nurse,
            a.day,
            a.shift_counts,
        )
        if a.workload + future_cap < min_work:
            return False
        return True

    def _prune_labels(self, labels: List[Label], nurse: str) -> List[Label]:
        buckets: Dict[tuple, List[Label]] = defaultdict(list)
        for lab in labels:
            min_work = int(self.inst["staff"][nurse]["min_total_min"])
            if lab.workload + self._max_additional_work_minutes(nurse, lab.day, lab.shift_counts) < min_work:
                self.stats["label_pruned"] += 1
                continue
            buckets[self._dominance_signature(lab)].append(lab)

        kept: List[Label] = []
        for bucket in buckets.values():
            bucket.sort(key=lambda l: (l.reduced_cost, l.max_fatigue, l.workload))
            local: List[Label] = []
            for lab in bucket:
                dominated = False
                for inc in local:
                    if self._dominates(nurse, inc, lab):
                        dominated = True
                        break
                if dominated:
                    self.stats["label_pruned"] += 1
                    continue

                local = [inc for inc in local if not self._dominates(nurse, lab, inc)]
                local.append(lab)
            kept.extend(local)

        return kept

    def solve_pricing_subproblem(
        self,
        nurse: str,
        dual_alpha: Mapping[Tuple[int, str], float],
        dual_fatigue: float,
        dual_convexity: float,
        deadline: Optional[float] = None,
        require_negative: bool = True,
        perturbation: Optional[Mapping[Tuple[int, str], float]] = None,
    ) -> Tuple[Optional["ScheduleColumn"], float, dict]:
        import time as _time

        start = _time.perf_counter()
        perturbation = perturbation or {}
        deadline = deadline if deadline is not None else float("inf")

        remaining_global = max(0.0, deadline - start)
        per_call_cap = (DEFAULT_PRICING_TIME_CAP_SMALL
                        if len(self.E) <= 30 and len(self.D) <= 56
                        else DEFAULT_PRICING_TIME_CAP)
        per_call_cap = min(per_call_cap, max(0.02, remaining_global / max(1, len(self.E))))
        local_deadline = min(deadline, start + per_call_cap, start + max(0.01, remaining_global))
        deadline = local_deadline

        incumbent_col = None
        incumbent_rc = float("inf")
        inc_rng = random.Random(self.seed + 51_713 * (self.shift_index.get(self.S[0], 0) + 1) + hash(nurse) % 100_000)

        if _time.perf_counter() < deadline:
            incumbent_col = self.construct_feasible_schedule(
                nurse,
                inc_rng,
                min(deadline, _time.perf_counter() + 0.10),
                mode="dual",
                dual_alpha=dual_alpha,
                dual_fatigue=dual_fatigue,
            )
            if incumbent_col is not None:
                incumbent_rc = self._true_reduced_cost(
                    nurse,
                    incumbent_col,
                    dual_alpha,
                    dual_fatigue,
                    dual_convexity,
                )

        labels = [self._initial_label()]
        total_extensions = 0

        future_lb = [0.0] * (len(self.D) + 1)
        running = 0.0
        for d in reversed(self.D):
            best_arc = 0.0  # OFF
            for s in self.S:
                a = float(dual_alpha.get((d, s), 0.0))
                pc = self._preference_arc_cost(nurse, d, s)
                best_arc = min(best_arc, pc - a + float(perturbation.get((d, s), 0.0)))
            running += best_arc
            future_lb[d] = running

        for d in self.D:
            next_labels = []
            for lab in labels:
                bound = (
                    lab.reduced_cost
                    + future_lb[d]
                    - float(dual_fatigue) * lab.max_fatigue
                    - float(dual_convexity)
                )
                if bound >= incumbent_rc - CG_RC_TOL:
                    self.stats["label_pruned"] += 1
                    continue

                for s in ("OFF",) + self.S:
                    if _time.perf_counter() >= deadline:
                        self.stats["pricing_truncated"] = True
                        elapsed = _time.perf_counter() - start
                        self.stats["pricing_calls"] += 1
                        self.stats["pricing_time"] += elapsed
                        self.stats["label_extensions"] += total_extensions
                        if incumbent_col is not None and (not require_negative or incumbent_rc < -CG_RC_TOL):
                            return incumbent_col, incumbent_rc, {
                                "status": "TimeLimitWithIncumbent",
                                "labels": len(labels),
                            }
                        return None, incumbent_rc, {
                            "status": "TimeLimit",
                            "labels": len(labels),
                        }

                    total_extensions += 1
                    nxt = self._extend(
                        nurse,
                        lab,
                        d,
                        s,
                        dual_alpha,
                        dual_fatigue,
                    )
                    if nxt is not None:
                        if (d, s) in perturbation and s != "OFF":
                            nxt.reduced_cost += float(perturbation[(d, s)])
                        next_labels.append(nxt)

            labels = self._prune_labels(next_labels, nurse)

            beam_width = DEFAULT_PRICING_BEAM_WIDTH if (len(self.E) <= 60 and len(self.D) <= 56) else DEFAULT_PRICING_BEAM_WIDTH_LARGE
            if len(labels) > beam_width:
                self.stats["pricing_truncated"] = True
                labels.sort(
                    key=lambda l: (
                        l.reduced_cost
                        + future_lb[d + 1]
                        - float(dual_fatigue) * l.max_fatigue
                    )
                )
                labels = labels[:beam_width]

            if not labels:
                self.stats["pricing_calls"] += 1
                self.stats["pricing_time"] += _time.perf_counter() - start
                self.stats["label_extensions"] += total_extensions
                if incumbent_col is not None:
                    return (
                        incumbent_col if (not require_negative or incumbent_rc < -CG_RC_TOL) else None,
                        incumbent_rc,
                        {"status": "BestIncumbent", "labels": 0},
                    )
                return None, 0.0, {"status": "Infeasible"}

        best = incumbent_col
        best_rc = incumbent_rc

        for lab in labels:
            rc = (
                lab.reduced_cost
                - float(dual_fatigue) * lab.max_fatigue
                - float(dual_convexity)
            )
            if rc < best_rc:
                best = lab
                best_rc = rc

        elapsed = _time.perf_counter() - start
        self.stats["pricing_calls"] += 1
        self.stats["pricing_time"] += elapsed
        self.stats["label_extensions"] += total_extensions

        if best is None:
            return None, 0.0, {"status": "NoColumn", "labels": len(labels)}

        if isinstance(best, Label):
            path = best.path
            shifts_dict = {d: path[d] for d in self.D}
            fatigue_array = list(best.fatigue_array)
            max_f = float(best.max_fatigue)
            base = sum(self._preference_arc_cost(nurse, d, shifts_dict[d]) for d in self.D)
            best_column = ScheduleColumn(
                nurse,
                shifts_dict,
                float(base),
                max_f,
                fatigue_array,
            )
        else:
            best_column = best

        if require_negative and best_rc >= -CG_RC_TOL:
            return None, float(best_rc), {"status": "NoNegativeColumn", "labels": len(labels)}

        return (
            best_column,
            float(best_rc),
            {
                "status": "Negative" if best_rc < -CG_RC_TOL else "Best",
                "labels": len(labels),
            },
        )

    def _column_signature(self, col: "ScheduleColumn") -> Tuple[str, ...]:
        return col.signature()

    def _true_reduced_cost(
        self,
        nurse: str,
        col: "ScheduleColumn",
        dual_alpha: Mapping[Tuple[int, str], float],
        dual_fatigue: float,
        dual_convexity: float,
    ) -> float:
        rc = float(col.base_cost) - float(dual_convexity)
        for d in self.D:
            s = col.shifts.get(d, "OFF")
            if s in self.S:
                rc -= float(dual_alpha.get((d, s), 0.0))
        rc -= float(dual_fatigue) * float(col.max_fatigue)
        return rc

    def construct_feasible_schedule(
        self,
        nurse: str,
        rng: random.Random,
        deadline: float,
        mode: str = "balanced",
        forbidden_signatures: Optional[set] = None,
        dual_alpha: Optional[Mapping[Tuple[int, str], float]] = None,
        dual_fatigue: float = 0.0,
    ) -> Optional["ScheduleColumn"]:
        forbidden_signatures = forbidden_signatures or set()
        staff = self.inst["staff"][nurse]
        horizon = len(self.D)
        days_off = set(self.inst.get("days_off", {}).get(nurse, []))
        lengths = {s: int(self.inst["shifts"][s]["length"]) for s in self.S}
        max_len = max(lengths.values(), default=1)
        min_len = min(lengths.values(), default=1)
        min_work = int(staff["min_total_min"])
        max_work = int(staff["max_total_min"])
        maxc = int(staff.get("max_cons", horizon) or horizon)
        minc = max(1, int(staff.get("min_cons", 1) or 1))
        minoff = max(0, int(staff.get("min_consec_days_off", 0) or 0))
        max_weekends = staff.get("max_weekends")
        max_weekends = int(max_weekends) if max_weekends is not None else None
        caps = {s: int(staff.get("maxshift_per_shift", {}).get(s, horizon)) for s in self.S}

        avail_days = [d for d in self.D if d not in days_off]

        remaining_need = max(0, min_work)
        min_workdays = 0
        for length, cap in sorted(((lengths[s], caps[s]) for s in self.S if caps[s] > 0), reverse=True):
            if remaining_need <= 0:
                break
            take = min(cap, int(math.ceil(remaining_need / max(1, length))))
            min_workdays += take
            remaining_need -= take * length
        if remaining_need > 0:
            return None

        max_workdays = min(
            len(avail_days),
            int(max_work // min_len) if min_len else len(avail_days),
            sum(caps.values()),
        )
        if min_workdays > max_workdays:
            return None

        def work_runs(mask):
            runs = []
            d = 0
            while d < horizon:
                if mask[d]:
                    a = d
                    while d + 1 < horizon and mask[d + 1]:
                        d += 1
                    runs.append((a, d))
                d += 1
            return runs

        def off_runs(mask):
            runs = []
            d = 0
            while d < horizon:
                if not mask[d]:
                    a = d
                    while d + 1 < horizon and not mask[d + 1]:
                        d += 1
                    runs.append((a, d))
                d += 1
            return runs

        def weekend_count(mask):
            return sum(
                1
                for w, ds in weekend_indices(horizon).items()
                if any(mask[d] for d in ds)
            )

        def repair_status(base_mask):
            mask = list(base_mask)

            for d in days_off:
                if 0 <= d < horizon:
                    mask[d] = False

            guard = 0
            while guard < horizon * 2:
                guard += 1
                changed = False
                for a, b in work_runs(mask):
                    if b - a + 1 > maxc:
                        cut = a + maxc
                        end = min(horizon - 1, cut + max(1, minoff) - 1)
                        for d in range(cut, end + 1):
                            if d not in days_off and mask[d]:
                                mask[d] = False
                                changed = True
                        if not changed and cut < horizon:
                            mask[cut] = False
                            changed = True
                        break
                if not changed:
                    break

            for a, b in off_runs(mask):
                if a == 0 or b == horizon - 1 or b - a + 1 >= minoff:
                    continue
                need = minoff - (b - a + 1)
                left_cost = sum(1 for d in range(max(0, a - need), a) if mask[d] and d not in days_off)
                right_cost = sum(1 for d in range(b + 1, min(horizon, b + 1 + need)) if mask[d] and d not in days_off)
                if left_cost <= right_cost:
                    for d in range(max(0, a - need), a):
                        if d not in days_off:
                            mask[d] = False
                else:
                    for d in range(b + 1, min(horizon, b + 1 + need)):
                        if d not in days_off:
                            mask[d] = False

            if max_weekends is not None:
                weekends = weekend_indices(horizon)
                ordered = list(weekends.items())
                rng.shuffle(ordered)
                for w, ds in ordered:
                    if weekend_count(mask) <= max_weekends:
                        break
                    if any(mask[d] for d in ds):
                        for d in ds:
                            if d not in days_off:
                                mask[d] = False

            target_workdays = min(max_workdays, min_workdays + (1 if attempt % 3 else 2))
            while sum(mask) > target_workdays:
                removal = None
                k = max(1, minoff)
                for a, b in work_runs(mask):
                    L = b - a + 1
                    if sum(mask) - L >= min_workdays and L > 0:
                        trial = mask[:]
                        for d in range(a, b + 1):
                            if d not in days_off:
                                trial[d] = False
                        if (max_weekends is None or weekend_count(trial) <= max_weekends) and all(
                            rb - ra + 1 <= maxc for ra, rb in work_runs(trial)
                        ):
                            removal = trial
                            break

                    possible_starts = []
                    if a == 0 and L - k >= minc:
                        possible_starts.append(a)
                    if b == horizon - 1 and L - k >= minc:
                        possible_starts.append(max(a, b - k + 1))
                    if L >= 2 * minc + k:
                        possible_starts.extend(range(a + minc, b - minc - k + 2))
                    rng.shuffle(possible_starts)
                    for x in possible_starts:
                        y = min(b, x + k - 1)
                        if any(d in days_off for d in range(x, y + 1)):
                            continue
                        trial = mask[:]
                        for d in range(x, y + 1):
                            trial[d] = False
                        good = True
                        if any(rb - ra + 1 > maxc for ra, rb in work_runs(trial)):
                            good = False
                        if good and minoff > 0:
                            for oa, ob in off_runs(trial):
                                if oa == 0 or ob == horizon - 1:
                                    continue
                                if ob - oa + 1 < minoff:
                                    good = False
                                    break
                        if good and max_weekends is not None and weekend_count(trial) > max_weekends:
                            good = False
                        if good:
                            removal = trial
                            break
                    if removal is not None:
                        break
                if removal is None:
                    break
                mask = removal

            while sum(mask) < min_workdays:
                candidates = []
                for a, b in off_runs(mask):
                    L = b - a + 1
                    if a == 0:
                        for d0 in range(a, min(b + 1, maxc)):
                            Lb = min(maxc, b - d0 + 1)
                            candidates.append((0, -(Lb), d0, Lb, (a, b)))
                    elif b == horizon - 1:
                        for d0 in range(max(a, b - maxc + 1), b + 1):
                            Lb = b - d0 + 1
                            candidates.append((0, -Lb, d0, Lb, (a, b)))
                    else:
                        if L > 2 * minoff:
                            Lb = min(maxc, L - 2 * minoff)
                            for d0 in range(a + minoff, b - minoff - Lb + 2):
                                candidates.append((1, -Lb, d0, Lb, (a, b)))
                rng.shuffle(candidates)
                candidates.sort(key=lambda x: (x[0], x[1]))
                added = False
                for _kind, _negL, d0, Lb, _run in candidates:
                    trial = mask[:]
                    ok = True
                    for d in range(d0, min(horizon, d0 + Lb)):
                        if d in days_off:
                            ok = False
                            break
                        trial[d] = True
                    if not ok:
                        continue
                    if any(b - a + 1 > maxc for a, b in work_runs(trial)):
                        continue
                    if max_weekends is not None and weekend_count(trial) > max_weekends:
                        continue
                    if minoff > 0:
                        good = True
                        for a, b in off_runs(trial):
                            if a == 0 or b == horizon - 1:
                                continue
                            if b - a + 1 < minoff:
                                good = False
                                break
                        if not good:
                            continue
                    mask = trial
                    added = True
                    break
                if not added:
                    break

            if sum(mask) < min_workdays:
                return None

            return mask

        def shift_score(d, s, history, counts, workload, remaining_workdays):
            if s == "OFF":
                return 0.0
            score = self._preference_arc_cost(nurse, d, s)
            req = float(self.inst["cover"].get((d, s), {}).get("req", 0.0))
            if mode == "coverage":
                score -= 1.50 * req
            elif mode == "dual" and dual_alpha is not None:
                score -= float(dual_alpha.get((d, s), 0.0))
            if self.use_fatigue and mode in {"fatigue", "balanced", "dual"}:
                f = self._fatigue_score(nurse, history, s)
                score += 0.01 * f
                if mode == "dual":
                    score -= 0.01 * float(dual_fatigue) * f
            if mode == "random":
                score = rng.random()
            if workload >= min_work:
                score += 0.002 * lengths[s]
            else:
                score -= 0.002 * lengths[s]
            return score

        def assign_shifts(mask):
            work_days = [d for d in self.D if mask[d]]
            if not work_days:
                return None

            beam_width = 128 if len(self.S) <= 8 else 64
            nodes = 0

            def remaining_capacity(idx, counts):
                rem = len(work_days) - idx
                if rem <= 0:
                    return 0, 0
                caps_left = []
                for j, s in enumerate(self.S):
                    left = max(0, caps[s] - int(counts[j]))
                    if left > 0:
                        caps_left.extend([lengths[s]] * left)
                caps_left.sort(reverse=True)
                max_cap = sum(caps_left[:rem])
                caps_low = sorted(caps_left)
                min_cap = sum(caps_low[:rem]) if len(caps_low) >= rem else 10**18
                return max_cap, min_cap

            def score_state(d, s, history, counts, workload, rem_after):
                base = shift_score(d, s, history, counts, workload, rem_after)
                if s == "OFF":
                    return base
                need = max(0, min_work - workload)
                if rem_after > 0:
                    target = need / rem_after
                    base += 0.004 * abs(float(lengths[s]) - float(target))
                    if lengths[s] < target:
                        base += 0.01 * (target - lengths[s])
                return base

            states = [(0.0, "OFF", tuple(0 for _ in self.S), 0, ("OFF", "OFF", "OFF", "OFF"), tuple(), 0.0, tuple())]
            for idx, d in enumerate(work_days):
                if time.perf_counter() >= deadline:
                    return None
                next_states = []
                rem_after = len(work_days) - idx - 1
                for score0, last_shift, counts_t, workload0, history0, path0, maxf0, farr0 in states:
                    counts = list(counts_t)
                    for s in self.S:
                        if counts[self.shift_index[s]] >= caps[s]:
                            continue
                        if last_shift in self.inst["shifts"] and s in self.inst["shifts"][last_shift].get("forbid_next", []):
                            continue
                        new_workload = workload0 + lengths[s]
                        if new_workload > max_work:
                            continue
                        counts[self.shift_index[s]] += 1
                        max_future, min_future = remaining_capacity(idx + 1, counts)
                        if new_workload + max_future < min_work:
                            counts[self.shift_index[s]] -= 1
                            continue
                        if rem_after > 0 and new_workload + min_future > max_work:
                            counts[self.shift_index[s]] -= 1
                            continue

                        f_today = 0.0
                        new_maxf = maxf0
                        if self.use_fatigue:
                            f_today = self._fatigue_score(nurse, history0, s)
                            new_maxf = max(maxf0, f_today)
                            if self.inst.get("fatigue_limit") is not None and new_maxf > float(self.inst["fatigue_limit"]) + EPS:
                                counts[self.shift_index[s]] -= 1
                                continue

                        score = score0 + score_state(d, s, history0, counts_t, workload0, rem_after)
                        new_hist = (history0 + (s,))[-4:]
                        next_states.append((
                            score,
                            s,
                            tuple(counts),
                            new_workload,
                            new_hist,
                            path0 + (s,),
                            new_maxf,
                            farr0 + (f_today,),
                        ))
                        counts[self.shift_index[s]] -= 1

                if not next_states:
                    return None

                buckets = {}
                for st in next_states:
                    key = (st[1], st[2], st[3], st[4], round(st[6], 8))
                    old = buckets.get(key)
                    if old is None or st[0] < old[0]:
                        buckets[key] = st
                states = sorted(buckets.values(), key=lambda x: (x[0], x[6], x[3]))[:beam_width]
                nodes += len(next_states)

            valid = [st for st in states if min_work <= st[3] <= max_work and len(st[5]) == len(work_days)]
            if valid:
                best = min(valid, key=lambda st: (st[0], st[6], abs(st[3] - min_work)))
                return {d: s for d, s in zip(work_days, best[5])}

            return None

        for attempt in range(18):
            if time.perf_counter() >= deadline:
                break
            mask = [d not in days_off for d in self.D]
            if attempt > 0:
                for d in self.D:
                    if d not in days_off and rng.random() < 0.08 + 0.01 * (attempt % 4):
                        mask[d] = False
            mask = repair_status(mask)
            if mask is None:
                continue
            shifts_dict = assign_shifts(mask)
            if shifts_dict is None:
                continue

            if self.use_fatigue:
                P = self.P_Score
                fatigue_array, max_f = compute_fatigue_profile(self.inst, nurse, shifts_dict, P)
            else:
                fatigue_array, max_f = [0.0] * horizon, 0.0

            base = sum(self._preference_arc_cost(nurse, d, shifts_dict.get(d, "OFF")) for d in self.D)
            col = ScheduleColumn(nurse, {d: shifts_dict.get(d, "OFF") for d in self.D}, float(base), float(max_f), list(fatigue_array))
            sig = col.signature()
            if sig in forbidden_signatures:
                continue
            ok, _ = validate_column(self.inst, nurse, col)
            if ok:
                return col

        return None

    def solve_rmp(self, pool: Mapping[str, Sequence["ScheduleColumn"]], time_limit: float) -> dict:
        rmp_build_start = time.perf_counter()
        m = gp.Model("RMP", env=self.env)
        m.Params.OutputFlag = 1
        m.Params.Method = 1  # Simplex: duals are needed for pricing.
        m.Params.TimeLimit = max(0.05, float(time_limit))
        if _ACTIVE_RUN_LOG is not None:
            native_path = f"{_ACTIVE_RUN_LOG}.rmp_{len(_NATIVE_RMP_LOGS)+1:04d}.native"
            m.Params.LogFile = native_path
            _NATIVE_RMP_LOGS.append(native_path)

        if any(not pool[e] for e in self.E):
            m.dispose()
            raise ValueError("RMP requires one valid real column per nurse.")
        theta = {}
        for e in self.E:
            for j, _col in enumerate(pool[e]):
                theta[e, j] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)

        under = {}
        over = {}
        for d in self.D:
            for s in self.S:
                under[d, s] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
                over[d, s] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS)

        agency_total = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS) if self.gamma > 0 else None
        surge_unmet = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS) if self.gamma > 0 else None
        z_budget = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS) if self.gamma > 0 else None
        p_dual = (
            {(d, s): m.addVar(lb=0.0, vtype=GRB.CONTINUOUS) for d in self.D for s in self.S}
            if self.gamma > 0 else {}
        )

        FGM = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS) if self.use_fatigue else None

        conv = {}
        for e in self.E:
            conv[e] = m.addConstr(
                gp.quicksum(theta[e, j] for j in range(len(pool[e]))) == 1.0
            )

        coverage_rows = {}
        fatigue_rows = {}

        for d in self.D:
            for s in self.S:
                theta_cov = gp.quicksum(
                    theta[e, j]
                    for e in self.E
                    for j, col in enumerate(pool[e])
                    if col.shifts.get(d, "OFF") == s
                )
                req = float(self.inst["cover"].get((d, s), {}).get("req", 0))
                coverage_rows[d, s] = m.addConstr(
                    theta_cov + under[d, s] - over[d, s] == req
                )

        if self.use_fatigue:
            for e in self.E:
                fatigue_rows[e] = m.addConstr(
                    gp.quicksum(
                        pool[e][j].max_fatigue * theta[e, j]
                        for j in range(len(pool[e]))
                    )
                    - FGM <= 0.0
                )

        robust_row = None
        if self.gamma > 0:
            surge = self.inst["synthetic"]["surges"]
            for d in self.D:
                for s in self.S:
                    m.addConstr(
                        z_budget + p_dual[d, s] >= float(surge.get((d, s), 0))
                    )

            pooled_surge = (
                float(self.gamma) * z_budget
                + gp.quicksum(p_dual[d, s] for d in self.D for s in self.S)
            )
            pooled_over = gp.quicksum(over[d, s] for d in self.D for s in self.S)
            robust_row = m.addConstr(
                pooled_over + agency_total + surge_unmet >= pooled_surge
            )

        obj = gp.LinExpr()
        for e in self.E:
            for j, col in enumerate(pool[e]):
                obj += col.base_cost * theta[e, j]

        for d in self.D:
            for s in self.S:
                ci = self.inst["cover"].get((d, s), {})
                obj += float(ci.get("w_under", 0.0)) * under[d, s]
                obj += float(ci.get("w_over", 0.0)) * over[d, s]

        if self.gamma > 0:
            obj += self.agency_cost * agency_total
            obj += float(self.inst.get("synthetic", {}).get(
                "surge_under_penalty", DEFAULT_SURGE_UNDER_PENALTY
            )) * surge_unmet

        if self.use_fatigue:
            obj += self.fatigue_w * FGM

        m.setObjective(obj, GRB.MINIMIZE)
        remaining_rmp = float(time_limit) - (time.perf_counter() - rmp_build_start)
        if remaining_rmp <= 0.02:
            m.dispose()
            return {"status": "NotOptimal", "gurobi_status": "BuildTimeLimit"}
        m.Params.TimeLimit = remaining_rmp
        print("\n" + "-" * 80)
        print("CG - RMP Gurobi solve")
        print("-" * 80)
        m.optimize()

        if m.Status != GRB.OPTIMAL:
            status = int(m.Status)
            m.dispose()
            return {"status": "NotOptimal", "gurobi_status": status}

        dual_alpha = {(d, s): float(coverage_rows[d, s].Pi) for d in self.D for s in self.S}
        dual_conv = {e: float(conv[e].Pi) for e in self.E}
        dual_fatigue = (
            {e: float(fatigue_rows[e].Pi) for e in self.E}
            if self.use_fatigue else
            {e: 0.0 for e in self.E}
        )

        theta_values = {
            (e, j): float(theta[e, j].X)
            for e in self.E
            for j in range(len(pool[e]))
        }
        result = {
            "status": "Optimal" if m.Status == GRB.OPTIMAL else "TimeLimit",
            "objective": float(m.ObjVal),
            "dual_alpha": dual_alpha,
            "dual_conv": dual_conv,
            "dual_fatigue": dual_fatigue,
            "theta_values": theta_values,
            "robust_row": robust_row.Pi if robust_row is not None else 0.0,
        }
        try:
            m.dispose()
        except Exception:
            pass
        return result

    def solve_nurse_mip(
        self, nurse, deadline, coverage=None, forbidden_signatures=None,
        dual_alpha=None, dual_fatigue=0.0, dual_convexity=0.0,
        require_negative=False, diversify=False,
    ):
        """Gurobi nurse-level MIP pricing. The RMP itself remains an LP.

        All nurse hard constraints are imposed explicitly. Fatigue scores depend
        on the five-day shift history and are enforced exactly via lazy cuts.
        The returned column is independently validated before entering the pool.
        """
        remaining = deadline - time.perf_counter()
        if remaining <= 0.02:
            return None, "TimeLimit"
        self.stats["nurse_mip_calls"] += 1
        h, states = len(self.D), ("OFF",) + self.S
        staff = self.inst["staff"][nurse]
        dual_alpha = dual_alpha or {}
        coverage = coverage or {}
        previous = list(forbidden_signatures or ())
        m = gp.Model(f"pricing_{nurse}", env=self.env)
        try:
            m.Params.OutputFlag = 0  # RMP logs are printed; nurse solves are summarized.
            m.Params.Threads = 1
            m.Params.MIPFocus = 1 if not require_negative else 0
            m.Params.TimeLimit = max(0.02, remaining)
            if self.use_fatigue:
                m.Params.LazyConstraints = 1
            x = {(d, s): m.addVar(vtype=GRB.BINARY, name=f"x_{d}_{s}")
                 for d in self.D for s in states}
            work = {d: 1 - x[d, "OFF"] for d in self.D}
            fmax = m.addVar(lb=0.0, name="nurse_max_fatigue") if self.use_fatigue else None

            for d in self.D:
                m.addConstr(gp.quicksum(x[d, s] for s in states) == 1)
            for d in self.inst.get("days_off", {}).get(nurse, []):
                if d in self.D:
                    m.addConstr(x[d, "OFF"] == 1)
            for d in range(h - 1):
                for s in self.S:
                    for nxt in self.inst["shifts"][s].get("forbid_next", []):
                        if nxt in self.S:
                            m.addConstr(x[d, s] + x[d + 1, nxt] <= 1)
            for s in self.S:
                cap = staff.get("maxshift_per_shift", {}).get(s)
                if cap is not None:
                    m.addConstr(gp.quicksum(x[d, s] for d in self.D) <= int(cap))
            minutes = gp.quicksum(
                int(self.inst["shifts"][s]["length"]) * x[d, s]
                for d in self.D for s in self.S
            )
            m.addConstr(minutes >= int(staff["min_total_min"]))
            m.addConstr(minutes <= int(staff["max_total_min"]))
            maxc = staff.get("max_cons")
            if maxc is not None:
                maxc = int(maxc)
                for d in range(max(0, h - maxc)):
                    m.addConstr(gp.quicksum(work[t] for t in range(d, d + maxc + 1)) <= maxc)
            minc = int(staff.get("min_cons") or 1)
            minoff = int(staff.get("min_consec_days_off") or 0)
            for d in range(1, h):
                if minc > 1 and d + minc <= h:
                    for t in range(d + 1, d + minc):
                        m.addConstr(work[d] - work[d - 1] <= work[t])
                if minoff > 1 and d + minoff <= h:
                    for t in range(d + 1, d + minoff):
                        m.addConstr(x[d, "OFF"] - x[d - 1, "OFF"] <= x[t, "OFF"])
            if staff.get("max_weekends") is not None:
                weekends = []
                for days in weekend_indices(h).values():
                    y = m.addVar(vtype=GRB.BINARY)
                    weekends.append(y)
                    for d in days:
                        m.addConstr(work[d] <= y)
                m.addConstr(gp.quicksum(weekends) <= int(staff["max_weekends"]))

            # A novel column is required: otherwise pricing can repeatedly return
            # a roster already present in the restricted master.
            for sig in previous:
                if len(sig) == h and all(sig[d] in states for d in self.D):
                    m.addConstr(gp.quicksum(x[d, sig[d]] for d in self.D) <= h - 1)

            objective = gp.LinExpr()
            for d in self.D:
                for shift in states:
                    c = float(self.arc_cost[nurse, d, shift])
                    if shift != "OFF":
                        c -= float(dual_alpha.get((d, shift), 0.0))
                        if diversify or not previous:
                            info = self.inst["cover"].get((d, shift), {})
                            req = float(info.get("req", 0))
                            count = float(coverage.get((d, shift), 0))
                            c += (-0.35 * float(info.get("w_under", 0))
                                  if count < req else 0.20 * float(info.get("w_over", 0)))
                    if diversify:
                        # Deterministic variation across pricing attempts.
                        c += 0.15 * (((hash((self.seed, nurse, d, shift, len(previous))) & 31) / 31) - 0.5)
                    objective += c * x[d, shift]
            if fmax is not None:
                objective += max(0.0, -float(dual_fatigue)) * fmax
            m.setObjective(objective, GRB.MINIMIZE)

            cutoff = self.inst.get("fatigue_limit")
            callback_cuts = {"max": 0, "limit": 0, "invalid": 0}
            def lazy_fatigue(model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                roster = {d: max(states, key=lambda shift: model.cbGetSolution(x[d, shift]))
                          for d in self.D}
                history = ("OFF", "OFF", "OFF", "OFF")
                value = model.cbGetSolution(fmax)
                for d in self.D:
                    score = self._fatigue_score(nurse, history, roster[d])
                    lo = max(0, d - 4)
                    matching = gp.quicksum(x[t, roster[t]] for t in range(lo, d + 1))
                    size = d - lo + 1
                    if cutoff is not None and score > float(cutoff) + EPS:
                        model.cbLazy(matching <= size - 1)
                        callback_cuts["limit"] += 1
                    elif score > value + 1e-5:
                        # When all shifts in the history match, fmax >= score.
                        model.cbLazy(fmax >= score * (matching - size + 1))
                        callback_cuts["max"] += 1
                    history = history[1:] + (roster[d],)

            time_left = deadline - time.perf_counter()
            if time_left <= 0.02:
                return None, "TimeLimit"
            m.Params.TimeLimit = time_left
            start = time.perf_counter()
            m.optimize(lazy_fatigue if self.use_fatigue else None)
            elapsed = time.perf_counter() - start
            self.stats["pricing_calls"] += 1
            self.stats["pricing_time"] += elapsed
            status = int(m.Status)
            if status == GRB.OPTIMAL:
                self.stats["gurobi_pricing_optimal"] += 1
            elif status == GRB.TIME_LIMIT:
                self.stats["gurobi_pricing_timeout"] += 1
                self.stats["pricing_truncated"] = True

            if m.SolCount < 1:
                return None, ("ProvenInfeasible" if status == GRB.INFEASIBLE and not previous
                              else "NoMoreColumns" if status == GRB.INFEASIBLE
                              else "TimeLimit" if status == GRB.TIME_LIMIT
                              else f"GurobiStatus{status}")
            roster = {d: max(states, key=lambda shift: x[d, shift].X) for d in self.D}
            if self.use_fatigue:
                scores, maximum = compute_fatigue_profile(self.inst, nurse, roster, self.P_Score)
            else:
                scores, maximum = [0.0] * h, 0.0
            col = ScheduleColumn(
                nurse, roster,
                sum(self.arc_cost[nurse, d, roster[d]] for d in self.D),
                float(maximum), list(scores),
            )
            valid, errors = validate_column(self.inst, nurse, col)
            if not valid:
                print(f"WARNING nurse {nurse} Gurobi pricing invalid: {errors}")
                return None, "ValidationFailure"
            rc = self._true_reduced_cost(nurse, col, dual_alpha, dual_fatigue, dual_convexity)
            if require_negative and rc >= -CG_RC_TOL:
                return None, "NoNegativeColumn" if status == GRB.OPTIMAL else "PricingUnproven"
            return col, "Optimal" if status == GRB.OPTIMAL else "FeasibleTimeLimit"
        finally:
            m.dispose()

    def mutate_column(self, nurse: str, columns, seen: set, rng, deadline):
        """Fast legal-column diversification using existing nurse schedules."""
        if not columns:
            return None
        shift_lengths = {s: int(v["length"]) for s, v in self.inst["shifts"].items()}
        for _ in range(45):
            if time.perf_counter() >= deadline:
                break
            parent = rng.choice(columns)
            shifts = dict(parent.shifts)
            workdays = [d for d in self.D if shifts[d] != "OFF"]
            if not workdays:
                break
            d = rng.choice(workdays)
            present = shifts[d]
            alternatives = [s for s in self.S if s != present and
                            shift_lengths[s] == shift_lengths[present]]
            if alternatives and rng.random() < 0.75:
                shifts[d] = rng.choice(alternatives)
            else:
                offdays = [day for day in self.D if shifts[day] == "OFF"]
                if not offdays:
                    continue
                other = rng.choice(offdays)
                shifts[d], shifts[other] = shifts[other], shifts[d]
            signature = tuple(shifts[d] for d in self.D)
            if signature in seen:
                continue
            if self.use_fatigue:
                fatigue, max_f = compute_fatigue_profile(
                    self.inst, nurse, shifts, self.P_Score)
            else:
                fatigue, max_f = [0.0] * len(self.D), 0.0
            base = sum(self.arc_cost[nurse, d, shifts[d]] for d in self.D)
            candidate = ScheduleColumn(nurse, shifts, base, max_f, fatigue)
            if validate_column(self.inst, nurse, candidate)[0]:
                return candidate
        return None

    def run_column_generation(self, pool: Dict[str, List["ScheduleColumn"]], deadline: float) -> dict:
        """Gurobi RMP LP + Gurobi nurse MIP pricing, with feasibility priority."""
        start = time.perf_counter()
        seen = {e: {col.signature() for col in pool[e]} for e in self.E}
        last_theta = None
        first_feasible = None
        rounds = 0
        rmp_rounds = 0
        no_progress = 0
        status = "TimeLimit"
        lp_certified = False
        rmp_objective = None
        # First obtain one independently valid nurse schedule per nurse. Soft
        # nominal coverage then makes their combination a feasible full roster.
        missing = [e for e in self.E if not pool[e]]
        while missing and time.perf_counter() < deadline:
            progress = False
            for e in list(missing):
                remaining = deadline - time.perf_counter()
                if remaining <= 0.03:
                    break
                others = sum(1 for n in self.E if pool[n])
                coverage = defaultdict(int)
                for n in self.E:
                    if pool[n]:
                        for d, shift in pool[n][0].shifts.items():
                            if shift in self.S:
                                coverage[d, shift] += 1
                slice_sec = min(12.0, max(0.20, remaining / max(1, len(missing)) * 0.85))
                col, why = self.solve_nurse_mip(
                    e, min(deadline, time.perf_counter() + slice_sec), coverage=coverage
                )
                self.stats.setdefault("nurse_mip_status", {})[e] = why
                if col is not None:
                    pool[e].append(col)
                    seen[e].add(col.signature())
                    missing.remove(e)
                    progress = True
                    print(f"Seed nurse {e}: valid column ({len(missing)} still missing)")
                elif why == "ProvenInfeasible":
                    self.stats.setdefault("infeasible_nurses", []).append(e)
                    status = "NurseInfeasible"
                    print(f"Seed nurse {e}: independently proven infeasible")
                    break
                else:
                    print(f"Seed nurse {e}: {why}; will retry if time remains")
            if status == "NurseInfeasible":
                break
            if not progress and time.perf_counter() < deadline:
                # Retry difficult nurses with the remaining budget, but avoid a
                # tight loop when every pricing attempt immediately fails.
                if all(self.stats.get("nurse_mip_status", {}).get(e) == "ValidationFailure"
                       for e in missing):
                    status = "ValidationFailure"
                    break
        if not missing:
            first_feasible = time.perf_counter() - start
            print(f"First complete feasible roster: {first_feasible:.3f}s")
        else:
            result = dict(self.stats)
            result.update(status="NoFeasibleRoster", reason=status, converged=False,
                          lp_optimal=False, missing_nurses=missing, iterations=0,
                          rmp_rounds=0, columns_per_nurse_min=0,
                          columns_per_nurse_max=max(map(len, pool.values()), default=0),
                          first_feasible_time=None, time=time.perf_counter()-start)
            print(f"CG: missing valid schedules for {missing}; reason={status}")
            return result

        # Enrich the pool using Gurobi LP duals. Every priced roster is a valid
        # integer nurse schedule; the LP RMP chooses fractional combinations.
        while time.perf_counter() < deadline and no_progress < CG_STAGNATION_LIMIT:
            rounds += 1
            remaining = deadline - time.perf_counter()
            if remaining <= 0.10:
                break
            rmp = self.solve_rmp(pool, remaining)
            if rmp["status"] != "Optimal":
                status = "RMPTimeLimit"
                break
            rmp_objective = rmp["objective"]
            rmp_rounds += 1
            last_theta = rmp["theta_values"]
            alpha, fatigue_duals, convex_duals = (
                rmp["dual_alpha"], rmp["dual_fatigue"], rmp["dual_conv"]
            )
            added = 0
            candidates = [e for e in self.E if len(pool[e]) < CG_MAX_COLUMNS_PER_NURSE]
            candidates.sort(key=lambda e: (len(pool[e]), e))
            # Certify optimality only when EXACT original reduced-cost pricing
            # was solved for every nurse, with no heuristic diversity bias.
            all_pricing_optimal = (len(candidates) == len(self.E) and
                                   all(len(pool[e]) >= CG_TARGET_COLUMNS for e in self.E))
            any_negative = False
            coverage = defaultdict(float)
            for n in self.E:
                for d, shift in pool[n][0].shifts.items():
                    if shift in self.S:
                        coverage[d, shift] += 1
            for i, e in enumerate(candidates):
                remaining = deadline - time.perf_counter()
                if remaining < 0.08:
                    all_pricing_optimal = False
                    break
                # First seek improving reduced-cost columns, then diversify a
                # nurse's pool if there are still fewer than target columns.
                wants_diversity = len(pool[e]) < CG_TARGET_COLUMNS
                slice_sec = min(1.75, max(0.08, remaining / (len(candidates)-i) * 0.72))
                col, why = self.solve_nurse_mip(
                    e, min(deadline, time.perf_counter()+slice_sec),
                    coverage=coverage, forbidden_signatures=seen[e],
                    dual_alpha=alpha, dual_fatigue=fatigue_duals[e],
                    dual_convexity=convex_duals[e],
                    require_negative=not wants_diversity,
                    diversify=wants_diversity,
                )
                if why not in ("Optimal", "NoNegativeColumn", "NoMoreColumns"):
                    all_pricing_optimal = False
                if col is not None and col.signature() not in seen[e]:
                    rc = self._true_reduced_cost(
                        e, col, alpha, fatigue_duals[e], convex_duals[e]
                    )
                    if rc < -CG_RC_TOL:
                        any_negative = True
                    pool[e].append(col)
                    seen[e].add(col.signature())
                    added += 1
            self.stats["iterations"] = rounds
            self.stats["columns_added"] += added
            no_progress = 0 if added else no_progress + 1
            smallest = min(map(len, pool.values()))
            print(f"CG iteration={rounds} LP={rmp_objective:.4f} "
                  f"added={added} total_columns={sum(map(len,pool.values()))} "
                  f"min_columns={smallest} time={time.perf_counter()-start:.2f}s")
            if added == 0 and all_pricing_optimal:
                lp_certified = True
                status = "LPOptimal"
                break
            if (smallest >= CG_TARGET_COLUMNS and
                    rmp_rounds >= CG_MIN_RMP_ROUNDS and not any_negative):
                status = "PoolTargetReached"
                break
            if rounds >= 60:
                status = "RoundLimit"
                break
        else:
            if no_progress >= CG_STAGNATION_LIMIT:
                status = "Stagnation"
        result = dict(self.stats)
        result.update(status=status, reason=status, converged=lp_certified,
                      lp_optimal=lp_certified, missing_nurses=[],
                      iterations=rounds, rmp_rounds=rmp_rounds,
                      rmp_objective=rmp_objective,
                      columns_per_nurse_min=min(map(len, pool.values())),
                      columns_per_nurse_max=max(map(len, pool.values())),
                      first_feasible_time=first_feasible,
                      time=time.perf_counter()-start)
        print(f"CG END: {status}; total={sum(map(len,pool.values()))} columns "
              f"in {result['time']:.3f}s")
        return result

class MatheuristicExtractor:
    def __init__(
        self,
        inst: Mapping,
        column_pool: Mapping[str, Sequence["ScheduleColumn"]],
        gamma: int,
        agency_cost: float,
        fatigue_weight: float,
        seed: int = 20261007,
    ):
        self.inst = inst
        self.pool = {e: list(cols) for e, cols in column_pool.items()}
        self.nurses = list(self.pool.keys())
        self.gamma = int(gamma)
        self.agency_cost = float(agency_cost)
        self.fatigue_w = float(fatigue_weight)
        self.surge_penalty = float(
            inst.get("synthetic", {}).get("surge_under_penalty", DEFAULT_SURGE_UNDER_PENALTY)
        )
        self.D = tuple(range(int(inst["horizon"])))
        self.S = tuple(inst["shifts"].keys())
        self.rng = random.Random(int(seed))
        self.cache: Dict[Tuple, dict] = {}

    def close(self):
        self.cache.clear()

    def evaluate_solution(self, solution):
        key = solution_signature(solution)
        if key not in self.cache:
            if len(self.cache) > 2000:
                self.cache.clear()
            self.cache[key] = evaluate_solution(
                self.inst,
                solution,
                self.gamma,
                self.agency_cost,
                self.fatigue_w,
            )
        return self.cache[key]

    def _all_pool_columns(self, nurse):
        return self.pool[nurse]
    
    def _column_preference_cost(self, nurse, col):
        total = 0.0
        for e, d, s, w in self.inst.get("shift_on_requests", []):
            if e == nurse and col.shifts.get(d, "OFF") != s:
                total += float(w)
        for e, d, s, w in self.inst.get("shift_off_requests", []):
            if e == nurse and col.shifts.get(d, "OFF") == s:
                total += float(w)
        return total

    def _coverage_penalty_for_count(self, d, s, staff_count):
        req = float(self.inst["cover"].get((d, s), {}).get("req", 0.0))
        info = self.inst["cover"].get((d, s), {})
        if staff_count < req:
            return (req - staff_count) * float(info.get("w_under", 0.0)), max(0.0, req - staff_count), 0.0
        return (staff_count - req) * float(info.get("w_over", 0.0)), 0.0, max(0.0, staff_count - req)

    def _evaluate_neighbor_fast(self, solution, current_eval, changes):
        coverage = current_eval["coverage"]
        affected = set()
        pref = float(current_eval["preference_penalty"])
        over_total = float(current_eval["over_total"])
        coverage_penalty = float(current_eval["coverage_penalty"])

        for n, new_col in changes.items():
            old_col = solution.get(n)
            if old_col is not None:
                pref += float(new_col.base_cost) - float(old_col.base_cost)
                for d in self.D:
                    os = old_col.shifts.get(d, "OFF")
                    ns = new_col.shifts.get(d, "OFF")
                    if os in self.S:
                        affected.add((d, os))
                    if ns in self.S:
                        affected.add((d, ns))
            else:
                pref += float(new_col.base_cost)
                for d in self.D:
                    ns = new_col.shifts.get(d, "OFF")
                    if ns in self.S:
                        affected.add((d, ns))

        old_counts = {k: coverage.get(k, 0.0) for k in affected}
        new_counts = dict(old_counts)
        for n, new_col in changes.items():
            old_col = solution.get(n)
            for d in self.D:
                os = old_col.shifts.get(d, "OFF") if old_col is not None else "OFF"
                ns = new_col.shifts.get(d, "OFF")
                if os in self.S:
                    new_counts[d, os] = new_counts.get((d, os), 0.0) - 1.0
                if ns in self.S:
                    new_counts[d, ns] = new_counts.get((d, ns), 0.0) + 1.0

        under_total = float(current_eval.get("under_total", 0.0))
        for k in affected:
            d, s = k
            old_pen, old_under, old_over = self._coverage_penalty_for_count(d, s, old_counts[k])
            new_pen, new_under, new_over = self._coverage_penalty_for_count(d, s, new_counts[k])
            coverage_penalty += new_pen - old_pen
            under_total += new_under - old_under
            over_total += new_over - old_over

        max_fatigue = float(current_eval.get("max_fatigue", 0.0))
        touched_old_max = any(
            solution.get(n) is not None and abs(float(solution[n].max_fatigue) - max_fatigue) <= 1e-9
            for n in changes
        )
        if touched_old_max:
            max_fatigue = max(
                (float(ch.max_fatigue) for n, ch in solution.items() if n not in changes),
                default=0.0,
            )
        max_fatigue = max(max_fatigue, *(float(ch.max_fatigue) for ch in changes.values()))

        recourse = pooled_recourse(
            self.inst, over_total, self.gamma, self.agency_cost, self.surge_penalty
        ) if self.gamma > 0 else {"recourse_cost": 0.0, "surge_total": 0.0, "worst_case_shifts": tuple(), "rearranged_total": 0.0, "agency_total": 0.0, "unmet_total": 0.0, "agency_allocation": {}}

        objective = pref + coverage_penalty + float(recourse["recourse_cost"]) + self.fatigue_w * max_fatigue
        ev = dict(current_eval)
        ev.update({
            "preference_penalty": pref,
            "coverage_penalty": coverage_penalty,
            "over_total": over_total,
            "under_total": under_total,
            "max_fatigue": max_fatigue,
            "surge_total": float(recourse.get("surge_total", 0.0)),
            "rearranged_total": float(recourse.get("rearranged_total", 0.0)),
            "agency_total": float(recourse.get("agency_total", 0.0)),
            "surge_unmet": float(recourse.get("unmet_total", 0.0)),
            "recourse_cost": float(recourse.get("recourse_cost", 0.0)),
            "worst_case_shifts": tuple(recourse.get("worst_case_shifts", tuple())),
            "agency_allocation": dict(recourse.get("agency_allocation", {})),
            "objective": float(objective),
        })
        cov = dict(coverage)
        for k, v in new_counts.items():
            cov[k] = v
        ev["coverage"] = cov
        return ev

    def _best_alternative_for_nurse(
        self,
        solution: Mapping[str, "ScheduleColumn"],
        nurse: str,
        candidate_count: int = 12,
    ):
        current = solution[nurse]
        candidates = [
            c for c in self.pool[nurse]
            if c.signature() != current.signature()
        ]
        if not candidates:
            return current, self.evaluate_solution(solution)

        candidates.sort(
            key=lambda c: (
                c.base_cost + self.fatigue_w * c.max_fatigue,
                c.max_fatigue,
            )
        )
        candidates = candidates[:candidate_count]

        best_col = current
        best_eval = self.evaluate_solution(solution)
        for col in candidates:
            neighbor = dict(solution)
            neighbor[nurse] = col
            ev = self.evaluate_solution(neighbor)
            if ev["objective"] < best_eval["objective"] - EPS:
                best_col, best_eval = col, ev
        return best_col, best_eval

    def local_search(self, solution, deadline: float, max_rounds: int = 3):
        current = dict(solution)
        current_eval = self.evaluate_solution(current)
        no_move_rounds = 0

        for _round in range(max_rounds):
            if time.perf_counter() >= deadline:
                break
            improved = False
            nurse_order = list(self.nurses)
            self.rng.shuffle(nurse_order)
            for n in nurse_order:
                if time.perf_counter() >= deadline:
                    break
                current_col = current[n]
                alternatives = [c for c in self.pool[n] if c.signature() != current_col.signature()]
                if not alternatives:
                    continue
                alternatives.sort(key=lambda c: c.base_cost + self.fatigue_w * c.max_fatigue)
                alternatives = alternatives[:min(10, len(alternatives))]
                best_col = current_col
                best_eval = current_eval
                for c in alternatives:
                    ev = self._evaluate_neighbor_fast(current, current_eval, {n: c})
                    if ev["objective"] < best_eval["objective"] - EPS:
                        best_col, best_eval = c, ev
                if best_col.signature() != current_col.signature():
                    current[n] = best_col
                    current_eval = best_eval
                    improved = True
            if not improved:
                no_move_rounds += 1
                if no_move_rounds >= 1:
                    break
            else:
                no_move_rounds = 0
        return current, current_eval

    def simulated_annealing(self, initial_solution, deadline: float):
        """Multi-nurse SA with best-incumbent retention and global stagnation limit."""
        start = time.perf_counter()
        current = dict(initial_solution)
        current_eval = self.evaluate_solution(current)
        best, best_eval = dict(current), current_eval
        temperature = max(1.0, 0.02 * best_eval["objective"])
        stagnation = accepted = iterations = restarts = 0
        movable = [e for e in self.nurses if len(self.pool[e]) > 1]
        print(f"SA START | objective={best_eval['objective']:.6f} | movable={len(movable)}")
        reason = "TimeLimit"
        if not movable:
            reason = "NoAlternativeColumns"

        while movable and time.perf_counter() < deadline:
            if stagnation >= HEURISTIC_STAGNATION_LIMIT:
                reason = "Stagnation"
                break
            iterations += 1
            # Occasionally modify two nurses together to cross one-move barriers.
            count = 2 if len(movable) > 1 and self.rng.random() < 0.25 else 1
            chosen = self.rng.sample(movable, count)
            changes = {}
            for n in chosen:
                choices = [c for c in self.pool[n]
                           if c.signature() != current[n].signature()]
                if not choices:
                    continue
                if self.rng.random() < 0.70:
                    # Favor coverage-aware marginal improvements rather than only preference cost.
                    sampled = self.rng.sample(choices, min(5, len(choices)))
                    changes[n] = min(sampled, key=lambda c: self._evaluate_neighbor_fast(
                        current, current_eval, {n: c})["objective"])
                else:
                    changes[n] = self.rng.choice(choices)
            if not changes:
                stagnation += 1
                continue
            neighbor_eval = self._evaluate_neighbor_fast(current, current_eval, changes)
            delta = neighbor_eval["objective"] - current_eval["objective"]
            if delta <= 0 or self.rng.random() < math.exp(-delta / max(temperature, 1e-9)):
                current.update(changes)
                current_eval = neighbor_eval
                accepted += 1
                if current_eval["objective"] < best_eval["objective"] - EPS:
                    best, best_eval = dict(current), current_eval
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            temperature *= 0.9995
            if stagnation > 0 and stagnation % 2500 == 0:
                current, current_eval = dict(best), best_eval
                temperature = max(2.0, 0.02 * best_eval["objective"])
                restarts += 1
            if iterations % 50 == 0:
                print(f"SA iter={iterations} current={current_eval['objective']:.6f} "
                      f"best={best_eval['objective']:.6f} stagnant={stagnation} "
                      f"elapsed={time.perf_counter()-start:.3f}s")

        # Independently verify the returned incumbent (not a partial repair).
        best_eval = evaluate_solution(self.inst, best, self.gamma, self.agency_cost, self.fatigue_w)
        if not best_eval["feasible"]:
            raise RuntimeError(f"SA returned invalid roster: {best_eval['problems']}")
        print(f"SA END | reason={reason} iterations={iterations} "
              f"best={best_eval['objective']:.6f} time={time.perf_counter()-start:.3f}s")
        return best, best_eval, {
            "iterations": iterations, "accepted": accepted,
            "acceptance_rate": accepted / iterations if iterations else 0.0,
            "restarts": restarts, "stagnation": stagnation, "stop_reason": reason,
            "time": time.perf_counter() - start,
        }

    def _destroy_nurses(self, solution, op: str, q: int) -> List[str]:
        q = min(max(1, q), len(self.nurses))
        ev = self.evaluate_solution(solution)

        if op == "D_fatigue":
            return sorted(
                self.nurses,
                key=lambda n: solution[n].max_fatigue,
                reverse=True,
            )[:q]

        if op == "D_preference":
            return sorted(
                self.nurses,
                key=lambda n: self._column_preference_cost(n, solution[n]),
                reverse=True,
            )[:q]

        if op in {"D_coverage", "D_uncertainty"}:
            target = []
            coverage = ev["coverage"]

            if op == "D_uncertainty":
                surge_values = self.inst["synthetic"]["surges"]
                selected_shifts = [
                    k for _v, k in sorted(
                        ((v, k) for k, v in surge_values.items() if v > 0),
                        key=lambda x: (-x[0], x[1]),
                    )[: self.gamma]
                ]
                target = selected_shifts
            else:
                over_targets = [
                    (d, s, max(0.0, coverage.get((d, s), 0.0) - float(self.inst["cover"].get((d, s), {}).get("req", 0))))
                    for d in self.D for s in self.S
                ]
                over_targets = sorted(over_targets, key=lambda x: x[2], reverse=True)
                target = [(d, s) for d, s, amount in over_targets if amount > EPS][:max(1, q * 2)]
                if not target:
                    target = [
                        (d, s)
                        for d in self.D
                        for s in self.S
                        if coverage.get((d, s), 0.0)
                        < float(self.inst["cover"].get((d, s), {}).get("req", 0))
                    ][:max(1, q * 2)]

            if target:
                contribution = {}
                for n in self.nurses:
                    score = 0.0
                    col = solution[n]
                    for d, s in target:
                        if col.shifts.get(d, "OFF") == s:
                            req = float(self.inst["cover"].get((d, s), {}).get("req", 0))
                            over_amt = max(0.0, coverage.get((d, s), 0.0) - req)
                            under_amt = max(0.0, req - coverage.get((d, s), 0.0))
                            score += 2.0 * over_amt + 0.5 * under_amt + 1.0
                    contribution[n] = score

                ranked = sorted(self.nurses, key=lambda n: contribution[n], reverse=True)
                if ranked and contribution[ranked[0]] > 0:
                    return ranked[:q]

        return self.rng.sample(self.nurses, q)

    def alns(self, initial_solution, deadline: float, q_min: int = 2, q_max: int = 8):
        """Adaptive large-neighborhood search with full-roster repairs and restarts."""
        start = time.perf_counter()
        current = dict(initial_solution)
        current_eval = self.evaluate_solution(current)
        best, best_eval = dict(current), current_eval
        operators = ["D_fatigue", "D_uncertainty", "D_coverage", "D_preference", "D_random"]
        weights = {op: 1.0 for op in operators}
        iterations = accepted = improvements = stagnation = restarts = 0
        temperature = max(2.0, 0.02 * best_eval["objective"])
        reason = "TimeLimit"
        movable = [n for n in self.nurses if len(self.pool[n]) > 1]
        print(f"ALNS START | objective={best_eval['objective']:.6f} | movable={len(movable)}")
        if not movable:
            reason = "NoAlternativeColumns"

        while movable and time.perf_counter() < deadline:
            if stagnation >= HEURISTIC_STAGNATION_LIMIT:
                reason = "Stagnation"
                break
            iterations += 1
            op = self.rng.choices(operators, [weights[o] for o in operators], k=1)[0]
            q = self.rng.randint(1, min(q_max, len(movable)))
            selected = [n for n in self._destroy_nurses(current, op, min(len(self.nurses), q*2))
                        if n in movable][:q]
            if not selected:
                selected = self.rng.sample(movable, min(q, len(movable)))
            self.rng.shuffle(selected)
            candidate = dict(current)  # Keep original valid columns as a fallback.
            candidate_eval = current_eval
            # Sequential regret/randomized repair, considering the entire legal column pool.
            for n in selected:
                if time.perf_counter() >= deadline:
                    break
                options = [c for c in self.pool[n] if c.signature() != candidate[n].signature()]
                if not options:
                    continue
                self.rng.shuffle(options)
                scored = sorted(
                    ((self._evaluate_neighbor_fast(candidate, candidate_eval, {n: c})["objective"], c)
                     for c in options[:min(18, len(options))]),
                    key=lambda item: item[0],
                )
                if not scored:
                    continue
                # Greedy most often, but occasional randomized repair preserves diversity.
                top = min(3, len(scored))
                _, new_col = (scored[0] if self.rng.random() < 0.75
                              else self.rng.choice(scored[:top]))
                candidate_eval = self._evaluate_neighbor_fast(candidate, candidate_eval, {n: new_col})
                candidate[n] = new_col
            delta = candidate_eval["objective"] - current_eval["objective"]
            reward = 0.2
            if candidate_eval["objective"] < best_eval["objective"] - EPS:
                best, best_eval = dict(candidate), candidate_eval
                current, current_eval = dict(candidate), candidate_eval
                accepted += 1
                improvements += 1
                stagnation = 0
                reward = 6.0
            else:
                stagnation += 1
                if delta <= 0 or self.rng.random() < math.exp(-delta / max(temperature, 1e-9)):
                    current, current_eval = dict(candidate), candidate_eval
                    accepted += 1
                    reward = 2.0 if delta < 0 else 0.5
            weights[op] = max(0.1, min(20.0, 0.9 * weights[op] + 0.1 * reward))
            temperature *= 0.9995
            if stagnation > 0 and stagnation % 2500 == 0:
                current, current_eval = dict(best), best_eval
                temperature = max(2.0, 0.02 * best_eval["objective"])
                restarts += 1
            if iterations % 50 == 0:
                print(f"ALNS iter={iterations} current={current_eval['objective']:.6f} "
                      f"best={best_eval['objective']:.6f} stagnant={stagnation} "
                      f"elapsed={time.perf_counter()-start:.3f}s")

        best_eval = evaluate_solution(self.inst, best, self.gamma, self.agency_cost, self.fatigue_w)
        if not best_eval["feasible"]:
            raise RuntimeError(f"ALNS returned invalid roster: {best_eval['problems']}")
        print(f"ALNS END | reason={reason} iterations={iterations} "
              f"best={best_eval['objective']:.6f} time={time.perf_counter()-start:.3f}s")
        return best, best_eval, {
            "iterations": iterations, "accepted": accepted,
            "acceptance_rate": accepted / iterations if iterations else 0.0,
            "improvements": improvements, "restarts": restarts,
            "stagnation": stagnation, "stop_reason": reason,
            "destroy_weights": weights, "time": time.perf_counter() - start,
        }


def format_shifts(keys: Sequence[Tuple[int, str]]) -> str:
    return ", ".join(f"Day {d} / {s}" for d, s in keys) if keys else "None"


def render_instance_details(inst: Mapping, use_equity: bool, gamma: int):
    st.header("Instance Details")

    with st.expander("📌 Translated Instance Details", expanded=True):
        st.write(f"**Horizon:** {inst['horizon']} days")
        st.write(f"**Nurses:** {len(inst['staff'])}")
        st.write(f"**Shift types:** {', '.join(inst['shifts'].keys())}")
        st.write(
            f"**Coverage rows:** {len(inst.get('cover', {}))} "
            f"(requirements are soft; under/over penalties are preserved from the benchmark instance)"
        )

        st.markdown("**Shift definitions**")
        shift_rows = []
        for s, info in inst["shifts"].items():
            shift_rows.append({
                "Shift": s,
                "Length (min)": info["length"],
                "Cannot Follow": ", ".join(info.get("forbid_next", [])) or "-",
                "Fatigue Type": info.get("fatigue_type", fatigue_type(s, inst["shifts"])),
            })
        st.dataframe(pd.DataFrame(shift_rows), use_container_width=True)

        staff_rows = []
        for e, info in inst["staff"].items():
            staff_rows.append({
                "Nurse": e,
                "Min Work (min)": info["min_total_min"],
                "Max Work (min)": info["max_total_min"],
                "Min Consecutive Work": info["min_cons"],
                "Max Consecutive Work": info["max_cons"],
                "Min Consecutive Off": info["min_consec_days_off"],
                "Max Weekends": info["max_weekends"],
                "Days Off": ", ".join(map(str, inst.get("days_off", {}).get(e, []))) or "-",
            })
        st.markdown("**Nurse constraints**")
        st.dataframe(pd.DataFrame(staff_rows), use_container_width=True)

    with st.expander("📌 Equity Details"):
        syn = inst.get("synthetic", {})
        if not use_equity:
            st.write("Equity padding is disabled.")
        else:
            rows = []
            for e in inst["staff"]:
                rows.append({
                    "Nurse": e,
                    "Original Weight": syn.get("original_preference_weights", {}).get(e, 0.0),
                    "Added Padding": syn.get("padding_added", {}).get(e, 0.0),
                    "Final Weight": syn.get("final_preference_weights", {}).get(e, 0.0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            padding_rows = syn.get("padding_requests", [])
            if padding_rows:
                st.markdown("**Added padding requests**")
                st.dataframe(pd.DataFrame(padding_rows), use_container_width=True)

    if gamma > 0:
        with st.expander("📌 Budget of Uncertainty Details", expanded=True):
            syn = inst["synthetic"]
            st.write(f"Gamma = **{gamma}**. Generated from the experiment seed **{syn['seed']}**.")
            surge_rows = []
            for d in range(inst["horizon"]):
                for s in inst["shifts"]:
                    surge_rows.append({
                        "Day": d,
                        "Shift": s,
                        "Nominal Demand": int(inst["cover"].get((d, s), {}).get("req", 0)),
                        "Worst-Case Surge": int(syn["surges"][(d, s)]),
                    })
            st.dataframe(pd.DataFrame(surge_rows), use_container_width=True)

def render_result(res: dict, use_fatigue: bool, gamma: int):
    if not res:
        st.error("No result was produced.")
        return

    status = res.get("status", "Unknown")

    if status == "NoFeasibleRoster":
        st.error(res.get("message", "No complete feasible roster was generated."))
        return

    if not res.get("assign"):
        st.error("No roster was produced.")
        return
    if status in {"Optimal", "Converged", "Success"}:
        st.success(f"Roster generated in {res.get('time', 0.0):.2f} seconds.")
    elif status == "TimeLimit":
        st.warning(f"Best roster found within the {res.get('time', 0.0):.2f}-second budget.")
    else:
        st.warning(f"Solver status: {status}")

    ev = res.get("evaluation", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Objective", f"{ev.get('objective', res.get('objective', 0.0)):,.2f}")
    c2.metric("Agency", f"{ev.get('agency_total', 0.0):.0f}")
    c3.metric("Surge Under", f"{ev.get('surge_unmet', 0.0):.0f}")
    c4.metric("Rearrangement Buffer", f"{ev.get('rearranged_total', 0.0):.0f}")
    c5.metric(
        "Global Max Fatigue",
        f"{ev.get('max_fatigue', 0.0):.2f}" if use_fatigue else "Disabled"
    )

    if gamma > 0:
        st.caption(
            f"Gamma worst-case shifts: {format_shifts(ev.get('worst_case_shifts', tuple()))} | "
            f"Surge total: {ev.get('surge_total', 0.0):.1f} | "
            f"Agency is used only when agency cost is below the surge-undercoverage penalty "
            f"({ev.get('surge_under_penalty', 100.0):.0f}); otherwise residual surge is left undercovered."
        )

    st.subheader("Final Schedule")
    rows = []
    for e in res["E"]:
        row = {"Nurse ID": e}
        for d in range(res["horizon"]):
            row[f"Day {d}"] = res["assign"][e][d]
        rows.append(row)

    if gamma > 0:
        row = {"Nurse ID": "AGENCY BUFFER"}
        agency_alloc = ev.get("agency_allocation", {})
        for d in range(res["horizon"]):
            vals = []
            for s in res["S"]:
                amount = float(agency_alloc.get((d, s), 0.0))
                if amount > EPS:
                    vals.append(f"+{amount:.0f}({s})")
            row[f"Day {d}"] = " ".join(vals) if vals else "-"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if use_fatigue:
        st.subheader("Fatigue Scores (4-Day Rolling Window)")
        frows = []
        for e in res["E"]:
            col = res["columns"][e]
            row = {"Nurse ID": e}
            for d, score in enumerate(col.fatigue_array):
                row[f"Day {d}"] = f"{score:.1f}" if score > EPS else "-"
            frows.append(row)
        st.dataframe(pd.DataFrame(frows), use_container_width=True)

    with st.expander("Objective Breakdown"):
        st.json({
            "preference_penalty": ev.get("preference_penalty", 0.0),
            "nominal_coverage_penalty": ev.get("coverage_penalty", 0.0),
            "nominal_undercoverage": ev.get("under_total", 0.0),
            "nominal_overcoverage": ev.get("over_total", 0.0),
            "rearrangement_buffer_used": ev.get("rearranged_total", 0.0),
            "agency_nurses": ev.get("agency_total", 0.0),
            "residual_surge_undercoverage": ev.get("surge_unmet", 0.0),
            "uncertainty_recourse_cost": ev.get("recourse_cost", 0.0),
            "global_max_fatigue": ev.get("max_fatigue", 0.0),
            "objective": ev.get("objective", 0.0),
            "best_bound": res.get("best_bound"),
            "gap": res.get("gap"),
            "objective_consistency_error": res.get("objective_consistency_error"),
        })

@contextmanager
def capture_run_log(method_name: str):
    """Tee Python output; append CG RMP Gurobi solver logs at run completion."""
    global _ACTIVE_RUN_LOG, _NATIVE_RMP_LOGS
    log_dir = os.path.join(BASE_DIR, "run_logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = method_name.replace(" ", "_").replace("+", "_")

    log_path = os.path.join(
        log_dir,
        f"{safe_name}_{timestamp}.log",
    )

    original_stdout = sys.__stdout__
    original_stderr = sys.__stderr__
    old_active, old_native = _ACTIVE_RUN_LOG, _NATIVE_RMP_LOGS
    _ACTIVE_RUN_LOG, _NATIVE_RMP_LOGS = log_path, []

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                try:
                    if stream is not None and not getattr(stream, "closed", False):
                        stream.write(data)
                except (ValueError, OSError):
                    pass
            return len(data)

        def flush(self):
            for stream in self.streams:
                try:
                    if stream is not None and not getattr(stream, "closed", False):
                        stream.flush()
                except (ValueError, OSError):
                    pass

        def isatty(self):
            return False

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        try:
            print("=" * 80)
            print(f"RUN STARTED: {datetime.now().isoformat()}")
            print(f"METHOD: {method_name}")
            print(f"LOG FILE: {log_path}")
            print("=" * 80)

            yield log_path

        except Exception:
            import traceback
            traceback.print_exc()
            raise

        finally:
            print("=" * 80)
            print(f"RUN ENDED: {datetime.now().isoformat()}")
            print(f"LOG SAVED TO: {log_path}")
            print("=" * 80)

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.flush()
            # Native Gurobi output is emitted directly by the C library, not sys.stdout.
            for native_path in _NATIVE_RMP_LOGS:
                try:
                    if os.path.exists(native_path):
                        log_file.write(f"\n----- GUROBI RMP LOG: {os.path.basename(native_path)} -----\n")
                        with open(native_path, "r", encoding="utf-8", errors="replace") as native:
                            log_file.write(native.read())
                        os.remove(native_path)
                except OSError as exc:
                    log_file.write(f"\nCould not append native log: {exc}\n")
            _ACTIVE_RUN_LOG, _NATIVE_RMP_LOGS = old_active, old_native

def main():
    st.set_page_config(
        page_title="Robust Fatigue-Aware Nurse Rostering",
        layout="wide",
        page_icon="🏥",
    )
    st.title("Robust Fatigue-Aware Nurse Rostering")

    st.sidebar.header("Solver")
    direct_solver = st.sidebar.radio("Direct MIP Solver", ["Gurobi", "HiGHS"])

    st.sidebar.header("Model Features")
    use_fatigue = st.sidebar.toggle("Enable Fatigue", value=False)
    fatigue_weight = st.sidebar.slider(
        "Global Max Fatigue Weight",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.5,
    )
    use_equity = st.sidebar.toggle("Enable Equity", value=False)
    gamma_budget = st.sidebar.slider("Budget of Uncertainty (Gamma)", 0, 10, 0)
    agency_cost = st.sidebar.number_input(
        "Agency Nurse Cost",
        min_value=1.0,
        max_value=10_000.0,
        value=20.0,
        step=1.0,
    )
    surge_randomization = st.sidebar.toggle("Randomize Uncertainty", value=True)
    experiment_seed = st.sidebar.number_input(
        "Experiment Seed",
        min_value=0,
        max_value=2_000_000_000,
        value=20261007,
        step=1,
    )
    time_budget = st.sidebar.number_input(
        "Time Limit (seconds)",
        min_value=10.0,
        max_value=3600.0,
        value=300.0,
        step=10.0,
    )

    st.sidebar.header("Data Source")
    data_mode = st.sidebar.radio(
        "Select Instance",
        ["Upload Benchmark .txt", "Custom Instance Builder"],
    )

    raw_inst = None

    if data_mode == "Upload Benchmark .txt":
        file = st.sidebar.file_uploader(
            "Upload one of Instances 1-24",
            type=["txt"],
        )
        if file is not None:
            raw_inst = parse_standard_nrp(file.getvalue().decode("utf-8-sig"))

    else:
        st.sidebar.caption("Debug/custom instances are for development, not benchmark reporting.")
        n_nurses = st.sidebar.number_input("Number of Nurses", 1, 50, 15)
        n_weeks = st.sidebar.number_input("Horizon (weeks)", 1, 8, 2)
        shifts = st.sidebar.multiselect(
            "Shift Types",
            ["Day", "Evening", "Night", "Early", "Late"],
            default=["Day", "Night"],
        )
        if st.sidebar.button("Generate Custom Instance"):
            text = build_custom_instance(n_weeks, n_nurses, shifts)
            st.session_state["custom_inst"] = parse_standard_nrp(text)
        raw_inst = st.session_state.get("custom_inst")

    if raw_inst is None:
        st.info("Upload one of the 24 benchmark instances to begin.")
        return

    inst = prepare_thesis_data(
        raw_inst,
        use_equity_proxy=use_equity,
        seed=int(experiment_seed),
        randomize_surges=surge_randomization,
    )

    render_instance_details(inst, use_equity, int(gamma_budget))

    st.header("Optimization")
    execution_mode = st.radio(
        "Execution Strategy",
        [
            "Direct MIP",
            "CG + SA",
            "CG + ALNS",
            "Compare all 3",
        ],
        horizontal=True,
    )

    if not st.button("🚀 Execute Schedule", type="primary"):
        return

    def run_direct():
        with st.spinner("Running direct MIP..."):
            return solve_thesis_model(
                inst,
                use_fatigue=use_fatigue,
                use_equity=use_equity,
                gamma_budget=int(gamma_budget),
                agency_cost=float(agency_cost),
                fatigue_weight=float(fatigue_weight),
                time_limit=float(time_budget),
                solver_choice=direct_solver,
            )

    def run_matheuristic(mode: str, seed_offset: int = 0):
        start = time.perf_counter()
        # The hybrid is capped at five minutes even when direct MIP gets an hour.
        budget = min(float(time_budget), HYBRID_MAX_SECONDS)
        deadline = start + budget
        pool = {n: [] for n in inst["staff"]}
        cg = None
        extractor = None
        cg_result, hstats = {}, {}
        cg_elapsed = heuristic_elapsed = 0.0
        try:
            cg_start = time.perf_counter()
            cg = ColumnGenerationEngine(
                inst, agency_cost=float(agency_cost), use_fatigue=use_fatigue,
                fatigue_weight=float(fatigue_weight), gamma_budget=int(gamma_budget),
                seed=int(experiment_seed) + int(seed_offset),
            )
            cg_deadline = min(deadline, cg_start + min(CG_MAX_SECONDS, 0.80 * budget))
            with st.spinner("Generating valid columns and improving the restricted master..."):
                cg_result = cg.run_column_generation(pool, cg_deadline)
            cg_elapsed = time.perf_counter() - cg_start
            print(f"CG COMPLETE | time={cg_elapsed:.6f}s | "
                  f"status={cg_result['status']} | LP optimal={cg_result.get('lp_optimal', False)} | "
                  f"minimum columns/nurse={cg_result['columns_per_nurse_min']}")

            missing = [n for n, cols in pool.items() if not cols]
            if missing:
                certified = sorted(set(cg.stats.get("infeasible_nurses", [])))
                explanation = (
                    f"Hard constraints are individually infeasible for: {', '.join(certified)}. "
                    if certified else
                    "Search exhausted its allocated time; infeasibility is not proven. "
                )
                return {
                    "status": "NoFeasibleRoster", "message":
                        f"No valid column for {len(missing)} nurse(s): "
                        f"{', '.join(map(str, missing[:15]))}. " + explanation,
                    "assign": {}, "columns": {}, "evaluation": {}, "objective": None,
                    "time": time.perf_counter() - start,
                    "E": list(inst["staff"]), "S": list(inst["shifts"]),
                    "horizon": inst["horizon"], "cg": cg_result,
                    "heuristic_stats": {}, "best_bound": None, "gap": None,
                }

            # Real columns only: soft nominal coverage guarantees this is a feasible roster.
            extractor = MatheuristicExtractor(
                inst, pool, int(gamma_budget), float(agency_cost), float(fatigue_weight),
                seed=int(experiment_seed) + 1_000_003 + int(seed_offset),
            )
            seed_solution = {n: min(pool[n], key=lambda c: c.base_cost +
                                 float(fatigue_weight) * c.max_fatigue) for n in pool}
            seed_solution, seed_eval = extractor.local_search(
                seed_solution, deadline, max_rounds=2)
            if not seed_eval["feasible"]:
                raise RuntimeError(f"Invalid generated roster: {seed_eval['problems']}")
            print(f"FIRST ROSTER | objective={seed_eval['objective']:.6f} "
                  f"elapsed={time.perf_counter()-start:.3f}s")
            heuristic_start = time.perf_counter()
            if mode == "SA":
                with st.spinner("Running simulated annealing..."):
                    best_sched, best_eval, hstats = extractor.simulated_annealing(seed_solution, deadline)
            else:
                with st.spinner("Running ALNS..."):
                    best_sched, best_eval, hstats = extractor.alns(seed_solution, deadline)
            heuristic_elapsed = time.perf_counter() - heuristic_start
            elapsed = time.perf_counter() - start
            print(f"HYBRID SUMMARY | CG={cg_elapsed:.6f}s | "
                  f"heuristic={heuristic_elapsed:.6f}s | total={elapsed:.6f}s | "
                  f"objective={best_eval['objective']:.6f} | "
                  f"stop={hstats['stop_reason']}")
            return {
                "status": "Success" if best_eval["feasible"] else "Failure",
                "assign": {n: dict(col.shifts) for n, col in best_sched.items()},
                "columns": best_sched, "objective": best_eval["objective"],
                "evaluation": best_eval, "time": elapsed,
                "E": list(cg.E), "S": list(cg.S), "horizon": inst["horizon"],
                "cg": cg_result, "heuristic_stats": dict(hstats,
                    initial_objective=seed_eval["objective"], cg_time=cg_elapsed,
                    heuristic_time=heuristic_elapsed),
                "best_bound": None, "gap": None, "objective_consistency_error": None,
            }
        finally:
            if extractor is not None:
                extractor.close()
            if cg is not None:
                cg.close()

    if execution_mode == "Direct MIP":
        with capture_run_log("Direct_MIP") as log_path:
            res = run_direct()

        st.caption(f"Console log saved: {log_path}")
        render_result(res, use_fatigue, int(gamma_budget))
        return

    if execution_mode in {"CG + SA", "CG + ALNS"}:
        mode = "SA" if execution_mode == "CG + SA" else "ALNS"

        with capture_run_log(f"CG_{mode}") as log_path:
            res = run_matheuristic(mode)

        st.caption(f"Console log saved: {log_path}")
        render_result(res, use_fatigue, int(gamma_budget))

        with st.expander("Algorithm statistics"):
            st.json({
                "cg": res.get("cg", {}),
                "heuristic": res.get("heuristic_stats", {}),
            })
        return

    st.subheader("Comparison")

    with capture_run_log("Direct_MIP") as direct_log:
        b_direct = run_direct()

    with capture_run_log("CG_SA") as sa_log:
        b_sa = run_matheuristic("SA", 10_001)

    with capture_run_log("CG_ALNS") as alns_log:
        b_alns = run_matheuristic("ALNS", 20_001)

    st.caption(
        f"Logs saved: {direct_log} | {sa_log} | {alns_log}"
    )

    benchmark = [
        ("Direct MIP", b_direct),
        ("CG + SA", b_sa),
        ("CG + ALNS", b_alns),
    ]

    rows = []
    for name, result in benchmark:
        ev = result.get("evaluation", {})
        rows.append({
            "Method": name,
            "Status": result.get("status"),
            "Objective": ev.get("objective"),
            "Time (s)": result.get("time"),
            "Best Bound": result.get("best_bound"),
            "Gap": result.get("gap"),
            "Agency": ev.get("agency_total", 0.0),
            "Surge Under": ev.get("surge_unmet", 0.0),
            "Rearrangement Buffer": ev.get("rearranged_total", 0.0),
            "Global Max Fatigue": ev.get("max_fatigue", 0.0),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.info("Detailed result: CG + ALNS")
    render_result(b_alns, use_fatigue, int(gamma_budget))


if __name__ == "__main__":
    main()
