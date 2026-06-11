import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import math
import re
import random
from collections import defaultdict
from datetime import datetime, timedelta

# ==============================================================================
# 1. PARSING, DATA PREPARATION & HELPERS
# ==============================================================================

def weekend_indices(horizon):
    """Helper function to find weekend day indices (assuming Day 0 is Monday)"""
    weekends = defaultdict(list)
    for d in range(horizon):
        if d % 7 in (5, 6): 
            weekends[d // 7].append(d)
    return dict(weekends)

def parse_standard_nrp(text):
    """Parses standard schedulingbenchmarks.org NRP format."""
    lines = [ln.replace('\ufeff', '').strip() for ln in text.splitlines() if ln.strip() and not ln.startswith(('#', '//'))]
    data = defaultdict(list)
    section = None
    for ln in lines:
        if ln.startswith('SECTION_'):
            section = ln[len('SECTION_'):].strip()
            continue
        if section is not None:
            data[section].append(ln)

    horizon = int(data['HORIZON'][0]) if 'HORIZON' in data and data['HORIZON'] else 14
    shifts = {}
    for ln in data.get('SHIFTS', []):
        parts = [p.strip() for p in ln.split(',')]
        shifts[parts[0]] = {'length': int(parts[1]), 'forbid_next': [x for x in parts[2].split('|') if x] if len(parts) > 2 else []}

    staff = {}
    for ln in data.get('STAFF', []):
        parts = [p.strip() for p in ln.split(',')]
        eid = parts[0]
        maxshift_per_shift = {}
        for token in (parts[1] if len(parts) > 1 else '').split('|'):
            if '=' in token:
                shiftid, val = token.split('=')
                maxshift_per_shift[shiftid.strip()] = int(val.strip())
        
        staff[eid] = {
            'maxshift_per_shift': maxshift_per_shift,
            'max_total_min': int(parts[2]) if len(parts) > 2 and parts[2] != '' else 10000,
            'min_total_min': int(parts[3]) if len(parts) > 3 and parts[3] != '' else 0,
            'max_cons': int(parts[4]) if len(parts) > 4 and parts[4] != '' else 6,
            'min_cons': int(parts[5]) if len(parts) > 5 and parts[5] != '' else 2,
            'min_consec_days_off': int(parts[6]) if len(parts) > 6 and parts[6] != '' else 2,
            'max_weekends': int(parts[7]) if len(parts) > 7 and parts[7] != '' else 2
        }

    days_off = defaultdict(list)
    for ln in data.get('DAYS_OFF', []):
        parts = [p.strip() for p in ln.split(',')]
        days_off[parts[0]] = [int(x) for x in parts[1:] if x != '']

    shift_on = [ (p[0], int(p[1]), p[2], int(p[3])) for ln in data.get('SHIFT_ON_REQUESTS', []) if len(p := [x.strip() for x in ln.split(',')]) >= 4 ]
    shift_off = [ (p[0], int(p[1]), p[2], int(p[3])) for ln in data.get('SHIFT_OFF_REQUESTS', []) if len(p := [x.strip() for x in ln.split(',')]) >= 4 ]

    cover = {}
    for ln in data.get('COVER', []):
        parts = [p.strip() for p in ln.split(',')]
        if len(parts) >= 5:
            cover[(int(parts[0]), parts[1])] = {'req': int(parts[2]), 'w_under': float(parts[3]), 'w_over': float(parts[4])}

    return {'horizon': horizon, 'shifts': shifts, 'staff': staff, 'days_off': dict(days_off), 'shift_on_requests': shift_on, 'shift_off_requests': shift_off, 'cover': cover}

def parse_inrc1_format(text):
    """Translates INRC-I (2010) format with dates and contracts into the universal dictionary."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    start_date = None
    horizon = 28
    shifts, contracts, staff, cover = {}, {}, {}, {}
    shift_off_requests = []
    
    mode = None
    for ln in lines:
        if ln.startswith('//'): continue
        ln = ln.rstrip(';')
        
        if ln.startswith("SCHEDULING_PERIOD"): mode = "PERIOD"; continue
        if ln.startswith("SKILLS"): mode = "SKILLS"; continue
        if ln.startswith("SHIFT_TYPES"): mode = "SHIFTS"; continue
        if ln.startswith("CONTRACTS"): mode = "CONTRACTS"; continue
        if ln.startswith("PATTERNS"): mode = "PATTERNS"; continue
        if ln.startswith("EMPLOYEES"): mode = "EMPLOYEES"; continue
        if ln.startswith("DAY_OF_WEEK_COVER"): mode = "DOW_COVER"; continue
        if ln.startswith("DATE_SPECIFIC_COVER"): mode = "DATE_COVER"; continue
        if ln.startswith("DAY_OFF_REQUESTS"): mode = "DAY_OFF_REQ"; continue
        if ln.startswith("DAY_ON_REQUESTS"): mode = "DAY_ON_REQ"; continue
        if ln.startswith("SHIFT_OFF_REQUESTS"): mode = "SHIFT_OFF_REQ"; continue
        if ln.startswith("SHIFT_ON_REQUESTS"): mode = "SHIFT_ON_REQ"; continue
        
        parts = [p.strip() for p in ln.split(',')]
        
        if mode == "PERIOD" and len(parts) >= 3:
            start_date = datetime.strptime(parts[1], "%Y-%m-%d")
            end_date = datetime.strptime(parts[2], "%Y-%m-%d")
            horizon = (end_date - start_date).days + 1
            
        elif mode == "SHIFTS" and len(parts) >= 2:
            shifts[parts[0]] = {'length': 480, 'forbid_next': []}
            
        elif mode == "CONTRACTS" and len(parts) > 3:
            c_id = parts[0]
            def ex(idx, default):
                if len(parts) > idx and '|' in parts[idx]:
                    vals = parts[idx].strip('()').split('|')
                    if len(vals) >= 3 and vals[1] == '1': return int(vals[2])
                return default
            contracts[c_id] = {
                'max_total_min': ex(3, 100) * 480, 'min_total_min': ex(4, 0) * 480,
                'max_cons': ex(5, 6), 'min_cons': ex(6, 2),
                'min_consec_days_off': ex(8, 2), 'max_weekends': ex(11, 4)
            }
            
        elif mode == "EMPLOYEES" and len(parts) >= 3:
            e_id = parts[0]
            c_id = parts[2]
            staff[e_id] = contracts.get(c_id, {'max_total_min': 40000, 'min_total_min': 0, 'max_cons': 6, 'min_cons': 2, 'min_consec_days_off': 2, 'max_weekends': 4}).copy()
            staff[e_id]['maxshift_per_shift'] = {s: horizon for s in shifts}
            
        elif mode == "DOW_COVER" and len(parts) >= 3:
            dow_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
            target_dow = dow_map.get(parts[0], 0)
            if start_date:
                for d in range(horizon):
                    if (start_date + timedelta(days=d)).weekday() == target_dow:
                        cover[(d, parts[1])] = {'req': int(parts[2]), 'w_under': 100.0, 'w_over': 10.0}
                        
        elif mode == "DAY_OFF_REQ" and len(parts) >= 3:
            if start_date:
                d_idx = (datetime.strptime(parts[1], "%Y-%m-%d") - start_date).days
                if 0 <= d_idx < horizon:
                    for s_id in shifts: shift_off_requests.append((parts[0], d_idx, s_id, int(parts[2])))
                    
        elif mode == "SHIFT_OFF_REQ" and len(parts) >= 4:
            if start_date:
                d_idx = (datetime.strptime(parts[1], "%Y-%m-%d") - start_date).days
                if 0 <= d_idx < horizon:
                    shift_off_requests.append((parts[0], d_idx, parts[2], int(parts[3])))

    return {'horizon': horizon, 'shifts': shifts, 'staff': staff, 'days_off': {}, 'shift_on_requests': [], 'shift_off_requests': shift_off_requests, 'cover': cover}

def parse_inrc2_format(text):
    """Translates INRC-II format (WEEK_DATA) into the universal dictionary."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    horizon = 7
    days_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    
    shifts = {'Early': {'length': 480, 'forbid_next': []}, 
              'Day': {'length': 480, 'forbid_next': []}, 
              'Late': {'length': 480, 'forbid_next': ['Early']}, 
              'Night': {'length': 480, 'forbid_next': ['Early', 'Day', 'Late']}}
    
    cover, shift_off, staff_ids, mode = {}, [], set(), None
    for ln in lines:
        if ln.startswith("REQUIREMENTS"): mode = "REQ"; continue
        elif ln.startswith("SHIFT_OFF_REQUESTS"): mode = "OFF"; continue
            
        if mode == "REQ" and not ln.startswith("SHIFT"):
            parts = ln.split()
            if len(parts) >= 9:
                shift_type = parts[0]
                for d in range(7):
                    req_tuple = parts[d+2].replace('(', '').replace(')', '').split(',')
                    min_req = int(req_tuple[0])
                    if (d, shift_type) not in cover: cover[(d, shift_type)] = {'req': 0, 'w_under': 100, 'w_over': 10}
                    cover[(d, shift_type)]['req'] += min_req
                    
        elif mode == "OFF":
            parts = ln.split()
            if "=" in ln: continue 
            if len(parts) >= 3:
                emp, shift_type, day_str = parts[0], parts[1], parts[2]
                day_idx = days_map.get(day_str, 0)
                staff_ids.add(emp)
                if shift_type == "Any":
                    for s in shifts.keys(): shift_off.append((emp, day_idx, s, 5))
                else: shift_off.append((emp, day_idx, shift_type, 5))

    header_match = re.search(r'n(\d+)w', text)
    if header_match:
        total_nurses = int(header_match.group(1))
        idx = 0
        while len(staff_ids) < total_nurses:
            staff_ids.add(f"Nurse_Fill_{idx}")
            idx += 1
    elif not staff_ids: staff_ids.add("Default_Nurse")

    staff = {}
    for eid in staff_ids:
        staff[eid] = {
            'maxshift_per_shift': {s: 5 for s in shifts.keys()}, 'max_total_min': 40 * 60, 'min_total_min': 0,
            'max_cons': 5, 'min_cons': 2, 'min_consec_days_off': 2, 'max_weekends': 1
        }
    return {'horizon': horizon, 'shifts': shifts, 'staff': staff, 'days_off': {}, 'shift_on_requests': [], 'shift_off_requests': shift_off, 'cover': cover}

def build_custom_instance(weeks: int, num_employees: int, shifts: list[str]) -> str:
    """Generates a standard NRP string using manual UI parameters, guaranteed feasible."""
    days = weeks * 7
    shift_length = 480 
    employees = [f"E{str(i).zfill(3)}" for i in range(num_employees)]

    # 1. Create a guaranteed feasible base schedule first
    base_schedule = {e: [] for e in employees}
    for i, e in enumerate(employees):
        offset = i % 6  
        for d in range(days):
            if (d + offset) % 6 < 4: 
                # Cycle through available shifts to ensure mixed coverage
                shift_idx = (d + i) % len(shifts)
                base_schedule[e].append(shifts[shift_idx]) 
            else: 
                base_schedule[e].append("OFF")

    out = []
    out.append("SECTION_HORIZON\n" + str(days) + "\n")
    out.append("SECTION_SHIFTS")
    for i, s in enumerate(shifts):
        cannot_follow = shifts[i - 1] if i > 0 else ""
        out.append(f"{s},{shift_length},{cannot_follow}")
    
    out.append("\nSECTION_STAFF")
    for e in employees:
        # Count exact shifts in base schedule
        shift_counts = {s: 0 for s in shifts}
        worked_days = 0
        for d in range(days):
            s = base_schedule[e][d]
            if s != "OFF":
                shift_counts[s] += 1
                worked_days += 1
        
        # Add a buffer (+2) to caps so solver has room to optimize
        shift_caps = [f"{s}={shift_counts[s] + 2}" for s in shifts]
        max_minutes = (worked_days + 3) * shift_length
        min_minutes = max(0, (worked_days - 3) * shift_length)
        out.append(f"{e},{'|'.join(shift_caps)},{max_minutes},{min_minutes},6,2,2,{weeks}")
        
    out.append("\nSECTION_DAYS_OFF")
    for e in employees:
        off_days = [d for d in range(days) if base_schedule[e][d] == "OFF"]
        if off_days:
            chosen_off = sorted(random.sample(off_days, min(2, len(off_days))))
            out.append(f"{e}," + ",".join(map(str, chosen_off)))
            
    out.append("\nSECTION_SHIFT_ON_REQUESTS")
    for e in employees:
        work_days = [d for d in range(days) if base_schedule[e][d] != "OFF"]
        if work_days:
            # Request 2 shifts they are already scheduled for in the feasible base
            chosen_on = sorted(random.sample(work_days, min(2, len(work_days))))
            for d in chosen_on:
                out.append(f"{e},{d},{base_schedule[e][d]},{random.randint(1, 3)}")

    out.append("\nSECTION_COVER")
    # Calculate exact coverage provided by the base schedule
    coverage = defaultdict(int)
    for d in range(days):
        for e in employees:
            s = base_schedule[e][d]
            if s != "OFF":
                coverage[(d, s)] += 1
                
    for d in range(days):
        for s in shifts:
            # Demand exactly what the feasible schedule provides
            req = max(1, coverage[(d, s)]) 
            out.append(f"{d},{s},{req},100,10")
            
    return "\n".join(out)

def generate_synthetic_thesis_data(inst):
    """Generates random tokens (EF1), biological profiles (Fatigue), and surges (BRO) explicitly BEFORE solving."""
    horizon = inst['horizon']
    E = list(inst['staff'].keys())
    S = list(inst['shifts'].keys())
    
    # 1. Envy Tokens (Game-Theoretic Bids - Strictly 100 per nurse)
    B = 100
    b_tokens = {}
    for idx_e, e in enumerate(E):
        np.random.seed(idx_e + 42) # Stable seed per nurse
        num_bins = horizon * len(S)
        
        # Dart-throwing method guarantees exactly 100 tokens are distributed
        tokens_array = np.zeros(num_bins, dtype=int)
        for _ in range(B):
            tokens_array[np.random.randint(0, num_bins)] += 1
            
        idx = 0
        for d in range(horizon):
            for s in S:
                b_tokens[(e, d, s)] = int(tokens_array[idx])
                idx += 1
                
    # 2. Assign Klyve et al. (2022) Biological Profiles
    for idx_e, e in enumerate(E):
        np.random.seed(idx_e + 100)
        inst['staff'][e]['biotype'] = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.64, 0.08, 0.08, 0.08, 0.01, 0.01, 0.08, 0.01, 0.01])
                
    # 3. Uncertainty Surges (R_hat)
    r_hat = {}
    for d in range(horizon):
        for s in S:
            req = inst['cover'].get((d, s), {}).get('req', 0)
            r_hat[(d, s)] = math.ceil(req * 0.4) 
            
    inst['synthetic'] = {'tokens': b_tokens, 'surges': r_hat}
    return inst


# ==============================================================================
# 2. GUROBI SOLVER ENGINE (All 10 Hard Constraints + 3 Novel)
# ==============================================================================

def solve_thesis_model(inst, use_fatigue, use_envy, gamma_budget, agency_cost=20, time_limit=300):
    mdl = gp.Model('Thesis_Engine')
    mdl.Params.OutputFlag = 1
    mdl.Params.LogToConsole = 1
    mdl.Params.TimeLimit = time_limit

    horizon = inst['horizon']
    E = list(inst['staff'].keys())
    S = list(inst['shifts'].keys())
    S_all = list(S) + ['OFF']
    staff = inst['staff']

    x = mdl.addVars(E, horizon, S, vtype=GRB.BINARY, name="x")
    
    # --- 10 ORIGINAL HARD CONSTRAINTS ---
    for e in E:
        for d in range(horizon):
            mdl.addConstr(gp.quicksum(x[e, d, s] for s in S) <= 1)

    for e, days in inst['days_off'].items():
        for d in days:
            if d < horizon:
                for s in S:
                    mdl.addConstr(x[e, d, s] == 0)

    for s, info in inst['shifts'].items():
        for t in info.get('forbid_next', []):
            if t == '': continue
            for e in E:
                for d in range(horizon - 1):
                    if t in S:
                        mdl.addConstr(x[e, d, s] + x[e, d + 1, t] <= 1)

    for e, info in staff.items():
        for s, limit in info['maxshift_per_shift'].items():
            if s in S:
                mdl.addConstr(gp.quicksum(x[e, d, s] for d in range(horizon)) <= limit)

    for e, info in staff.items():
        total_mins = gp.quicksum(x[e, d, s] * inst['shifts'][s]['length'] for d in range(horizon) for s in S)
        mdl.addConstr(total_mins <= info['max_total_min'])
        mdl.addConstr(total_mins >= info['min_total_min'])

    for e, info in staff.items():
        maxc = info['max_cons']
        if maxc is not None:
            for start_d in range(0, horizon - maxc):
                mdl.addConstr(gp.quicksum(x[e, d, s] for d in range(start_d, start_d + maxc + 1) for s in S) <= maxc)

    start = mdl.addVars(E, horizon, vtype=GRB.BINARY, name="start")
    for e in E:
        for d in range(horizon):
            prev_on = gp.quicksum(x[e, d - 1, s] for s in S) if d > 0 else 0
            curr_on = gp.quicksum(x[e, d, s] for s in S)
            mdl.addConstr(start[e, d] >= curr_on - prev_on)
            mdl.addConstr(start[e, d] <= 1)
            
    for e, info in staff.items():
        minc = info['min_cons']
        if minc is not None and minc > 1:
            for d in range(horizon):
                if d + minc - 1 < horizon:
                    mdl.addConstr(gp.quicksum(x[e, d + k, s] for k in range(minc) for s in S) >= minc * start[e, d])

    start_off = mdl.addVars(E, horizon, vtype=GRB.BINARY, name="start_off")
    for e in E:
        for d in range(horizon):
            prev_on = gp.quicksum(x[e, d - 1, s] for s in S) if d > 0 else 0
            curr_on = gp.quicksum(x[e, d, s] for s in S)
            curr_off = 1 - curr_on
            mdl.addConstr(start_off[e, d] >= curr_off + prev_on - 1)
            
    for e, info in staff.items():
        mindoff = info['min_consec_days_off']
        if mindoff is not None and mindoff > 0:
            for d in range(horizon):
                if d + mindoff - 1 < horizon:
                    mdl.addConstr(gp.quicksum(1 - gp.quicksum(x[e, d + k, s] for s in S) for k in range(mindoff)) >= mindoff * start_off[e, d])

    weekends = weekend_indices(horizon)
    y_weekend = mdl.addVars(E, weekends.keys(), vtype=GRB.BINARY, name="y_weekend")
    for e in E:
        if staff[e]['max_weekends'] is not None:
            for w, days in weekends.items():
                for d in days:
                    mdl.addConstr(y_weekend[e, w] >= gp.quicksum(x[e, d, s] for s in S))
            mdl.addConstr(gp.quicksum(y_weekend[e, w] for w in weekends.keys()) <= staff[e]['max_weekends'])

    under = mdl.addVars(horizon, S, vtype=GRB.CONTINUOUS, lb=0, name="under")
    over = mdl.addVars(horizon, S, vtype=GRB.CONTINUOUS, lb=0, name="over")
    for d in range(horizon):
        for s in S:
            req = inst['cover'].get((d, s), {}).get('req', 0)
            mdl.addConstr(gp.quicksum(x[e, d, s] for e in E) - over[d, s] + under[d, s] == req)


    # --- NOVEL CONSTRAINT 1: EXACT KLYVE ET AL. (2022) LOOKUP TABLE ---
    night_shift_key = None
    for k in S:
        if 'N' in k.upper():
            night_shift_key = k
            break
    if not night_shift_key:
        for k in S:
            if 'LATE' in k.upper():
                night_shift_key = k
                break
    if not night_shift_key:
        night_shift_key = list(S)[-1] 
        
    def compute_pattern_fatigue(b, w, s1, s2, s3, s4):
        sleep_time = {1:7, 2:5, 3:9, 4:7, 5:5, 6:9, 7:7, 8:5, 9:9}[b]
        chrono = {1:'D', 2:'D', 3:'D', 4:'M', 5:'M', 6:'M', 7:'E', 8:'E', 9:'E'}[b]
        
        def get_times(s):
            if s == 'OFF': return None, None
            name = s.upper()
            length = inst['shifts'][s]['length'] / 60.0
            if 'N' in name or 'LATE' in name: return 23.0, 23.0 + length
            if 'E' in name: return 15.0, 15.0 + length
            return 7.0, 7.0 + length
            
        def get_alpha(s):
            if s == 'OFF': return 0.0
            name = s.upper()
            alpha = 1.0
            if 'N' in name or 'LATE' in name: alpha = 1.5
            elif 'E' in name: alpha = 1.2
            
            if chrono == 'M' and ('N' in name or 'E' in name): alpha += 0.3
            if chrono == 'E' and ('D' in name or 'EARLY' in name): alpha += 0.3
            return alpha
            
        score = 0.0
        seq = [ (night_shift_key, get_times(night_shift_key)) if w == 1 else ('OFF', (None, None)) ]
        seq += [ (s, get_times(s)) for s in [s1, s2, s3, s4] ]
        
        for i in range(1, 5):
            s_curr, t_curr = seq[i]
            s_prev, t_prev = seq[i-1]
            
            if s_curr == 'OFF':
                score = max(0.0, score - 8.0) # Recovery
                continue
                
            score += get_alpha(s_curr) * (t_curr[1] - t_curr[0]) # Base
            
            if s_prev != 'OFF':
                rest = (t_curr[0] + 24.0) - t_prev[1]
                target_rest = 14.0 + (sleep_time - 7.0)
                if rest < target_rest:
                    score += (target_rest - rest) * 1.5 # Transition Penalty
        return score

    present_biotypes = set(inst['staff'][e]['biotype'] for e in E)
    P_Score = {}
    for b in present_biotypes:
        for w in [0, 1]:
            for s1 in S_all:
                for s2 in S_all:
                    for s3 in S_all:
                        for s4 in S_all:
                            P_Score[(b, w, s1, s2, s3, s4)] = compute_pattern_fatigue(b, w, s1, s2, s3, s4)

    fatigue_slack = mdl.addVars(E, horizon, vtype=GRB.CONTINUOUS, lb=0)
    f_max = mdl.addVars(E, horizon, vtype=GRB.CONTINUOUS, lb=0)
    
    obj_fatigue_penalty = 0
    obj_fatigue_pushdown = 0 # Ensures f_max reflects exact score without floating up

    if use_fatigue:
        Phi_max = 48.0 
        
        def get_y(e, d, s):
            if d < 0: return 1.0 if s == 'OFF' else 0.0
            if s == 'OFF': return 1.0 - gp.quicksum(x[e, d, s_type] for s_type in S)
            return x[e, d, s]

        for e in E:
            b_e = inst['staff'][e]['biotype']
            for d in range(horizon):
                for w in [0, 1]:
                    for s1 in S_all:
                        for s2 in S_all:
                            for s3 in S_all:
                                for s4 in S_all:
                                    score = P_Score[(b_e, w, s1, s2, s3, s4)]
                                    if score > 0:
                                        y_w = get_y(e, d-4, night_shift_key) if w==1 else (1.0 - get_y(e, d-4, night_shift_key))
                                        pattern_match = y_w + get_y(e, d-3, s1) + get_y(e, d-2, s2) + get_y(e, d-1, s3) + get_y(e, d, s4) - 4.0
                                        
                                        # When the 4-day pattern perfectly matches, pattern_match == 1.0
                                        mdl.addConstr(f_max[e, d] >= score * pattern_match)
                
                mdl.addConstr(f_max[e, d] - fatigue_slack[e, d] <= Phi_max)
                obj_fatigue_penalty += fatigue_slack[e, d] * 5000 
                obj_fatigue_pushdown += f_max[e, d] * 0.0001 # Microscopic penalty stops f_max from floating

    # --- NOVEL CONSTRAINT 2: EF1 ENVY ---
    obj_envy_slack = 0
    Delta_envy = mdl.addVars(E, E, vtype=GRB.CONTINUOUS, lb=0)
    if use_envy and len(E) > 1:
        b_tokens = inst['synthetic']['tokens']
        h = mdl.addVars(E, E, horizon, S, vtype=GRB.BINARY)
        
        for i in E:
            for j in E:
                if i != j:
                    for d in range(horizon):
                        for s in S:
                            mdl.addConstr(h[i, j, d, s] <= x[i, d, s])
                    mdl.addConstr(gp.quicksum(h[i, j, d, s] for d in range(horizon) for s in S) <= 1)
                    
                    U_i_own = gp.quicksum(b_tokens[(i, d, s)] * x[i, d, s] for d in range(horizon) for s in S)
                    U_i_peer = gp.quicksum(b_tokens[(i, d, s)] * x[j, d, s] for d in range(horizon) for s in S)
                    removed = gp.quicksum(b_tokens[(i, d, s)] * h[i, j, d, s] for d in range(horizon) for s in S)
                    
                    mdl.addConstr(U_i_own - removed <= U_i_peer + Delta_envy[i, j])
                    obj_envy_slack += Delta_envy[i, j] * 1000

    # --- NOVEL CONSTRAINT 3: BRO UNCERTAINTY ---
    obj_agency = 0
    v_agency = mdl.addVars(horizon, S, vtype=GRB.CONTINUOUS, lb=0)
    if gamma_budget > 0:
        r_hat = inst['synthetic']['surges']
        z = mdl.addVar(lb=0)
        p_dual = mdl.addVars(horizon, S, lb=0)
        
        for d in range(horizon):
            for s in S:
                mdl.addConstr(z + p_dual[d, s] >= r_hat[(d, s)])
                mdl.addConstr(v_agency[d, s] <= r_hat[(d, s)]) 
                
                obj_agency += v_agency[d, s] * agency_cost 
                
        total_staff = gp.quicksum(x[e, d, s] for e in E for d in range(horizon) for s in S)
        total_under = gp.quicksum(under[d, s] for d in range(horizon) for s in S)
        total_agcy = gp.quicksum(v_agency[d, s] for d in range(horizon) for s in S)
        total_nom = sum(inst['cover'].get((d, s), {}).get('req', 0) for d in range(horizon) for s in S)
        dual_surge = z * gamma_budget + gp.quicksum(p_dual[d, s] for d in range(horizon) for s in S)
        
        # Exact Original Working Constraint containing total_under (Decoupling)
        mdl.addConstr(total_staff + total_under + total_agcy >= total_nom + dual_surge)

    # --- EXACT ORIGINAL OBJECTIVE CALCULATION ---
    obj_base_calc = []
    
    for d in range(horizon):
        for s in S:
            w_under = inst['cover'].get((d, s), {}).get('w_under', 0.0)
            w_over = inst['cover'].get((d, s), {}).get('w_over', 0.0)
            obj_base_calc.append(under[d, s] * w_under)
            obj_base_calc.append(over[d, s] * w_over)
            
    for e, d, s, w in inst['shift_on_requests']:
        if d < horizon and (e, d, s) in x: 
            obj_base_calc.append(w * (1 - x[e, d, s]))
            
    for e, d, s, w in inst['shift_off_requests']:
        if d < horizon and (e, d, s) in x: 
            obj_base_calc.append(w * x[e, d, s])

    # The Base Score aligns perfectly with the original benchmark score
    obj_base = gp.quicksum(obj_base_calc)
    mdl.setObjective(obj_base + obj_envy_slack + obj_agency + obj_fatigue_penalty + obj_fatigue_pushdown, GRB.MINIMIZE)
    mdl.optimize()

    if mdl.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        assign = defaultdict(dict)
        for e in E:
            for d in range(horizon):
                assigned = [s for s in S if x[e, d, s].X > 0.5]
                assign[e][d] = assigned[0] if assigned else "OFF"
                
        final_base_score = sum(val.getValue() for val in obj_base_calc)
        final_fatigue_penalty = obj_fatigue_penalty.getValue() if use_fatigue else 0
        final_envy_penalty = obj_envy_slack.getValue() if use_envy else 0
        final_agency_penalty = obj_agency.getValue() if gamma_budget > 0 else 0
        
        # Clean the objective value by removing the mathematical trick used to stop floats
        final_pushdown_val = obj_fatigue_pushdown.getValue() if use_fatigue else 0
        clean_obj_val = mdl.ObjVal - final_pushdown_val

        agency_data = defaultdict(list)
        if gamma_budget > 0:
            for d in range(horizon):
                for s in S:
                    val = v_agency[d, s].X
                    if val > 0.5:
                        agency_data[d].append(f"+{val:.0f}({s})")
                        
        fatigue_used = sum(fatigue_slack[e, d].X for e in E for d in range(horizon)) if use_fatigue else 0
        envy_used = sum(Delta_envy[i, j].X for i in E for j in E if i != j) if use_envy and len(E) > 1 else 0
        
        f_max_extracted = {}
        if use_fatigue:
            for e in E:
                for d in range(horizon):
                    f_max_extracted[(e, d)] = f_max[e, d].X

        return {
            'status': 'Success', 
            'assign': assign, 
            'agency': agency_data, 
            'base_score': final_base_score, 
            'fatigue_pen': final_fatigue_penalty,
            'envy_pen': final_envy_penalty,
            'agency_pen': final_agency_penalty,
            'obj_val': clean_obj_val, 
            'fatigue_slack': fatigue_used,
            'envy_slack': envy_used,
            'time': mdl.Runtime, 
            'E': E, 
            'horizon': horizon,
            'f_max_extracted': f_max_extracted
        }
    else:
        return {'status': 'Infeasible'}


# ==============================================================================
# 3. STREAMLIT UI & HUMAN-READABLE TRANSLATOR
# ==============================================================================

st.set_page_config(page_title="NRP", layout="wide", page_icon="🏥")
st.title("Nurse Scheduling Interface")

st.sidebar.header("⚙️ Model Constraints")
use_fatigue = st.sidebar.toggle("Enable Fatigue Constraint", value=True)
use_envy = st.sidebar.toggle("Enable EF1 Envy-Freeness Constraint", value=True)
gamma_budget = st.sidebar.slider("Uncertainty Budget (Gamma)", 0, 10, 2)
agency_cost = st.sidebar.number_input("Agency Nurse Cost", min_value=1, max_value=10000, value=20, step=10)
st.sidebar.markdown("---")

st.sidebar.header("📂 Data Source")
data_mode = st.sidebar.radio("Select Instance Type:", [
    "Upload Standard (.txt)", 
    "Upload INRC-I (.txt)", 
    "Upload INRC-II (.txt)", 
    "Custom Instance Builder"
])

raw_inst = None

# Handle Data Input Modes
if data_mode == "Upload Standard (.txt)":
    file = st.sidebar.file_uploader("Upload Standard .txt file", type=["txt"])
    if file: 
        raw_inst = parse_standard_nrp(file.getvalue().decode("utf-8"))

elif data_mode == "Upload INRC-I (.txt)":
    file = st.sidebar.file_uploader("Upload INRC-I file", type=["txt"])
    if file: 
        raw_inst = parse_inrc1_format(file.getvalue().decode("utf-8"))

elif data_mode == "Upload INRC-II (.txt)":
    file = st.sidebar.file_uploader("Upload INRC-II file", type=["txt", "roster"])
    if file: 
        raw_inst = parse_inrc2_format(file.getvalue().decode("utf-8"))

elif data_mode == "Custom Instance Builder":
    st.sidebar.subheader("Instance Parameters")
    n_nurses = st.sidebar.number_input("Number of Nurses", min_value=1, max_value=50, value=10)
    n_weeks = st.sidebar.number_input("Horizon (Weeks)", min_value=1, max_value=8, value=2)
    s_types = st.sidebar.multiselect("Shift Types", ['Day', 'Evening', 'Night', 'Early', 'Late'], default=['Day', 'Night'])
    
    if st.sidebar.button("Generate Custom Matrix"):
        custom_txt = build_custom_instance(n_weeks, n_nurses, s_types)
        st.session_state['custom_inst'] = parse_standard_nrp(custom_txt)
        
    if 'custom_inst' in st.session_state:
        raw_inst = st.session_state['custom_inst']

if raw_inst is not None:
    # 1. Parse and Generate Synthetic Data
    inst = generate_synthetic_thesis_data(raw_inst)
    
    # 2. Human-Readable Constraints Translation
    st.header("Constraints Details")
    
    with st.expander("📌 Basic Hard Constraints (Translated from Dataset)"):
        st.markdown("**Strict Unavailability:**")
        for e, days in inst['days_off'].items():
            st.write(f"- Nurse **{e}** cannot work on days: {days}")
            
        st.markdown("**Shift Requests (Penalty in Objective):**")
        for req in inst['shift_on_requests']:
            st.write(f"- Nurse **{req[0]}** requested to work **Shift {req[2]}** on **Day {req[1]}** (Penalty if denied: {req[3]})")
        for req in inst['shift_off_requests']:
            st.write(f"- Nurse **{req[0]}** requested OFF for **Shift {req[2]}** on **Day {req[1]}** (Penalty if denied: {req[3]})")
            
        st.markdown("**Workload:**")
        if inst['staff']:
            sample_nurse = list(inst['staff'].keys())[0]
            st.write(f"- Nurse **{sample_nurse}**: Must work between {inst['staff'][sample_nurse]['min_total_min']} and {inst['staff'][sample_nurse]['max_total_min']} minutes. Max consecutive days: {inst['staff'][sample_nurse]['max_cons']}. Min consecutive days: {inst['staff'][sample_nurse]['min_cons']}.")

    if use_fatigue:
        with st.expander("📌 Fatigue Constraint"):
            st.write(" ")

    if use_envy:
        with st.expander("📌 EF1 Envy-Freeness"):
            st.write("- Every nurse is given exactly **100 tokens**. They distribute them over the shifts they most want to avoid.")
            
            b_tokens = inst['synthetic']['tokens']
            E_list = list(inst['staff'].keys())[:10] 
            
            pivot_data = []
            for e in E_list:
                row = {"Nurse": e}
                for d in range(min(5, inst['horizon'])): 
                    for s in list(inst['shifts'].keys()):
                        row[f"D{d}_{s}"] = b_tokens[(e, d, s)]
                
                # Prove it sums to 100 across the *entire* horizon
                row["Total Tokens (All Days)"] = sum(b_tokens[(e, d, s)] for d in range(inst['horizon']) for s in list(inst['shifts'].keys()))
                pivot_data.append(row)
                
            st.dataframe(pd.DataFrame(pivot_data))

    if gamma_budget > 0:
        with st.expander(f"📌 Uncertainty Constraints (Gamma = {gamma_budget})"):
            st.write(f"- Method: Budgeted Robust Optimization ensures the schedule stays feasible even if the **{gamma_budget} worst-case surges** happen simultaneously.")
            st.write("- We try to use existing nurses first. Agency nurses are used as a last resort.")
            st.write("- Synthesized Surges:")
            
            r_hat = inst['synthetic']['surges']
            surge_df = []
            for d in range(inst['horizon']): 
                for s in list(inst['shifts'].keys()):
                    surge_df.append({"Day": d, "Shift": s, "Nominal Demand": inst['cover'].get((d,s), {}).get('req', 0), "Worst-Case Surge": f"+{r_hat[(d, s)]} Nurses Needed (worse case)"})
            st.dataframe(pd.DataFrame(surge_df))

    # 3. Execution Engine
    st.header("Optimize")
    if st.button("🚀 Execute Gurobi Solver", type="primary"):
        with st.spinner("Gurobi is optimizing the matrix..."):
            res = solve_thesis_model(inst, use_fatigue, use_envy, gamma_budget, agency_cost=agency_cost)
            
        if res['status'] == 'Success':
            st.success(f"Optimal Roster Generated in {res['time']:.2f} seconds.")
            
            # --- EXACT 4 METRIC COLUMN LAYOUT ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Objective Value", f"{res['obj_val']:,.2f}")
            
            total_agcy = sum(int(str(s).split('(')[0].replace('+', '')) for lst in res['agency'].values() for s in lst) if gamma_budget > 0 else 0
            col2.metric("Agency Nurses Hired", f"{total_agcy}")
            
            col3.metric("Slack Fatigue Used", f"{res['fatigue_slack']:.2f}" if use_fatigue else "Disabled")
            col4.metric("Slack Envy-Freeness Used", f"{res['envy_slack']:.2f}" if use_envy else "Disabled")
            
            st.subheader("Final Schedule")
            df_data = []
            for e in res['E']:
                row = {"Nurse ID": e}
                for d in range(res['horizon']):
                    row[f"Day {d}"] = res['assign'][e][d]
                df_data.append(row)
                
            if gamma_budget > 0:
                row = {"Nurse ID": "🛑 AGENCY BUFFER"}
                for d in range(res['horizon']):
                    shifts_hired = res['agency'].get(d, [])
                    row[f"Day {d}"] = " ".join(shifts_hired) if shifts_hired else "-"
                df_data.append(row)

            def color_shifts(val):
                if val == 'OFF' or val == '-': return 'color: #94a3b8;'
                if str(val).startswith('+'): return 'background-color: #fca5a5; color: #991b1b; font-weight: bold;'
                if 'Nurse' in str(val) or 'AGENCY' in str(val): return 'color: #334155; font-weight: bold;'
                return 'background-color: #e0f2fe; color: #0369a1; font-weight: bold;'
                
            st.dataframe(pd.DataFrame(df_data).style.map(color_shifts), use_container_width=True)
            
            # --- EXACT FATIGUE HEATMAP REVERT ---
            if use_fatigue:
                st.subheader("Fatigue Scores (4-Day Rolling Window)")
                
                fatigue_df_data = []
                f_max_extracted = res['f_max_extracted']
                
                for e in res['E']:
                    row = {"Nurse ID": e}
                    for d in range(res['horizon']):
                        score = f_max_extracted[(e, d)]
                        if score > 48.0:
                            row[f"Day {d}"] = f"🔥 {score:.1f}"
                        elif score == 0.0:
                            row[f"Day {d}"] = "-"
                        else:
                            row[f"Day {d}"] = f"{score:.1f}"
                            
                    fatigue_df_data.append(row)
                
                def color_fatigue(val):
                    if isinstance(val, str) and '🔥' in val:
                        return 'background-color: #fca5a5; color: #991b1b; font-weight: bold;'
                    elif isinstance(val, str) and val.replace('.','',1).isdigit():
                        if float(val) == 0.0: return 'color: #94a3b8;'
                        if float(val) >= 48.0 * 0.8: return 'background-color: #fef08a; color: #9a3412;'
                    return ''
                    
                st.dataframe(pd.DataFrame(fatigue_df_data).style.map(color_fatigue), use_container_width=True)

        else:
            st.error("🚨 Infeasible Model. The constraints are too tight.")
else:
    st.info("👈 Please select a data mode and upload or generate an instance from the sidebar to begin.")