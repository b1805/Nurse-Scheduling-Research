import numpy as np
from scipy.integrate import solve_ivp
import itertools
import pickle
import csv  # Added for CSV export

# --- 1. PHILLIPS ET AL. (2010) TABLE 1 PARAMETERS ---
# Nominal parameters for the Sleep-Wake Switch and Circadian Oscillator
PARAMS = {
    'v_vc': -5.8, 'v_vh': 1.0, 'chi': 45.0, 'mu': 4.4, 'D_0': -10.2, 
    'Q_max': 100.0, 'theta': 10.0, 'sigma_prime': 3.0, 'A': 1.3, 
    'v_vm': -2.1, 'v_mv': -1.8, 'tau_v': 10/3600, 'tau_m': 10/3600, # tau in hours
    'lambda_val': 60.0, 'alpha_0': 0.16, 'I_0': 9500.0, 'p': 0.6, 
    'beta': 0.013, 'b': 0.4, 'G': 19.9, 'k': 0.55, 
    'tau_c': 24.2, 'f': 0.99669, 'gamma': 0.23, 'kappa': 12/np.pi
}

# --- 2. THE ODE SYSTEM ---
def sleep_model_odes(t, state, light_intensity, on_shift, biotype_params):
    V_v, V_m, H, n, x, y = state
    params = PARAMS.copy() 
    
    params['tau_c'] = biotype_params['tau_c']
    params['D_0'] = biotype_params['D_0']

    # Sigmoid firing rates
    sigma = params['sigma_prime'] * np.pi / np.sqrt(3)
    Q_v = params['Q_max'] / (1.0 + np.exp(-(V_v - params['theta']) / sigma))
    Q_m = params['Q_max'] / (1.0 + np.exp(-(V_m - params['theta']) / sigma))

    # --- THE FIX: FORCED WAKEFULNESS ON SHIFT ---
    # If the nurse is on shift, we override the neurons to force the awake state.
    # This forces the homeostatic drive (H) to keep climbing brutally.
    if on_shift:
        Q_m = params['Q_max']  # Maximum wake-active firing
        Q_v = 0.0              # Suppress sleep-active firing

    Q_m_th = 1.0
    I_tilde = light_intensity if Q_m > Q_m_th else 0.0

    # Retinal Processing 
    alpha = params['alpha_0'] * (I_tilde / params['I_0'])**params['p'] if I_tilde > 0 else 0.0
    B_hat = params['G'] * alpha * (1 - n)
    B = (1 - params['b']*x) * (1 - params['b']*y) * B_hat

    # Circadian Drive
    C = (1 + y) / 2.0
    D = params['v_vc'] * C + params['v_vh'] * H + params['D_0']

    # Derivatives
    dV_v = (params['v_vm'] * Q_m + D - V_v) / params['tau_v']
    dV_m = (params['v_mv'] * Q_v + params['A'] - V_m) / params['tau_m']
    dH = (params['mu'] * Q_m - H) / params['chi']
    dn = params['lambda_val'] * (alpha * (1 - n) - params['beta'] * n)
    dx = (params['gamma'] * (x - (4 * x**3)/3) - y * ((24 / (params['f'] * params['tau_c']))**2 + params['k'] * B)) / params['kappa']
    dy = (x + B) / params['kappa']

    return [dV_v, dV_m, dH, dn, dx, dy]

# --- 3. SHIFT SCHEDULE & LIGHT FUNCTION ---
def get_light_and_shift_state(t, shifts):
    """
    Returns (environmental_light, is_on_shift).
    Commute times (45 mins) are included in the forced wakefulness window.
    Uses absolute hours (d_idx * 24) to properly span crossing midnights.
    """
    t_mod = t % 24
    environmental_light = 10000 * (1 + np.cos(2 * np.pi * (t_mod - 12) / 24)) / 2
    
    is_on_shift = False
    
    for d_idx, s in enumerate(shifts):
        if s == 'D':
            if (d_idx * 24 + 6.25) <= t <= (d_idx * 24 + 15.75):
                is_on_shift = True
        elif s == 'E':
            if (d_idx * 24 + 13.75) <= t <= (d_idx * 24 + 23.25):
                is_on_shift = True
        elif s == 'N':
            # Night shifts start at 21:15 and end the next day at 8:25 (+24 hours)
            if (d_idx * 24 + 21.25) <= t <= (d_idx * 24 + 32.25):
                is_on_shift = True
                
    # If awake for a shift, hit them with hospital lighting
    if is_on_shift:
        environmental_light = 10000.0
        
    return environmental_light, is_on_shift

# --- 4. PRECOMPUTE THE LOOKUP TABLE ---
def generate_lookup_table():
    print("Generating offline ODE lookup table with true biological entrainment...")
    lookup_table = {}
    S_all = ['D', 'E', 'N', 'OFF']
    
    biotypes = {
        1: {'tau_c': 24.2, 'D_0': -10.2}, # Normal Sleep, Day Chronotype
    }

    # 1. First, spin the model up to a true entrained steady-state using 30 days of OFF shifts
    initial_state = [0.0, 0.0, 15.0, 0.1, 0.0, 0.0]
    b_params = {'tau_c': 24.2, 'D_0': -10.2}
    
    def odefunc_spin(t, state):
        light, on_shift = get_light_and_shift_state(t, ['OFF']*35)
        return sleep_model_odes(t, state, light, on_shift, b_params)

    sol_spin = solve_ivp(odefunc_spin, (0, 720), initial_state, method='Radau', max_step=1.0)
    entrained_steady_state = sol_spin.y[:, -1]

    # 2. Evaluate all patterns starting from the true entrained state
    for b, b_params in biotypes.items():
        for w in [0, 1]:
            for s1, s2, s3, s4 in itertools.product(S_all, repeat=4):
                
                shifts = ['N' if w == 1 else 'OFF', s1, s2, s3, s4]
                
                def odefunc_eval(t, state):
                    light, on_shift = get_light_and_shift_state(t, shifts)
                    return sleep_model_odes(t, state, light, on_shift, b_params)

                t_span = (0, 120)
                t_eval = np.linspace(0, 120, 500)
                
                sol = solve_ivp(odefunc_eval, t_span, entrained_steady_state, t_eval=t_eval, method='Radau')
                
                V_v, V_m, H, n, x, y = sol.y
                C = (1 + y) / 2.0
                D = PARAMS['v_vc'] * C + PARAMS['v_vh'] * H + b_params['D_0']
                
                # Extract the *maximum* fatigue during D4 (hours 96 to 120)
                day_4_idx = np.where(sol.t >= 96)[0]
                max_fatigue = np.max(D[day_4_idx]) if len(day_4_idx) > 0 else 0
                
                lookup_table[(b, w, s1, s2, s3, s4)] = max_fatigue / 30.0
                
    # Save the computed table to disk as .pkl (so the main dashboard still works)
    with open('P_Score_Table.pkl', 'wb') as f:
        pickle.dump(lookup_table, f)
        
    # Save the computed table to disk as .csv (for easy viewing)
    with open('P_Score_Table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Biotype', 'Prior_Night_w', 'D1', 'D2', 'D3', 'D4', 'Peak_Fatigue_mV'])
        
        for (b, w, s1, s2, s3, s4), score in lookup_table.items():
            writer.writerow([b, w, s1, s2, s3, s4, round(score, 2)])
            
    print("Table saved to P_Score_Table.pkl and P_Score_Table.csv!")

if __name__ == "__main__":
    generate_lookup_table()