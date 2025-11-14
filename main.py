# main.py


import matplotlib.pyplot as plt
import numpy as np


from signal_rainflow.carica_dati import App  # Importa la GUI
from signal_rainflow.S_t import calcola_S, plot_S_t, calcola_S_p, plot_S_p_step  # Importa calcola_S e plot_S_t da S_t.py
from signal_rainflow.rainflow import rainflow_counting, plot_rainflow_3d, plot_rainflow_map, mostra_tabella, salva_rainflow  # Importa le funzioni Rainflow da rainflow.py
from ISO19902.parameters import ask_parameters
from ISO19902.neuber import calcola_sigma_p, calcola_sigma_r
#from ISO19902.ramberg_osgood import ramberg_osgood_amplitude, ramberg_osgood_range
from ISO19902.stress_strain_re import calcola_sigma_re, calcola_epsilon_re, plot_ramberg_osgood
from ISO19902.initiation_life import calcola_N_f
from ISO19902.damage import calcola_D, calcola_n_b, plot_damage_3d, plot_damage_map
from ISO19902.stress_strain_re import plot_sigma_re, plot_epsilon_re
from result.result_export import export_fatica


# --- STEP 1: CARICA DATI ---
app = App()  # Avvia la GUI per caricare i dati
app.mainloop()  # Esegui l'app
if not getattr(app, "data", None):
    print("Nessun dato confermato. Operazione annullata.")
    raise SystemExit  # <-- esci in modo pulito

# I dati vengono già salvati come vettori nella variabile globale
time, VM, P11, P22 = app.data  # Ottieni i dati da `carica_dati.py`


# --- STEP 2: CALCOLA IL VETTORE S ---
# Calcola il vettore S usando la funzione di `S_t.py`
S = calcola_S(VM, P11, P22)
#plot_S_t(time, S)  # Usa plot_S_t da `S_t.py`
print(f"S[0:5]: {S[:5]}")


# --- STEP 3: "PULIRE" IL SEGNALE" (picchi e valli) ---
S_p, step = calcola_S_p(S, time)
#plot_S_p_step(step, S_p)
print(f"n_Sp: {len(S_p)}")


# --- STEP 4: ESEGUI RAINFLOW COUNTING ---
S_r, S_0, n_i = rainflow_counting(S_p)
salva_rainflow(S_r, S_0, n_i)  # Salva i risultati usando salva_rainflow da `rainflow.py`

# 3D con assi di default, ma Z fra 0 e 50 con tick ogni 5
plot_rainflow_3d(S_r, S_0, n_i,
                 delta_S_r=50, delta_S_0=5,
                 tick_Sr=200, tick_S0=20,
                 nmin=0, nmax=None, tick_n=0.5)
# Mappa 2D con stessa discretizzazione e scala colori 0–50
plot_rainflow_map(S_r, S_0, n_i,
                 delta_S_r=50, delta_S_0=5,
                 tick_Sr=200, tick_S0=20,
                  nmin=0, nmax=None, tick_n=0.5)

#mostra_tabella(S_r, S_0, n_i)  # Mostra la tabella dei risultati

print(f"Sr: {len(S_r)}")
print(f"S0: {len(S_0)}")
print(f"ni: {len(n_i)}")


# --- STEP 5: PARAMETRI PER ANALISI A FATICA ---
# 0) Acquisizione parametri tramite finestra iniziale (GUI).
#    Ritorna un dict e aggiorna anche le variabili globali in config per retro-compatibilità.
params = ask_parameters()
E = params["E"]
Kf = params["Kf"]
K_prime = params["K_prime"]
n_prime = params["n_prime"]
sigma_prime_f = params["sigma_prime_f"]
epsilon_prime_f = params["epsilon_prime_f"]
b = params["b"]
c = params["c"]
gamma_M2 = params["gamma_M2"]
gamma_ov = params["gamma_ov"]
gamma_I = params["gamma_I"]
gamma_FE = params["gamma_FE"]

if all(value == 1 for value in [
    params["gamma_M2"], params["gamma_ov"], params["gamma_I"], params["gamma_FE"]
]):
    params["gamma_ov"] = 1/1.1
print("=== Parameters ===")
for key, value in params.items():
    print(f"{key}: {value}")
print("==================")


# --- STEP 6: NEUBER'S METHOD ---> [sigma_p sigma_r sigma_0] ---
sigma_r = np.array(calcola_sigma_r(Kf, list(S_r), E, K_prime, n_prime, gamma_M2, gamma_FE))
sigma_p = np.array(calcola_sigma_p(list(S_0), list(S_r), Kf, E, K_prime, n_prime, gamma_M2, gamma_FE))
sigma_0 = sigma_p - 0.5 * sigma_r

print("=== Neuber's Rule ===")
print(f"sigma_r[0:5]: {sigma_r[:5]}")
print(f"sigma_p[0:5]: {sigma_p[:5]}")
print(f"sigma_0[0:5]: {sigma_0[:5]}")

# --- STEP 7: STRESS & STRAIN FULLY REVERSED CYCLE (R=-1) --->  [sigma_re epsilon_re] ---
sigma_a, sigma_re = np.array(calcola_sigma_re(sigma_r, sigma_0, sigma_prime_f))
epsilon_re_el, epsilon_re, epsilon_re_p = calcola_epsilon_re(E, K_prime, n_prime, sigma_re)

print("===  Equivalent fully-reversible elasto-plastic ===")
print("===    stress & strain amplitude at the notch   ===")
print(f"sigma_a[0:5]: {sigma_a[:5]}")
print(f"sigma_re[0:5]: {sigma_re[:5]}")
print(f"epsilon_re[0:5]: {epsilon_re[:5]}")

print(f"sigma_a: {len(sigma_a)}")
print(f"sigma_re: {len(sigma_re)}")
print(f"epsilon_re: {len(epsilon_re)}")

#plot_sigma_re(sigma_re, n_i, sort_desc=True)
#plot_epsilon_re(epsilon_re, n_i, sort_desc=True)
#plot_ramberg_osgood(E, K_prime, n_prime, sigma_re)

# --- STEP 8: STRAIN LIFE (SWT) --->  [N_f] ---
N_f = np.asarray(
    calcola_N_f(sigma_re=sigma_re, epsilon_re=epsilon_re, E=E,
                sigma_f_prime=sigma_prime_f, epsilon_f_prime=epsilon_prime_f,
                b=b, c=c, Nf_guess=1.0),dtype=float)

k = min(5, N_f.size)
print("=== Initiation Life Method (SWT) ===")
print(f"n: {N_f.size}")
print(f"N_f[0:{k}]: {N_f[:k].tolist()}")


# --- STEP 9: DANNO TOTALE: MINER'S RULE ---> [D] ---j
D_tot, D_ni_d, n_i, n_tot = calcola_D(N_f, n_i, gamma_I, gamma_ov)
n_b, sum_D_ni = calcola_n_b(D_ni_d, gamma_I, gamma_ov)

print(f"n_i[0:{k}]: {n_i[:k].tolist()}")
print(f"D_ni[0:{k}]: {D_ni_d[:k].tolist()}")
print(f"sum_D_ni: {sum_D_ni}")
print(f"n_tot: {n_tot}")
print(f"D_tot: {D_tot}")
print(f"n_b: {n_b:.1f} blocks")

plot_damage_3d(S_r, S_0, D_ni_d,
    delta_S_r=250, delta_S_0=20,        # range dell'istogramma
    tick_Sr=250, tick_S0=20 ,           # range dei valori sugli assi x e y
    Dmin=0.0, Dmax=None, tick_D=0.5)

plot_damage_map(S_r, S_0, D_ni_d,
    delta_S_r=250, delta_S_0=20,
    tick_Sr=250, tick_S0=20,
    Dmin=0.0, Dmax=None, tick_D=0.5)

# --- STEP 10: EXPORT DATA ---
xlsx_path = export_fatica(
    time=time, VM=VM, P11=P11, P22=P22, S=S,
    step=step, S_p=S_p,
    S_r=S_r, S_0=S_0, n_i=n_i,
    sigma_r=sigma_r, sigma_p=sigma_p, sigma_0=sigma_0,
    sigma_a=sigma_a, sigma_re=sigma_re,
    epsilon_re_el=epsilon_re_el, epsilon_re=epsilon_re, epsilon_re_p=epsilon_re_p,
    N_f=N_f, D_ni_d=D_ni_d, D_tot=D_tot, n_tot=n_tot,
    params=params,
    outdir="export", xlsx_name="risultati_fatica.xlsx", also_csv=True
)
print(f"Export scritto: {xlsx_path}")
