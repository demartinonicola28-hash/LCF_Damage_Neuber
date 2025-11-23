# S_t.py
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np


def segno_traccia(P11, P22):
    """
    Calcola il segno della somma di P11 + P22
    """
    return [1 if (p11 + p22) > 0 else -1 for p11, p22 in zip(P11, P22)]

def calcola_S(VM, P11, P22):
    """
    Calcola il vettore S come VM * segno_traccia(P11 + P22)
    """
    segni = segno_traccia(P11, P22)
    return [vm * segno for vm, segno in zip(VM, segni)]

def plot_S_t(time, S):
    """
    Plotta il grafico S (Stress) vs Time.
    """
    plt.figure(figsize=(10, 6))
    plt.gca().set_facecolor('whitesmoke')
    plt.plot(time, S, label="Nominal stress - S [MPa]", color='#0066CC', linewidth=1.5)
    plt.xlabel('Time - t [s]')
    plt.ylabel('Nominal stress - S [MPa]')
    x_min = min(time)
    x_max = max(time)
    y_min = np.floor(np.nanmin(S)/100)*100
    y_max = np.ceil( np.nanmax(S)/100)*100
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Stress Spectrum away 1,5t from Notch')
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.tight_layout()
    plt.savefig("plot/S_vs_time.png", dpi=600)
    plt.show()

# S_t.py
def calcola_S_p(S, time) -> Tuple[List[float], List[int]]:
    """
    Input:
      - S: vettore delle tensioni nel tempo
      - time: vettore tempi (stessa lunghezza di S)
    Output:
      - S_p: soli picchi/valli di S, inclusi primo e ultimo
      - n:   [1..len(S_p)]
    """
    S = np.asarray(S, dtype=float)
    t = np.asarray(time, dtype=float)
    if S.size == 0 or t.size == 0 or S.size != t.size:
        return [], []

    # 1) Rimuovi plateau consecutivi (basato su S)
    keep = [0]
    for i in range(1, S.size):
        if S[i] != S[keep[-1]]:
            keep.append(i)
    S_red = S[keep]
    # t_red = t[keep]  # utile se in futuro vuoi anche i tempi dei picchi

    if S_red.size == 1:
        return [float(S_red[0])], [1]

    # 2) Seleziona inversioni (massimi o minimi locali)
    k2 = [0]
    for i in range(1, S_red.size - 1):
        if (S_red[i] - S_red[i - 1]) * (S_red[i + 1] - S_red[i]) < 0:
            k2.append(i)
    if k2[-1] != S_red.size - 1:
        k2.append(S_red.size - 1)

    S_p = S_red[k2].tolist()
    step = list(range(1, len(S_p) + 1))
    return S_p, step

def plot_S_p_step(step: List[int], S_p: List[float], title: str = "Reduced Stress Spectrum for Rainflow"):
    step = np.asarray(step)
    S_p = np.asarray(S_p, dtype=float)
    if step.size != S_p.size:
        raise ValueError("step e S_p devono avere la stessa lunghezza.")
    if step.size == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.gca().set_facecolor('whitesmoke')
    plt.plot(step, S_p, "-o", markersize=5, linewidth=1.5, color='#0066CC', markerfacecolor="None", markeredgecolor="#C62828", markeredgewidth=1)
    plt.xlabel("Cycle - n")
    plt.ylabel("Nominal stress - S [MPa]")
    x_min = min(step)
    x_max = max(step)
    y_min = np.floor(np.nanmin(S_p)/100)*100
    y_max = np.ceil( np.nanmax(S_p)/100)*100
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.tight_layout()
    plt.savefig("plot/S_vs_step.png", dpi=600)
    plt.show()