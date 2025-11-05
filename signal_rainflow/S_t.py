# S_t.py
import matplotlib.pyplot as plt
from typing import List, Tuple

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
    plt.figure(figsize=(8, 6))
    plt.plot(time, S, label="Stress (S)", color='b')
    plt.xlabel('Time')
    plt.ylabel('Stress (S)')
    plt.title('Vettore S vs Time')
    plt.grid(True)
    plt.legend()
    plt.savefig("plot/S_vs_time.png", dpi=600, format="jpg")
    plt.show()

# S_t.py
from typing import List, Tuple
import numpy as np

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

def plot_S_step(step: List[int], S_p: List[float], title: str = "S-step (picchi e valli)"):
    step = np.asarray(step)
    S_p = np.asarray(S_p, dtype=float)
    if step.size != S_p.size:
        raise ValueError("step e S_p devono avere la stessa lunghezza.")
    if step.size == 0:
        return

    plt.figure()
    plt.plot(step, S_p, "-o", markersize=4, linewidth=1.4)  # linea diretta tra punti
    plt.xlabel("step")
    plt.ylabel("S")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.tight_layout()
    plt.savefig("plot/S_vs_time.png", dpi=600, format="jpg")
    plt.show()