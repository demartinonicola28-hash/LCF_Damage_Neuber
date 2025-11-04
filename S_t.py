# S_t.py
import matplotlib.pyplot as plt

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
