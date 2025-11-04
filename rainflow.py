# rainflow.py
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import numpy as np
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Funzione per calcolare il segno della somma P11 + P22
def segno_traccia(P11, P22):
    return [1 if (p11 + p22) > 0 else -1 for p11, p22 in zip(P11, P22)]

# Funzione che calcola il vettore S come VM * segno_traccia(P11 + P22)
def calcola_S(VM, P11, P22):
    segni = segno_traccia(P11, P22)
    return [vm * segno for vm, segno in zip(VM, segni)]

# Funzione per trovare i punti di inversione (reversal points)
def reversals(series, left=False, right=False):
    """Itera sui punti di inversione nella serie"""
    series = iter(series)
    x_last, x = next(series), next(series)
    d_last = (x - x_last)

    if left:
        yield x_last
    for x_next in series:
        if x_next == x:
            continue
        d_next = x_next - x
        if d_last * d_next < 0:
            yield x
        x_last, x = x, x_next
        d_last = d_next
    if right:
        yield x_next

# Funzione per estrarre i cicli
def extract_cycles(series, left=False, right=False):
    """Estrai i cicli dalla serie"""
    points = deque()
    for x in reversals(series, left=left, right=right):
        points.append(x)
        while len(points) >= 3:
            X = abs(points[-2] - points[-1])
            Y = abs(points[-3] - points[-2])
            if X < Y:
                break
            elif len(points) == 3:
                yield points[-3], points[-2], 0.5
                points.popleft()
            else:
                yield points[-3], points[-2], 1.0
                last = points.pop()
                points.pop()
                points.pop()
                points.append(last)
    else:
        while len(points) > 1:
            yield points[-2], points[-1], 0.5
            points.pop()

# Funzione per contare i cicli
def count_cycles(series, ndigits=None, left=False, right=False):
    """Conta i cicli nella serie e ritorna i range e il numero di cicli"""
    counts = defaultdict(float)
    for low, high, mult in extract_cycles(series, left=left, right=right):
        delta = round(abs(high - low), ndigits) if ndigits else abs(high - low)
        counts[delta] += mult
    return sorted(counts.items())

# Funzione per eseguire il Rainflow Counting
def rainflow_counting(S):
    """
    Esegui il Rainflow Counting per ottenere i range di tensione (S_r),
    la tensione media (S_0) e il numero di cicli (n_i)
    """
    S_r = []
    S_0 = []
    n_i = []

    # Esegui il Rainflow counting sulla lista S
    cycle_counts = count_cycles(S)

    # Popola le liste S_r, S_0, e n_i
    for (S_range, cycles) in cycle_counts:
        S_r.append(S_range)
        S_0.append(np.mean(S[:len(S_r)]))  # Calcola la tensione media S_0
        n_i.append(cycles)

    return S_r, S_0, n_i

# Funzione per plottare la matrice 3D
def plot_rainflow(S_r, S_0, n_i, delta_S_r=10, delta_S_0=5):
    """
    Plotta un istogramma 3D raggruppando le righe in base a intervalli specifici di S_r e S_0.
    La somma di n_i è calcolata per ogni gruppo e plottata come barra 3D.
    """
    # Crea gli intervalli per S_r (Stress Range) e S_0 (Mean Stress) in base ai delta
    S_r_bins = np.arange(0, np.max(S_r) + delta_S_r, delta_S_r)  # Creazione intervallo per S_r
    S_0_bins = np.arange(0, np.max(S_0) + delta_S_0, delta_S_0)  # Creazione intervallo per S_0

    # Crea un istogramma 2D con i pesi come n_i (somma dei reversals per ogni cella)
    H, xedges, yedges = np.histogram2d(S_r, S_0, bins=[S_r_bins, S_0_bins], weights=n_i)

    # Calcola la posizione di ogni barra nel grafico 3D
    xpos, ypos = np.meshgrid(xedges[:-1] + delta_S_r/2, yedges[:-1] + delta_S_0/2, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    # Calcola le dimensioni delle barre (delta_S_r e delta_S_0)
    dx = delta_S_r * np.ones_like(zpos)
    dy = delta_S_0 * np.ones_like(zpos)
    dz = H.ravel()  # L'altezza delle barre è la somma dei valori di n_i per ogni cella

    # Maschera per escludere valori di dz uguali a zero (se non ci sono reversals per quel gruppo)
    mask = dz > 0
    xpos = xpos[mask]
    ypos = ypos[mask]
    zpos = zpos[mask]
    dx = dx[mask]
    dy = dy[mask]
    dz = dz[mask]

    # Definisci la colormap
    colormap = plt.cm.plasma  # Colormap scelta (viridis, plasma, inferno, ecc.)

    # Crea il grafico
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Crea le barre 3D, applicando il colore in base ai valori di dz (numero di reversals)
    img = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colormap(dz / np.max(dz)), zsort='average')

    # Imposta le etichette degli assi
    ax.set_xlabel('S_r (Stress Range)')
    ax.set_ylabel('S_0 (Mean Stress)')
    ax.set_zlabel('n (Reversals)')
    ax.set_title('Matrice Rainflow')

    # Aggiungi la barra dei colori con scala in base al massimo valore di dz
    cbar = plt.colorbar(img, ax=ax, orientation='vertical')
    cbar.set_label('Numero di Cicli (Reversals)', fontsize=12)
    
    # Mostra il grafico
    plt.show()






# Funzione per mostrare la tabella GUI dei risultati Rainflow
def mostra_tabella(S_r, S_0, n_i):
    """
    Crea e visualizza una finestra GUI che mostra la tabella dei risultati Rainflow
    con le colonne: n Reversals, S_r Stress Range, S_0 Mean Stress, aggiungendo una 
    barra di scorrimento per navigare tra i dati.
    """
    # Crea la finestra principale
    window = tk.Tk()
    window.title("Rainflow Cycle Counting - Tabella")

    # Aggiungi la label con il titolo
    label = tk.Label(window, text="Tab. Rainflow Result", font=("Courier new", 16))
    label.pack(pady=10)

    # Crea il frame per la tabella e la barra di scorrimento
    frame = ttk.Frame(window)
    frame.pack(fill="both", expand=True, padx=12, pady=6)

    # Crea la Treeview (tabella) per mostrare i dati
    tree = ttk.Treeview(frame, columns=("n", "S_r", "S_0"), show="headings", height=15)
    tree.pack(side="left", fill="both", expand=True)

    # Definisci le intestazioni delle colonne
    tree.heading("n", text="n Reversals")
    tree.heading("S_r", text="S_r Stress Range (MPa)")
    tree.heading("S_0", text="S_0 Mean Stress (MPa)")

    # Imposta la larghezza delle colonne
    tree.column("n", width=100, anchor="center")
    tree.column("S_r", width=150, anchor="center")
    tree.column("S_0", width=150, anchor="center")

    # Aggiungi i dati alla tabella
    for i in range(len(S_r)):
        tree.insert("", "end", values=(n_i[i], S_r[i], S_0[i]))

    # Aggiungi una barra di scorrimento verticale
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    # Aggiungi un pulsante per chiudere la finestra
    ttk.Button(window, text="Chiudi", command=window.destroy).pack(pady=10)

    # Avvia il loop della finestra
    window.mainloop()

def salva_rainflow(S_r, S_0, n_i, filename="n_Sr_S0.txt"):
    """
    Salva i valori di n, S_r e S_0 in un file di testo.
    Ogni riga contiene i valori n (reversals), S_r (stress range), e S_0 (mean stress).
    
    Parameters:
        S_r (list): Lista dei valori di S_r (stress range).
        S_0 (list): Lista dei valori di S_0 (mean stress).
        n_i (list): Lista dei valori di n (numero di reversals).
        filename (str): Nome del file in cui salvare i dati (default: "n_Sr_S0.txt").
    """
    try:
        with open(filename, "w") as f:
            # Scrivi l'intestazione del file
            f.write("n Reversals | S_r Stress Range (MPa) | S_0 Mean Stress (MPa)\n")
            f.write("-" * 60 + "\n")
            
            # Scrivi i dati
            for n, sr, s0 in zip(n_i, S_r, S_0):
                f.write(f"{n:.6f} | {sr:.6f} | {s0:.6f}\n")
        
        print(f"File '{filename}' creato con successo!")
    except Exception as e:
        print(f"Errore durante la scrittura del file: {e}")
