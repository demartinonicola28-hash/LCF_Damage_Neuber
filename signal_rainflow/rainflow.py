# rainflow.py
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import numpy as np
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
def plot_rainflow(S_r, S_0, n_i, delta_S_r=10.0, delta_S_0=5.0):
    """
    Istogramma 3D della matrice rainflow con:
      - colormap 'Spectral_r'
      - colorbar al 50% di altezza, esterna a destra
      - tick colorbar ogni 0.5
      - assi X (S_r) e Y (S_0) da floor(min) a ceil(max)
    """
    # --- import locali
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # ----- 1) Sanitizzazione input
    S_r = np.asarray(S_r, dtype=float).ravel()
    S_0 = np.asarray(S_0, dtype=float).ravel()
    n_i = np.asarray(n_i, dtype=float).ravel()
    m = np.isfinite(S_r) & np.isfinite(S_0) & np.isfinite(n_i)
    S_r, S_0, n_i = S_r[m], S_0[m], n_i[m]

    # ----- 2) Estremi reali e arrotondati a integer
    sr_min, sr_max = float(np.min(S_r)), float(np.max(S_r))
    s0_min, s0_max = float(np.min(S_0)), float(np.max(S_0))
    sr_lo, sr_hi = np.floor(sr_min), np.ceil(sr_max)   # ⌊min⌋, ⌈max⌉ per S_r
    s0_lo, s0_hi = np.floor(s0_min), np.ceil(s0_max)   # ⌊min⌋, ⌈max⌉ per S_0

    # ----- 3) Bin edges allineati agli estremi arrotondati
    # Se range nullo, forzo almeno un bin usando i delta
    if sr_hi == sr_lo:
        S_r_bins = np.array([sr_lo, sr_lo + delta_S_r])
    else:
        S_r_bins = np.arange(sr_lo, sr_hi + delta_S_r, delta_S_r)

    if s0_hi == s0_lo:
        S_0_bins = np.array([s0_lo, s0_lo + delta_S_0])
    else:
        S_0_bins = np.arange(s0_lo, s0_hi + delta_S_0, delta_S_0)

    # ----- 4) Istogramma 2D pesato
    H, xedges, yedges = np.histogram2d(S_r, S_0, bins=[S_r_bins, S_0_bins], weights=n_i)

    # ----- 5) Centri cella e dimensioni barre
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    DXv = np.diff(xedges); DYv = np.diff(yedges)
    Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")
    DX, DY = np.meshgrid(DXv, DYv, indexing="ij")

    xpos = Xc.ravel(); ypos = Yc.ravel(); zpos = np.zeros_like(xpos)
    dx = DX.ravel(); dy = DY.ravel(); dz = H.ravel()

    # ----- 6) Filtra celle vuote
    mask = dz > 0
    if not np.any(mask):
        fig = plt.figure(figsize=(12, 8)); ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('S_r (Stress Range)'); ax.set_ylabel('S_0 (Mean Stress)'); ax.set_zlabel('n (Reversals)')
        ax.set_title('Matrice Rainflow (vuota)')
        # Limiti assi anche se vuoto
        ax.set_xlim(sr_lo, sr_hi); ax.set_ylim(s0_lo, s0_hi)
        plt.tight_layout(); plt.show(); return

    xpos, ypos, zpos = xpos[mask], ypos[mask], zpos[mask]
    dx, dy, dz = dx[mask], dy[mask], dz[mask]

    # ----- 7) Colori con 'Spectral_r'
    cmap = plt.cm.get_cmap('Spectral_r')
    vmin, vmax = float(np.min(dz)), float(np.max(dz))
    if vmin == vmax: vmin = 0.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = sm.to_rgba(dz)

    # ----- 8) Figura e barre
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')

    # ----- 9) Etichette e limiti assi arrotondati
    ax.set_xlabel('S_r (Stress Range)')
    ax.set_ylabel('S_0 (Mean Stress)')
    ax.set_zlabel('n (Reversals)')
    ax.set_title('Matrice Rainflow')
    ax.set_xlim(sr_lo, sr_hi)   # imposta X da ⌊min⌋ a ⌈max⌉
    ax.set_ylim(s0_lo, s0_hi)   # imposta Y da ⌊min⌋ a ⌈max⌉
    ax.set_zlim(0, 5)           # imposta z da ⌊min⌋ a ⌈max⌉

    # ----- 10) Colorbar al 50% fuori a destra
    cb_ax = inset_axes(
        ax,
        width="3%", height="50%",          # dimensioni desiderate
        bbox_to_anchor=(1.05, 0.0, 1, 1),  # tutta a destra, esterna
        bbox_transform=ax.transAxes,
        loc="center left",
        borderpad=0
    )
    cbar = plt.colorbar(sm, cax=cb_ax, orientation='vertical')
    cbar.set_label('Numero di cicli (peso n_i)', fontsize=11)

    # ----- 11) Tick colorbar ogni 0.5
    step = 0.5
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    cbar.set_ticks(np.arange(lo, hi + 1e-12, step))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # ----- 12) Layout e show
    plt.tight_layout()
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
