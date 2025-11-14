# rainflow.py
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import numpy as np
import tkinter as tk
from tkinter import ttk
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Sequence, Optional
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401



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


def plot_rainflow_3d(S_r: Sequence[float],
                     S_0: Sequence[float],
                     n_i: Sequence[float],
                     delta_S_r: float = 10.0,
                     delta_S_0: float = 5.0,
                     tick_Sr: Optional[float] = None,
                     tick_S0: Optional[float] = None,
                     zmin: Optional[float] = None,
                     zmax: Optional[float] = None,
                     tick_n: Optional[float] = None) -> None:
    """
    Istogramma 3D della matrice rainflow:
      - asse X: S_0 (mean stress)
      - asse Y: S_r (stress range)
      - asse Z: somma n_i nel quadrato (S_r, S_0).
    """

    S_r = np.asarray(S_r, dtype=float).ravel()
    S_0 = np.asarray(S_0, dtype=float).ravel()
    n_i = np.asarray(n_i, dtype=float).ravel()

    if not (S_r.size == S_0.size == n_i.size):
        raise ValueError("S_r, S_0 e n_i devono avere la stessa lunghezza.")

    m = np.isfinite(S_r) & np.isfinite(S_0) & np.isfinite(n_i) & (n_i > 0.0)
    S_r, S_0, n_i = S_r[m], S_0[m], n_i[m]

    if S_r.size == 0:
        raise ValueError("Nessun punto valido per il plot (n_i non finiti o non positivi).")

    # estremi (X = S_0, Y = S_r)
    s0_min, s0_max = float(np.min(S_0)), float(np.max(S_0))
    sr_min, sr_max = float(np.min(S_r)), float(np.max(S_r))

    s0_min = np.floor(s0_min / delta_S_0) * delta_S_0
    s0_max = np.ceil(s0_max  / delta_S_0) * delta_S_0
    sr_min = np.floor(sr_min / delta_S_r) * delta_S_r
    sr_max = np.ceil(sr_max  / delta_S_r) * delta_S_r

    S_0_bins = np.arange(s0_min, s0_max + delta_S_0, delta_S_0)  # X
    S_r_bins = np.arange(sr_min, sr_max + delta_S_r, delta_S_r)  # Y

    # istogramma 2D pesato: primo argomento X (S_0), secondo Y (S_r)
    H, xedges, yedges = np.histogram2d(
        S_0, S_r,
        bins=[S_0_bins, S_r_bins],
        weights=n_i
    )

    # colonne ancorate agli spigoli
    x_left = xedges[:-1]      # S_0
    y_front = yedges[:-1]     # S_r
    Xl, Yf = np.meshgrid(x_left, y_front, indexing="ij")

    xpos = Xl.ravel()
    ypos = Yf.ravel()
    zpos = np.zeros_like(xpos)

    dz = H.ravel()
    nonzero = dz > 0.0
    xpos, ypos, zpos, dz = xpos[nonzero], ypos[nonzero], zpos[nonzero], dz[nonzero]

    if dz.size == 0:
        raise ValueError("Tutti i quadrati hanno n_i nullo (dz == 0).")

    dx = (xedges[1] - xedges[0]) * np.ones_like(dz)  # in S_0
    dy = (yedges[1] - yedges[0]) * np.ones_like(dz)  # in S_r

    # limiti Z / scala colori
    auto_zmin = float(np.min(dz))
    auto_zmax = float(np.max(dz))
    if zmin is None:
        zmin_plot = 0.0
        zmin_norm = auto_zmin
    else:
        zmin_plot = zmin
        zmin_norm = zmin
    if zmax is None:
        zmax_plot = auto_zmax * 1.05
        zmax_norm = auto_zmax
    else:
        zmax_plot = zmax
        zmax_norm = zmax

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.Spectral_r
    norm = colors.Normalize(vmin=zmin_norm, vmax=zmax_norm)
    bar_colors = cmap(norm(dz))

    # ordinamento back-to-front
    depth = xpos + ypos
    order = np.argsort(depth)

    for i in order:
        ax.bar3d(
            xpos[i], ypos[i], zpos[i],
            dx[i], dy[i], dz[i],
            color=bar_colors[i],
            shade=False,
            edgecolor="k",
            linewidth=0.3,
        )

    ax.set_xlabel(r"Mean $S_0$")
    ax.set_ylabel(r"Range $S_r$")
    ax.set_zlabel(r"n (reversals)")
    ax.set_title("Matrice rainflow – istogramma 3D")

    # ticks X/Y
    if tick_S0 is not None and tick_S0 > 0.0:
        xticks = np.arange(s0_min, s0_max + tick_S0, tick_S0)
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(S_0_bins)

    if tick_Sr is not None and tick_Sr > 0.0:
        yticks = np.arange(sr_min, sr_max + tick_Sr, tick_Sr)
        ax.set_yticks(yticks)
    else:
        ax.set_yticks(S_r_bins)

    ax.set_xlim(s0_min, s0_max)
    ax.set_ylim(sr_min, sr_max)
    ax.set_zlim(zmin_plot, zmax_plot)

    if tick_n is not None and tick_n > 0.0:
        zticks = np.arange(zmin_plot, zmax_plot + tick_n, tick_n)
        ax.set_zticks(zticks)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(dz)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7)
    cbar.set_label(r"n (reversals)")

    ax.view_init(elev=25, azim=45)
    ax.dist = 10

    plt.tight_layout()
    plt.show()


def plot_rainflow_map(S_r: Sequence[float],
                      S_0: Sequence[float],
                      n_i: Sequence[float],
                      delta_S_r: float = 10.0,
                      delta_S_0: float = 5.0,
                      tick_Sr: Optional[float] = None,
                      tick_S0: Optional[float] = None,
                      nmin: Optional[float] = None,
                      nmax: Optional[float] = None,
                      tick_n: Optional[float] = None) -> None:
    """
    Mappa 2D della matrice rainflow:
      - asse X: S_0 (mean stress)
      - asse Y: S_r (stress range)
      - colore: somma n_i nel quadrato (S_r, S_0).
    """

    S_r = np.asarray(S_r, dtype=float).ravel()
    S_0 = np.asarray(S_0, dtype=float).ravel()
    n_i = np.asarray(n_i, dtype=float).ravel()

    if not (S_r.size == S_0.size == n_i.size):
        raise ValueError("S_r, S_0 e n_i devono avere la stessa lunghezza.")

    m = np.isfinite(S_r) & np.isfinite(S_0) & np.isfinite(n_i) & (n_i > 0.0)
    S_r, S_0, n_i = S_r[m], S_0[m], n_i[m]

    if S_r.size == 0:
        raise ValueError("Nessun punto valido per il plot (n_i non finiti o non positivi).")

    # estremi (X = S_0, Y = S_r)
    s0_min, s0_max = float(np.min(S_0)), float(np.max(S_0))
    sr_min, sr_max = float(np.min(S_r)), float(np.max(S_r))

    s0_min = np.floor(s0_min / delta_S_0) * delta_S_0
    s0_max = np.ceil(s0_max  / delta_S_0) * delta_S_0
    sr_min = np.floor(sr_min / delta_S_r) * delta_S_r
    sr_max = np.ceil(sr_max  / delta_S_r) * delta_S_r

    S_0_bins = np.arange(s0_min, s0_max + delta_S_0, delta_S_0)  # X
    S_r_bins = np.arange(sr_min, sr_max + delta_S_r, delta_S_r)  # Y

    # istogramma 2D pesato
    H, xedges, yedges = np.histogram2d(
        S_0, S_r,
        bins=[S_0_bins, S_r_bins],
        weights=n_i
    )

    H = np.where(H > 0.0, H, np.nan)

    auto_nmin = np.nanmin(H)
    auto_nmax = np.nanmax(H)
    if nmin is None:
        nmin_plot = auto_nmin
    else:
        nmin_plot = nmin
    if nmax is None:
        nmax_plot = auto_nmax
    else:
        nmax_plot = nmax

    fig, ax = plt.subplots(figsize=(8, 6))

    # pcolormesh: X = xedges (S_0), Y = yedges (S_r), H.T per (Y,X)
    pcm = ax.pcolormesh(xedges, yedges, H.T,
                        cmap="Spectral_r", shading="auto",
                        vmin=nmin_plot, vmax=nmax_plot)

    ax.set_xlabel(r"Mean $S_0$")
    ax.set_ylabel(r"Range $S_r$")
    ax.set_title("Matrice rainflow – mappa 2D")

    # ticks X/Y
    if tick_S0 is not None and tick_S0 > 0.0:
        xticks = np.arange(s0_min, s0_max + tick_S0, tick_S0)
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(S_0_bins)

    if tick_Sr is not None and tick_Sr > 0.0:
        yticks = np.arange(sr_min, sr_max + tick_Sr, tick_Sr)
        ax.set_yticks(yticks)
    else:
        ax.set_yticks(S_r_bins)

    ax.set_xlim(s0_min, s0_max)
    ax.set_ylim(sr_min, sr_max)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"n (reversals)")

    if tick_n is not None and tick_n > 0.0:
        nticks = np.arange(nmin_plot, nmax_plot + tick_n, tick_n)
        cbar.set_ticks(nticks)

    ax.grid(True, linestyle=":", linewidth=0.5)
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
    window.geometry("750x750")
    window.resizable(False, False)          # <-- blocca le frecce/drag

    # Aggiungi la label con il titolo
    label = tk.Label(window, text="Tab. Rainflow Result", font=(16))
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

def salva_rainflow(S_r, S_0, n_i, filename="signal_rainflow/n_Sr_S0.txt"):
    """
    Salva i valori di n, S_r e S_0 in un file di testo.
    Ogni riga contiene i valori n (reversals), S_r (stress range), e S_0 (mean stress).
    
    Parameters:
        S_r (list): Lista dei valori di S_r (stress range).
        S_0 (list): Lista dei valori di S_0 (mean stress).
        n_i (list): Lista dei valori di n (numero di reversals).
        filename (str): Nome del file in cui salvare i dati (default: "signal_rainflow/n_Sr_S0.txt").
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
