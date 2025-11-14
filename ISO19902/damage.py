# damage.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm, colors
from typing import Sequence, List, Tuple, Optional


def calcola_D(N_f: Sequence[float], n_i: Sequence[float], gamma_I: float, gamma_ov: float) -> Tuple[float, List[float], List[float], float]:
    """
    Miner's rule
      - N_f : vite a fatica per classe (S_r, S_0)
      - n_i : numero di reversals per classe (S_r, S_0)
      - n_tot : numero di reversal nel blocco
      - D_ni : danno per classe (S_r, S_0) considerando il numero di reversal associati
      - n_b : numero di blocchi / stress history
    formulas:
      - n_b   = 1
      - D_ni  = n_i / N_f --> Sequence[float]
      - D_ni_d = D_ni * gamma_I * 1.1 * gamma_ov
      - D_tot = n_b * sum(D_ni_d)
      - n_tot = sum(n_i)
    """
    N_f = np.asarray(N_f, dtype=float)
    n_i = np.asarray(n_i, dtype=float)
    if N_f.shape != n_i.shape:
        raise ValueError("N_f e n_i devono avere la stessa lunghezza.")

    n_tot = float(np.sum(n_i))
    if n_tot <= 0:
        raise ValueError("Somma n_i nulla o negativa.")


    # D_ni = n_i / N_f, con protezione da N_f<=0 o non finiti
    with np.errstate(divide="ignore", invalid="ignore"):
        D_ni = np.divide(n_i, N_f, out=np.full_like(N_f, np.inf, dtype=float),
                          where=(N_f > 0) & np.isfinite(N_f))
        
    D_ni_d = D_ni * gamma_I * 1.1 *gamma_ov
    D_tot = float(np.sum(D_ni_d))
    return D_tot, D_ni_d, n_i, n_tot

def calcola_n_b(D_ni_d: Sequence[float], gamma_I: float, gamma_ov: float) -> Tuple[float, float]:
    """
    Miner's rule
      - N_f : vite a fatica per classe (S_r, S_0)
      - n_i : numero di reversals per classe (S_r, S_0)
      - n_tot : numero di reversal nel blocco
      - D_ni_d : danno per classe (S_r, S_0) considerando il numero di reversal associati
      - n_b : numero di blocchi / stress history
    formulas:
      - D_tot = 1
      - D_ni_d  = D_ni * gamma_I * 1.1 * gamma_ov
      - n_b   = D_tot / sum(D_ni_d)
    """
    D_tot = 1 
    D_ni_d = np.asarray(D_ni_d, dtype=float)
    sum_D_ni = np.nansum(D_ni_d)
    n_b = np.divide(D_tot, sum_D_ni, out=np.array(np.inf, dtype=float),
                          where=(sum_D_ni != 0) & np.isfinite(sum_D_ni))
    #print(f"sum D_ni: {sum_D_ni}")
    return n_b, sum_D_ni


def plot_damage_3d(S_r: Sequence[float],
                   S_0: Sequence[float],
                   D_ni_d: Sequence[float],
                   delta_S_r: float = 50.0,
                   delta_S_0: float = 50.0,
                   tick_Sr: Optional[float] = None,
                   tick_S0: Optional[float] = None,
                   zmin: Optional[float] = None,
                   zmax: Optional[float] = None,
                   tick_D: Optional[float] = None) -> None:
    """
    Istogramma 3D del danno:
      - asse X: S_0 (mean)
      - asse Y: S_r (range)
      - asse Z: somma D_ni_d nel quadrato (S_r, S_0).

    I quadrati in piano hanno lato delta_S_0 (X) e delta_S_r (Y).

    tick_S0, tick_Sr (se forniti) definiscono il passo delle tacche sugli assi X e Y.
    zmin, zmax (se forniti) fissano i limiti dell'asse Z e della scala colori.
    tick_D (se fornito) è il passo delle tacche sull'asse Z.
    """

    # --- conversione e filtri base ---
    S_r = np.asarray(S_r, dtype=float).ravel()
    S_0 = np.asarray(S_0, dtype=float).ravel()
    D_ni_d = np.asarray(D_ni_d, dtype=float).ravel()

    if not (S_r.size == S_0.size == D_ni_d.size):
        raise ValueError("S_r, S_0 e D_ni_d devono avere la stessa lunghezza.")

    m = np.isfinite(S_r) & np.isfinite(S_0) & np.isfinite(D_ni_d) & (D_ni_d > 0.0)
    S_r = S_r[m]
    S_0 = S_0[m]
    D_ni_d = D_ni_d[m]

    if S_r.size == 0:
        raise ValueError("Nessun punto valido per il plot (danni non finiti o non positivi).")

    # --- definizione dei bin: ORA X = S_0, Y = S_r ---
    s0_min, s0_max = np.min(S_0), np.max(S_0)
    sr_min, sr_max = np.min(S_r), np.max(S_r)

    s0_min = np.floor(s0_min / delta_S_0) * delta_S_0
    s0_max = np.ceil(s0_max / delta_S_0) * delta_S_0
    sr_min = np.floor(sr_min / delta_S_r) * delta_S_r
    sr_max = np.ceil(sr_max / delta_S_r) * delta_S_r

    S_0_bins = np.arange(s0_min, s0_max + delta_S_0, delta_S_0)  # X
    S_r_bins = np.arange(sr_min, sr_max + delta_S_r, delta_S_r)  # Y

    # --- istogramma 2D pesato dal danno ---
    H, xedges, yedges = np.histogram2d(
        S_0, S_r,
        bins=[S_0_bins, S_r_bins],
        weights=D_ni_d
    )

    # colonne ancorate agli spigoli dei quadrati: X = S_0, Y = S_r
    x_left = xedges[:-1]      # estremi sinistri dei bin in S_0 (asse X)
    y_front = yedges[:-1]     # estremi anteriori dei bin in S_r (asse Y)

    Xl, Yf = np.meshgrid(x_left, y_front, indexing="ij")

    xpos = Xl.ravel()
    ypos = Yf.ravel()
    zpos = np.zeros_like(xpos)

    dz = H.ravel()
    nonzero = dz > 0.0
    xpos = xpos[nonzero]
    ypos = ypos[nonzero]
    zpos = zpos[nonzero]
    dz = dz[nonzero]

    if dz.size == 0:
        raise ValueError("Tutti i quadrati hanno danno nullo (dz == 0).")

    dx = (xedges[1] - xedges[0]) * np.ones_like(dz)  # in S_0
    dy = (yedges[1] - yedges[0]) * np.ones_like(dz)  # in S_r

    # --- limiti Z / scala colori ---
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # colori in funzione del danno (norm stessa di asse Z)
    norm = colors.Normalize(vmin=zmin_norm, vmax=zmax_norm)
    cmap = cm.viridis
    bar_colors = cmap(norm(dz))

    # ordinamento manuale "back-to-front"
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
    ax.set_zlabel(r"Danno $D_{ni,d}$")
    ax.set_title("Istogramma 3D del danno per quadrati (S_r, S_0)")

    # --- ticks sugli assi (X = S_0, Y = S_r) ---
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

    # ticks Z se richiesti
    if tick_D is not None and tick_D > 0.0:
        zticks = np.arange(zmin_plot, zmax_plot + tick_D, tick_D)
        ax.set_zticks(zticks)

    # colorbar legata al danno
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(dz)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7)
    cbar.set_label(r"Danno $D_{ni,d}$")

    ax.view_init(elev=25, azim=45)
    ax.dist = 10

    plt.tight_layout()
    plt.show()


def plot_damage_map(S_r: Sequence[float],
                    S_0: Sequence[float],
                    D_ni_d: Sequence[float],
                    delta_S_r: float = 50.0,
                    delta_S_0: float = 50.0,
                    tick_Sr: Optional[float] = None,
                    tick_S0: Optional[float] = None,
                    Dmin: Optional[float] = None,
                    Dmax: Optional[float] = None,
                    tick_D: Optional[float] = None) -> None:
    """
    Mappa 2D del danno:
      - asse X: Mean S_0
      - asse Y: Range S_r
      - colore: somma D_ni_d nel quadrato (S_r, S_0).

    delta_S_r, delta_S_0: dimensione dei quadrati.
    tick_S0, tick_Sr: passo delle tacche sugli assi X e Y (se forniti).
    Dmin, Dmax: limiti della scala colori (se forniti).
    tick_D: passo delle tacche sulla colorbar (se fornito).
    """

    # --- conversione e filtri base ---
    S_r = np.asarray(S_r, dtype=float).ravel()
    S_0 = np.asarray(S_0, dtype=float).ravel()
    D_ni_d = np.asarray(D_ni_d, dtype=float).ravel()

    if not (S_r.size == S_0.size == D_ni_d.size):
        raise ValueError("S_r, S_0 e D_ni_d devono avere la stessa lunghezza.")

    m = np.isfinite(S_r) & np.isfinite(S_0) & np.isfinite(D_ni_d) & (D_ni_d > 0.0)
    S_r = S_r[m]
    S_0 = S_0[m]
    D_ni_d = D_ni_d[m]

    if S_r.size == 0:
        raise ValueError("Nessun punto valido per il plot (danni non finiti o non positivi).")

    # --- definizione dei bin (X = S_0, Y = S_r) ---
    s0_min, s0_max = np.min(S_0), np.max(S_0)
    sr_min, sr_max = np.min(S_r), np.max(S_r)

    s0_min = np.floor(s0_min / delta_S_0) * delta_S_0
    s0_max = np.ceil(s0_max  / delta_S_0) * delta_S_0
    sr_min = np.floor(sr_min / delta_S_r) * delta_S_r
    sr_max = np.ceil(sr_max  / delta_S_r) * delta_S_r

    S_0_bins = np.arange(s0_min, s0_max + delta_S_0, delta_S_0)  # X
    S_r_bins = np.arange(sr_min, sr_max + delta_S_r, delta_S_r)  # Y

    # --- istogramma 2D pesato dal danno ---
    H, xedges, yedges = np.histogram2d(
        S_0, S_r,
        bins=[S_0_bins, S_r_bins],
        weights=D_ni_d
    )

    H = np.where(H > 0.0, H, np.nan)

    # limiti automatici del danno
    auto_Dmin = np.nanmin(H)
    auto_Dmax = np.nanmax(H)
    if Dmin is None:
        Dmin_plot = auto_Dmin
    else:
        Dmin_plot = Dmin
    if Dmax is None:
        Dmax_plot = auto_Dmax
    else:
        Dmax_plot = Dmax

    fig, ax = plt.subplots()

    pcm = ax.pcolormesh(xedges, yedges, H.T,
                        cmap="Spectral_r", shading="auto",
                        vmin=Dmin_plot, vmax=Dmax_plot)

    ax.set_xlabel(r"Mean $S_0$")
    ax.set_ylabel(r"Range $S_r$")
    ax.set_title("Mappa 2D del danno per quadrati (S_r, S_0)")

    # ticks sugli assi
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
    cbar.set_label(r"Danno $D_{ni,d}$")

    if tick_D is not None and tick_D > 0.0:
        Dticks = np.arange(Dmin_plot, Dmax_plot + tick_D, tick_D)
        cbar.set_ticks(Dticks)

    ax.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()
