# stress_strain_re.py

import numpy as np
from typing import List, Tuple
from ISO19902.ramberg_osgood import ramberg_osgood_amplitude
import os
import matplotlib.pyplot as plt

def calcola_sigma_re(sigma_r: List[float],
                     sigma_0: List[float] | float,
                     sigma_f_prime: float
                    ) -> Tuple[List[float], List[float]]:
    """
    Calcola:
      σ_a = 0.5 * |σ_r|
      σ_re = [ (σ_a / (1 - σ_0/σ'_f)) * sqrt((σ_0 + σ_a)*σ_a) ]^{0.5}
    Interpreto la formula come potenza 0.5 dell'intero termine tra parentesi.

    Input: liste/array della stessa lunghezza (σ_0 può essere scalare).
    Output: (sigma_a, sigma_re) come liste.
    """
    if sigma_f_prime <= 0:
        raise ValueError("σ'_f deve essere > 0.")

    sr = np.asarray(sigma_r, dtype=float)
    s0 = np.asarray(sigma_0, dtype=float)
    if s0.ndim == 0:
        s0 = np.full_like(sr, s0)

    if sr.shape != s0.shape:
        raise ValueError("sigma_r e sigma_0 devono avere la stessa lunghezza.")

    sigma_a = 0.5 * np.abs(sr)

    # evita divisioni per zero
    denom = 1.0 - s0 / float(sigma_f_prime)
    eps = 1e-12
    denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)

    inner = (sigma_a / denom) * np.sqrt(np.maximum(0.0, (s0 + sigma_a) * sigma_a))
    sigma_re = np.sqrt(np.maximum(0.0, inner))

    return sigma_a.tolist(), sigma_re.tolist()

def calcola_epsilon_re(E: float, K_prime: float, n_prime: float,
                       sigma_re: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """
    Decomposizione di ε_re in:
      ε_re_el = σ_re / E              (parte elastica, con σ_a = σ_re)
      ε_re    = RO(σ_re)              (totale da Ramberg–Osgood in ampiezza)
      ε_re_p = ε_re - ε_re_el        (parte plastica)

    Ritorna (epsilon_re_el, epsilon_re, epsilon_re_p) come liste.
    """
    import numpy as np
    # parte elastica
    epsilon_re_el = (sigma_re / float(E)).tolist()
    # totale con Ramberg–Osgood
    epsilon_re = ramberg_osgood_amplitude(E, K_prime, n_prime, sigma_re.tolist())
    # parte plastica
    epsilon_re_p = (np.asarray(epsilon_re) - np.asarray(epsilon_re_el)).tolist()
    return epsilon_re_el, epsilon_re, epsilon_re_p

def calcola_epsilon_a(E: float, K_prime: float, n_prime: float,
                       sigma_a: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """
    Decomposizione di ε_re in:
      ε_a_el = σ_a / E              (parte elastica, con σ_a = σ_re)
      ε_a    = RO(σ_a)              (totale da Ramberg–Osgood in ampiezza)
      ε_a_p = ε_a - ε_a_el        (parte plastica)

    Ritorna (epsilon_re_el, epsilon_re, epsilon_re_p) come liste.
    """
    import numpy as np
    # totale con Ramberg–Osgood
    epsilon_a = ramberg_osgood_amplitude(E, K_prime, n_prime, sigma_a.tolist())
    return epsilon_a


def plot_sigma_re(sigma_re, n_i, sort_desc=True):
    """
    Spettro per σ_re con segni alternati:
      y = [0, +σ_re(1) ... (2*n1 volte alternando), +/−σ_re(2) ...]
    L'alternanza è globale e parte positiva dopo lo 0.
    Ritorna (x, y).
    """
    v = np.asarray(sigma_re, dtype=float).ravel()
    n = np.asarray(n_i, dtype=float).ravel()
    if v.size != n.size:
        raise ValueError(f"Dimensioni non coerenti: sigma_re={v.size}, n_i={n.size}")

    mask = ~(np.isnan(v) | np.isnan(n)) & (n > 0)
    v, n = v[mask], n[mask]

    if sort_desc and v.size:
        order = np.argsort(v)[::-1]
        v, n = v[order], n[order]

    seq = []
    sign = +1  # primo valore dopo lo 0 è positivo
    for val, ni in zip(v, n):
        k = int(round(2.0 * float(ni)))
        for _ in range(max(k, 0)):
            seq.append(sign * float(val))
            sign *= -1  # alterna ad ogni step

    y = np.array([0.0] + seq, dtype=float)
    x = np.arange(y.size, dtype=int)

    plt.figure(figsize=(10, 6))
    plt.gca().set_facecolor('whitesmoke')
    plt.plot(x, y, color="#00CC0A", linewidth=1.5)
    plt.xlabel(r"Cicli $n_{i}$")
    plt.ylabel(r"$\sigma_{re}$")
    plt.title(r"Spettro per $\sigma_{re}$ ordinati")
    x_min = min(x)
    x_max = max(x)
    y_min = np.floor(np.nanmin(y)/100)*100
    y_max = np.ceil( np.nanmax(y)/100)*100
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.savefig("plot/plot_epsilon_re.png", dpi=600)
    plt.show()
    return x, y

def plot_epsilon_re(epsilon_re, n_i, sort_desc=True):
    """
    Spettro per ε_re con segni alternati:
      y = [0, +ε_re(1) ... (2*n1 volte alternando), +/−ε_re(2) ...]
    L'alternanza è globale e parte positiva dopo lo 0.
    Ritorna (x, y).
    """
    v = np.asarray(epsilon_re, dtype=float).ravel()
    n = np.asarray(n_i, dtype=float).ravel()
    if v.size != n.size:
        raise ValueError(f"Dimensioni non coerenti: epsilon_re={v.size}, n_i={n.size}")

    mask = ~(np.isnan(v) | np.isnan(n)) & (n > 0)
    v, n = v[mask], n[mask]

    if sort_desc and v.size:
        order = np.argsort(v)[::-1]
        v, n = v[order], n[order]

    seq = []
    sign = +1  # primo valore dopo lo 0 è positivo
    for val, ni in zip(v, n):
        k = int(round(2.0 * float(ni)))
        for _ in range(max(k, 0)):
            seq.append(sign * float(val))
            sign *= -1  # alterna ad ogni step

    y = np.array([0.0] + seq, dtype=float)
    x = np.arange(y.size, dtype=int)

    plt.figure(figsize=(10, 6))
    plt.gca().set_facecolor('whitesmoke')
    plt.plot(x, y, color="#00CC0A", linewidth=1.5)
    plt.xlabel(r"Cicli $n_{i}$")
    plt.ylabel(r"$\epsilon_{re}$")
    plt.title(r"Spettro per $\epsilon_{re}$ ordinati")
    x_min = min(x)
    x_max = max(x)
    y_min = np.floor(np.nanmin(y)/100)*0.025 if np.nanmax(np.abs(y)) >= 0.02 else np.floor(np.nanmin(y)/0.1)*0.02
    y_max = np.ceil( np.nanmax(y)/100)*0.025  if np.nanmax(np.abs(y)) >= 0.02 else np.ceil( np.nanmax(y)/0.1)*0.02
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.savefig("plot/plot_epsilon_re.png", dpi=600)
    plt.show()
    return x, y

def plot_ramberg_osgood(E, K_prime, n_prime, sigma_re):
    """
    Plotta ε_re_el, ε_re_p ed ε_re in funzione di σ_re.
      ε_re_el = σ_re / E
      ε_re    = RO(σ_re) = σ_re/E + (σ_re/K')^{1/n'}
      ε_re_p = ε_re - ε_re_el
    Ritorna (sigma_re_sorted, epsilon_re_el, epsilon_re, epsilon_re_ep).
    """
    s = np.asarray(sigma_re, dtype=float).ravel()
    m = ~np.isnan(s)
    s = s[m]
    
    sort_x=False

    if sort_x:
        s = np.sort(s)

    eps_el = s / float(E)
    eps_tot = np.asarray(ramberg_osgood_amplitude(E, K_prime, n_prime, s.tolist()), dtype=float)
    eps_p = eps_tot - eps_el

    plt.figure(figsize=(10, 6))
    plt.gca().set_facecolor("whitesmoke")
    plt.plot(eps_el, s, linewidth=1.5, label=r"$\epsilon_{re,el}=\frac{\sigma_{re}}{E}$")
    plt.plot(eps_p, s, linewidth=1.5, label=r"$\epsilon_{re,p}=(\frac{\sigma_{re}}{K^\prime})^\frac{1}{n^\prime}$")
    plt.plot(eps_tot, s, linewidth=1.8, label=r"$\epsilon_{re} = \epsilon_{re,el} + \epsilon_{re,p}$")
    plt.xlabel(r"$\epsilon_{re}$")
    plt.ylabel(r"$\sigma_{re}$")
    plt.title("Stabilized Cyclic Curve")
    x_min = min(eps_tot)
    x_max = max(eps_tot)
    y_min = np.floor(np.nanmin(s)/100)*100
    y_max = np.ceil( np.nanmax(s)/100)*100
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend(title= "Ramber-Osgood strain component")
    plt.tight_layout()
    plt.savefig("plot/plot_ramberg_osgood.png", dpi=600)
    plt.show()

    return s, eps_el, eps_tot, eps_p


def plot_sigma_a(sigma_a, n_i, sort_desc=True):
    v = np.asarray(sigma_a, dtype=float).ravel()
    n = np.asarray(n_i, dtype=float).ravel()
    if v.size != n.size:
        raise ValueError(f"Dimensioni non coerenti: sigma_re={v.size}, n_i={n.size}")

    mask = ~(np.isnan(v) | np.isnan(n)) & (n > 0)
    v, n = v[mask], n[mask]

    if sort_desc and v.size:
        order = np.argsort(v)[::-1]
        v, n = v[order], n[order]

    seq = []
    sign = +1  # primo valore dopo lo 0 è positivo
    for val, ni in zip(v, n):
        k = int(round(2.0 * float(ni)))
        for _ in range(max(k, 0)):
            seq.append(sign * float(val))
            sign *= -1  # alterna ad ogni step

    y = np.array([0.0] + seq, dtype=float)
    x = np.arange(y.size, dtype=int)

    plt.figure(figsize=(10, 6))
    plt.gca().set_facecolor('whitesmoke')
    plt.plot(x, y, color="#CC0000", linewidth=1.5)
    plt.xlabel(r"Cicli $n_{i}$")
    plt.ylabel(r"$\sigma_{a}$")
    plt.title(r"Spettro per $\sigma_{a}$ ordinati")
    x_min = min(x)
    x_max = max(x)
    y_min = np.floor(np.nanmin(y)/100)*100
    y_max = np.ceil( np.nanmax(y)/100)*100
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.savefig("plot/plot_epsilon_re.png", dpi=600)
    plt.show()
    return x, y


def plot_epsilon_a(epsilon_a, n_i, sort_desc=True):
    v = np.asarray(epsilon_a, dtype=float).ravel()
    n = np.asarray(n_i, dtype=float).ravel()
    if v.size != n.size:
        raise ValueError(f"Dimensioni non coerenti: epsilon_re={v.size}, n_i={n.size}")

    mask = ~(np.isnan(v) | np.isnan(n)) & (n > 0)
    v, n = v[mask], n[mask]

    if sort_desc and v.size:
        order = np.argsort(v)[::-1]
        v, n = v[order], n[order]

    seq = []
    sign = +1  # primo valore dopo lo 0 è positivo
    for val, ni in zip(v, n):
        k = int(round(2.0 * float(ni)))
        for _ in range(max(k, 0)):
            seq.append(sign * float(val))
            sign *= -1  # alterna ad ogni step

    y = np.array([0.0] + seq, dtype=float)
    x = np.arange(y.size, dtype=int)

    plt.figure(figsize=(10, 6))
    plt.gca().set_facecolor('whitesmoke')
    plt.plot(x, y, color="#CC0000", linewidth=1.5)
    plt.xlabel(r"Cicli $n_{i}$")
    plt.ylabel(r"$\epsilon_{a}$")
    plt.title(r"Spettro per $\epsilon_{a}$ ordinati")
    x_min = min(x)
    x_max = max(x)
    y_min = np.floor(np.nanmin(y)/100)*0.025 if np.nanmax(np.abs(y)) >= 0.02 else np.floor(np.nanmin(y)/0.1)*0.02
    y_max = np.ceil( np.nanmax(y)/100)*0.025  if np.nanmax(np.abs(y)) >= 0.02 else np.ceil( np.nanmax(y)/0.1)*0.02
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.savefig("plot/plot_epsilon_a.png", dpi=600)
    plt.show()
    return x, y

