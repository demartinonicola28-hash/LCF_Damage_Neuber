# damage.py

import numpy as np
from typing import Sequence, List, Tuple

def calcola_D(N_f: Sequence[float], n_i: Sequence[float]) -> Tuple[float, List[float], List[float], float]:
    """
    Miner normalizzato per frequenze relative.
    Input:
      - N_f : vite a fatica per ogni classe/ciclo
      - n_i : occorrenze per ogni classe/ciclo
      - n_tot : numero di cicli nel blocco
      - n_b : numero di blocchi
    Ritorna:
      - n_b = 1
      - D_tot         : n_tot * n_b * (somma di D_ni)
      - D_ni  : vettore f_i / N_f
      - f_i           : n_i / n_tot
      - n_tot         : somma di n_i
    """
    Nf = np.asarray(N_f, dtype=float)
    ni = np.asarray(n_i, dtype=float)
    if Nf.shape != ni.shape:
        raise ValueError("N_f e n_i devono avere la stessa lunghezza.")

    n_tot = float(np.sum(ni))
    if n_tot <= 0:
        raise ValueError("Somma n_i nulla o negativa.")

    f_i = ni / n_tot

    # D_ni = f_i / N_f, con protezione da N_f<=0 o non finiti
    with np.errstate(divide="ignore", invalid="ignore"):
        D_ni = np.divide(f_i, Nf, out=np.full_like(f_i, np.inf),
                          where=(Nf > 0) & np.isfinite(Nf))

    D_tot = n_tot * float(np.sum(D_ni))
    return D_tot, D_ni.tolist(), f_i.tolist(), n_tot

def calcola_N_f_tot(N_f: Sequence[float], n_i: Sequence[float]) -> Tuple[float, List[float], List[float], float]:
    """
    Miner normalizzato per frequenze relative.
    Input:
      - N_f : vite a fatica per ogni classe/ciclo
      - n_i : occorrenze per ogni classe/ciclo
      - n_tot : numero di cicli nel blocco
      - n_b : numero di blocchi
    Ritorna:
      - D_tot = 1
      - D_ni  : vettore f_i / N_f
      - f_i           : n_i / n_tot
      - n_b           : D_tot / [(somma di D_ni) * n_tot]
    """
    Nf = np.asarray(N_f, dtype=float)
    ni = np.asarray(n_i, dtype=float)
    if Nf.shape != ni.shape:
        raise ValueError("N_f e n_i devono avere la stessa lunghezza.")
    n_tot = float(np.sum(ni))
    if n_tot <= 0:
        raise ValueError("Somma n_i nulla o negativa.")

    f_i = ni / n_tot
    D_tot = 1

    # D_ni = f_i / N_f, con protezione da N_f<=0 o non finiti
    with np.errstate(divide="ignore", invalid="ignore"):
        D_ni = np.divide(f_i, Nf, out=np.full_like(f_i, np.inf),
                          where=(Nf > 0) & np.isfinite(Nf))
    n_b = D_tot / (float(np.sum(D_ni)) * n_tot)

