# damage.py

import numpy as np
from typing import Sequence, List, Tuple

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
      - D_tot = n_b * sum(D_ni) * gamma_I * 1.1 * gamma_ov
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

    D_tot = float(np.sum(D_ni)) * gamma_I * 1.1 *gamma_ov
    return D_tot, D_ni, n_i, n_tot

def calcola_n_b(D_ni: Sequence[float], gamma_I: float, gamma_ov: float) -> Tuple[float, float]:
    """
    Miner's rule
      - N_f : vite a fatica per classe (S_r, S_0)
      - n_i : numero di reversals per classe (S_r, S_0)
      - n_tot : numero di reversal nel blocco
      - D_ni : danno per classe (S_r, S_0) considerando il numero di reversal associati
      - n_b : numero di blocchi / stress history
    formulas:
      - D_tot = 1 * gamma_I * 1.1 * gamma_ov
      - D_ni  = n_i / N_f --> Sequence[float]
      - n_b   = D_tot / sum(D_ni)
    """
    D_tot = 1 
    D_ni = np.asarray(D_ni, dtype=float)
    sum_D_ni = np.nansum(D_ni)
    n_b = np.divide(D_tot, sum_D_ni * gamma_I * 1.1 * gamma_ov, out=np.array(np.inf, dtype=float),
                          where=(sum_D_ni != 0) & np.isfinite(sum_D_ni))
    #print(f"sum D_ni: {sum_D_ni}")
    return n_b, sum_D_ni
