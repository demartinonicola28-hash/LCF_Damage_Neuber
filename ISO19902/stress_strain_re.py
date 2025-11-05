# stress_strain_re.py

import numpy as np
from typing import List, Tuple
from ISO19902.ramberg_osgood import ramberg_osgood_amplitude

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
                       sigma_re: List[float]) -> List[float]:
    """
    ε_re tramite Ramberg–Osgood in ampiezza:
      ε_a = σ_a/E + (σ_a/K')^{1/n'}
    qui σ_a = σ_re.
    """
    return ramberg_osgood_amplitude(E, K_prime, n_prime, sigma_re)
