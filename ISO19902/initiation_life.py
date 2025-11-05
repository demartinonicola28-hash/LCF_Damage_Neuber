# strain_life_curve.py

import numpy as np
from typing import Sequence, List
from scipy.optimize import fsolve

def calcola_N_f(sigma_re: Sequence[float],
                epsilon_re: Sequence[float],
                E: float,
                sigma_f_prime: float,
                epsilon_f_prime: float,
                b: float,
                c: float,
                Nf_guess: float = 1.0) -> List[float]:
    """
    Risolve per N_f:
      left  = σ_re * ε_re * E
      right = (σ'_f)^2 (2N_f)^(2b) + σ'_f ε'_f E (2N_f)^(b+c)
      right - left = 0

    Usa fsolve su y=log(N_f) per avere N_f>0.
    Ritorna una lista di N_f (cicli).
    """
    s = np.asarray(sigma_re, dtype=float)
    e = np.asarray(epsilon_re, dtype=float)
    if s.shape != e.shape:
        raise ValueError("sigma_re ed epsilon_re devono avere la stessa lunghezza.")
    if E <= 0 or sigma_f_prime <= 0 or epsilon_f_prime <= 0:
        raise ValueError("E, σ'_f, ε'_f devono essere > 0.")

    out: List[float] = []
    for sig, eps in zip(s, e):
        left = float(sig * eps * E)
        if not np.isfinite(left) or left <= 0.0:
            out.append(np.inf)  # nessun danno
            continue

        def equation(y):
            Nf = np.exp(y)              # Nf > 0
            x = 2.0 * Nf
            right = (sigma_f_prime**2.0) * (x**(2.0*b)) + \
                    (sigma_f_prime*epsilon_f_prime*E) * (x**(b+c))
            return right - left

        y0 = np.log(max(Nf_guess, 1e-12))
        y = fsolve(equation, x0=y0)[0]
        out.append(float(np.exp(y)))

    return out
