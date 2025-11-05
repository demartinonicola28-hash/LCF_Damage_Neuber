import numpy as np
from scipy.optimize import fsolve
from typing import List, Tuple

def calcola_sigma_r(Kf: float, S_r: List[float], E: float, K_prime: float,
                    n_prime: float, gamma_M2: float, gamma_FE: float) -> List[float]:
    """
    Ritorna σ_r per ogni sr in S_r risolvendo:
      σ_r [ σ_r/E + 2(σ_r/(2K'))^(1/n') ] = (Kf*γ_M2*γ_FE*sr)^2 / E
    """
    if not S_r:
        return []

    sigma_r_values: List[float] = []
    for sr in S_r:
        if sr == 0.0:
            sigma_r_values.append(0.0)
            continue

        right_side = (Kf * gamma_M2 * gamma_FE * sr) ** 2 / E

        def equation(sigma_r):
            if sigma_r <= 0:
                return 1e6
            return sigma_r * (sigma_r / E + 2.0 * (sigma_r / (2.0 * K_prime)) ** (1.0 / n_prime)) - right_side

        x0 = max(1.0, abs(Kf * gamma_M2 * gamma_FE * sr))  # guess elastico
        sigma_r = float(fsolve(equation, x0=x0)[0])
        sigma_r_values.append(max(sigma_r, 0.0))

    return sigma_r_values

def calcola_sigma_p(S_0: List[float], S_r: List[float], Kf: float, E: float,
                    K_prime: float, n_prime: float, gamma_M2: float, gamma_FE: float) -> List[float]:
    """
    σ_p per step risolvendo:
      σ_p [ σ_p/E + (σ_p/K')^(1/n') ] = (Kf*γ_M2*γ_FE)^2/E * (S0_i + 0.5*Sr_i)^2
    dove S0_i e Sr_i variano a ogni step.
    """
    if len(S_0) != len(S_r):
        raise ValueError("S_0 e S_r devono avere la stessa lunghezza.")
    sigma_p_values = []
    for s0, sr in zip(S_0, S_r):
        right_side = (Kf * gamma_M2 * gamma_FE) ** 2 * (s0 + 0.5 * sr) ** 2 / E

        def equation(sigma_p):
            if sigma_p <= 0:
                return 1e6
            return sigma_p * (sigma_p / E + (sigma_p / K_prime) ** (1.0 / n_prime)) - right_side

        x0 = max(1.0, abs(Kf * gamma_M2 * gamma_FE * (s0 + 0.5*sr)))
        sigma_p = float(fsolve(equation, x0=x0)[0])
        sigma_p_values.append(max(sigma_p, 0.0))
    return sigma_p_values
