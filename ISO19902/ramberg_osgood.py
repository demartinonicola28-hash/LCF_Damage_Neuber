# ramberg_osgood.py

import numpy as np
from typing import List

def ramberg_osgood_amplitude(E: float, K_prime: float, n_prime: float,
                             stress: List[float]) -> List[float]:
    if E <= 0 or K_prime <= 0 or n_prime <= 0:
        raise ValueError("E, K' e n' devono essere > 0.")

    s_a = np.asarray(stress, dtype=float)

    """
    Ramberg–Osgood: ε = σ/E + (σ/K')^n'
    """

    elastic = s_a / E
    plastic = (s_a / K_prime) ** (1.0 / n_prime)


    strain = elastic + plastic
    return strain.tolist()

def ramberg_osgood_range(E: float, K_prime: float, n_prime: float,
                             stress: List[float]) -> List[float]:
    if E <= 0 or K_prime <= 0 or n_prime <= 0:
        raise ValueError("E, K' e n' devono essere > 0.")

    s_r = np.asarray(stress, dtype=float)

    """
    Ramberg–Osgood: Δε = Δσ/E + K * 2[Δσ/(2K')]^n'
    """

    elastic = s_r / E
    plastic = 2 * (s_r / (2 * K_prime) ) ** (1.0 / n_prime)


    strain = elastic + plastic
    return strain.tolist()
