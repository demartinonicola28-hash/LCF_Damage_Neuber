import numpy as np
from scipy.optimize import fsolve
from typing import List, Tuple

def calcola_sigma_r(Kf: float, S_r: float, E: float, K_prime: float, n_prime: float, gamma_M2: float, gamma_FE: float) -> float:
    """
    Calcola σ_r ad ogni step risolvendo:
        σ_r * [ σ_r/E + 2(σ_r/(2K'))^(1/n') ] = (Kf * gamma_M2 * gamma_FE * S_r)^2 / E
    """
    if S_r == 0:
        return 0.0
    
    right_side = (Kf * gamma_M2 * gamma_FE * S_r) ** 2 / E

    def equation(sigma_r):
        if sigma_r <= 0:
            return 1e6  # restituisce un valore grande per evitare sigma_r <= 0
        return sigma_r * (sigma_r / E + 2 * (sigma_r / (2 * K_prime)) ** (1 / n_prime)) - right_side

    sigma_r_initial_guess = 1
    sigma_r = fsolve(equation, x0=sigma_r_initial_guess)[0]
    return max(sigma_r, 0)  # Assicurati che sigma_r non sia negativo

def calcola_sigma_p(S_0: float, S_r: List[float], Kf: float, E: float, K_prime: float, n_prime: float, gamma_M2: float, gamma_FE: float) -> List[float]:
    """
    Calcola σ_p ad ogni step risolvendo:
        σ_p * [ σ_p/E + (σ_p/K')^(1/n') ] = (Kf * gamma_M2 * gamma_FE)^2 / E * (S_0 + S_r/2)^2
    - S_0 è unico (media delle tensioni nominali)
    - S_r contiene i range delle tensioni nominali
    """
    sigma_p_values = []
    for S_r_value in S_r:
        right_side = (Kf * gamma_M2 * gamma_FE) ** 2 * (S_0 + 0.5 * S_r_value) ** 2 / E

        def equation(sigma_p):
            return sigma_p * (sigma_p / E + (sigma_p / K_prime) ** (1.0 / n_prime)) - right_side
        
        sigma_p_initial_guess = 1
        sigma_p = fsolve(equation, x0=sigma_p_initial_guess)[0]
        sigma_p_values.append(sigma_p)
    
    return sigma_p_values
