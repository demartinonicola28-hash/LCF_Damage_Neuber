# result_export.py
"""
Funzione unica per esportare i risultati in un XLSX multi‑foglio e CSV separati.
Gestisce lunghezze diverse con pandas.Series. Include foglio parametri.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Any

import numpy as np
import pandas as pd


def export_fatica(
    *,
    # Serie temporale
    time: Iterable[float] | None = None,
    VM: Iterable[float] | None = None,
    P11: Iterable[float] | None = None,
    P22: Iterable[float] | None = None,
    S: Iterable[float] | None = None,
    # Picchi/Valli
    step: Iterable[float] | None = None,
    S_p: Iterable[float] | None = None,
    # Rainflow
    S_r: Iterable[float] | None = None,
    S_0: Iterable[float] | None = None,
    n_i: Iterable[float] | None = None,
    # Neuber
    sigma_r: Iterable[float] | None = None,
    sigma_p: Iterable[float] | None = None,
    sigma_0: Iterable[float] | None = None,
    # Ampiezze e strain
    sigma_a: Iterable[float] | None = None,
    sigma_re: Iterable[float] | None = None,
    epsilon_re_el: Iterable[float] | None = None,
    epsilon_re: Iterable[float] | None = None,
    epsilon_re_p: Iterable[float] | None = None,
    # Vita e danno
    N_f: Iterable[float] | None = None,
    D_ni_d: Iterable[float] | None = None,
    D_tot: float | None = None,
    n_tot: float | None = None,
    # Parametri
    params: Dict[str, Any] | None = None,
    # Output
    outdir: str = "export",
    xlsx_name: str = "risultati_fatica.xlsx",
    also_csv: bool = True,
) -> str:
    """
    Crea cartella outdir, salva un file Excel con fogli multipli e, se richiesto, CSV singoli.
    Ritorna il path dell'XLSX.
    """
    os.makedirs(outdir, exist_ok=True)

    def _series_dict(**kwargs):
        return {k: pd.Series(v) for k, v in kwargs.items() if v is not None}

    # Costruzione DataFrame per ciascun blocco
    sheets: Dict[str, pd.DataFrame] = {}

    if any(v is not None for v in (time, VM, P11, P22, S)):
        sheets["time_series"] = pd.DataFrame(_series_dict(time=time, VM=VM, P11=P11, P22=P22, S=S))

    if any(v is not None for v in (step, S_p)):
        sheets["peaks_valleys"] = pd.DataFrame(_series_dict(step=step, S_p=S_p))

    if any(v is not None for v in (S_r, S_0, n_i)):
        sheets["rainflow"] = pd.DataFrame(_series_dict(S_r=S_r, S_0=S_0, n_i=n_i))

    if any(v is not None for v in (sigma_r, sigma_p, sigma_0)):
        sheets["neuber"] = pd.DataFrame(_series_dict(sigma_r=sigma_r, sigma_p=sigma_p, sigma_0=sigma_0))

    if any(v is not None for v in (sigma_a, sigma_re, epsilon_re_el, epsilon_re, epsilon_re_p)):
        sheets["strain_stress"] = pd.DataFrame(_series_dict(
            sigma_a=sigma_a, sigma_re=sigma_re,
            epsilon_re_el=epsilon_re_el, epsilon_re=epsilon_re, epsilon_re_p=epsilon_re_p
        ))

    if any(v is not None for v in (N_f, D_ni_d)):
        sheets["fatigue_damage"] = pd.DataFrame(_series_dict(N_f=N_f, D_ni_d=D_ni_d))

    # Parametri e scalari
    params = params or {}
    df_params = pd.DataFrame({"chiave": list(params.keys()), "valore": list(params.values())})
    scalari_keys, scalari_vals = [], []
    if D_tot is not None:
        scalari_keys.append("D_tot")
        scalari_vals.append(D_tot)
    if n_tot is not None:
        scalari_keys.append("n_tot")
        scalari_vals.append(n_tot)
    if scalari_keys:
        df_scalari = pd.DataFrame({"chiave": scalari_keys, "valore": scalari_vals})
        df_params = pd.concat([df_params, df_scalari], ignore_index=True)
    sheets["parametri"] = df_params

    # Scrittura XLSX
    xlsx_path = os.path.join(outdir, xlsx_name)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)

    # CSV opzionali
    if also_csv:
        for name, df in sheets.items():
            df.to_csv(os.path.join(outdir, f"{name}.csv"), sep=";", index=False)

    return xlsx_path


if __name__ == "__main__":
    # Mini test fittizio
    xlsx = export_fatica(time=[0,1], VM=[1,2], P11=[0.1,0.2], P22=[0.3,0.4], S=[0.5,0.6],
                         step=[0,1,2], S_p=[10,20,30],
                         S_r=[5,4], S_0=[2,1], n_i=[0.5,1.0],
                         sigma_r=[100,80], sigma_p=[120,90], sigma_0=[60,45],
                         sigma_a=[50,40], sigma_re=[55,44],
                         epsilon_re_el=[2e-4, 1.8e-4], epsilon_re=[3e-4, 2.5e-4], epsilon_re_p=[1e-4, 7e-5],
                         N_f=[1e6, 2e6], D_ni_d=[1e-6, 5e-7], D_tot=0.12, n_tot=1.0,
                         params={"E": 2.1e11, "K'": 900e6, "n'": 0.2})
    print("Scritto:", xlsx)
