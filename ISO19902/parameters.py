# parameters.py
"""
GUI parametri con opzione UML (Bäumel–Seeger).
- Se UML attivo:
    K′ = 1.65 * fu
    n′ = 0.15
    σ′f = 1.5 * fu
    ψ = 1              se fu/E ≤ 0.003
        1.375 − 125 fu/E  altrimenti
    ε′f = 0.59 * ψ
    b = −0.087
    c = −0.58
  I campi sopra sono bloccati e mostrano il valore calcolato.
- Se UML disattivo: i campi restano precompilati ma modificabili.

Altre funzioni:
- Kf = Δσc_smooth / Δσc_notch (auto-update, read-only).
- Icona e immagine con percorsi robusti (PyInstaller compatibile).
"""

from pathlib import Path
import sys
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, Tk

# ==========================
# DEFAULTS
# ==========================
DEFAULTS = {
    # EN 1993-1-9 categorie (per Kf)
    "delta_sigma_smooth": 160.0,   # MPa
    "delta_sigma_notch": 56.0,     # MPa

    # Parametri base
    "E": 210000.0,                 # MPa
    "fu": 510.0,                   # MPa (σu)

    # Stabilized cyclic curve - Ramberg–Osgood
    "K_prime": 841.5,              # MPa (K′)
    "n_prime": 0.15,               # n′

    # Smith–Watson–Topper's approch
    "sigma_prime_f": 765.0,        # MPa (σ′f)
    "epsilon_prime_f": 0.59,       # (ε′f)
    "b": -0.087,
    "c": -0.58,

    # Fattori parziali
    "gamma_M2": 1.25,
    "gamma_ov": 1.25,
    "gamma_I": 1.2,
    "gamma_FE": 1.05,
}

# Esporta globali compatibili con il resto del progetto
E = DEFAULTS["E"]
fu = DEFAULTS["fu"]
delta_sigma_smooth = DEFAULTS["delta_sigma_smooth"]
delta_sigma_notch = DEFAULTS["delta_sigma_notch"]
Kf = delta_sigma_smooth / delta_sigma_notch
K_prime = DEFAULTS["K_prime"]
n_prime = DEFAULTS["n_prime"]
sigma_prime_f = DEFAULTS["sigma_prime_f"]
epsilon_prime_f = DEFAULTS["epsilon_prime_f"]
b = DEFAULTS["b"]
c = DEFAULTS["c"]
gamma_M2 = DEFAULTS["gamma_M2"]
gamma_ov = DEFAULTS["gamma_ov"]
gamma_I = DEFAULTS["gamma_I"]
gamma_FE = DEFAULTS["gamma_FE"]


# ==========================
# UTILITY: risorse e icona
# ==========================
def resource_path(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name

def set_app_icon(root: Tk, icon_name: str = "icona.ico") -> None:
    for p in (resource_path(icon_name),
              resource_path("assets") / icon_name,
              Path.cwd() / icon_name):
        if p.exists():
            try:
                if sys.platform.startswith("win") and p.suffix.lower() == ".ico":
                    root.iconbitmap(str(p))
                    return
            except Exception:
                pass
            img = Image.open(p)
            photo = ImageTk.PhotoImage(img)
            root.iconphoto(True, photo)
            root._icon_ref = photo  # evita GC
            return


# ==========================
# UML: formule
# ==========================
def compute_uml_params(E_val: float, fu_val: float):
    """Ritorna un dict con i parametri calcolati via UML."""
    Kp = 1.65 * fu_val
    np = 0.15
    sigf = 1.5 * fu_val
    ratio = fu_val / E_val
    psi = 1.0 if ratio <= 0.003 else (1.375 - 125.0 * ratio)
    epsf = 0.59 * psi
    return {
        "K_prime": Kp,
        "n_prime": np,
        "sigma_prime_f": sigf,
        "epsilon_prime_f": epsf,
        "b": -0.087,
        "c": -0.58,
    }


# ==========================
# GUI principale
# ==========================
def ask_parameters():
    # Avvio finestra o fallback headless
    try:
        root = tk.Tk()
        ENTRY_COL_PX = 260  # scegli il valore che preferisci
        LABEL_COL_PX = 180  # larghezza uniforme colonna etichette
    except Exception:
        out = DEFAULTS.copy()
        out["Kf"] = out["delta_sigma_smooth"] / out["delta_sigma_notch"]
        globals().update(out)
        return out

    root.title("Low Cycle Fatigue Analysis with Neuber's Method")
    root.geometry("1080x850")
    root.resizable(False, False)          # <-- blocca le frecce/drag
    #root.minsize(1200, 500)
    set_app_icon(root, "icona.ico")

    container = ttk.Frame(root, padding=20)
    container.grid(sticky="nsew")
    container.columnconfigure(0, weight=1)                        # colonna sinistra
    container.columnconfigure(1, minsize=ENTRY_COL_PX, weight=0)  # colonna destra (immagine)

    # Stile per riquadro immagine
    style = ttk.Style()
    style.configure("White.TLabelframe")
    style.configure("White.TLabelframe.Label")


    # -------------------------------------------------
    # Riquadro: EN 1993-1-9 (Δσc e Kf auto)
    # -------------------------------------------------
    lf_en = ttk.LabelFrame(container, text="EN 1993-1-9: 2025", padding=10)
    lf_en.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    lf_en.columnconfigure(0, minsize=LABEL_COL_PX, weight=0)
    lf_en.columnconfigure(1, minsize=ENTRY_COL_PX, weight=0)

    # Etichette e entry per Δσc
    ttk.Label(lf_en, text="Smooth detail category Δσc").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_ds = ttk.Entry(lf_en, justify="right", width=20)
    ent_ds.grid(row=0, column=1, sticky="e")
    ent_ds.insert(0, str(DEFAULTS["delta_sigma_smooth"]))

    ttk.Label(lf_en, text="Notch detail category Δσc").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_dn = ttk.Entry(lf_en, justify="right", width=20)
    ent_dn.grid(row=1, column=1, sticky="e")
    ent_dn.insert(0, str(DEFAULTS["delta_sigma_notch"]))

    # Funzione per calcolare Kf da Δσc_smooth e Δσc_notch
    def calculate_Kf(delta_sigma_smooth, delta_sigma_notch):
        return delta_sigma_smooth / delta_sigma_notch

    # Etichetta e entry per Kf (read-only)
    ttk.Label(lf_en, text="Fatigue notch factor Kf").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_kf = ttk.Entry(lf_en, justify="right", width=20, state=tk.DISABLED)
    ent_kf.grid(row=2, column=1, sticky="e")

    # Calcola Kf subito dopo il caricamento dei valori predefiniti
    Kf_value = calculate_Kf(DEFAULTS["delta_sigma_smooth"], DEFAULTS["delta_sigma_notch"])
    ent_kf.insert(0, f"{float(Kf_value):.2f}")  # Inserisci Kf calcolato

    # Etichette e entry per Δσc
    ttk.Label(lf_en, text="Smooth detail category Δσc").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_ds = ttk.Entry(lf_en, justify="right", width=20)
    ent_ds.grid(row=0, column=1, sticky="e")
    ent_ds.insert(0, str(DEFAULTS["delta_sigma_smooth"]))

    ttk.Label(lf_en, text="Notch detail category Δσc").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_dn = ttk.Entry(lf_en, justify="right", width=20)
    ent_dn.grid(row=1, column=1, sticky="e")
    ent_dn.insert(0, str(DEFAULTS["delta_sigma_notch"]))

    # Etichetta e entry per Kf (read-only)
    ttk.Label(lf_en, text="Fatigue notch factor Kf").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_kf = ttk.Entry(lf_en, justify="right", width=20, state=tk.DISABLED)
    ent_kf.grid(row=2, column=1, sticky="e")

    # Funzione di aggiornamento di Kf
    def update_kf(*_):
        try:
            # Calcola Kf da Δσc_smooth e Δσc_notch
            ds = float(ent_ds.get().replace(",", ".").strip())
            dn = float(ent_dn.get().replace(",", ".").strip())
            if dn == 0:  # Proteggi da divisione per zero
                val = ""
            else:
                val = calculate_Kf(ds, dn)  # Usa la funzione per calcolare Kf
            # Aggiorna Kf
            ent_kf.configure(state=tk.NORMAL)
            ent_kf.delete(0, tk.END)
            ent_kf.insert(0, "" if val == "" else f"{val:.2f}")
            ent_kf.configure(state=tk.DISABLED)
        except ValueError:
            pass

    # Calcola Kf subito dopo il caricamento dei valori predefiniti
    update_kf()  # Aggiungi questa riga per calcolare Kf appena la finestra si apre.

    # Eventi per aggiornare Kf quando l'utente modifica i campi
    ent_ds.bind("<KeyRelease>", update_kf)
    ent_dn.bind("<KeyRelease>", update_kf)

    # -------------------------------------------------
    # Riquadro: UML + fu (prima di Ramberg–Osgood)
    # -------------------------------------------------
    lf_pr = ttk.LabelFrame(container, text="Steels's Properties", padding=10)
    lf_pr.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    lf_pr.columnconfigure(0, minsize=LABEL_COL_PX, weight=0)
    lf_pr.columnconfigure(1, minsize=ENTRY_COL_PX, weight=0)

    uml_var = tk.BooleanVar(value=True)
    chk_uml = ttk.Checkbutton(lf_pr, text="Uniform Material Law (UML) – Bäumel & Seeger", variable=uml_var)
    chk_uml.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

    ttk.Label(lf_pr, text="Young's modulus E").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_E = ttk.Entry(lf_pr, justify="right", width=20); ent_E.grid(row=1, column=1, sticky="e"); ent_E.insert(0, str(DEFAULTS["E"]))

    ttk.Label(lf_pr, text="Ultimate tensile strength fu").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_fu = ttk.Entry(lf_pr, justify="right", width=20); ent_fu.grid(row=2, column=1, sticky="e"); ent_fu.insert(0, str(DEFAULTS["fu"]))

    # -------------------------------------------------
    # Riquadro: Ramberg–Osgood
    # -------------------------------------------------
    lf_ro = ttk.LabelFrame(container, text="Stabilized Cyclic Curve (Ramberg – Osgood)", padding=10)
    lf_ro.grid(row=2, column=0, sticky="ew", pady=(0, 10))
    lf_ro.columnconfigure(0, minsize=LABEL_COL_PX, weight=0)
    lf_ro.columnconfigure(1, minsize=ENTRY_COL_PX, weight=0)

    ttk.Label(lf_ro, text="Cyclic hardening coefficient K′").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_Kp = ttk.Entry(lf_ro, justify="right", width=20); ent_Kp.grid(row=0, column=1, sticky="e"); ent_Kp.insert(0, str(DEFAULTS["K_prime"]))

    ttk.Label(lf_ro, text="Cyclic hardening exponent n′").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_np = ttk.Entry(lf_ro, justify="right", width=20); ent_np.grid(row=1, column=1, sticky="e"); ent_np.insert(0, str(DEFAULTS["n_prime"]))

    # -------------------------------------------------
    # Riquadro: Smith–Watson–Topper
    # -------------------------------------------------
    lf_swt = ttk.LabelFrame(container, text="Initiation Life Method – ISO 19902:2020", padding=10)
    lf_swt.grid(row=3, column=0, sticky="ew", pady=(0, 10))
    lf_swt.columnconfigure(0, minsize=LABEL_COL_PX, weight=0)
    lf_swt.columnconfigure(1, minsize=ENTRY_COL_PX, weight=0)

    ttk.Label(lf_swt, text="Fatigue strength coefficient σ′f").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_sigf = ttk.Entry(lf_swt, justify="right", width=20); ent_sigf.grid(row=0, column=1, sticky="e"); ent_sigf.insert(0, str(DEFAULTS["sigma_prime_f"]))

    ttk.Label(lf_swt, text="Fatigue ductility coefficient ε′f").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_epsf = ttk.Entry(lf_swt, justify="right", width=20); ent_epsf.grid(row=1, column=1, sticky="e"); ent_epsf.insert(0, str(DEFAULTS["epsilon_prime_f"]))

    ttk.Label(lf_swt, text="Fatigue strength exponent b").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_b = ttk.Entry(lf_swt, justify="right", width=20); ent_b.grid(row=2, column=1, sticky="e"); ent_b.insert(0, str(DEFAULTS["b"]))

    ttk.Label(lf_swt, text="Fatigue ductility exponent c").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_c = ttk.Entry(lf_swt, justify="right", width=20); ent_c.grid(row=3, column=1, sticky="e"); ent_c.insert(0, str(DEFAULTS["c"]))

    # -------------------------------------------------
    # Riquadro: fattori EN 1993 e EN 1998
    # -------------------------------------------------
    lf_fac = ttk.LabelFrame(container, text="EN 1993 e EN 1998", padding=10)
    lf_fac.grid(row=4, column=0, sticky="ew", pady=(0, 10))
    lf_fac.columnconfigure(0, minsize=LABEL_COL_PX, weight=0)
    lf_fac.columnconfigure(1, minsize=ENTRY_COL_PX, weight=0)

    ttk.Label(lf_fac, text="Partial factor γM2").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_gM2 = ttk.Entry(lf_fac, justify="right", width=20); ent_gM2.grid(row=0, column=1, sticky="e"); ent_gM2.insert(0, str(DEFAULTS["gamma_M2"]))

    ttk.Label(lf_fac, text="Overstrength factor γov").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_gov = ttk.Entry(lf_fac, justify="right", width=20); ent_gov.grid(row=1, column=1, sticky="e"); ent_gov.insert(0, str(DEFAULTS["gamma_ov"]))

    ttk.Label(lf_fac, text="Importance factor γI").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_gI = ttk.Entry(lf_fac, justify="right", width=20); ent_gI.grid(row=2, column=1, sticky="e"); ent_gI.insert(0, str(DEFAULTS["gamma_I"]))

    ttk.Label(lf_fac, text="Model factor γFE").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
    ent_gFE = ttk.Entry(lf_fac, justify="right", width=20); ent_gFE.grid(row=3, column=1, sticky="e"); ent_gFE.insert(0, str(DEFAULTS["gamma_FE"]))

    # -------------------------------------------------
    # Colonna destra: immagine formule
    # -------------------------------------------------
    lf_img = ttk.LabelFrame(container, text="Reference formulas", padding=10, style="White.TLabelframe")
    lf_img.grid(row=0, column=1, rowspan=6, sticky="n", padx=(12, 0))
    try:
        img_path = resource_path("formulas.png")
        if not img_path.exists():
            img_path = resource_path("assets/formulas.png")
        img = Image.open(img_path)
        img.thumbnail((int(260*2.068), int(500*2.068)))
        photo = ImageTk.PhotoImage(img)
        lbl_img = ttk.Label(lf_img, image=photo); lbl_img.image = photo; lbl_img.pack()
    except Exception as e:
        ttk.Label(lf_img, text=f"Immagine non trovata: {e}").pack()

    # -------------------------------------------------
    # Logica UML: calcolo, blocco/sblocco, binding
    # -------------------------------------------------
    uml_targets = [ent_Kp, ent_np, ent_sigf, ent_epsf, ent_b, ent_c]

    def set_state(widgets, state):
        for w in widgets:
            w.configure(state=state)

    def recalc_from_uml(*_):
        if not uml_var.get():
            return
        try:
            E_val = float(ent_E.get().replace(",", "."))
            fu_val = float(ent_fu.get().replace(",", "."))
        except ValueError:
            return
        vals = compute_uml_params(E_val, fu_val)
        # Aggiorna contenuti (temporaneamente abilita per scrivere)
        for w in uml_targets:
            w.configure(state=tk.NORMAL)
        ent_Kp.delete(0, tk.END);   ent_Kp.insert(0, f"{vals['K_prime']:.2f}")
        ent_np.delete(0, tk.END);   ent_np.insert(0, f"{vals['n_prime']}")
        ent_sigf.delete(0, tk.END); ent_sigf.insert(0, f"{vals['sigma_prime_f']}")
        ent_epsf.delete(0, tk.END); ent_epsf.insert(0, f"{vals['epsilon_prime_f']}")
        ent_b.delete(0, tk.END);    ent_b.insert(0, f"{vals['b']}")
        ent_c.delete(0, tk.END);    ent_c.insert(0, f"{vals['c']}")
        # Riblocca
        set_state(uml_targets, tk.DISABLED)

    def apply_uml_toggle(*_):
        if uml_var.get():
            recalc_from_uml()
            set_state(uml_targets, tk.DISABLED)
        else:
            set_state(uml_targets, tk.NORMAL)

    chk_uml.configure(command=apply_uml_toggle)
    ent_fu.bind("<KeyRelease>", recalc_from_uml)
    ent_E.bind("<KeyRelease>", recalc_from_uml)
    apply_uml_toggle()  # inizializza secondo stato corrente

    # -------------------------------------------------
    # Pulsanti e callbacks
    # -------------------------------------------------
    confirmed = {"ok": False}
    result = {}

    def on_ok():
        # Conversioni numeriche
        try:
            result["delta_sigma_smooth"] = float(ent_ds.get().replace(",", "."))
            result["delta_sigma_notch"]  = float(ent_dn.get().replace(",", "."))
            result["Kf"] = float(ent_kf.get().replace(",", "."))
            result["fu"] = float(ent_fu.get().replace(",", "."))
            result["E"]  = float(ent_E.get().replace(",", "."))
            result["K_prime"] = float(ent_Kp.get().replace(",", "."))
            result["n_prime"] = float(ent_np.get().replace(",", "."))
            result["sigma_prime_f"]   = float(ent_sigf.get().replace(",", "."))
            result["epsilon_prime_f"] = float(ent_epsf.get().replace(",", "."))
            result["b"] = float(ent_b.get().replace(",", "."))
            result["c"] = float(ent_c.get().replace(",", "."))
            result["gamma_M2"] = float(ent_gM2.get().replace(",", "."))
            result["gamma_ov"] = float(ent_gov.get().replace(",", "."))
            result["gamma_I"]  = float(ent_gI.get().replace(",", "."))
            result["gamma_FE"] = float(ent_gFE.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Errore", "Inserisci solo numeri (usa . come separatore).")
            return

        globals().update(result)
        confirmed["ok"] = True
        root.destroy()

    def on_cancel():
        confirmed["ok"] = False
        root.destroy()

    btns = ttk.Frame(container); btns.grid(row=5, column=0, sticky="e")
    ttk.Button(btns, text="Annulla", command=on_cancel).grid(row=0, column=0, padx=(0, 8), pady=(4, 0))
    ttk.Button(btns, text="OK", command=on_ok).grid(row=0, column=1, pady=(4, 0))

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.bind("<Return>", lambda e: on_ok())
    root.bind("<Escape>", lambda e: on_cancel())
    ent_ds.focus_set()

    root.mainloop()

    if not confirmed["ok"]:
        print("Analisi annullata dall'utente.")
        sys.exit(0)

    return result


# Test manuale
if __name__ == "__main__":
    params = ask_parameters()
    print(params)
