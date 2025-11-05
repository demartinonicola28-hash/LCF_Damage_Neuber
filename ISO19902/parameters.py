# parameters.py
"""
- Parametri di default
- Finestra iniziale per l'inserimento con tre riquadri:
    1) Neuber parameters
    2) Ramberg–Osgood parameters
    3) Smith–Watson–Topper parameters
- Etichette con simboli: K′, n′, σ′f, ε′f
- Se Annulla o X: interrompe l'analisi
- Se un campo è vuoto o non numerico: mostra errore e non prosegue
"""

import sys

# === DEFAULTS: modificabili a piacere ===
DEFAULTS = {
    "Kf": 160/56,   # Fattore di concentrazione della tensione (calcolato come Δσc liscia / Δσc intagliata)
    # St-52-3 (1,0841) - pagina 169 MATERIALS DATA FOR CYCLIC LOADING - Chr. Boller, T. Seeger
    "E": 210000.0,                # MPa
    "K_prime": 841.5,        # MPa (K′)
    "n_prime": 0.15,              # n′
    "sigma_prime_f": 765,   # MPa (σ′f)
    "epsilon_prime_f": 0.59,      # (ε'f)
    "b": -0.087,
    "c": -0.58,
    # Fattori di sicurezza parziali (EN 1993 e EN 1998)
    "gamma_M2": 1.05,
    "gamma_ov": 1.25,
    "gamma_I": 1.2,
    "gamma_FE": 1.05,
    "delta_sigma_smooth": 160,
    "delta_sigma_notch": 56,
}

# Esporta anche variabili globali per compatibilità col resto del codice
E = DEFAULTS["E"]
Kf = DEFAULTS["Kf"]
K_prime = DEFAULTS["K_prime"]
n_prime = DEFAULTS["n_prime"]
sigma_prime_f = DEFAULTS["sigma_prime_f"]
epsilon_prime_f = DEFAULTS["epsilon_prime_f"]
b = DEFAULTS["b"]
c = DEFAULTS["c"]
# Fattori di sicurezza parziali (EN 1993 e EN 1998)
gamma_M2 = DEFAULTS["gamma_M2"]
gamma_ov = DEFAULTS["gamma_ov"]
gamma_I = DEFAULTS["gamma_I"]
gamma_FE = DEFAULTS["gamma_FE"]
delta_sigma_smooth = DEFAULTS["delta_sigma_smooth"]
delta_sigma_notch = DEFAULTS["delta_sigma_notch"]

def ask_parameters():
    """
    GUI con tre riquadri:
      - Neuber parameters: Kf
      - Ramberg–Osgood:    E, K′ (K_prime), n′ (n_prime)
      - Smith–Watson–Topper: σ′f (sigma_prime_f), ε′f (epsilon_prime_f), b, c

    Mostra i default. Campi obbligatori e numerici.
    Annulla/X: interrompe l'analisi.
    In assenza di Tkinter o GUI: usa i default.
    """
    # Import lazy per poter funzionare anche in ambienti headless
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception:
        globals().update(DEFAULTS)
        return DEFAULTS.copy()

    # Prova ad aprire la finestra
    try:
        root = tk.Tk()
    except Exception:
        globals().update(DEFAULTS)
        return DEFAULTS.copy()

    # Imposta l'icona della finestra
    root.iconbitmap(r"C:\Users\demnic15950\Downloads\LCF_Damage_Neuber\ISO19902\icona.ico")  # Sostituisci "icona.ico" con il percorso corretto del tuo file icona    
    
    # C:\Users\demnic15950\Downloads\LCF_Damage_Neuber\icona.ico
    # D:\Utente\Downloads\LCF_Damage_Neuber\icona.ico
    # === Finestra ===
    root.title("Low-Cycle Fatigue Analysis")
    root.resizable(True, True)
    root.geometry("400x650")     # larghezza x altezza
    root.minsize(885, 600)

    # Contenitore principale
    container = ttk.Frame(root, padding=20)
    container.grid(sticky="nsew")
    container.columnconfigure(0, weight=1)

    # === Definizione gruppi (label visibili → nome variabile) ===
    # Le etichette usano i simboli richiesti; i dati vengono salvati nei nomi variabile standard
    groups = [
        ("EN 1993-1-9: 2025", [
            ("Smooth detail category Δσc           ", "delta_sigma_smooth"),
            ("Notch detail category Δσc           ", "delta_sigma_notch"),
            ("Fatigue notch factor  Kf           ", "Kf")  # Mostra Kf come calcolato da A/B
        ]),
        ("Stabilized Cyclic Curve - Ramberg–Osgood", [
            ("Young's modulus   E           ", "E"),
            ("Cyclic hardening coefficient K′            ", "K_prime"),
            ("Cyclic hardening exponent n′            ", "n_prime"),
        ]),
        ("Initiaton Life Method ISO 19902: 2020", [
            ("Fatigue strenght coefficient σ′f           ", "sigma_prime_f"),
            ("Fatigue ductility coefficiet ε′f           ", "epsilon_prime_f"),
            ("atigue strenght exponent b             ", "b"),
            ("Fatigue ductility exponent c             ", "c"),
        ]),
        ("EN 1993 and EN 1998", [
            ("Partial factor for steel γM2            ", "gamma_M2"),
            ("Overstrength factor γov           ", "gamma_ov"),
            ("Importance factor γI            ", "gamma_I"),
            ("Model factor γFE           ", "gamma_FE"),
        ])
    ]

    entries = {}   # mappa: nome_variabile -> widget Entry

    # Crea i tre riquadri con i campi e inserisce i default visibili
    row_g = 0
    for title, items in groups:
        lf = ttk.LabelFrame(container, text=title, padding=10)
        lf.grid(row=row_g, column=0, sticky="ew", pady=(0, 10))
        lf.columnconfigure(1, weight=1)

        for i, (label_text, var_name) in enumerate(items):
            ttk.Label(lf, text=label_text).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=4)
            e = ttk.Entry(lf, width=20)
            e.grid(row=i, column=1, sticky="ew")
            # mostra default
            e.insert(0, str(DEFAULTS[var_name]))
            entries[var_name] = e
        row_g += 1

    # --- Colonna destra con immagine e sfondo bianco --------------------------
    right_lf = ttk.LabelFrame(container, text="Reference formulas", padding=10)
    right_lf.grid(row=0, column=1, rowspan=row_g+1, sticky="n", padx=(12, 0))

    # imposta sfondo bianco
    right_lf.configure(style="White.TLabelframe")
    style = ttk.Style()
    style.configure("White.TLabelframe")            # background="white")
    style.configure("White.TLabelframe.Label")      # background="white")

    try:
        from PIL import Image, ImageTk
        img = Image.open("formulas.png")
        img.thumbnail((260*1.73, 500*1.73))
        photo = ImageTk.PhotoImage(img)
        img_label = ttk.Label(right_lf, image=photo) 
        img_label.image = photo
        img_label.pack()
    except Exception:
        ttk.Label(right_lf, text="formulas.png non trovata", background="white").pack()
    # --------------------------------------------------------------------------


    # Disabilita la casella di testo di Kf per non permettere modifiche
    entries["Kf"].config(state=tk.DISABLED)

    # Funzione di aggiornamento in tempo reale per Kf
    def update_Kf():
        try:
            delta_sigma_smooth = float(entries["delta_sigma_smooth"].get().replace(",", "."))
            delta_sigma_notch = float(entries["delta_sigma_notch"].get().replace(",", "."))
            # Calcola Kf come delta_sigma_smooth / delta_sigma_notch
            Kf_value = delta_sigma_smooth / delta_sigma_notch
            entries["Kf"].config(state=tk.NORMAL)  # Rendi la casella Kf modificabile temporaneamente
            entries["Kf"].delete(0, tk.END)
            entries["Kf"].insert(0, str(Kf_value))
            entries["Kf"].config(state=tk.DISABLED)  # Rendi la casella Kf non modificabile
        except ValueError:
            # In caso di errore (ad esempio, se non sono numeri validi), non aggiornare Kf
            pass

    # Aggiungi eventi per aggiornare Kf ogni volta che delta_sigma_smooth o delta_sigma_notch vengono modificati
    entries["delta_sigma_smooth"].bind("<KeyRelease>", lambda event: update_Kf())
    entries["delta_sigma_notch"].bind("<KeyRelease>", lambda event: update_Kf())

    # Focus al primo campo
    first_var = groups[0][1][0][1]  # "delta_sigma_smooth"
    entries[first_var].focus_set()

    # Stato di conferma
    confirmed = {"ok": False}
    result = {}

    # === Callbacks ===
    def on_ok():
        # Campi non vuoti
        for var_name, ent in entries.items():
            if not ent.get().strip():
                messagebox.showerror("Errore", "Dati mancanti. Compila tutti i campi.")
                return
        # Conversione numerica
        try:
            for var_name, ent in entries.items():
                val = float(ent.get().replace(",", "."))
                result[var_name] = val
        except ValueError:
            messagebox.showerror("Errore", "Inserisci solo numeri (usa . come separatore).")
            return

        # Aggiorna le variabili globali del modulo
        globals().update(result)
        confirmed["ok"] = True
        root.destroy()

    def on_cancel():
        confirmed["ok"] = False
        root.destroy()

    # Pulsanti
    btns = ttk.Frame(container)
    btns.grid(row=row_g, column=0, sticky="e")
    ttk.Button(btns, text="Annulla", command=on_cancel).grid(row=0, column=0, padx=(0, 8), pady=(4, 0))
    ttk.Button(btns, text="OK", command=on_ok).grid(row=0, column=1, pady=(4, 0))

    # Eventi finestra
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.bind("<Return>", lambda e: on_ok())
    root.bind("<Escape>", lambda e: on_cancel())

    # Avvio GUI
    root.mainloop()

    # Uscita secondo scelta utente
    if not confirmed["ok"]:
        print("Analisi annullata dall'utente.")
        sys.exit(0)

    return result
