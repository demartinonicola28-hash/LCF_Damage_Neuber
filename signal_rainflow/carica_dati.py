# carica_dati.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re

# Vettori per i dati (liste)
time = []
VM = []
P11 = []
P22 = []

# Regex per trovare le sezioni delle tensioni
SECTION_RE = re.compile(r"^\s*Plate\s+Stress:\s*(von\s*Mises|Principal\s*11|Principal\s*22)\b", re.IGNORECASE)
FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

def _norm_label(s: str):
    s = s.lower()
    if "von" in s:
        return "Von Mises"
    if "11" in s:
        return "Principal 11"
    if "22" in s:
        return "Principal 22"
    return None

def _as_float(tok: str) -> float:
    return float(tok.replace(",", "."))

def aggiungi_zero_iniziale(t, vm, p11, p22, tol=1e-12):
    """
    Se necessario, aggiunge come primo punto (0.0, 0.0, 0.0, 0.0).
    - Non aggiunge nulla se il primo tempo è già ~0 e tutte le tensioni sono ~0.
    Restituisce nuove liste (non modifica gli argomenti in-place).
    """
    if t:
        cond_t = abs(t[0]) <= tol
        cond_vm = abs(vm[0]) <= tol if vm else False
        cond_p11 = abs(p11[0]) <= tol if p11 else False
        cond_p22 = abs(p22[0]) <= tol if p22 else False
        if cond_t and cond_vm and cond_p11 and cond_p22:
            return t, vm, p11, p22  # già ok

    # Prepend 0.0 a tutte le liste
    return [0.0] + t, [0.0] + vm, [0.0] + p11, [0.0] + p22

def carica_dati(path: str):
    """
    Carica i dati dal file e restituisce i vettori: time, VM, P11, P22.
    Prepend (0,0,0,0) se non presente.
    """
    global time, VM, P11, P22
    # Resetta le liste
    time, VM, P11, P22 = [], [], [], []

    current = None

    # Legge il file di testo
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Elabora le linee del file
    for ln in lines:
        msec = SECTION_RE.match(ln)
        if msec:
            current = _norm_label(msec.group(1))
            continue
        if current is None:
            continue  # ignora righe fuori sezione
        if re.match(r"\s*\d+\b", ln):  # riga con numeri (indice, X, Y)
            nums = FLOAT_RE.findall(ln.replace(",", "."))
            if len(nums) < 3:
                continue
            t = _as_float(nums[1])  # Tempo (X)
            y = _as_float(nums[2])  # Tensione (Y)
            if current == "Von Mises":
                time.append(t)
                VM.append(y)
            elif current == "Principal 11":
                P11.append(y)
            elif current == "Principal 22":
                P22.append(y)

    # Debug: Stampa le lunghezze prima dell'aggiunta dello zero
    print(f"[Prima] time={len(time)}, VM={len(VM)}, P11={len(P11)}, P22={len(P22)}")

    # Verifica lunghezze coerenti
    if not (len(time) == len(VM) == len(P11) == len(P22)):
        raise ValueError("I vettori time, VM, P11 e P22 non hanno la stessa lunghezza!")

    # Aggiunge (0,0,0,0) in testa se mancante
    time2, VM2, P112, P222 = aggiungi_zero_iniziale(time, VM, P11, P22)

    # Debug: Stampa le lunghezze dopo l'aggiunta dello zero
    print(f"[Dopo ] time={len(time2)}, VM={len(VM2)}, P11={len(P112)}, P22={len(P222)}")

    return time2, VM2, P112, P222

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Carica dati tensione-tempo (Straus7)")
        self.geometry("740x480")
        self.minsize(620, 400)

        self.path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Seleziona il .txt e premi Carica.")

        # Frame per file input
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="File dati:").pack(side="left")
        ttk.Entry(top, textvariable=self.path_var, width=60).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(top, text="Sfoglia…", command=self.pick_file).pack(side="left")

        # Frame per bottoni di azione
        btns = ttk.Frame(self, padding=(12, 0, 12, 12))
        btns.pack(fill="x")
        ttk.Button(btns, text="Carica", command=self.load_clicked).pack(side="left")
        ttk.Button(btns, text="OK", command=self.ok_clicked).pack(side="left", padx=8)

        # Barra di scorrimento
        self.scrollbar = tk.Scrollbar(self)
        self.scrollbar.pack(side="right", fill="y")

        # Treeview per mostrare i dati
        self.tree = ttk.Treeview(self, columns=("t", "vm", "p11", "p22"),
                                 show="headings", height=14, yscrollcommand=self.scrollbar.set)
        self.tree.pack(fill="both", expand=True, padx=12, pady=6)
        for c, h in zip(("t", "vm", "p11", "p22"), ("Time", "Von_Mises", "Principal_11", "Principal_22")):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=130, anchor="center")

        self.scrollbar.config(command=self.tree.yview)

        # Status della finestra
        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(12, 6))
        status.pack(fill="x")

        self.data = None

    def pick_file(self):
        p = filedialog.askopenfilename(
            title="Seleziona file dati",
            filetypes=[("Text files", "*.txt *.dat *.out"), ("All files", "*.*")]
        )
        if p:
            self.path_var.set(p)

    def load_clicked(self):
        p = self.path_var.get().strip()
        if not p:
            messagebox.showwarning("Errore", "Seleziona prima un file.")
            return
        try:
            # Carica i dati (già con eventuale (0,0,0,0) prepended)
            t, vm, p11, p22 = carica_dati(p)
            self.data = (t, vm, p11, p22)

            # Popola la GUI con i dati caricati
            for i in self.tree.get_children():
                self.tree.delete(i)
            for row in zip(t, vm, p11, p22):
                self.tree.insert("", "end", values=tuple(f"{x:.10g}" for x in row))

            self.status_var.set("Dati caricati.")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante il parsing:\n{e}")
            self.status_var.set("Errore.")

    def ok_clicked(self):
        """
        Salva i vettori globali e CHIUDE la finestra.
        """
        global time, VM, P11, P22
        if not self.data:
            messagebox.showinfo("Errore", "Carica i dati prima di premere OK.")
            return

        time, VM, P11, P22 = self.data
        self.destroy()  # chiude la finestra e termina il mainloop
