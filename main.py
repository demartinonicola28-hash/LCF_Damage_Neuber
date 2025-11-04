# main.py
from carica_dati import App  # Importa la GUI
from S_t import calcola_S, plot_S_t  # Importa calcola_S e plot_S_t da S_t.py
from rainflow import rainflow_counting, plot_rainflow, mostra_tabella, salva_rainflow  # Importa le funzioni Rainflow da rainflow.py
import matplotlib.pyplot as plt

"""
Funzione principale che esegue la GUI, carica i dati, calcola S, esegue Rainflow e plottiamo la matrice 3D.
"""
# --- STEP 1: CARICA DATI ---
app = App()  # Avvia la GUI per caricare i dati
app.mainloop()  # Esegui l'app
if not getattr(app, "data", None):
    print("Nessun dato confermato. Operazione annullata.")

# I dati vengono già salvati come vettori nella variabile globale
time, VM, P11, P22 = app.data  # Ottieni i dati da `carica_dati.py`

# --- STEP 2: CALCOLA IL VETTORE S ---
## Calcola il vettore S usando la funzione di `S_t.py`
S = calcola_S(VM, P11, P22)
#plot_S_t(time, S)  # Usa plot_S_t da `S_t.py`

# --- STEP 3: ESEGUI RAINFLOW COUNTING ---
S_r, S_0, n_i = rainflow_counting(S)
salva_rainflow(S_r, S_0, n_i)  # Salva i risultati usando salva_rainflow da `rainflow.py`
plot_rainflow(S_r, S_0, n_i)  # Usa plot_rainflow da `rainflow.py`
mostra_tabella(S_r, S_0, n_i)  # Mostra la tabella dei risultati
