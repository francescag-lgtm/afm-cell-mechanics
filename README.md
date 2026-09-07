# AFM cell mechanics — pipeline di analisi

Codice di analisi sviluppato per la tesi magistrale in Fisica, Università di Genova (relatore: Prof. Claudio Canale).

Il progetto studia la risposta meccanica di singole cellule (linee umane
MCF10.DCIS.com e murine D2A1/D2_0R) misurata mediante microscopia a forza
atomica (AFM), confrontando condizioni di controllo e condizioni di
induzione di RAB5A associate alla transizione tissutale jammed → unjammed.
Le misure comprendono **curve forza-indentazione** (analizzate con il
modello di Ting) e **curve di stress-relaxation** (analizzate con modelli
di rilassamento SLS, legge di potenza, Maxwell biesponenziale e
poroelastico semplificato).

La pipeline è organizzata in **due rami paralleli**, ciascuno su tre
livelli — modulo di funzioni, analisi di una singola cartella/cellula,
aggregazione statistica di popolazione — descritti di seguito.

---

## Struttura del repository

```
.
├── ting_utils.py                  # modulo: fit di Hertz e Ting sulle curve forza-indentazione
├── prova.py                       # modulo: utility di supporto per ting_utils.py (selezione cartelle,
│                                   #   raccolta curve, correzione viscous drag)
├── Ting_single_curve.ipynb        # analisi di una curva singola o batch su una cartella (una linea/condizione)
├── Ting_aggregate_summary.ipynb   # aggregazione multi-cartella + confronto statistico tra condizioni
│
├── sr_utils.py                    # modulo: caricamento, fit e diagnostica delle curve di stress-relaxation
├── Rilassamento.ipynb             # analisi della curva media di stress-relaxation di una cartella/giornata
├── MediaDelleMediae_SR.ipynb      # grand mean multi-giornata + confronto statistico tra condizioni
│
├── sr_utils.ipynb                 # versione precedente, autocontenuta, dell'analisi di stress-relaxation
│                                   #   (non dipende da sr_utils.py; mantenuta per riferimento/tracciabilità)
│
└── requirements.txt               # dipendenze Python (da aggiungere, vedi sezione "Da completare")
```

---

## Ramo 1 — Curve forza-indentazione (modello di Ting)

**`ting_utils.py`** — libreria di funzioni condivise: ricerca del punto di
contatto, correzione di baseline/tilt/viscous drag, fit hertziano
preliminare, fit viscoelastico di Ting (con confronto automatico tra
reologia a legge di potenza e modello di Maxwell biesponenziale tramite
criterio BIC), calcolo delle metriche di qualità del fit ed export dei
risultati.

**`Ting_single_curve.ipynb`** — notebook operativo che richiama
`ting_utils.py`. Accetta in input:
- una singola curva forza-distanza, oppure
- una cartella contenente più cellule (batch), restituendo un file
  riassuntivo `ting_batch_summary.csv` con un modulo elastico, esponente
  di fluidità (βE), tempo di contatto e indicatori di qualità del fit
  per ciascuna curva analizzata.

**`Ting_aggregate_summary.ipynb`** — prende in ingresso i vari
`ting_batch_summary.csv` prodotti da più cartelle/condizioni sperimentali
(es. linee A1 vs 0R), li unisce assegnando manualmente un'etichetta di
gruppo, e produce statistiche per cellula e di popolazione (media ± SD),
boxplot per parametro e un report PDF riassuntivo, oltre ai CSV aggregati.

**Flusso d'uso:** `ting_utils.py` → `Ting_single_curve.ipynb` (una volta
per ciascuna cartella/condizione) → `Ting_aggregate_summary.ipynb` (una
volta, per il confronto finale tra condizioni).

---

## Ramo 2 — Curve di stress-relaxation

**`sr_utils.py`** — libreria di funzioni
condivise: caricamento delle curve grezze, individuazione del segmento di
hold, normalizzazione rispetto al picco iniziale, rimozione di spike e
rumore, calcolo della curva media per cellula/area, fit dei quattro
modelli di rilassamento discussi in tesi (SLS, PLR, Maxwell biesponenziale,
poroelastico semplificato) con selezione automatica del modello migliore.

**`Rilassamento.ipynb`** — notebook operativo che richiama `sr_utils.py`.
Presa in ingresso una cartella di curve grezze di una giornata/area,
calcola la curva media normalizzata, esegue il fit dei modelli di
rilassamento, stima il modulo elastico statico di controllo, e produce un
report PDF unico. Include inoltre una cella di confronto diretto tra due
condizioni (es. EV vs RAB5) con test di Mann–Whitney.

**`MediaDelleMediae_SR.ipynb`** — prende in ingresso i file
`*-media-norm.jpk-force` prodotti da `Rilassamento.ipynb` su più
giornate/cartelle, calcola la *grand mean* per ciascuna condizione
sperimentale (con criteri di esclusione "dura" e "morbida" per le curve
anomale, come descritto nel Cap. 6 della tesi), e produce la tabella
riassuntiva finale (modulo di Young, βE, ΔF, tempo di transizione
poroelastico/viscoelastico) con confronto statistico tra i gruppi.

**`sr_utils.ipynb`** — versione precedente e autocontenuta della stessa
analisi (tutte le funzioni definite localmente nel notebook, senza
dipendere da un modulo esterno). Non fa parte del flusso di lavoro
corrente, mantenuta nel repository a scopo di tracciabilità dello
sviluppo.

**Flusso d'uso:** `sr_utils.py` → `Rilassamento.ipynb` (una volta per
ciascuna cartella/giornata) → `MediaDelleMediae_SR.ipynb` (una volta, per
la grand mean e il confronto finale tra condizioni).

---

## Dipendenze

- NumPy, SciPy, Pandas, Matplotlib
- [lmfit](https://lmfit.github.io/lmfit-py/)
- pyFMRheo, pyFMReader, pyFMGUI — il fit di Hertz e Ting si appoggia su
  [PyFMGUI](https://github.com/jlopezalo/PyFMGUI), da cui questo lavoro ha
  preso spunto; nello specifico è stato usato il fork
  [PyFMlab_DyNaMo](https://github.com/DyNaMo-INSERM/PyFMlab_DyNaMo)
  (che include anche il sotto-modulo PyFMRheo_DyNaMo) e
  [PyFMReader_DyNaMo](https://github.com/DyNaMo-INSERM/PyFMReader_DyNaMo)
- scikit-learn (opzionale, solo per `r2_score`; la pipeline include un
  fallback che non richiede la dipendenza)

Installazione:
```bash
pip install -r requirements.txt
```

## Riferimenti

Codice sviluppato a supporto della tesi magistrale in Fisica, a.a.
2024/2025–2025/2026. Tesi completa disponibile su richiesta.
