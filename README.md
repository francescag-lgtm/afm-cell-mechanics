# Analisi AFM di cellule tumorali: curve forza–indentazione e stress-relaxation

Codice di analisi sviluppato per la tesi di laurea magistrale in Fisica
(Università di Genova, relatore: Prof. Claudio Canale) di Francesca Elisabetta Gianelli.

Il codice analizza curve di microscopia a forza atomica (AFM) acquisite su
cellule vive:

- **curve forza–indentazione**: correzione del segnale (linea di base, punto di
  contatto, trascinamento viscoso, chiusura al contatto), fit di Hertz e fit di
  Ting con modulo di rilassamento PLR e SLS, correzione per lo spessore finito;
- **curve di stress-relaxation**: correzione dello zero, normalizzazione, fit
  dei modelli SLS, PLR, biesponenziale e poroelastico, scelta del tempo di
  transizione;
- **confronto tra linee cellulari**: una osservazione per cellula, criteri di
  qualità del fit, test di Mann–Whitney, tabelle, report e figure.

## Struttura

```
Analisi_murine.ipynb      configurazione e avvio dell'analisi (linee A1 e 0R)
Analisi_umane.ipynb       configurazione e avvio dell'analisi (linee EV e RAB5)
analisi_per_cellula.py    pipeline comune: raccolta delle curve, raggruppamento
                          per cellula, criteri di qualità, confronto tra le linee,
                          tabelle, report e figure
ting_utils.py             curve forza–indentazione
sr_utils.py               curve di stress-relaxation
prova.py                  funzioni di supporto (ricerca dei file, trascinamento viscoso)
requirements.txt          dipendenze Python
```

## Uso

1. Installare le dipendenze: `pip install -r requirements.txt`
2. Aprire il notebook del sistema da analizzare, indicare nella prima cella le
   cartelle dei dati e la cartella dei risultati, ed eseguire tutte le celle.

L'analisi produce, nella cartella dei risultati, le tabelle per curva e per
cellula (CSV), il registro delle esclusioni, le tabelle riassuntive con il
confronto tra le linee, un report PDF e le figure del capitolo Risultati
(`fig_tesi_*.pdf`).

I dati sperimentali non sono inclusi nel repository.

## Dipendenze

NumPy, SciPy, pandas, Matplotlib e lmfit per il calcolo e la visualizzazione.
La lettura dei file AFM e il fit di Hertz e Ting si appoggiano a
[PyFMGUI](https://github.com/jlopezalo/PyFMGUI); nello specifico è stato usato
il fork [PyFMlab_DyNaMo](https://github.com/DyNaMo-INSERM/PyFMlab_DyNaMo)
(con il sotto-modulo PyFMRheo_DyNaMo) e
[PyFMReader_DyNaMo](https://github.com/DyNaMo-INSERM/PyFMReader_DyNaMo).
Le versioni esatte sono elencate in `requirements.txt`.
