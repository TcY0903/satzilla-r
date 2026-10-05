# SATZilla-R

SATZilla-R è un sistema di **algorithm selection** per SAT: data un'istanza CNF, predice quale solver sarà il più veloce e lo esegue.

Il sistema è ispirato a SATZilla e usa:
- il **feature extractor del 2024** (Revisiting SATZilla Features in 2024)
- **5 solver moderni**: AE-kissat2025-MAB, Cadical-sc2025, Dynamiccadical, IsaSAT, YalSAT
- un modello di predizione con classificatori pairwise

---

## Uso del sistema (predizione)

#### 1. Costruire l'immagine Docker
```
docker build -t satzilla-r .
```
#### 2. Eseguire la predizione su un'istanza CNF
```
docker run --rm -v $(pwd)/data:/app/data satzilla-r data/instance.cnf
```

---

## Addestramento del modello

#### 1. Installare le dipendenze
```
pip install -r requirements.txt
```
#### 2. Preparare i dati ed eseguire i solver sulle istanze
```
python scripts/run_solvers.py instance_dir/
```
Genera data/results/instance_dir/times.csv
#### 3. Estrarre le feature
```
python scripts/extract_features.py instance_dir/
```
Genera data/results/instance_dir/features.csv
#### 4. Costruire il dataset
```
python data_scripts/build_dataset.py
```
Genera training_data/ con train/validation/test e i dataset pairwise.
#### 5. Addestrare il modello
```
python data_scripts/train_model.py
```
Genera training_data/pairwise_forests.pkl, il modello usato dal sistema.