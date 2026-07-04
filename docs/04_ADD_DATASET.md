# 4. Neuen Datensatz hinzufügen, ohne den Loader umzuschreiben

Der generische Loader versteht das Standardmanifest. Deshalb ist das Hinzufügen eines Datasets in fast allen Fällen reine Datenaufbereitung.

## Standardfall: Du hast Audio + Tabellenannotation

Du schreibst eine kleine einmalige Importdatei unter `tools/`, zum Beispiel:

```text
tools/build_watkins_manifest.py
tools/build_cbi_manifest.py
```

Diese Datei darf alles Datasetspezifische enthalten:

- Original-CSV lesen
- Codes zu Klassennamen mappen
- offizielle Splits übernehmen
- Pfade erzeugen
- lange Aufnahmen mit `start_s` / `end_s` referenzieren

Sie schreibt am Ende nur das Standardformat nach `data/manifests/<dataset>.csv`.

Die Hauptpipeline wird nicht verändert.

## Vorlage für einen Importer

```python
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/my_dataset")
OUT = Path("data/manifests/my_dataset.csv")

rows = []
# Hier: eigene Annotation lesen und rows füllen.
rows.append({
    "clip_id": "my_dataset_000001",
    "audio_path": str(RAW / "audio" / "example.wav"),
    "split": "train",
    "task": "multiclass",
    "labels": "orca",
    "recording_id": "recording_001",
    "start_s": None,
    "end_s": None,
    "source_id": "original-id",
    "notes": "",
})

pd.DataFrame(rows).to_csv(OUT, index=False)
```

Danach:

```bash
biobench manifest validate data/manifests/my_dataset.csv
```

## Sonderfall: mehrere Clips aus derselben Aufnahme

Trage dieselbe `recording_id` in allen zugehörigen Zeilen ein. Verteile solche Clips nicht über verschiedene Splits, wenn die offizielle Aufteilung das nicht bereits vorgibt. Sonst kann ein Modell die Aufnahmebedingungen statt der Tierklasse lernen.

## Sonderfall: Detektion / Mehrlabel

Setze:

```csv
task,labels
multilabel,bird;mammal
```

Im Manifest eines Datensatzes darf `task` nicht wechseln. Für Multilabel nutzt der Linear Probe automatisch One-vs-Rest Logistic Regression und wählt die Labelschwelle auf Validation.

## Was nicht in den Importer gehört

- kein Resampling
- kein Denoising
- keine MFCCs
- kein Modellcode
- kein Train/Test-Splitting nach Bauchgefühl

Diese Dinge würden die Vergleichbarkeit gefährden oder gehören in die einheitliche Pipeline.
