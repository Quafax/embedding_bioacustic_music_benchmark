# 1. Wo kommt was hin?

```text
audio_transfer_benchmark/
├── configs/experiments/    ← nur hier änderst du normalerweise etwas
├── data/
│   ├── raw/                ← heruntergeladene Originaldaten; nie Git
│   ├── prepared/           ← optional vorverarbeitete Clips; nie Git
│   └── manifests/          ← kleine CSV-Dateien; gehören in Git
├── artifacts/
│   ├── embeddings/         ← automatisch erzeugte Embedding-Caches; nie Git
│   └── runs/               ← automatisch erzeugte Ergebnisse; nie Git
├── src/biobench/
│   ├── audio.py            ← ein generischer Audio-Lader für alle Datasets
│   ├── manifest.py         ← ein generischer Manifest-Validator
│   ├── encoders/           ← genau ein Adapter pro Modell
│   └── protocols/          ← Linear Probe und Few-shot, nicht datasetspezifisch
└── docs/                   ← Arbeitsanleitungen
```

## Daten

Lege heruntergeladene Dateien unter einem eindeutigen Namen ab:

```text
data/raw/watkins/
data/raw/cbi/
data/raw/dcase/
```

Du darfst die originale Ordnerstruktur innerhalb dieser Ordner behalten. Der Benchmark ist davon unabhängig, denn das Manifest enthält die tatsächlichen Pfade.

## Manifeste

Für jedes Dataset eine Datei:

```text
data/manifests/watkins.csv
data/manifests/cbi.csv
data/manifests/dcase.csv
```

Diese Dateien sind klein, nachvollziehbar und sollten versioniert werden. Sie dokumentieren zugleich deine exakte Split-Zuordnung.

## Configs

Ein Experiment ist nur eine kurze YAML-Datei. Beispiel:

```yaml
run:
  name: watkins_mert_linear
  seed: 42

data:
  manifest: data/manifests/watkins.csv

encoder: mert_95m

protocol:
  kind: linear_probe
  c_values: [0.01, 0.1, 1.0, 10.0, 100.0]
  threshold_values: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  max_iter: 5000
```

Beim normalen Vergleich kopierst du diese Datei und änderst höchstens:

- `run.name`
- `data.manifest`
- `encoder`
- `protocol.kind`

Du musst **nicht** Sample Rate, Checkpoint, Batch Size, Device oder Loader konfigurieren. Diese sind zentral in der Encoder-Registry festgelegt und werden in jeden Run geschrieben.

## Ergebnisse

Nach einem Run findest du alles unter:

```text
artifacts/runs/<run-name>/
├── config.yaml             ← exakte verwendete Config
├── cache_reference.json    ← welcher Cache verwendet wurde
├── metrics.json            ← wichtigste Metriken
├── predictions.csv         ← pro Clip: Wahrheit, Vorhersage, Split
└── fewshot.csv             ← nur bei Few-shot
```

Niemals Ergebnisse manuell umbenennen. Ändere stattdessen `run.name`, damit jeder Lauf einen klaren Ordner bekommt.
