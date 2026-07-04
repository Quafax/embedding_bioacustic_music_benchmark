# Audio Transfer Benchmark

Ein schlankes, reproduzierbares Benchmark-Projekt für die Frage:

> Wie gut übertragen sich **musiktrainierte Audio-Embeddings** (MERT, MuQ) auf Bioakustik im Vergleich zu einem bioakustischen Referenzmodell (Perch 2.0), wenn alle Modelle mit derselben Downstream-Auswertung verglichen werden?

Die Pipeline ist absichtlich in feste, getrennte Schritte geteilt:

```text
Rohdaten + offizielle Splits
        ↓
kanonisches Manifest (eine Zeile = ein Clip)
        ↓
Encoder-Adapter → Embedding-Cache
        ↓
Linear Probe / Few-shot Probe
        ↓
Ergebnisse mit Config, Cache-ID und Metriken
```

## Start hier

Lies in dieser Reihenfolge:

1. [`docs/00_START_HERE.md`](docs/00_START_HERE.md) – einmalige Installation
2. [`docs/01_WHERE_THINGS_GO.md`](docs/01_WHERE_THINGS_GO.md) – wo Daten, Manifeste und Ergebnisse liegen
3. [`docs/02_FIRST_PILOT.md`](docs/02_FIRST_PILOT.md) – erster echter Watkins-Pilot mit MERT
4. [`docs/03_DAILY_WORKFLOW.md`](docs/03_DAILY_WORKFLOW.md) – was du bei jedem neuen Experiment machst
5. [`docs/04_ADD_DATASET.md`](docs/04_ADD_DATASET.md) – nur wenn du einen neuen Datensatz hinzufügst
6. [`docs/05_ENCODERS.md`](docs/05_ENCODERS.md) – MERT, MuQ und Perch 2.0

## Die wichtigsten Regeln

- Du bearbeitest im Normalfall **nur** eine Datei in `configs/experiments/`.
- Neue Datensätze werden über ein CSV-Manifest hinzugefügt; der generische Loader wird nicht verändert.
- Neue Encoder werden ausschließlich in `src/biobench/encoders/` ergänzt; der Dataset-Code bleibt unangetastet.
- Der Testsplit wird nie zum Aussuchen von Layer, C, Schwelle oder Few-shot-Seed benutzt.
- Embeddings werden nach `Manifest + Encoder-Spezifikation` gecacht. Ändert sich eines davon, entsteht automatisch ein neuer Cache.

## Die typischen Befehle

```bash
# 1. Manifest prüfen
biobench manifest validate data/manifests/watkins.csv

# 2. Experiment-Config prüfen
biobench check configs/experiments/watkins_mert_linear.yaml

# 3. Nur Embeddings berechnen oder Cache wiederverwenden
biobench extract configs/experiments/watkins_mert_linear.yaml

# 4. Cache auswerten
biobench evaluate configs/experiments/watkins_mert_linear.yaml

# 5. Beides hintereinander
biobench run configs/experiments/watkins_mert_linear.yaml

# 6. Few-shot-Auswertung
biobench fewshot configs/experiments/watkins_mert_fewshot.yaml
```

## Was bereits funktioniert

- generische, getestete Manifest-Validierung
- Audio-Laden, Mono-Mix, Segment-Ausschnitt und Resampling
- Encoder-Registry
- `dummy`-Encoder für einen kompletten Smoke Test
- MERT-Adapter über Hugging Face
- MuQ-Adapter nach der offiziellen MuQ-API
- Embedding-Cache mit Fingerprint
- Multiclass Linear Probe mit Validierungs-Auswahl von `C`
- Multilabel Linear Probe mit validierungsbasierter Auswahl von `C` und Entscheidungsschwelle
- Multiclass Few-shot Prototype-Probe
- Ergebnisordner mit kopierter Config und Metadaten

## Was für Perch 2.0 absichtlich getrennt bleibt

Perch 2.0 wird aktuell über ein klar abgegrenztes Adaptermodul vorbereitet. Die aktuelle Perch-Tooling-Landschaft nutzt Perch-Hoplite, TensorFlow und modellabhängige Downloads; die genaue Lade-API sollte vor dem ersten Perch-Lauf einmal gegen den konkret gewählten offiziellen Checkpoint geprüft werden. Deshalb muss nur **eine** Datei angepasst werden: `src/biobench/encoders/perch2.py`. Der Loader, die Manifeste, der Cache und die Evaluation ändern sich dabei nicht.

Details: [`docs/05_ENCODERS.md`](docs/05_ENCODERS.md).
