# 0. Einmalige Installation

Dieses Projekt ist absichtlich **nicht** von deinem alten `whaledetection`-Environment abhängig. Dein früheres Projekt brauchte TensorFlow; dieses Benchmark-Projekt braucht für MERT und MuQ zunächst PyTorch. Das vermeidet, dass TensorFlow, PyTorch, Perch und ältere Audio-Abhängigkeiten in einer einzigen Umgebung kollidieren.

## 0.1 Python-Version

Nutze **Python 3.11**. Nicht Python 3.13 wie in deinem alten Repository.

Grund: MERT und MuQ laufen gut im modernen PyTorch-Stack. Perch-Tooling kann zusätzlich TensorFlow benötigen; Python 3.11 ist dafür gegenwärtig die konservativste gemeinsame Wahl.

## 0.2 Projekt entpacken und Environment erzeugen

### Windows PowerShell

```powershell
cd C:\Pfad\zu\audio_transfer_benchmark
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,mert]"
```

### macOS/Linux

```bash
cd /pfad/zu/audio_transfer_benchmark
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,mert]"
```

Für MuQ später zusätzlich:

```bash
pip install -e ".[muq]"
```

## 0.3 Funktioniert die Installation?

```bash
pytest -q
biobench --help
```

Dann noch ein kompletter Smoke Test ohne großes Modell:

```bash
python -m biobench.demo
```

Der Demo-Lauf erzeugt zwei synthetische WAV-Dateien, ein Manifest, Dummy-Embeddings und ein Mini-Ergebnis unter `artifacts/demo/`.

## 0.4 Was du nicht machen sollst

- Nicht `data/raw/` in Git committen.
- Nicht echte Daten direkt in `src/` ablegen.
- Nicht für MERT, MuQ und Perch drei getrennte Datenloader schreiben.
- Nicht den Testsplit zum Auswählen von Hyperparametern verwenden.
