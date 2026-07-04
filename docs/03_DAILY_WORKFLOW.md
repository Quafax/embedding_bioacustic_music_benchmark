# 3. Was machst du wann?

## Fall A – Du willst dasselbe Dataset mit einem anderen Encoder vergleichen

Du kopierst nur eine Config:

```text
watkins_mert_linear.yaml
→ watkins_muq_linear.yaml
```

und änderst nur:

```yaml
run:
  name: watkins_muq_linear
encoder: muq_large
```

Dann:

```bash
biobench run configs/experiments/watkins_muq_linear.yaml
```

Du änderst **weder** den Loader **noch** das Manifest **noch** den Probe-Code.

## Fall B – Du willst Few-shot statt Standardtransfer

Du nutzt die vorhandene Few-shot-Config:

```bash
biobench fewshot configs/experiments/watkins_mert_fewshot.yaml
```

Die Few-shot-Auswertung verwendet nur den Trainingssplit; Validation bleibt für andere Entscheidungen unberührt und Test nur für die Schlussauswertung.

## Fall C – Du willst einen neuen Datensatz ergänzen

1. Rohdaten nach `data/raw/<dataset>/`.
2. Ein neues Manifest nach `data/manifests/<dataset>.csv`.
3. `biobench manifest validate ...`.
4. Eine Config aus `watkins_mert_linear.yaml` kopieren.
5. Den Manifestpfad und `run.name` ändern.

Nur wenn der Datensatz keine tabellarische Annotation besitzt und du die Manifest-Erzeugung automatisieren möchtest, schreibst du einen kleinen einmaligen Importer unter `tools/`. Details: [`04_ADD_DATASET.md`](04_ADD_DATASET.md).

## Fall D – Du willst Einstellungen eines Encoders verändern

Das soll selten passieren. Beispiele:

- andere MERT-Schicht
- anderes MuQ-Modell
- anderes Perch-Checkpoint-Exportformat

Dann änderst du ausschließlich die zentrale Spezifikation in `src/biobench/encoders/registry.py` oder erstellst eine neue Encoder-ID. Du änderst nie die Experiment-Config für technische Details, die alle Runs derselben Encoder-Variante betreffen.

Beispiel: Du willst MERT Layer 6 und Layer 12 vergleichen. Dann legst du zwei klare IDs an:

```python
"mert_95m_layer6"
"mert_95m_last"
```

Und verwendest sie transparent in zwei Configs. So weißt du später, was verglichen wurde.

## Fall E – Du änderst die Daten oder Segmentierung

Dann ist dein alter Cache nicht mehr wissenschaftlich gültig. Das ist kein Problem: Das Manifest ändert sich, sein Hash ändert sich ebenfalls, und das Projekt schreibt automatisch einen neuen Cache.

## Fall F – Ein Experiment ist fertig

1. Schau in `metrics.json` und `predictions.csv`.
2. Prüfe offensichtliche Fehlerfälle anhand von `predictions.csv`.
3. Committe: Manifest, Config, Codeänderungen und eventuell eine kleine Ergebniszusammenfassung.
4. Committe **nicht** Rohdaten oder Embeddings.
