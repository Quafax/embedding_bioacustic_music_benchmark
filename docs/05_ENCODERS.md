# 5. Encoder: Was ist schon fertig und wann änderst du etwas?

## Encoder-ID statt technischer Config

In Experiment-Configs verwendest du kurze IDs:

```yaml
encoder: mert_95m
```

Die technische Spezifikation steckt zentral in:

```text
src/biobench/encoders/registry.py
```

Das verhindert, dass unterschiedliche YAML-Dateien versehentlich leicht unterschiedliche Sample Rates oder Checkpoints für „dasselbe“ Modell verwenden.

## MERT

`mert_95m` ist direkt implementiert.

Installation:

```bash
pip install -e ".[mert]"
```

Der Adapter lädt `m-a-p/MERT-v1-95M`, verwendet 24 kHz und standardmäßig den letzten Hidden Layer. Die Modellkarte beschreibt dieses Modell als 12-lagiges 95M-Parameter-Modell mit 24-kHz-Audio und 75-Hz-Feature-Rate. MERT benötigt beim Laden `trust_remote_code=True`.

Wenn du Schichten vergleichen willst, erstelle klare neue Encoder-IDs in der Registry, zum Beispiel `mert_95m_layer6`. Ändere nicht still denselben Namen.

## MuQ

`muq_large` ist nach der offiziellen API implementiert.

Installation:

```bash
pip install -e ".[muq]"
```

MuQ erwartet strikt 24-kHz-Audio. Der Adapter lädt standardmäßig `OpenMuQ/MuQ-large-msd-iter` und bittet um Hidden States. Die offizielle MuQ-Dokumentation warnt, dass der offene Checkpoint gegenüber dem im Paper berichteten Modell abweichen kann; diese Checkpoint-ID wird deshalb in die Cache- und Ergebnis-Metadaten geschrieben.

## Perch 2.0

Perch 2.0 ist der einzige Adapter, den du vor dem ersten realen Perch-Lauf einmal konkretisieren musst. Das ist keine Schwäche des Benchmarks, sondern bewusstes Isolieren einer beweglichen externen Abhängigkeit:

- das alte Perch-Repository warnt selbst, dass Teile veraltet sind;
- Perch-Hoplite ist die aktuellere Tooling-Empfehlung;
- für Perch in Hoplite wird TensorFlow benötigt;
- der tatsächliche Encoder hängt vom gewählten offiziellen Model/Checkpoint ab.

Die einzige vorgesehene Datei ist:

```text
src/biobench/encoders/perch2.py
```

Dort muss `Perch2Encoder.encode()` am Ende einen PyTorch-Tensor der Form `[batch, time, dimension]` oder `[batch, dimension]` zurückgeben. Alles andere—Audio-Laden, Caching, Probes, Few-shot und Ergebnisdateien—bleibt gleich.

### Vorgehen für Perch

1. Entscheide dich vorab für den genauen offiziellen Perch-2.0-Export/Checkpoint.
2. Erstelle ein eigenes Environment oder installiere die nötigen Perch-Hoplite/TensorFlow-Abhängigkeiten zusätzlich.
3. Teste den Perch-Adapter zuerst mit einem Clip und prüfe Shape, Sample Rate und Determinismus.
4. Erst danach `biobench run configs/experiments/watkins_perch2_linear.yaml`.

## Dummy

`dummy` ist nur für Tests. Er hat keine wissenschaftliche Bedeutung, aber ermöglicht die komplette Pipeline ohne Modell-Download:

```yaml
encoder: dummy
```

## Wichtig für den fairen Vergleich

Dass MERT und MuQ 24 kHz benötigen und ein Perch-Export möglicherweise eine andere Native Sample Rate erwartet, ist okay. Du hältst die Clipgrenzen, Splits, Labels, Pooling-Regel und Probe gleich; nur das notwendige modellnative Resampling ist erlaubt und wird gespeichert.
