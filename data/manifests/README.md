# Manifest: die einzige Schnittstelle zwischen Dataset und Benchmark

Der Benchmark kennt keine Klassenordner, keine datasetspezifische Dateistruktur und keine speziellen Loader. Er liest nur dieses einheitliche CSV-Format.

Eine Zeile ist **ein auswertbarer Clip**. Ein Clip kann eine komplette Audiodatei oder ein Ausschnitt daraus sein.

## Pflichtspalten

| Spalte | Bedeutung | Beispiel |
|---|---|---|
| `clip_id` | stabiler eindeutiger Name | `watkins_000012` |
| `audio_path` | Pfad relativ zum Projektordner oder absolut | `data/raw/watkins/a.wav` |
| `split` | offizieller Split: `train`, `val`, `test` | `train` |
| `task` | überall im Manifest gleich: `multiclass` oder `multilabel` | `multiclass` |
| `labels` | eine Klasse oder mehrere Klassen mit `;` | `orca` oder `bird;mammal` |

## Optionale Spalten

| Spalte | Zweck |
|---|---|
| `recording_id` | Gruppierung/Leakage-Check und spätere Analyse |
| `start_s`, `end_s` | Ausschnitt aus einer längeren Datei in Sekunden |
| `source_id` | originale Dataset-ID |
| `notes` | Freitext, wird nicht für das Training verwendet |

## Regel für die Splits

Nutze die offiziellen Splits unverändert. Wenn ein Dataset keine offizielle Validation besitzt, erzeuge nicht heimlich einen neuen Testsplit. Stattdessen dokumentierst du die festgelegte Validation-Aufteilung vorab, zum Beispiel gruppiert nach Aufnahme.

## Vor dem ersten Lauf

```bash
biobench manifest validate data/manifests/watkins.csv
```

Die Validierung prüft Pflichtspalten, eindeutige IDs, erlaubte Splits, einheitlichen Tasktyp, vorhandene Labels und fehlende Dateien.
