# 2. Dein erster echter Pilot: Watkins + MERT

Das Ziel ist zunächst **nicht**, sofort BEANS vollständig auszuwerten. Das Ziel ist, eine vertrauenswürdige Pipeline für ein Dataset und einen Encoder zu etablieren.

## Schritt 1 – Rohdaten ablegen

Lege die Watkins-Dateien unter `data/raw/watkins/` ab. Du kannst die Originalunterordner beibehalten.

## Schritt 2 – Manifest erstellen

Kopiere die Vorlage:

```bash
cp data/manifests/watkins.example.csv data/manifests/watkins.csv
```

Unter Windows kannst du die Datei im Explorer kopieren und umbenennen.

Dann ersetzt du jede Beispielzeile durch echte Clips. Jede Zeile muss einen echten Dateipfad und den offiziellen Split enthalten.

Beispiel:

```csv
clip_id,audio_path,split,task,labels,recording_id,start_s,end_s,source_id,notes
watkins_0001,data/raw/watkins/orca/a_01.wav,train,multiclass,orca,rec_a,,,original_a01,
watkins_0002,data/raw/watkins/humpback/b_07.wav,val,multiclass,humpback,rec_b,,,original_b07,
watkins_0003,data/raw/watkins/blue/c_03.wav,test,multiclass,blue_whale,rec_c,,,original_c03,
```

`recording_id` ist besonders wichtig, wenn mehrere Clips aus derselben Aufnahme stammen. Er hilft dir später zu prüfen, ob es Leakage zwischen Splits gibt.

## Schritt 3 – Manifest prüfen

```bash
biobench manifest validate data/manifests/watkins.csv
```

Erst wenn das ohne Fehler durchläuft, machst du weiter.

## Schritt 4 – Config prüfen

```bash
biobench check configs/experiments/watkins_mert_linear.yaml
```

Die Ausgabe zeigt, welches Manifest und welche Encoder-Spezifikation tatsächlich benutzt wird.

## Schritt 5 – Embeddings extrahieren

```bash
biobench extract configs/experiments/watkins_mert_linear.yaml
```

Beim ersten Lauf lädt MERT die Gewichte und erzeugt einen Cache. Beim zweiten Lauf mit demselben Manifest und Encoder wird der Cache wiederverwendet.

## Schritt 6 – Linear Probe auswerten

```bash
biobench evaluate configs/experiments/watkins_mert_linear.yaml
```

Oder beides zusammen:

```bash
biobench run configs/experiments/watkins_mert_linear.yaml
```

## Schritt 7 – Ergebnis lesen

Öffne:

```text
artifacts/runs/watkins_mert_linear/metrics.json
```

Bei Multiclass sind besonders wichtig:

- `validation_accuracy`: zur Wahl von `C`
- `test_accuracy`: primäres Testergebnis
- `test_macro_f1`: wichtig bei unbalancierten Klassen
- `selected_C`: darf nur aus Validation stammen

## Erst danach

1. MERT auf Watkins stabil zum Laufen bringen.
2. Identische Watkins-Config mit `encoder: muq_large` kopieren.
3. Danach Perch 2.0 integrieren.
4. Erst wenn alle drei auf Watkins laufen: CBI und DCASE ergänzen.
