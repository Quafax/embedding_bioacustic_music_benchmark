# 6. Methodische Checkliste vor Ergebnissen

## Vor der Extraktion

- [ ] offizielle Splits im Manifest, nicht neu gemischt
- [ ] jede `clip_id` genau einmal
- [ ] keine `recording_id` in mehreren Splits, sofern du die Splitzuordnung kontrollierst
- [ ] kein Denoising nur für ein Modell
- [ ] gleiche Clipgrenzen für alle Encoder
- [ ] verwendeter Encoder-ID und Checkpoint festgelegt

## Vor der Evaluation

- [ ] Cache stammt vom richtigen Manifest und Encoder-Fingerprint
- [ ] `C` nur auf Validation ausgewählt
- [ ] bei Multilabel: Schwelle nur auf Validation ausgewählt
- [ ] Test nur einmal als Schlussauswertung gelesen

## Für Few-shot

- [ ] Stichproben nur aus `train`
- [ ] gleiche Shots und Seeds für alle Encoder
- [ ] mehrere Wiederholungen pro Shot
- [ ] nicht mögliche Shot-Stufen transparent als `skipped` berichten

## Zum Paper

- [ ] Tabelle pro Dataset statt nur ein globaler Score
- [ ] Klassifikation und Mehrlabel-Detektion getrennt
- [ ] Ergebnisdateien enthalten Modell-ID, Checkpoint, Sample Rate, Cache-Fingerprint und Config
- [ ] Fehlerschwerpunkte nach Taxon, Frequenzbereich, Signaltyp und Sample-Rate-Limit diskutieren
