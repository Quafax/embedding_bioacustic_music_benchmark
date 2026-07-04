from __future__ import annotations

import math
import shutil
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from biobench.cli import main
from biobench.paths import project_root


def _write_tone(path: Path, frequency: float, seconds: float = 0.4, sample_rate: int = 8_000) -> None:
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    samples = (0.2 * np.sin(2 * math.pi * frequency * time) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def main_demo() -> None:
    root = project_root()
    demo = root / "artifacts" / "demo"
    if demo.exists():
        shutil.rmtree(demo)
    audio_dir = demo / "audio"
    audio_dir.mkdir(parents=True)
    rows = []
    for split, label, frequency in [
        ("train", "low", 220.0), ("train", "high", 880.0),
        ("val", "low", 240.0), ("val", "high", 860.0),
        ("test", "low", 260.0), ("test", "high", 840.0),
    ]:
        clip_id = f"{split}_{label}"
        path = audio_dir / f"{clip_id}.wav"
        _write_tone(path, frequency)
        rows.append({
            "clip_id": clip_id,
            "audio_path": str(path),
            "split": split,
            "task": "multiclass",
            "labels": label,
            "recording_id": clip_id,
        })
    manifest = demo / "demo.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = demo / "demo.yaml"
    config.write_text(
        "\n".join([
            "run:", "  name: demo_dummy", "  seed: 42", "", "data:", f"  manifest: {manifest}",
            "", "encoder: dummy", "", "protocol:", "  kind: linear_probe",
            "  c_values: [0.1, 1.0]", "  threshold_values: [0.5]", "  max_iter: 1000", "",
        ]),
        encoding="utf-8",
    )
    print("Running deterministic demo...")
    main(["run", str(config)])
    print("Demo complete. See artifacts/runs/demo_dummy/metrics.json")


if __name__ == "__main__":
    main_demo()
