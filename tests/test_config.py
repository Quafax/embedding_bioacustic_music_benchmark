from __future__ import annotations

from pathlib import Path

from biobench.config import load_experiment


def test_load_linear_config(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[build-system]\nrequires=[]\n")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "clip_id,audio_path,split,task,labels\na,a.wav,train,multiclass,x\nb,b.wav,val,multiclass,x\nc,c.wav,test,multiclass,x\n"
    )
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "run:\n  name: test\n  seed: 1\ndata:\n  manifest: manifest.csv\nencoder: dummy\nprotocol:\n  kind: linear_probe\n  c_values: [1.0]\n  threshold_values: [0.5]\n  max_iter: 100\n"
    )
    monkeypatch.chdir(tmp_path)
    loaded = load_experiment(config)
    assert loaded.run.name == "test"
    assert loaded.encoder == "dummy"
