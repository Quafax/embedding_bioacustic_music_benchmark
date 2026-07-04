from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from biobench.cache import cache_location, load_cache
from biobench.config import ExperimentConfig, FewShotConfig, LinearProbeConfig
from biobench.encoders import encoder_spec
from biobench.paths import artifact_dir
from biobench.protocols.fewshot import evaluate_fewshot_prototype
from biobench.protocols.labels import encode_targets
from biobench.protocols.linear import evaluate_linear_probe


def _run_dir(config: ExperimentConfig) -> Path:
    path = artifact_dir() / "runs" / config.run.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_common(config: ExperimentConfig, cache_dir: Path) -> Path:
    run_dir = _run_dir(config)
    shutil.copy2(config.source_path, run_dir / "config.yaml")
    (run_dir / "cache_reference.json").write_text(
        json.dumps({"cache_directory": str(cache_dir)}, indent=2), encoding="utf-8"
    )
    return run_dir


def evaluate(config: ExperimentConfig) -> Path:
    spec = encoder_spec(config.encoder)
    location = cache_location(config.data.manifest, spec)
    embeddings, metadata, provenance = load_cache(location)
    targets, classes, task = encode_targets(metadata)
    splits = metadata["split"].astype(str).to_numpy()
    run_dir = _write_common(config, location.directory)

    if isinstance(config.protocol, LinearProbeConfig):
        result = evaluate_linear_probe(embeddings, targets, splits, task, config.protocol)
        test_rows = metadata.loc[metadata["split"] == "test", ["clip_id", "labels", "split"]].copy()
        test_rows["prediction"] = [str(value) for value in result.test_predictions]
        if result.test_probabilities is not None:
            test_rows["probabilities"] = [json.dumps(row.tolist()) for row in result.test_probabilities]
        test_rows.to_csv(run_dir / "predictions.csv", index=False)
        payload = {
            "run": config.run.name,
            "encoder": spec.as_dict(),
            "cache_key": location.key,
            "task": task,
            "classes": classes,
            "selected_C": result.selected_c,
            "selected_threshold": result.selected_threshold,
            "validation": result.validation,
            "test": result.test,
            "cache_provenance": provenance,
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved evaluation: {run_dir}")
        return run_dir

    if isinstance(config.protocol, FewShotConfig):
        if task != "multiclass":
            raise ValueError("fewshot_prototype currently supports multiclass manifests only.")
        records = evaluate_fewshot_prototype(
            embeddings,
            targets,
            splits,
            config.protocol.shots,
            config.protocol.repeats,
            config.run.seed,
        )
        frame = pd.DataFrame(records)
        frame.to_csv(run_dir / "fewshot.csv", index=False)
        summary = frame.loc[frame["status"] == "ok"].groupby("shots")[["test_accuracy", "test_macro_f1"]].agg(["mean", "std"])
        payload = {
            "run": config.run.name,
            "encoder": spec.as_dict(),
            "cache_key": location.key,
            "task": task,
            "classes": classes,
            "summary": json.loads(summary.to_json()),
            "cache_provenance": provenance,
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved few-shot evaluation: {run_dir}")
        return run_dir

    raise TypeError(f"Unsupported protocol type: {type(config.protocol)!r}")
