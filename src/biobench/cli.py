from __future__ import annotations

import argparse
import json
from pathlib import Path

from biobench.cache import cache_location
from biobench.config import load_experiment
from biobench.encoders import available_encoders, encoder_spec
from biobench.evaluate import evaluate
from biobench.extract import extract_embeddings
from biobench.manifest import read_manifest


def _config(path: str):
    config = load_experiment(path)
    read_manifest(config.data.manifest, check_files=False)
    encoder_spec(config.encoder)
    return config


def command_check(args: argparse.Namespace) -> int:
    config = _config(args.config)
    manifest = read_manifest(config.data.manifest, check_files=args.check_files)
    spec = encoder_spec(config.encoder)
    print(json.dumps({
        "run": config.run.name,
        "manifest": str(config.data.manifest),
        "n_manifest_rows": len(manifest),
        "encoder": spec.as_dict(),
        "protocol": config.protocol.__dict__,
    }, indent=2, default=str))
    return 0


def command_manifest_validate(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest, check_files=True)
    print(f"Valid manifest: {args.manifest} ({len(manifest)} rows; task={manifest['task'].iloc[0]})")
    return 0


def command_extract(args: argparse.Namespace) -> int:
    config = _config(args.config)
    extract_embeddings(config, device=args.device, force=args.force)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config = _config(args.config)
    evaluate(config)
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = _config(args.config)
    extract_embeddings(config, device=args.device, force=args.force)
    evaluate(config)
    return 0


def command_fewshot(args: argparse.Namespace) -> int:
    config = _config(args.config)
    if config.protocol.kind != "fewshot_prototype":
        raise ValueError("Use a config with protocol.kind: fewshot_prototype for the fewshot command.")
    extract_embeddings(config, device=args.device, force=args.force)
    evaluate(config)
    return 0


def command_encoders(_: argparse.Namespace) -> int:
    for encoder_id in available_encoders():
        spec = encoder_spec(encoder_id)
        print(f"{encoder_id}: {spec.implementation}; sr={spec.sample_rate}; model={spec.model_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biobench", description="Frozen embedding benchmark for audio transfer.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("manifest", help="Manifest tools")
    validate_sub = validate.add_subparsers(dest="manifest_command", required=True)
    validate_manifest = validate_sub.add_parser("validate", help="Validate a canonical manifest CSV")
    validate_manifest.add_argument("manifest")
    validate_manifest.set_defaults(handler=command_manifest_validate)

    check = sub.add_parser("check", help="Validate config and print resolved settings")
    check.add_argument("config")
    check.add_argument("--check-files", action="store_true", help="Also fail if referenced audio files are missing")
    check.set_defaults(handler=command_check)

    for name, handler, help_text in [
        ("extract", command_extract, "Extract embeddings or reuse cache"),
        ("run", command_run, "Extract then evaluate"),
    ]:
        item = sub.add_parser(name, help=help_text)
        item.add_argument("config")
        item.add_argument("--device", default="auto", help="auto, cpu, cuda, or explicit torch device")
        item.add_argument("--force", action="store_true", help="Recompute cache even if it exists")
        item.set_defaults(handler=handler)

    evaluate_parser = sub.add_parser("evaluate", help="Evaluate an existing embedding cache")
    evaluate_parser.add_argument("config")
    evaluate_parser.set_defaults(handler=command_evaluate)

    fewshot = sub.add_parser("fewshot", help="Extract/reuse cache and run few-shot protocol")
    fewshot.add_argument("config")
    fewshot.add_argument("--device", default="auto")
    fewshot.add_argument("--force", action="store_true")
    fewshot.set_defaults(handler=command_fewshot)

    encoders = sub.add_parser("encoders", help="List registered encoders")
    encoders.set_defaults(handler=command_encoders)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
