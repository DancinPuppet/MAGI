from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magi")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--mode", choices=("evaluate", "train"), default="evaluate")
    parser.add_argument("--datasets", nargs="+", default=["dolphins", "netscience", "cora_ml", "power_grid"])
    parser.add_argument("--mechanisms", nargs="+", default=["IC", "LT", "USE"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[127])
    parser.add_argument("--expected-samples", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--rag-topk", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--source-latent-dim", type=int, default=16)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--source-epochs", type=int, default=150)
    parser.add_argument("--source-learning-rate", type=float, default=1e-3)
    parser.add_argument("--source-kl-weight", type=float, default=0.02)
    parser.add_argument("--source-kl-warmup", type=int, default=30)
    parser.add_argument("--source-free-bits", type=float, default=0.02)
    parser.add_argument("--proposal-corruption", type=float, default=0.15)
    parser.add_argument("--source-early-stopping-checks", type=int, default=999)
    parser.add_argument("--replay-draws", type=int, default=8)
    parser.add_argument("--forward-weight", type=float, default=0.25)
    parser.add_argument("--weight-temperature", type=float, default=1.0)
    parser.add_argument("--mean-loss-weight", type=float, default=0.5)
    parser.add_argument("--residual-penalty", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--early-stopping-checks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-hidden", type=int, default=64)
    parser.add_argument("--memory-embedding-dim", type=int, default=32)
    parser.add_argument("--memory-latent-dim", type=int, default=8)
    parser.add_argument("--generator-epochs", type=int, default=35)
    parser.add_argument("--generator-learning-rate", type=float, default=1e-3)
    parser.add_argument("--vae-kl-weight", type=float, default=0.01)
    parser.add_argument("--vae-kl-warmup", type=int, default=15)
    parser.add_argument("--patch-batch-size", type=int, default=4096)
    parser.add_argument("--minimum-generated-fraction", type=float, default=0.25)
    parser.add_argument("--memory-topk", type=int, default=32)
    parser.add_argument("--kernel-temperature", type=float, default=0.1)
    parser.add_argument("--retrieval-chunk-size", type=int, default=256)
    parser.add_argument("--memory-weight-grid", nargs="+", type=float, default=[0.5])
    parser.add_argument("--generation-mix-grid", nargs="+", type=float, default=[0.75])
    parser.add_argument("--memory-reconstruction-tolerance", type=float, default=2e-5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
