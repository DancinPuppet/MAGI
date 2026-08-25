from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["dataset"], row["mechanism"], int(row["seed"])): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--tolerance", type=float, default=2e-7)
    args = parser.parse_args()
    expected = rows(ROOT / "results" / "expected_seed127.csv")
    actual = rows(args.result)
    failures = []
    for key, target in expected.items():
        if key not in actual:
            continue
        for expected_name, actual_name in (("f1", "test_f1"), ("auc", "test_auc")):
            difference = abs(float(target[expected_name]) - float(actual[key][actual_name]))
            if difference > args.tolerance:
                failures.append((key, actual_name, difference))
    if failures:
        raise SystemExit("Result mismatch: " + repr(failures))
    print(f"Matched {len(set(expected) & set(actual))} scenarios")


if __name__ == "__main__":
    main()
