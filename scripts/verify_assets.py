from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ASSET_MANIFEST.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    failures = []
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            failures.append(relative)
        checked += 1
    if failures:
        raise SystemExit("Asset verification failed: " + ", ".join(failures))
    print(f"Verified {checked} files")


if __name__ == "__main__":
    main()
