#!/usr/bin/env python3
"""Build and validate the deterministic evidence assets used by the presentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "detection_comparison" / "experiment_manifest.json"


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover() -> None:
    manifest = json.loads(MANIFEST.read_text())
    local_paths = {
        "dream_multi_fusion.py": ROOT / "dream_multi_fusion.py",
        "land_segmentation.py": ROOT / "land_segmentation.py",
        "radar_bounded_full.jsonl": (
            ROOT / "experiments" / "detection_comparison" / "radar_bounded_full.jsonl"
        ),
    }
    for name, path in local_paths.items():
        expected = manifest["artifacts"][name]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{path} failed SHA-256 validation")
    print("Recovered experiment artifacts match the recorded server hashes.")


def stills() -> None:
    run("uv", "run", "python", "presentation/stills/generate.py")


def videos() -> None:
    run("uv", "run", "python", "presentation/media/base/build_presentation_media.py")
    run("uv", "run", "python", "presentation/media/vision/build_vision_media.py")
    run("uv", "run", "python", "presentation/media/gate/generate_gate_media.py")
    run("uv", "run", "python", "presentation/media/radar_bounded/build_media.py")


def errors() -> None:
    run("uv", "run", "python", "presentation/errors/mine_errors.py")


def plots() -> None:
    run("uv", "run", "python", "presentation/metrics/pipeline.py")
    run("bash", "presentation/metrics/build_figures.sh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("recover", "stills", "videos", "errors", "plots", "all"),
    )
    args = parser.parse_args()

    actions = {
        "recover": recover,
        "stills": stills,
        "videos": videos,
        "errors": errors,
        "plots": plots,
    }
    if args.command == "all":
        for action in (recover, stills, videos, errors, plots):
            action()
    else:
        actions[args.command]()


if __name__ == "__main__":
    main()
