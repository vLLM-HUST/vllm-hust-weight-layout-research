from __future__ import annotations

import argparse
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path


def load_lock() -> dict:
    path = files("vllm_hust_operator_optimizations").joinpath(
        "compatibility/verified-line.json"
    )
    return json.loads(path.read_text())


def _public_version(value: str) -> str:
    return value.split("+")[0]


def check_packages(lock: dict) -> list[str]:
    errors = []
    package_names = {
        "torch": "torch",
        "torch_npu": "torch-npu",
        "triton_ascend": "triton-ascend",
    }
    for key, distribution in package_names.items():
        expected = lock["packages"][key]
        try:
            actual = _public_version(version(distribution))
        except PackageNotFoundError:
            errors.append(f"missing package: {distribution}")
            continue
        if actual != expected:
            errors.append(f"{distribution}: expected {expected}, got {actual}")
    return errors


def check_revision(path: str, expected: str, name: str) -> list[str]:
    checkout = Path(path)
    if not checkout.is_dir():
        return [f"{name}: checkout not found: {checkout}"]
    try:
        actual = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return [f"{name}: cannot inspect checkout: {error}"]
    errors = []
    if actual != expected:
        errors.append(f"{name}: expected {expected}, got {actual}")
    if dirty:
        errors.append(f"{name}: checkout is dirty")
    return errors


def verify(vllm_src: str, vllm_ascend_src: str, triton_ascend_src: str | None) -> list[str]:
    lock = load_lock()
    errors = check_packages(lock)
    errors += check_revision(vllm_src, lock["repositories"]["vllm"], "vllm")
    errors += check_revision(
        vllm_ascend_src,
        lock["repositories"]["vllm_ascend_hust"],
        "vllm-ascend-hust",
    )
    if triton_ascend_src:
        errors += check_revision(
            triton_ascend_src,
            lock["repositories"]["triton_ascend"],
            "triton-ascend",
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the verified vLLM-HUST release line")
    parser.add_argument("--vllm-src", required=True)
    parser.add_argument("--vllm-ascend-src", required=True)
    parser.add_argument("--triton-ascend-src")
    args = parser.parse_args()
    errors = verify(args.vllm_src, args.vllm_ascend_src, args.triton_ascend_src)
    if errors:
        raise SystemExit("incompatible environment:\n" + "\n".join(errors))
    print("compatible: ascend-910b2-cann91-torch210")


if __name__ == "__main__":
    main()
