"""Read-only FoldFlow environment audit."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any
import numpy as np


def _version(module: str, distribution: str) -> str | None:
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", None)
        if isinstance(version, str):
            return version
    except Exception:
        pass
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_foldflow_environment() -> dict[str, Any]:
    """Return machine-readable dependency and compatibility diagnostics."""
    configured_root = os.environ.get("OSPEDIT_FOLDFLOW_ROOT")
    root_path = Path(configured_root).expanduser() if configured_root else None
    root_exists = root_path is not None and root_path.is_dir()
    root_text = str(root_path.resolve()) if root_path is not None else None
    added_root = False
    if root_exists and root_text is not None and root_text not in sys.path:
        sys.path.insert(0, root_text)
        added_root = True
    try:
        packages = {
            name: {
                "installed": importlib.util.find_spec(module) is not None,
                "version": _version(module, distribution),
            }
            for name, module, distribution in (
                ("numpy", "numpy", "numpy"),
                ("torch", "torch", "torch"),
                ("foldflow", "foldflow", "foldflow"),
                ("openfold", "openfold", "openfold"),
                ("einops", "einops", "einops"),
                ("dm_tree", "tree", "dm-tree"),
            )
        }
    finally:
        if added_root and root_text is not None:
            sys.path.remove(root_text)
    missing = [name for name, info in packages.items() if not info["installed"]]
    py_ok = (3, 8) <= sys.version_info[:2] < (3, 11)
    torch_version = packages["torch"]["version"]
    torch_major = int(torch_version.split(".", 1)[0]) if isinstance(torch_version, str) else None
    numpy_trapz_available = hasattr(np, "trapz")
    compatibility_warnings = []
    if not py_ok:
        compatibility_warnings.append(
            "Python version is outside the official FoldFlow 0.2 environment range (3.8-3.10)"
        )
    if torch_major is None:
        compatibility_warnings.append("Torch is not installed; the official FoldFlow environment cannot be used")
    elif torch_major >= 2:
        compatibility_warnings.append(
            f"Torch {torch_version} is outside the official FoldFlow 0.2 environment range (<2.0)"
        )
    if not numpy_trapz_available:
        compatibility_warnings.append("numpy.trapz is unavailable; legacy geomstats imports may fail")
    if configured_root and not root_exists:
        compatibility_warnings.append(
            f"OSPEDIT_FOLDFLOW_ROOT does not exist or is not a directory: {configured_root}"
        )
    return {
        "python": platform.python_version(),
        "foldflow_root": root_text,
        "foldflow_root_exists": bool(root_exists),
        "python_supported_by_official_foldflow_env": py_ok,
        "packages": packages,
        "missing_packages": missing,
        "torch_supported_by_official_foldflow_env": torch_major is not None and torch_major < 2,
        "numpy_trapz_available": numpy_trapz_available,
        "compatibility_warnings": compatibility_warnings,
        "ready_for_foldflow_import": not missing and py_ok and (torch_major is not None and torch_major < 2) and numpy_trapz_available,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit optional FoldFlow runtime dependencies")
    parser.add_argument("--strict", action="store_true", help="exit nonzero unless the official environment is usable")
    parser.add_argument("--foldflow-root", help="upstream FoldFlow source checkout to include in the audit")
    args = parser.parse_args()
    if args.foldflow_root:
        os.environ["OSPEDIT_FOLDFLOW_ROOT"] = args.foldflow_root
    report = inspect_foldflow_environment()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["ready_for_foldflow_import"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
