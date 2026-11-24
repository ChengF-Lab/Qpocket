from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def find_pymol(env_var: str = "PYMOL_BIN") -> str:

    override = os.environ.get(env_var)
    if override and Path(override).is_file():
        return override
    which = shutil.which("pymol")
    if which:
        return which
    raise FileNotFoundError(
        "PyMOL executable not found. Set "
        f"{env_var}=/path/to/pymol or add 'pymol' to PATH."
    )


def run_pml(
    pml_text: str,
    timeout_s: float = 300.0,
    pymol_bin: Optional[str] = None,
) -> None:
    """Write ``pml_text`` to a temp file and execute it under ``pymol -cq``."""
    binary = pymol_bin or find_pymol()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pml", delete=False) as f:
        f.write(pml_text)
        tmp = f.name
    try:
        proc = subprocess.run(
            [binary, "-cq", tmp],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "PyMOL exited non-zero").strip()[:2500]
            raise RuntimeError(msg)
    finally:
        Path(tmp).unlink(missing_ok=True)
