"""Extract the project version from pyproject.toml.

This is the single source of truth for the version: ``[project].version`` in
pyproject.toml. meson.build calls this script so the native extension and the
Python metadata never drift apart.

meson runs this with whatever ``python3`` it finds first (often the system
interpreter, e.g. 3.9), so it must stay compatible with older Pythons: the
``from __future__`` import keeps the ``X | None`` annotations from being
evaluated at definition time on 3.9.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - older build interpreters
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


def _find_pyproject() -> Path | None:
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir.parent / "pyproject.toml", Path.cwd() / "pyproject.toml"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _version_from_toml(text: str) -> str | None:
    if tomllib is not None:
        try:
            data = tomllib.loads(text)
        except Exception:
            data = {}
        version = data.get("project", {}).get("version")
        if isinstance(version, str):
            return version
    # Fallback: match version only inside the [project] table, so unrelated
    # version = "..." lines (e.g. dependency pins) are never picked up.
    project_section = re.search(
        r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL
    )
    scope = project_section.group(1) if project_section else text
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"', scope, re.MULTILINE)
    return match.group(1) if match else None


def main() -> None:
    pyproject = _find_pyproject()
    if pyproject is None:
        print("0.0.0")
        return
    version = _version_from_toml(pyproject.read_text(encoding="utf-8"))
    print(version if version else "0.0.0")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
