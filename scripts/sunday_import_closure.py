"""Does a freeze list cover every project file the campaign imports? (Sunday freeze-day check, read-only.)

Walks the import statements of the campaign entry points through the project (orch.*, scripts/*.py) and compares the
files it reaches with one or more freeze lists. The walk is static: it cannot see files opened by name at run time
(configs, Tcl, recipe registry); those are named by hand in the lists.

Usage: sunday_import_closure.py --list notes/freeze_candidate_sunday.sha256 --list <other list> [--entry scripts/x.py ...]
A list is a sha256sum-style text file or a JSON file with a "files" object. Exit 0 when nothing reached is unlisted.
"""
from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable
from pathlib import Path

ENTRY_POINTS = (
    "scripts/campaign_run.py", "scripts/campaign_harness.py", "scripts/test_campaign.py",
    "scripts/campaign_analysis.py", "scripts/campaign_manifest.py",
)


def module_files(root: Path, name: str) -> list[Path]:
    parts = name.split(".")
    found: list[Path] = []
    for base in (root, root / "scripts"):          # scripts import each other by bare name
        for candidate in (base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts, "__init__.py")):
            if candidate.is_file():
                found.append(candidate)
    return found


def imported_names(root: Path, path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = path.parent
                for _ in range(node.level - 1):
                    package = package.parent
                prefix = ".".join(package.relative_to(root).parts)
                base = ".".join(part for part in (prefix, node.module) if part)
            else:
                base = node.module or ""
            if base:
                names.add(base)
                names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def closure(root: Path, entries: Iterable[str]) -> list[str]:
    """Project files reached by import from the entry points, entry points included, as sorted relative paths."""

    seen: set[Path] = set()
    queue = [root / entry for entry in entries]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        for name in imported_names(root, current):
            for module in module_files(root, name):
                queue.append(module)
                for parent in module.relative_to(root).parents:     # importing a module imports its packages
                    init = root / parent / "__init__.py"
                    if init.is_file():
                        queue.append(init)
    return sorted(path.relative_to(root).as_posix() for path in seen)


def listed(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return set(json.loads(text)["files"])
    paths: set[str] = set()
    for line in text.splitlines():
        if line.strip() and not line.startswith("#") and len(line.split(None, 1)) == 2:
            paths.add(line.split(None, 1)[1].strip().lstrip("*"))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--entry", action="append", help="entry point relative to the root (repeatable)")
    parser.add_argument("--list", action="append", required=True, dest="lists", help="a freeze list (repeatable)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    entries = args.entry or list(ENTRY_POINTS)
    reached = closure(root, entries)
    covered: set[str] = set()
    for item in args.lists:
        covered |= listed(Path(item))
    missing = [path for path in reached if path not in covered]
    print(f"{len(reached)} project files reached by import from {len(entries)} entry points; {len(covered)} files listed")
    for path in missing:
        print(f"NOT LISTED: {path}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
