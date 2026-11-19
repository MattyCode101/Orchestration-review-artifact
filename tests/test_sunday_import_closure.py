"""The import-closure check: it finds the project files the campaign imports and reports any that a freeze list omits."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/sunday_import_closure.py"


def _tool():
    spec = importlib.util.spec_from_file_location("sunday_import_closure", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project(root: Path) -> Path:
    (root / "pkg").mkdir()
    (root / "scripts").mkdir()
    files = {
        "pkg/__init__.py": "", "pkg/b.py": "", "pkg/c.py": "", "pkg/unused.py": "", "scripts/helper.py": "",
        "pkg/a.py": "import json\nfrom pkg import b\nimport helper\nfrom . import c\n",
        "scripts/entry.py": "from pkg.a import x\n",
    }
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_the_closure_follows_absolute_relative_and_script_imports_and_leaves_the_rest_out(tmp_path) -> None:
    root = _project(tmp_path)
    assert _tool().closure(root, ["scripts/entry.py"]) == [
        "pkg/__init__.py", "pkg/a.py", "pkg/b.py", "pkg/c.py", "scripts/entry.py", "scripts/helper.py",
    ]


def test_a_file_the_lists_omit_is_reported_and_fails_the_check(tmp_path, capsys) -> None:
    root = _project(tmp_path)
    listing = tmp_path / "list.sha256"
    listing.write_text("# comment\n" + "".join(f"{'0' * 64}  {name}\n" for name in (
        "pkg/__init__.py", "pkg/a.py", "pkg/b.py", "scripts/entry.py", "scripts/helper.py")), encoding="utf-8")
    code = _tool().main(["--root", str(root), "--entry", "scripts/entry.py", "--list", str(listing)])
    assert code == 1 and "NOT LISTED: pkg/c.py" in capsys.readouterr().out


def test_a_sha256_list_and_a_json_list_together_can_cover_the_closure(tmp_path, capsys) -> None:
    root = _project(tmp_path)
    shas = tmp_path / "list.sha256"
    shas.write_text(f"{'0' * 64}  pkg/__init__.py\n{'0' * 64} *pkg/a.py\n", encoding="utf-8")
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps({"files": {"pkg/b.py": "x", "pkg/c.py": "x", "scripts/entry.py": "x", "scripts/helper.py": "x"}}), encoding="utf-8")
    code = _tool().main(["--root", str(root), "--entry", "scripts/entry.py", "--list", str(shas), "--list", str(draft)])
    assert code == 0 and "NOT LISTED" not in capsys.readouterr().out
