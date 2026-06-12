"""Tests for the `emboss id` CLI (`emboss._cli.main`)."""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap

from emboss._cli import main

_MODULE_TEMPLATE = """\
import diskcache
from emboss import cached

cache = diskcache.Cache("{cache_dir}")


@cached(cache)
def f(x: int) -> int:
    return x * {factor}


def plain(x: int) -> int:
    return x
"""

_ID_PATTERN = re.compile(
    r"""(?x)        # verbose
    f               # the function name
    :               # identity separator
    [0-9a-f]{32}    # md5 body hash
    """
)


def _write_module(tmp_path, stem: str, factor: int = 2):
    path = tmp_path / f"{stem}.py"
    path.write_text(
        _MODULE_TEMPLATE.format(cache_dir=tmp_path / "cache", factor=factor)
    )
    return path


def test_id_file_target(tmp_path, capsys):
    path = _write_module(tmp_path, "climod_file")
    assert main(["id", f"{path}:f"]) == 0
    out = capsys.readouterr().out.strip()
    assert _ID_PATTERN.fullmatch(out)


def test_id_dotted_module_target(tmp_path, capsys, monkeypatch):
    _write_module(tmp_path, "climod_dotted")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert main(["id", "climod_dotted:f"]) == 0
    out = capsys.readouterr().out.strip()
    assert _ID_PATTERN.fullmatch(out)


def test_id_unwrapped_function_errors(tmp_path, capsys):
    path = _write_module(tmp_path, "climod_plain")
    assert main(["id", f"{path}:plain"]) == 1
    err = capsys.readouterr().err
    assert "not an @cached-wrapped function" in err


def test_id_malformed_target_errors(capsys):
    assert main(["id", "no-colon-anywhere"]) == 1
    err = capsys.readouterr().err
    assert "pkg.mod:func" in err


def test_id_missing_attribute_errors(tmp_path, capsys):
    path = _write_module(tmp_path, "climod_missing")
    assert main(["id", f"{path}:nope"]) == 1
    err = capsys.readouterr().err
    assert "no attribute 'nope'" in err


def test_id_rev_recovers_pre_edit_identity(tmp_path, capsys):
    """`--rev` reads the file out of git, so the OLD id is recoverable after an edit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    # Identity config scoped to the throwaway fixture repo (hermetic on CI).
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test Fixture"], cwd=repo, check=True)

    path = repo / "climod_rev.py"
    path.write_text(
        textwrap.dedent(_MODULE_TEMPLATE).format(cache_dir=tmp_path / "cache", factor=2)
    )
    subprocess.run(["git", "add", "climod_rev.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "v1"], cwd=repo, check=True)

    # Capture the identity BEFORE the edit — `--rev` must recover exactly this.
    assert main(["id", f"{path}:f"]) == 0
    pre_edit_id = capsys.readouterr().out.strip()

    # Edit the working tree: the committed identity and the current one diverge.
    path.write_text(
        textwrap.dedent(_MODULE_TEMPLATE).format(cache_dir=tmp_path / "cache", factor=3)
    )

    assert main(["id", f"{path}:f"]) == 0
    new_id = capsys.readouterr().out.strip()
    assert main(["id", "--rev", "HEAD", f"{path}:f"]) == 0
    old_id = capsys.readouterr().out.strip()

    assert _ID_PATTERN.fullmatch(old_id)
    assert _ID_PATTERN.fullmatch(new_id)
    assert old_id == pre_edit_id  # --rev recovered the exact pre-edit identity
    assert old_id != new_id  # the body edit re-keyed


def test_id_rev_requires_file_target(capsys):
    assert main(["id", "--rev", "HEAD", "some.module:f"]) == 1
    err = capsys.readouterr().err
    assert "--rev requires a file-path target" in err


def test_id_rev_unknown_revision_errors(tmp_path, capsys):
    path = _write_module(tmp_path, "climod_badrev")
    assert main(["id", "--rev", "not-a-rev", f"{path}:f"]) == 1
    err = capsys.readouterr().err
    assert "git show" in err


def test_id_module_import_failure_errors(tmp_path, capsys):
    """README: 'if the module's imports fail, it reports the error and exits 1'."""
    path = tmp_path / "climod_importfail.py"
    path.write_text("import does_not_exist_xyz\n")
    path_before = list(sys.path)
    assert main(["id", f"{path}:f"]) == 1
    err = capsys.readouterr().err
    assert "ModuleNotFoundError" in err
    assert "does_not_exist_xyz" in err
    # The half-initialized module must not linger and shadow later imports,
    # and the sibling-dir sys.path entry must not leak past the failure.
    assert "_emboss_file_climod_importfail" not in sys.modules
    assert sys.path == path_before


def test_id_file_named_like_stdlib_module(tmp_path, capsys):
    """A file target named `json.py` must not clobber the real `json` module."""
    import json as real_json

    path = _write_module(tmp_path, "json")
    path_before = list(sys.path)
    assert main(["id", f"{path}:f"]) == 0
    out = capsys.readouterr().out.strip()
    assert _ID_PATTERN.fullmatch(out)
    assert sys.modules["json"] is real_json
    assert sys.path == path_before  # in-process main() must not leak path entries
