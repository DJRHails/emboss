"""Internal: the `emboss` console script (`emboss id`). Best-effort convenience tooling.

`emboss id <target>` imports the module containing an `@cached` function and
prints its cache identity (`"name:body_hash"`) — the token `also_accept`
consumes. `--rev <git-rev>` reads a *file* target out of git instead of the
working tree, recovering the pre-edit identity after the fact.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import subprocess
import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

from emboss._cached import cache_id


def _import_module_from_file(path: Path, module_name: str) -> types.ModuleType:
    """Import a module from a file path via `importlib.util.spec_from_file_location`."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # A broken, half-initialized module must not shadow later imports.
        del sys.modules[module_name]
        raise
    return module


def _import_file_target(path: Path, module_name: str, sibling_dir: Path) -> types.ModuleType:
    """Import `path` under a namespaced module name with `sibling_dir` importable.

    The sibling directory is prepended to `sys.path` only for the duration of
    the import (so the target's own imports resolve against its directory) and
    removed afterwards — `main()` runs in-process under tests and must not leak
    path entries. The `_emboss_*` namespace keeps a file target like `json.py`
    from clobbering a real module in `sys.modules`.
    """
    sys.path.insert(0, str(sibling_dir))
    try:
        return _import_module_from_file(path, module_name=module_name)
    finally:
        sys.path.remove(str(sibling_dir))


def _git_file_at_rev(rev: str, path: Path) -> bytes:
    """Return the bytes of `path` as of git revision `rev` (via `git show`).

    Runs git from the file's directory with a `./`-relative path spec, so both
    absolute and cwd-relative paths resolve regardless of where in the repo
    the file lives.
    """
    result = subprocess.run(
        ["git", "-C", str(path.parent or Path(".")), "show", f"{rev}:./{path.name}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git show {rev}:{path} failed: {stderr}")
    return result.stdout


def _resolve_function(target: str, rev: str | None) -> Callable[..., Any]:
    """Import `target` (`pkg.mod:func` or `path/to/mod.py:func`) and return the function.

    With `rev`, the target must be a file path; its content is taken from git
    at that revision (written to a temp file and imported from there). The
    original file's directory is prepended to `sys.path` for file targets so
    sibling imports resolve against the working tree (best-effort).
    """
    module_part, sep, func_name = target.rpartition(":")
    if not sep or not module_part or not func_name:
        raise ValueError(
            f"target {target!r} is not of the form 'pkg.mod:func' or 'path/to/mod.py:func'"
        )
    looks_like_path = module_part.endswith(".py") or "/" in module_part
    if rev is not None:
        if not looks_like_path:
            raise ValueError(
                f"--rev requires a file-path target ('path/to/mod.py:func'), got {target!r}"
            )
        path = Path(module_part)
        source = _git_file_at_rev(rev, path)
        with tempfile.NamedTemporaryFile(
            "wb", suffix=".py", prefix=f"emboss_rev_{path.stem}_", delete=False
        ) as tmp:
            tmp.write(source)
            tmp_path = Path(tmp.name)
        try:
            module = _import_file_target(
                tmp_path,
                module_name=f"_emboss_rev_{path.stem}",
                sibling_dir=path.resolve().parent,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    elif looks_like_path:
        path = Path(module_part)
        module = _import_file_target(
            path,
            module_name=f"_emboss_file_{path.stem}",
            sibling_dir=path.resolve().parent,
        )
    else:
        module = importlib.import_module(module_part)
    try:
        return getattr(module, func_name)
    except AttributeError:
        raise AttributeError(
            f"module {module.__name__!r} has no attribute {func_name!r}"
        ) from None


def _cmd_id(target: str, rev: str | None) -> int:
    """Print the cache identity of the targeted function; 1 + stderr on any failure."""
    try:
        func = _resolve_function(target, rev)
        print(cache_id(func))
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report any failure as exit 1
        print(f"emboss id: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `emboss` console script.

    `emboss id [--rev GIT_REV] TARGET` — print `cache_id()` of an `@cached`
    function. TARGET is `pkg.mod:func` (dotted import) or `path/to/mod.py:func`
    (file import); `--rev` reads a file target from git at that revision.
    """
    parser = argparse.ArgumentParser(prog="emboss", description="emboss cache tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    id_parser = subparsers.add_parser(
        "id",
        help="print the cache identity ('name:body_hash') of an @cached function",
        description=(
            "Import the target module and print cache_id() of the named @cached "
            "function — the token that also_accept consumes for cache migration."
        ),
    )
    id_parser.add_argument("target", help="'pkg.mod:func' or 'path/to/mod.py:func'")
    id_parser.add_argument(
        "--rev",
        default=None,
        help="git revision to read a file target from (e.g. HEAD~1) — "
        "recovers the pre-edit identity",
    )
    args = parser.parse_args(argv)
    return _cmd_id(args.target, args.rev)


if __name__ == "__main__":
    raise SystemExit(main())
