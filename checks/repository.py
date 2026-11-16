"""Git comparisons and command-line reporting for repository checks."""

import argparse
import subprocess
import sys
import tarfile
import tomllib
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import batched
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Self, override


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    @classmethod
    def empty(cls, subject: str) -> Self:
        return cls(subject, 1, "empty-subject", "no files or measurements were examined")

    @override
    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.message}"


@dataclass(frozen=True)
class SourceRoots:
    paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.paths or any(
            not (path := PurePosixPath(root)).parts or path.is_absolute() or ".." in path.parts
            for root in self.paths
        ):
            raise ValueError(
                "tool.quality.source-roots must be a nonempty list of "
                "repository-relative directories; "
                'write [tool.quality] with source-roots = ["library"] in pyproject.toml'
            )

    @classmethod
    def from_project(cls, text: str) -> Self:
        tools = tomllib.loads(text).get("tool", {})
        quality = tools.get("quality") if isinstance(tools, dict) else None
        roots = quality.get("source-roots") if isinstance(quality, dict) else None
        if not isinstance(roots, list) or not all(isinstance(root, str) for root in roots):
            return cls(())
        return cls(tuple(roots))

    def contains(self, path: str) -> bool:
        candidate = PurePosixPath(path)
        return any(candidate.is_relative_to(root) for root in self.paths)


def command(arguments: Sequence[str], *, cwd: Path, accepts_findings: bool = False) -> str:
    """Run a fixed argument list, accepting a tool's findings exit status.

    Returns:
        Standard output of the completed process.

    Raises:
        RuntimeError: If the tool cannot complete its measurement.
    """
    result = subprocess.run(arguments, cwd=cwd, capture_output=True, text=True, check=False)
    allowed = (0, 1) if accepts_findings else (0,)
    if result.returncode not in allowed:
        raise RuntimeError(f"{arguments[0]} failed: {result.stderr or result.stdout}")
    return result.stdout


class Repository:
    def __init__(self, base: str | None = None) -> None:
        self.root = Path(command(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()).strip())
        self.base = command(
            ["git", "rev-parse", "--verify", "--end-of-options", base]
            if bool(base)
            else ["git", "merge-base", "HEAD", "origin/main"],
            cwd=self.root,
        ).strip()
        if not self.base:
            raise RuntimeError(
                "Cannot resolve comparison base; fetch origin/main or pass --base REF"
            )
        self.source_roots = SourceRoots.from_project((self.root / "pyproject.toml").read_text())

    def paths(self) -> list[str]:
        output = command(["git", "ls-files", "-z"], cwd=self.root)
        return [path for path in output.split("\0") if path]

    def ignored_paths(self) -> list[str]:
        output = command(["git", "ls-files", "-ci", "--exclude-standard", "-z"], cwd=self.root)
        return [path for path in output.split("\0") if path]

    def base_paths(self) -> set[str]:
        output = command(["git", "ls-tree", "-r", "--name-only", "-z", self.base], cwd=self.root)
        return set(output.strip("\0").split("\0"))

    def base_text(self, path: str) -> str:
        return command(["git", "show", f"{self.base}:{path}"], cwd=self.root)

    def files(self) -> dict[str, bytes]:
        return {
            path: (self.root / path).read_bytes()
            for path in self.paths()
            if (self.root / path).is_file()
        }

    def texts(self, suffixes: tuple[str, ...]) -> dict[str, str]:
        return {
            path: data.decode("utf-8")
            for path, data in self.files().items()
            if path.endswith(suffixes)
        }

    def utf8_texts(self) -> dict[str, str]:
        return self.decode_utf8(self.files(), self.paths())

    @staticmethod
    def decode_utf8(files: dict[str, bytes], paths: list[str]) -> dict[str, str]:
        texts = {}
        for path in paths:
            data = files.get(path, b"")
            try:
                texts[path] = data.decode("utf-8")
            except UnicodeDecodeError:
                texts[path] = ""
        return texts

    def changed_paths(self) -> set[str]:
        output = command(["git", "diff", "--name-only", "-z", self.base, "--"], cwd=self.root)
        return set(output.strip("\0").split("\0")) - {""}

    def renamed_paths(self, suffixes: tuple[str, ...]) -> dict[str, str]:
        """Find the files Git reads as renamed since the base.

        A renamed file keeps the allowances of its origin and no others: its comments and
        functions are compared with the origin's, exactly as if it had been edited in place.

        Returns:
            Each renamed path mapped to the base path it came from.
        """
        arguments = ["git", "diff", "--name-status", "-z", "--find-renames", "--diff-filter=R"]
        output = command([*arguments, self.base, "--"], cwd=self.root)
        return self.renames(output, suffixes)

    @staticmethod
    def renames(name_status: str, suffixes: tuple[str, ...]) -> dict[str, str]:
        """Read rename records, keeping those whose origin was a checked file too.

        Returns:
            Each renamed path mapped to its origin, when both carry one of the suffixes.
        """
        fields = [field for field in name_status.split("\0") if field]
        return {
            current: origin
            for _, origin, current in batched(fields, 3, strict=True)
            if origin.endswith(suffixes) and current.endswith(suffixes)
        }

    @contextmanager
    def base_tree(self) -> Generator[Path]:
        with TemporaryDirectory(prefix="quality-base-") as directory:
            root = Path(directory).resolve()
            archive = root / "tree.tar"
            command(["git", "archive", self.base, "-o", str(archive)], cwd=self.root)
            with tarfile.open(archive) as tree:
                tree.extractall(root, filter="data")
            archive.unlink()
            yield root


def comparison_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--base", help="Compare with REF instead of the merge base with origin/main"
    )
    return parser


def report(findings: Sequence[Finding]) -> int:
    for finding in findings:
        sys.stdout.write(f"{finding}\n")
    return int(bool(findings))
