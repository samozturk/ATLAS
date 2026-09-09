"""Read-only, path-safe access to a local Obsidian Markdown vault."""

import re
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field


class ObsidianError(Exception):
    """A safe failure while accessing the locally configured vault."""


class ObsidianNoteNotFoundError(ObsidianError):
    """The requested Markdown note does not exist inside the vault."""


class ObsidianNoteExistsError(ObsidianError):
    """A write would replace an existing note without explicit authorization."""


class ObsidianSearchResult(BaseModel):
    """A compact result the model can use to select a note to read."""

    path: str
    title: str
    excerpt: str


class ObsidianNote(BaseModel):
    """The text of one Markdown note resolved inside the vault."""

    path: str
    content: str


class ObsidianWriteResult(BaseModel):
    """The durable result of creating or explicitly replacing one Markdown note."""

    path: str
    action: str


class ObsidianVault:
    """Search, read, and explicitly write Markdown without escaping the vault root."""

    _max_note_bytes = 512 * 1024

    def __init__(self, root_path: str) -> None:
        # Path("") resolves to the current working directory, which would make
        # an unconfigured vault silently expose application files.
        self._configured_root = (
            Path(root_path).expanduser()
            if root_path.strip()
            else Path("/__atlas_unconfigured_obsidian_vault__")
        )

    def search(self, query: str, *, limit: int = 5) -> list[ObsidianSearchResult]:
        """Return path, title, and a bounded excerpt for matching Markdown notes."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ObsidianError("A note search needs a non-empty query.")
        matcher = re.compile(re.escape(normalized_query), re.IGNORECASE)
        matches: list[ObsidianSearchResult] = []
        root = self._root()
        for candidate in sorted(root.rglob("*.md")):
            note_path = self._safe_markdown_path(candidate)
            if note_path is None:
                continue
            relative_path = note_path.relative_to(root).as_posix()
            content = self._read_text(note_path)
            found = matcher.search(relative_path) or matcher.search(content)
            if found is None:
                continue
            matches.append(
                ObsidianSearchResult(
                    path=relative_path,
                    title=_title_from(relative_path, content),
                    excerpt=_excerpt(content if matcher.search(content) else relative_path, found),
                )
            )
            if len(matches) >= limit:
                break
        return matches

    def read(self, relative_path: str) -> ObsidianNote:
        """Read one requested relative Markdown path after resolving it safely."""
        root = self._root()
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ObsidianNoteNotFoundError("ATLAS could not find that Obsidian note.")
        resolved = self._safe_markdown_path(root / candidate)
        if resolved is None or not resolved.is_file():
            raise ObsidianNoteNotFoundError("ATLAS could not find that Obsidian note.")
        return ObsidianNote(
            path=resolved.relative_to(root).as_posix(),
            content=self._read_text(resolved),
        )

    def write(
        self, relative_path: str, content: str, *, overwrite: bool = False
    ) -> ObsidianWriteResult:
        """Create a note atomically, replacing an existing note only when requested."""
        if len(content.encode("utf-8")) > self._max_note_bytes:
            raise ObsidianError("That Obsidian note is too large for ATLAS to write safely.")
        root = self._root()
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix.lower() != ".md":
            raise ObsidianError("ATLAS can write only Markdown notes inside the configured Obsidian vault.")
        try:
            parent = (root / candidate.parent).resolve(strict=True)
        except OSError as error:
            raise ObsidianError("ATLAS could not find the target Obsidian folder.") from error
        if not parent.is_relative_to(root) or not parent.is_dir():
            raise ObsidianError("ATLAS could not find the target Obsidian folder.")
        target = parent / candidate.name
        if target.exists():
            if self._safe_markdown_path(target) is None:
                raise ObsidianError("ATLAS can write only Markdown notes inside the configured Obsidian vault.")
            if not overwrite:
                raise ObsidianNoteExistsError(
                    "That Obsidian note already exists. Ask ATLAS explicitly to replace it."
                )
            action = "updated"
        else:
            action = "created"
        self._atomic_write(target, content)
        return ObsidianWriteResult(path=(candidate.parent / target.name).as_posix(), action=action)

    def _root(self) -> Path:
        try:
            root = self._configured_root.resolve(strict=True)
        except OSError as error:
            raise ObsidianError("The configured Obsidian vault is unavailable.") from error
        if not root.is_dir():
            raise ObsidianError("The configured Obsidian vault is unavailable.")
        return root

    def _safe_markdown_path(self, path: Path) -> Path | None:
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return None
        root = self._root()
        if resolved.suffix.lower() != ".md" or not resolved.is_relative_to(root):
            return None
        return resolved

    def _read_text(self, path: Path) -> str:
        try:
            if path.stat().st_size > self._max_note_bytes:
                raise ObsidianError("That Obsidian note is too large for ATLAS to read safely.")
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise ObsidianError("ATLAS could not read that Obsidian note.") from error

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        temporary_path: str | None = None
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                dir=target.parent, prefix=".atlas-", suffix=".tmp"
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, target)
        except OSError as error:
            raise ObsidianError("ATLAS could not write that Obsidian note.") from error
        finally:
            if temporary_path is not None:
                try:
                    Path(temporary_path).unlink(missing_ok=True)
                except OSError:
                    pass


def _title_from(relative_path: str, content: str) -> str:
    for line in content.splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                return title
    return Path(relative_path).stem


def _excerpt(content: str, match: re.Match[str], *, maximum: int = 240) -> str:
    start = max(0, match.start() - 80)
    end = min(len(content), match.end() + 160)
    excerpt = " ".join(content[start:end].split())
    if start:
        excerpt = f"…{excerpt}"
    if end < len(content):
        excerpt = f"{excerpt}…"
    return excerpt[:maximum]
