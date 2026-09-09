import asyncio
import json

import pytest

from atlas.llm.models import ToolCall, ToolFunction
from atlas.obsidian import (
    ObsidianError,
    ObsidianNoteExistsError,
    ObsidianNoteNotFoundError,
    ObsidianVault,
)
from atlas.tools import (
    ReadObsidianNoteTool,
    SearchObsidianNotesTool,
    ToolRegistry,
    WriteObsidianNoteTool,
)


def test_obsidian_vault_searches_and_reads_only_markdown_inside_its_root(tmp_path) -> None:
    vault = tmp_path / "vault"
    projects = vault / "Projects"
    projects.mkdir(parents=True)
    (projects / "Garden.md").write_text("# Garden plan\nThe greenhouse needs watering.")
    (vault / "scratch.txt").write_text("greenhouse outside markdown")
    outside = tmp_path / "outside.md"
    outside.write_text("This must not be visible.")

    reader = ObsidianVault(str(vault))
    results = reader.search("greenhouse")

    assert [result.model_dump() for result in results] == [
        {
            "path": "Projects/Garden.md",
            "title": "Garden plan",
            "excerpt": "# Garden plan The greenhouse needs watering.",
        }
    ]
    assert reader.read("Projects/Garden.md").content == "# Garden plan\nThe greenhouse needs watering."
    with pytest.raises(ObsidianNoteNotFoundError):
        reader.read("../outside.md")


def test_unconfigured_obsidian_vault_never_falls_back_to_the_current_directory() -> None:
    with pytest.raises(ObsidianError, match="vault is unavailable"):
        ObsidianVault("").search("atlas")


def test_obsidian_vault_creates_notes_atomically_and_requires_explicit_overwrite(tmp_path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    reader = ObsidianVault(str(vault))

    created = reader.write("ATLAS ideas.md", "# First idea\nKeep this local.")

    assert created.model_dump() == {"path": "ATLAS ideas.md", "action": "created"}
    assert reader.read("ATLAS ideas.md").content == "# First idea\nKeep this local."
    with pytest.raises(ObsidianNoteExistsError, match="already exists"):
        reader.write("ATLAS ideas.md", "Replacement")
    updated = reader.write("ATLAS ideas.md", "Replacement", overwrite=True)
    assert updated.action == "updated"
    assert reader.read("ATLAS ideas.md").content == "Replacement"
    with pytest.raises(ObsidianError, match="only Markdown notes"):
        reader.write("../outside.md", "Nope")


def test_obsidian_tools_return_search_and_note_content(tmp_path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Ideas.md").write_text("# Ideas\nBuild an ATLAS event feed.")
    reader = ObsidianVault(str(vault))
    registry = ToolRegistry(
        [SearchObsidianNotesTool(reader), ReadObsidianNoteTool(reader), WriteObsidianNoteTool(reader)],
        execution_timeout_seconds=1,
    )

    async def exercise():
        search = await registry.execute(
            ToolCall(function=ToolFunction(name="search_obsidian_notes", arguments={"query": "event"}))
        )
        read = await registry.execute(
            ToolCall(function=ToolFunction(name="read_obsidian_note", arguments={"path": "Ideas.md"}))
        )
        write = await registry.execute(
            ToolCall(
                function=ToolFunction(
                    name="write_obsidian_note",
                    arguments={"path": "Saved.md", "content": "# Saved\nBy ATLAS."},
                )
            )
        )
        return search, read, write

    search, read, write = asyncio.run(exercise())

    assert search.ok is True
    assert json.loads(search.content)[0]["path"] == "Ideas.md"
    assert read.ok is True
    assert json.loads(read.content)["content"] == "# Ideas\nBuild an ATLAS event feed."
    assert write.ok is True
    assert json.loads(write.content) == {"path": "Saved.md", "action": "created"}
