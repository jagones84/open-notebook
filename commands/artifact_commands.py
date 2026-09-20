"""``generate_artifact`` command: notebook -> rendered report/deck (#203).

Drives the ``open_notebook.artifacts`` pipeline and persists both halves of
the result:

1. a ``Note`` inside the notebook, so the readable document lives with the
   sources it was written from;
2. a ``GeneratedArtifact`` row plus the rendered files on disk under
   ``ARTIFACTS_FOLDER``, so the API can list it and hand out a download.

The row is created BEFORE the job is submitted (see ``api/artifact_service``)
so the API can return an ``artifact_id`` immediately; this command fills in
``output_path``, ``note_id`` and ``sections`` when it finishes.

Progress is logged per section by the pipeline (``composition.build_document``)
because a long artifact run is otherwise a black box in the worker log.
"""

import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from loguru import logger
from pydantic import Field
from surreal_commands import CommandInput, CommandOutput, command

from open_notebook.artifacts.composition import (
    CompositionConfig,
    RetrievalConfig,
    build_document,
    make_notebook_search,
    short_id,
)
from open_notebook.artifacts.diagrams import make_renderer, render_diagrams
from open_notebook.artifacts.outline import complete_text, normalize_kind
from open_notebook.artifacts.paths import to_relative_artifact_path
from open_notebook.artifacts.render import (
    SUPPORTED_FORMATS,
    build_base_name,
    render_artifact,
)
from open_notebook.config import ARTIFACTS_FOLDER
from open_notebook.domain.artifact import GeneratedArtifact
from open_notebook.domain.notebook import Note, Notebook

#: Formats rendered when the request asks for none.
DEFAULT_FORMATS = ["md", "docx"]


class ArtifactGenerationInput(CommandInput):
    """Job payload for one artifact generation."""

    artifact_id: str
    notebook_id: str
    kind: str = "report"
    formats: list[str] = Field(default_factory=list)
    language: str = "en"
    title: str = ""
    instructions: Optional[str] = None
    sections: int = 10
    model_id: Optional[str] = None


class ArtifactGenerationOutput(CommandOutput):
    """Job result: where the document and its files ended up."""

    success: bool
    artifact_id: str
    output_path: Optional[str] = None
    note_id: Optional[str] = None
    sections: int = 0
    processing_time: float
    error_message: Optional[str] = None


def _requested_formats(formats: Optional[list[str]]) -> list[str]:
    """Keep only supported formats, defaulting when nothing usable is left.

    Args:
        formats: Formats requested by the user.

    Returns:
        A non-empty list of supported formats.
    """
    chosen = [fmt for fmt in (formats or []) if fmt in SUPPORTED_FORMATS]
    return chosen or list(DEFAULT_FORMATS)


def _used_source_ids(report: dict[str, Any]) -> list[str]:
    """Collect the distinct source ids a run actually wrote from.

    Args:
        report: Run report produced by ``build_document``.

    Returns:
        Source record ids in first-seen order.
    """
    ids: list[str] = []
    for section in report.get("sections") or []:
        for chunk in section.get("chunks") or []:
            source_id = str(chunk.get("source_id") or "")
            if source_id and source_id not in ids:
                ids.append(source_id)
    return ids


def _sources_section(report: dict[str, Any], titles: dict[str, str]) -> str:
    """Build a trailing ``## Sources`` section from the cited sources.

    Args:
        report: Run report produced by ``build_document``.
        titles: Mapping of source record id -> source title.

    Returns:
        The section Markdown, or an empty string when nothing was cited.
    """
    ids = _used_source_ids(report)
    if not ids:
        return ""
    lines = ["## Sources", ""]
    for source_id in ids:
        label = titles.get(source_id) or short_id(source_id)
        lines.append(f"- {label} [source:{short_id(source_id)}]")
    return "\n".join(lines)


async def _create_notebook_note(notebook_id: str, title: str, markdown: str) -> str:
    """Persist the readable document as a Note tied to the notebook.

    Args:
        notebook_id: Notebook the note belongs to.
        title: Note title.
        markdown: Document body.

    Returns:
        The created note record id.
    """
    note = Note(title=title, content=markdown, note_type="ai")
    await note.save()
    await note.add_to_notebook(notebook_id)
    logger.info(f"Artifact written to note {note.id}")
    return str(note.id)


@command("generate_artifact", app="open_notebook", retry={"max_attempts": 1})
async def generate_artifact_command(
    input_data: ArtifactGenerationInput,
) -> ArtifactGenerationOutput:
    """Generate a report or deck from a notebook and store every artifact.

    Args:
        input_data: Job payload; ``artifact_id`` points at the row created by
            the API before submission.

    Returns:
        The job result with the primary file path and the note id.

    Raises:
        ValueError: If the artifact or the notebook cannot be loaded.
        RuntimeError: If the pipeline produced no content or no file.
    """
    start_time = time.time()

    artifact = await GeneratedArtifact.get(input_data.artifact_id)

    try:
        notebook = await Notebook.get(input_data.notebook_id)
        kind: Literal["report", "deck"] = (
            "deck" if normalize_kind(input_data.kind) == "deck" else "report"
        )
        formats = _requested_formats(input_data.formats)
        language = input_data.language or "en"
        title = (input_data.title or "").strip() or artifact.title or notebook.name
        max_sections = max(1, int(input_data.sections or 10))

        sources = await notebook.get_sources()
        source_titles = [source.title or "" for source in sources]
        allowed_ids = [str(source.id) for source in sources if source.id]
        titles_by_id = {
            str(source.id): (source.title or "") for source in sources if source.id
        }
        logger.info(
            f"Artifact {artifact.id}: kind={kind} formats={formats} "
            f"sources={len(allowed_ids)}"
        )

        out_dir = Path(ARTIFACTS_FOLDER) / str(uuid.uuid4())
        out_dir.mkdir(parents=True, exist_ok=True)

        retrieval = RetrievalConfig(outline_max_sections=max_sections)
        composition = CompositionConfig(
            language=language,
            instructions=input_data.instructions or "",
            allow_diagrams=kind == "deck",
            model_id=input_data.model_id,
        )

        markdown, report = await build_document(
            complete_text,
            make_notebook_search(input_data.notebook_id),
            source_titles,
            allowed_ids,
            title,
            kind,
            retrieval,
            composition,
        )
        # Check BEFORE the Sources section is appended: an empty body must
        # fail the job, not be rescued into a document that is only a
        # bibliography with nothing to cite.
        if not markdown.strip():
            raise RuntimeError("Artifact generation produced no content")

        renderer, engine = make_renderer()
        if renderer is not None:
            logger.info(f"Mermaid engine: {engine}")
        markdown, diagram_report = render_diagrams(markdown, out_dir, renderer)
        if diagram_report.get("failures"):
            logger.warning(f"Diagram failures: {diagram_report['failures']}")

        sources_md = _sources_section(report, titles_by_id)
        if sources_md:
            markdown = f"{markdown.rstrip()}\n\n{sources_md}\n"

        base_name = build_base_name(title, kind)
        outputs = render_artifact(markdown, out_dir, base_name, formats)
        if not outputs:
            raise RuntimeError("Artifact rendering produced no output file")

        primary = outputs.get("md") or next(iter(outputs.values()))
        output_rel = to_relative_artifact_path(primary)

        note_id = await _create_notebook_note(input_data.notebook_id, title, markdown)

        artifact.kind = kind
        artifact.formats = list(outputs.keys())
        artifact.language = language
        artifact.title = title
        artifact.sections = int(report.get("sections_written") or 0)
        artifact.output_path = output_rel
        artifact.note_id = note_id
        await artifact.save()

        processing_time = time.time() - start_time
        logger.info(
            f"Artifact {artifact.id} ready in {processing_time:.2f}s "
            f"({output_rel}, note {note_id})"
        )
        return ArtifactGenerationOutput(
            success=True,
            artifact_id=str(artifact.id),
            output_path=output_rel,
            note_id=note_id,
            sections=artifact.sections,
            processing_time=processing_time,
        )

    except ValueError:
        raise
    except Exception as exc:
        logger.error(f"Artifact generation failed: {exc}")
        logger.exception(exc)
        raise RuntimeError(str(exc)) from exc
