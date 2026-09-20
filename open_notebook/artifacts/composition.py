"""Section retrieval, grounded composition and citation verification.

One section at a time: run the planner's queries through vector search, keep
the best in-notebook excerpts, write the section from THOSE excerpts only, then
verify every citation against the excerpts actually passed to the model.

Citations are resolved by NUMBER (``[source:2]``) because small models mangle
long record ids (``source:jyqsntmzovlau8ccbk35``); the number maps back to the
excerpt deterministically and the real source id is restored afterwards.
Anything that does not resolve is REMOVED from the text and reported - never
silently trusted.

The search callable is injected and awaited, so the whole layer is unit-tested
without a database or a network.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

from loguru import logger

from open_notebook.artifacts.outline import (
    DIAGRAM_RULES,
    SECTION_PROMPT,
    SECTION_RULES_DECK,
    SECTION_RULES_REPORT,
    Complete,
    Section,
    instructions_block,
    kind_label,
    language_name,
    normalize_kind,
    plan_outline,
)
from open_notebook.domain.notebook import vector_search

#: One search call: query, hit limit, minimum similarity -> normalised hits.
Search = Callable[[str, int, float], Awaitable[list[dict[str, Any]]]]

#: ``[source:N]`` citations, by number or by full record id.
CITATION_RE = re.compile(r"\[source:([^\]\s]+)\]")

#: Text appended to an excerpt cut to ``max_chars_per_chunk``.
TRUNCATION_MARK = " [truncated]"

_NO_CHUNKS_PLACEHOLDER = "(no excerpt retrieved)"


@dataclass
class Chunk:
    """One retrieved excerpt, ready to be pasted into a section prompt.

    Attributes:
        source_id: Source record id the excerpt comes from.
        title: Source title, used as a readable citation label.
        text: Excerpt text.
        similarity: Vector similarity reported by the search.
    """

    source_id: str
    title: str
    text: str
    similarity: float = 0.0


@dataclass
class RetrievalConfig:
    """Tuning knobs for the retrieval layer.

    Defaults mirror the standalone ``config.yaml`` ``retrieval:`` block; the
    command passes overrides from the artifact request.
    """

    enabled: bool = True
    outline_max_sections: int = 10
    chunks_per_query: int = 6
    max_chunks_per_section: int = 8
    max_chars_per_chunk: int = 1200
    max_chars_per_section: int = 6000
    min_score: float = 0.2
    verify_citations: bool = True


@dataclass
class CompositionConfig:
    """Tuning knobs for the writing stage.

    Attributes:
        language: Language the document body is written in.
        instructions: Optional user brief passed to every prompt.
        allow_diagrams: Ask the model for ```mermaid``` fences per section.
        model_id: Optional explicit Model record id.
        max_section_attempts: Calls per section; the retry halves the material.
    """

    language: str = "en"
    instructions: str = ""
    allow_diagrams: bool = False
    model_id: Optional[str] = None
    max_section_attempts: int = 2


@dataclass
class SectionOutcome:
    """What one section produced, for the run report.

    Attributes:
        section: The planned section.
        chunks: Excerpts the section was written from.
        text: Verified section Markdown (empty when the model gave nothing).
        unknown_citations: Citation tokens dropped during verification.
    """

    section: Section
    chunks: list[Chunk] = field(default_factory=list)
    text: str = ""
    unknown_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome for the run report.

        Returns:
            A JSON-serializable summary of the section.
        """
        return {
            "title": self.section.title,
            "thesis": self.section.thesis,
            "queries": list(self.section.queries),
            "chunks": [
                {
                    "source_id": chunk.source_id,
                    "chars": len(chunk.text),
                    "similarity": chunk.similarity,
                }
                for chunk in self.chunks
            ],
        }


def config_from_dict(raw: Optional[dict[str, Any]]) -> RetrievalConfig:
    """Build a RetrievalConfig from a plain mapping.

    Args:
        raw: Mapping with any subset of the RetrievalConfig fields.

    Returns:
        The retrieval configuration, missing keys taken from the defaults.
    """
    raw = raw or {}
    defaults = RetrievalConfig()
    return RetrievalConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        outline_max_sections=int(
            raw.get("outline_max_sections", defaults.outline_max_sections)
        ),
        chunks_per_query=int(raw.get("chunks_per_query", defaults.chunks_per_query)),
        max_chunks_per_section=int(
            raw.get("max_chunks_per_section", defaults.max_chunks_per_section)
        ),
        max_chars_per_chunk=int(
            raw.get("max_chars_per_chunk", defaults.max_chars_per_chunk)
        ),
        max_chars_per_section=int(
            raw.get("max_chars_per_section", defaults.max_chars_per_section)
        ),
        min_score=float(raw.get("min_score", defaults.min_score)),
        verify_citations=bool(raw.get("verify_citations", defaults.verify_citations)),
    )


def normalize_hit(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one ``vector_search()`` row to the internal hit shape.

    ``fn::vector_search`` returns ``{id, parent_id, title, similarity,
    matches}`` where ``matches`` is the flattened list of matched chunk texts;
    the retrieval layer works on ``{source_id, title, similarity, text}``.

    Args:
        row: Raw row from ``vector_search()``.

    Returns:
        The normalised hit.
    """
    matches = row.get("matches") or []
    if isinstance(matches, str):
        matches = [matches]
    text = "\n".join(str(match) for match in matches if isinstance(match, str))
    return {
        "source_id": str(row.get("parent_id") or row.get("source_id") or ""),
        "title": str(row.get("title") or ""),
        "similarity": row.get("similarity"),
        "text": text,
    }


def make_notebook_search(notebook_id: str) -> Search:
    """Build the section search callable for one notebook.

    Uses the domain's ``vector_search()`` scoped to the notebook, so results
    are limited server-side instead of being filtered client-side out of a
    global result set.

    Args:
        notebook_id: Notebook record id to scope the search to.

    Returns:
        An async callable ``(query, limit, min_score) -> hits``.
    """

    async def search(query: str, limit: int, min_score: float) -> list[dict[str, Any]]:
        rows = await vector_search(
            query,
            limit,
            source=True,
            note=False,
            minimum_score=min_score,
            notebook_ids=[notebook_id],
        )
        return [normalize_hit(row) for row in rows or []]

    return search


def short_id(source_id: str) -> str:
    """Drop the ``source:`` prefix so citations stay readable.

    Args:
        source_id: Full record id.

    Returns:
        The id without its table prefix.
    """
    sid = (source_id or "").strip()
    return sid.split(":", 1)[1] if sid.startswith("source:") else sid


def select_chunks(
    hits: Sequence[dict[str, Any]],
    allowed_source_ids: Sequence[str],
    cfg: RetrievalConfig,
) -> list[Chunk]:
    """Filter hits to the notebook, dedupe, truncate and cap the budget.

    Args:
        hits: Normalised hits from one or more queries.
        allowed_source_ids: Source ids belonging to the notebook.
        cfg: Retrieval configuration.

    Returns:
        The excerpts kept for one section, best first.
    """
    allowed = {sid for sid in allowed_source_ids if sid}
    selected: list[Chunk] = []
    seen: set[tuple[str, str]] = set()
    total_chars = 0

    for hit in hits:
        source_id = str(hit.get("source_id") or "")
        if allowed and source_id not in allowed:
            continue
        text = str(hit.get("text") or "").strip()
        if not text:
            continue
        if len(text) > cfg.max_chars_per_chunk:
            text = text[: cfg.max_chars_per_chunk] + TRUNCATION_MARK
        key = (source_id, text[:100])
        if key in seen:
            continue
        seen.add(key)
        if len(selected) >= cfg.max_chunks_per_section:
            break
        if total_chars + len(text) > cfg.max_chars_per_section and selected:
            break
        try:
            similarity = float(hit.get("similarity") or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        selected.append(
            Chunk(
                source_id=source_id,
                title=str(hit.get("title") or ""),
                text=text,
                similarity=similarity,
            )
        )
        total_chars += len(text)
    return selected


async def retrieve_for_section(
    search: Search,
    section: Section,
    allowed_source_ids: Sequence[str],
    cfg: RetrievalConfig,
) -> list[Chunk]:
    """Run the section's queries and keep the best in-notebook excerpts.

    Args:
        search: Injected search callable.
        section: Planned section.
        allowed_source_ids: Source ids belonging to the notebook.
        cfg: Retrieval configuration.

    Returns:
        The excerpts for the section; a failed query is logged and skipped.
    """
    hits: list[dict[str, Any]] = []
    for query in section.queries:
        try:
            hits.extend(await search(query, cfg.chunks_per_query, cfg.min_score))
        except Exception as exc:
            logger.warning(f"Search failed for {query!r}: {exc}")
    chunks = select_chunks(hits, allowed_source_ids, cfg)
    logger.info(f"Section {section.title!r}: {len(chunks)} excerpt(s)")
    return chunks


def chunks_block(chunks: Sequence[Chunk]) -> str:
    """Render excerpts with a SHORT citation label the model can copy.

    Args:
        chunks: Excerpts for one section.

    Returns:
        The excerpt block for the prompt, numbered from 1.
    """
    if not chunks:
        return _NO_CHUNKS_PLACEHOLDER
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.title or short_id(chunk.source_id)
        parts.append(f"[source:{index}] (source: {title})\n{chunk.text}")
    return "\n\n".join(parts)


def build_section_prompt(
    section: Section,
    chunks: Sequence[Chunk],
    kind: str,
    language: str,
    instructions: str = "",
    allow_diagrams: bool = False,
) -> str:
    """Assemble the prompt for one section.

    Args:
        section: Planned section.
        chunks: Excerpts retrieved for the section.
        kind: Artifact kind.
        language: Language the section is written in (code or name).
        instructions: Optional user brief.
        allow_diagrams: Ask for a ```mermaid``` fence when the section fits.

    Returns:
        The complete section prompt.
    """
    rules = (
        SECTION_RULES_DECK if normalize_kind(kind) == "deck" else SECTION_RULES_REPORT
    )
    return SECTION_PROMPT.format(
        kind_label=kind_label(kind),
        title=section.title,
        thesis=section.thesis or "(none)",
        instructions_block=instructions_block(instructions),
        chunks_block=chunks_block(chunks),
        language=language_name(language),
        rules=rules,
        diagrams=DIAGRAM_RULES if allow_diagrams else "",
    )


async def generate_section(
    complete: Complete,
    section: Section,
    chunks: Sequence[Chunk],
    kind: str,
    cfg: CompositionConfig,
) -> str:
    """Write one section; a failure degrades instead of killing the run.

    Args:
        complete: Injected model callable.
        section: Planned section.
        chunks: Excerpts the section is written from.
        kind: Artifact kind.
        cfg: Composition configuration.

    Returns:
        The section Markdown, or an empty string when the call failed.
    """
    prompt = build_section_prompt(
        section,
        chunks,
        kind,
        cfg.language,
        cfg.instructions,
        cfg.allow_diagrams,
    )
    try:
        return await complete(prompt)
    except Exception as exc:
        logger.warning(
            f"Section {section.title!r} call failed ({type(exc).__name__}: {exc})"
        )
        return ""


async def compose_section(
    complete: Complete,
    section: Section,
    chunks: list[Chunk],
    kind: str,
    cfg: CompositionConfig,
) -> tuple[str, list[Chunk]]:
    """Write one section, retrying with half the material when it comes back empty.

    A reasoning model can burn its whole token budget thinking or stall past
    the read timeout (measured: one section at 192 s while its siblings took
    ~5 s), so an empty reply is retried with half the excerpts - the retry is
    genuinely lighter. Returns the excerpts that were actually used.

    Args:
        complete: Injected model callable.
        section: Planned section.
        chunks: Excerpts retrieved for the section.
        kind: Artifact kind.
        cfg: Composition configuration.

    Returns:
        ``(section_markdown, chunks_used)``; the Markdown is empty when every
        attempt failed.
    """
    text = await generate_section(complete, section, chunks, kind, cfg)
    attempts = max(cfg.max_section_attempts, 1)
    if text.strip() or attempts < 2 or len(chunks) <= 2:
        return text, chunks

    halves = chunks[: max(len(chunks) // 2, 1)]
    logger.warning(
        f"Section {section.title!r} produced nothing with {len(chunks)} excerpt(s)"
        f" - retrying with {len(halves)}"
    )
    text = await generate_section(complete, section, halves, kind, cfg)
    return (text, halves) if text.strip() else (text, chunks)


def verify_citations(markdown: str, chunks: Sequence[Chunk]) -> tuple[str, list[str]]:
    """Keep only citations that point at an excerpt actually provided.

    Args:
        markdown: Section Markdown as returned by the model.
        chunks: Excerpts passed to the model, in prompt order.

    Returns:
        ``(cleaned_markdown, unknown_tokens)`` where ``unknown_tokens`` holds
        the citation tokens that were removed.
    """
    allowed = {short_id(chunk.source_id) for chunk in chunks}
    unknown: list[str] = []

    def replace(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        if token.isdigit():
            index = int(token)
            if 1 <= index <= len(chunks) and chunks[index - 1].source_id:
                return f"[source:{short_id(chunks[index - 1].source_id)}]"
            unknown.append(token)
            return ""
        short = short_id(token)
        if short in allowed:
            return f"[source:{short}]"
        unknown.append(short)
        return ""

    cleaned = CITATION_RE.sub(replace, markdown)
    if unknown:
        logger.warning(
            f"Removed {len(unknown)} citation(s) pointing outside the excerpts:"
            f" {sorted(set(unknown))}"
        )
    return cleaned, sorted(set(unknown))


def assemble_document(title: str, sections_markdown: Sequence[str]) -> str:
    """Concatenate the per-section Markdown into one document.

    Args:
        title: Document title.
        sections_markdown: Section bodies, in order.

    Returns:
        The full Markdown document.
    """
    body = "\n\n".join(s.strip() for s in sections_markdown if s and s.strip())
    header = f"# {title}\n" if title else ""
    return f"{header}\n{body}\n".lstrip("\n")


async def build_document(
    complete: Complete,
    search: Search,
    source_titles: Sequence[str],
    allowed_source_ids: Sequence[str],
    title: str,
    kind: str,
    cfg: RetrievalConfig,
    composition: Optional[CompositionConfig] = None,
    sections: Optional[Sequence[Section]] = None,
) -> tuple[str, dict[str, Any]]:
    """Outline -> retrieve -> write -> verify -> assemble.

    Args:
        complete: Injected model callable.
        search: Injected search callable.
        source_titles: Titles of the sources in the notebook.
        allowed_source_ids: Source ids belonging to the notebook.
        title: Document title.
        kind: Artifact kind.
        cfg: Retrieval configuration.
        composition: Writing configuration; defaults are used when omitted.
        sections: Pre-planned sections; the outline is planned when omitted.

    Returns:
        ``(markdown_document, report)`` where the report lists every section,
        its excerpts and the citations that were dropped.
    """
    comp = composition or CompositionConfig()
    planned = (
        list(sections)
        if sections
        else await plan_outline(
            source_titles,
            title,
            kind,
            comp.instructions,
            cfg.outline_max_sections,
            model_id=comp.model_id,
            language=comp.language,
        )
    )

    outcomes: list[SectionOutcome] = []
    total = len(planned)
    for position, section in enumerate(planned, start=1):
        logger.info(f"Section {position}/{total}: {section.title}")
        chunks = await retrieve_for_section(search, section, allowed_source_ids, cfg)
        text, used = await compose_section(complete, section, chunks, kind, comp)
        outcome = SectionOutcome(section=section, chunks=used)
        if not text.strip():
            logger.warning(f"Section {section.title!r} produced no content - skipped")
        else:
            if cfg.verify_citations:
                text, unknown = verify_citations(text, used)
                outcome.unknown_citations = unknown
            outcome.text = text
        outcomes.append(outcome)

    written = [outcome.text for outcome in outcomes if outcome.text.strip()]
    document = assemble_document(title, written)
    unknown_all = sorted(
        {token for outcome in outcomes for token in outcome.unknown_citations}
    )
    report: dict[str, Any] = {
        "mode": "retrieval",
        "sections": [outcome.to_dict() for outcome in outcomes],
        "sections_written": len(written),
        "unknown_citations": unknown_all,
        "chars": len(document),
    }
    return document, report
