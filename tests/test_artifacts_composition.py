"""Unit tests for section retrieval, grounded composition and citations.

The model call and the vector search are injected, so no test here touches the
network, the database or an embedding model.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from open_notebook.artifacts import composition, outline


def make_hit(source_id="source:aaa", title="Source A", text="body", similarity=0.9):
    """Build one normalised search hit."""
    return {
        "source_id": source_id,
        "title": title,
        "similarity": similarity,
        "text": text,
    }


class FakeSearch:
    """Search callable returning canned hits and recording the queries."""

    def __init__(self, hits):
        self.hits = list(hits)
        self.calls: list[tuple[str, int, float]] = []

    async def __call__(self, query, limit, min_score):
        self.calls.append((query, limit, min_score))
        return list(self.hits)


class FakeComplete:
    """Model callable answering canned replies in order, recording prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


def make_search(hits):
    """Build an injected search callable returning the given hits."""
    return FakeSearch(hits)


def make_complete(replies):
    """Build an injected model callable answering the given replies in order."""
    return FakeComplete(replies)


class TestSelectChunks:
    def test_hits_outside_the_notebook_are_dropped(self):
        chunks = composition.select_chunks(
            [make_hit(), make_hit(source_id="source:other")],
            ["source:aaa"],
            composition.RetrievalConfig(),
        )
        assert [c.source_id for c in chunks] == ["source:aaa"]

    def test_empty_allowed_list_keeps_every_hit(self):
        chunks = composition.select_chunks(
            [make_hit(), make_hit(source_id="source:other")],
            [],
            composition.RetrievalConfig(),
        )
        assert len(chunks) == 2

    def test_empty_text_is_dropped(self):
        chunks = composition.select_chunks(
            [make_hit(text="   "), make_hit(text="real")],
            ["source:aaa"],
            composition.RetrievalConfig(),
        )
        assert [c.text for c in chunks] == ["real"]

    def test_duplicates_are_dropped(self):
        chunks = composition.select_chunks(
            [make_hit(), make_hit()], ["source:aaa"], composition.RetrievalConfig()
        )
        assert len(chunks) == 1

    def test_long_excerpts_are_truncated(self):
        cfg = composition.RetrievalConfig(max_chars_per_chunk=5)
        chunks = composition.select_chunks([make_hit(text="abcdefghij")], [], cfg)
        assert chunks[0].text == "abcde" + composition.TRUNCATION_MARK

    def test_chunk_count_is_capped(self):
        cfg = composition.RetrievalConfig(max_chunks_per_section=2)
        hits = [make_hit(text=f"body {i}") for i in range(5)]
        assert len(composition.select_chunks(hits, [], cfg)) == 2

    def test_section_char_budget_is_respected(self):
        cfg = composition.RetrievalConfig(max_chars_per_section=20)
        hits = [make_hit(text="aaaaaa"), make_hit(text="bbbbbbbb")]
        assert len(composition.select_chunks(hits, [], cfg)) == 2

    def test_section_char_budget_stops_the_section(self):
        cfg = composition.RetrievalConfig(max_chars_per_section=10)
        hits = [make_hit(text="aaaaaaa"), make_hit(text="bbbbbbb")]
        assert len(composition.select_chunks(hits, [], cfg)) == 1

    def test_unparsable_similarity_becomes_zero(self):
        hit = make_hit()
        hit["similarity"] = "not-a-number"
        chunks = composition.select_chunks([hit], [], composition.RetrievalConfig())
        assert chunks[0].similarity == 0.0


class TestNormalizeHit:
    def test_vector_search_row_is_mapped_to_the_internal_shape(self):
        row = {
            "id": "source_embedding:1",
            "parent_id": "source:aaa",
            "title": "Source A",
            "similarity": 0.77,
            "matches": ["first", "second"],
        }
        hit = composition.normalize_hit(row)
        assert hit == {
            "source_id": "source:aaa",
            "title": "Source A",
            "similarity": 0.77,
            "text": "first\nsecond",
        }

    def test_missing_matches_become_empty_text(self):
        assert composition.normalize_hit({"parent_id": "source:aaa"})["text"] == ""


class TestMakeNotebookSearch:
    @pytest.mark.asyncio
    async def test_search_is_scoped_to_the_notebook(self, monkeypatch):
        rows = [
            {
                "parent_id": "source:aaa",
                "title": "Source A",
                "similarity": 0.8,
                "matches": ["excerpt"],
            }
        ]
        mock_search = AsyncMock(return_value=rows)
        monkeypatch.setattr(composition, "vector_search", mock_search)

        search = composition.make_notebook_search("notebook:nb1")
        hits = await search("alpha", 6, 0.2)

        await_args = mock_search.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["notebook_ids"] == ["notebook:nb1"]
        assert kwargs["source"] is True and kwargs["note"] is False
        assert kwargs["minimum_score"] == 0.2
        assert hits[0]["source_id"] == "source:aaa"
        assert hits[0]["text"] == "excerpt"


class TestRetrieveForSection:
    @pytest.mark.asyncio
    async def test_every_query_is_run_and_hits_are_filtered(self):
        section = outline.Section(title="One", queries=["alpha", "beta"])
        search = make_search([make_hit(), make_hit(source_id="source:other")])
        chunks = await composition.retrieve_for_section(
            search, section, ["source:aaa"], composition.RetrievalConfig()
        )
        assert [call[0] for call in search.calls] == ["alpha", "beta"]
        assert [c.source_id for c in chunks] == ["source:aaa"]

    @pytest.mark.asyncio
    async def test_a_failing_query_does_not_kill_the_section(self):
        async def flaky(query, limit, min_score):
            if query == "bad":
                raise RuntimeError("search down")
            return [make_hit()]

        section = outline.Section(title="One", queries=["bad", "good"])
        chunks = await composition.retrieve_for_section(
            flaky, section, ["source:aaa"], composition.RetrievalConfig()
        )
        assert len(chunks) == 1


class TestVerifyCitations:
    def test_numbers_are_resolved_to_real_source_ids(self):
        chunks = [
            composition.Chunk(source_id="source:aaa", title="A", text="a"),
            composition.Chunk(source_id="source:bbb", title="B", text="b"),
        ]
        cleaned, unknown = composition.verify_citations(
            "Fact [source:1] and other [source:2].", chunks
        )
        assert cleaned == "Fact [source:aaa] and other [source:bbb]."
        assert unknown == []

    def test_unresolvable_citations_are_dropped(self):
        chunks = [composition.Chunk(source_id="source:aaa", title="A", text="a")]
        cleaned, unknown = composition.verify_citations(
            "Kept [source:1]. Gone [source:9]. Gone [source:zzz].", chunks
        )
        assert cleaned == "Kept [source:aaa]. Gone . Gone ."
        assert unknown == ["9", "zzz"]

    def test_a_full_source_id_is_accepted(self):
        chunks = [composition.Chunk(source_id="source:aaa", title="A", text="a")]
        cleaned, unknown = composition.verify_citations("[source:aaa]", chunks)
        assert cleaned == "[source:aaa]"
        assert unknown == []

    def test_an_index_outside_the_chunk_list_is_dropped(self):
        chunks = [composition.Chunk(source_id="source:aaa", title="A", text="a")]
        cleaned, unknown = composition.verify_citations("[source:2]", chunks)
        assert cleaned == ""
        assert unknown == ["2"]

    def test_text_without_citations_is_untouched(self):
        cleaned, unknown = composition.verify_citations("plain text", [])
        assert cleaned == "plain text"
        assert unknown == []


class TestComposeSection:
    @pytest.mark.asyncio
    async def test_text_is_returned_with_the_chunks_used(self):
        section = outline.Section(title="One", queries=["alpha"])
        chunks = [composition.Chunk(source_id="source:aaa", title="A", text="a")]
        complete = make_complete(["## One\nbody"])

        text, used = await composition.compose_section(
            complete, section, chunks, "report", composition.CompositionConfig()
        )
        assert text == "## One\nbody"
        assert used == chunks

    @pytest.mark.asyncio
    async def test_empty_reply_is_retried_with_half_the_material(self):
        section = outline.Section(title="One", queries=["alpha"])
        chunks = [
            composition.Chunk(source_id=f"source:{i}", title="T", text=f"t{i}")
            for i in range(4)
        ]
        complete = make_complete(["", "## One\nbody"])

        text, used = await composition.compose_section(
            complete, section, chunks, "report", composition.CompositionConfig()
        )
        assert text == "## One\nbody"
        assert [c.source_id for c in used] == ["source:0", "source:1"]

    @pytest.mark.asyncio
    async def test_a_single_attempt_config_never_retries(self):
        section = outline.Section(title="One", queries=["alpha"])
        chunks = [
            composition.Chunk(source_id=f"source:{i}", title="T", text=f"t{i}")
            for i in range(4)
        ]
        complete = make_complete(["", "## One\nbody"])
        cfg = composition.CompositionConfig(max_section_attempts=1)

        text, used = await composition.compose_section(
            complete, section, chunks, "report", cfg
        )
        assert text == ""
        assert used == chunks
        assert len(complete.prompts) == 1

    @pytest.mark.asyncio
    async def test_a_failing_call_degrades_to_empty_text(self):
        async def failing(prompt):
            raise RuntimeError("model down")

        section = outline.Section(title="One", queries=["alpha"])
        text, _ = await composition.compose_section(
            failing,
            section,
            [composition.Chunk(source_id="source:aaa", title="A", text="a")],
            "report",
            composition.CompositionConfig(),
        )
        assert text == ""


class TestBuildSectionPrompt:
    def test_deck_and_report_use_their_own_rules(self):
        section = outline.Section(title="One", thesis="t", queries=["alpha"])
        chunks = [composition.Chunk(source_id="source:aaa", title="A", text="body")]
        deck = composition.build_section_prompt(section, chunks, "deck", "it")
        report = composition.build_section_prompt(section, chunks, "report", "it")
        assert "bullets per section" in deck
        assert "bullets per section" not in report
        # A raw ISO code ("it") is ignored by the model; the prompt must carry
        # the language NAME.
        assert "Write in Italian" in deck
        assert "Write in it" not in deck

    def test_chunks_are_numbered_for_citation(self):
        section = outline.Section(title="One", queries=["alpha"])
        chunks = [
            composition.Chunk(source_id="source:aaa", title="A", text="first"),
            composition.Chunk(source_id="source:bbb", title="B", text="second"),
        ]
        prompt = composition.build_section_prompt(section, chunks, "report", "en")
        assert "[source:1] (source: A)\nfirst" in prompt
        assert "[source:2] (source: B)\nsecond" in prompt

    def test_no_chunks_is_stated_explicitly(self):
        section = outline.Section(title="One", queries=["alpha"])
        prompt = composition.build_section_prompt(section, [], "report", "en")
        assert "(no excerpt retrieved)" in prompt

    def test_diagram_rules_only_when_allowed(self):
        section = outline.Section(title="One", queries=["alpha"])
        assert "mermaid" not in composition.build_section_prompt(
            section, [], "report", "en"
        )
        assert "mermaid" in composition.build_section_prompt(
            section, [], "report", "en", allow_diagrams=True
        )


class TestBuildDocument:
    @pytest.mark.asyncio
    async def test_document_resolves_citations_and_drops_unknown_ones(self):
        sections = [outline.Section(title="One", queries=["alpha"])]
        search = make_search([make_hit()])
        complete = make_complete(["## One\nBody [source:1]. Bad [source:7]."])

        document, report = await composition.build_document(
            complete,
            search,
            ["Source A"],
            ["source:aaa"],
            "My title",
            "report",
            composition.RetrievalConfig(),
            composition.CompositionConfig(),
            sections,
        )

        assert document.startswith("# My title")
        assert "[source:aaa]" in document
        assert "[source:7]" not in document
        assert report["mode"] == "retrieval"
        assert report["sections_written"] == 1
        assert report["unknown_citations"] == ["7"]
        assert report["sections"][0]["chunks"][0]["source_id"] == "source:aaa"

    @pytest.mark.asyncio
    async def test_verification_can_be_disabled(self):
        cfg = composition.RetrievalConfig(verify_citations=False)
        document, _ = await composition.build_document(
            make_complete(["## One\n[source:9]"]),
            make_search([make_hit()]),
            ["Source A"],
            ["source:aaa"],
            "T",
            "report",
            cfg,
            composition.CompositionConfig(),
            [outline.Section(title="One", queries=["alpha"])],
        )
        assert "[source:9]" in document

    @pytest.mark.asyncio
    async def test_sections_producing_nothing_are_skipped(self):
        document, report = await composition.build_document(
            make_complete([""]),
            make_search([make_hit()]),
            ["Source A"],
            ["source:aaa"],
            "T",
            "report",
            composition.RetrievalConfig(),
            composition.CompositionConfig(max_section_attempts=1),
            [outline.Section(title="One", queries=["alpha"])],
        )
        assert "## One" not in document
        assert report["sections_written"] == 0

    @pytest.mark.asyncio
    async def test_outline_is_planned_when_no_sections_are_given(self, monkeypatch):
        reply = '{"sections": [{"title": "Planned", "queries": ["alpha"]}]}'

        class _Model:
            async def ainvoke(self, prompt):
                return SimpleNamespace(content=reply)

        async def fake_provision(content, model_id, default_type, **kwargs):
            return _Model()

        monkeypatch.setattr(outline, "provision_langchain_model", fake_provision)

        complete = make_complete(["## Planned\nBody [source:1]"])
        document, report = await composition.build_document(
            complete,
            make_search([make_hit()]),
            ["Source A"],
            ["source:aaa"],
            "T",
            "report",
            composition.RetrievalConfig(),
        )
        assert "## Planned" in document
        assert report["sections"][0]["title"] == "Planned"


class TestAssembleAndConfig:
    def test_document_has_a_title_and_no_leading_blank_line(self):
        document = composition.assemble_document("T", ["a", "  ", "b"])
        assert document == "# T\n\na\n\nb\n"

    def test_document_without_a_title_starts_with_the_body(self):
        assert composition.assemble_document("", ["body"]) == "body\n"

    def test_config_defaults_match_the_standalone_config(self):
        cfg = composition.config_from_dict(None)
        assert cfg.outline_max_sections == 10
        assert cfg.chunks_per_query == 6
        assert cfg.max_chunks_per_section == 8
        assert cfg.min_score == 0.2

    def test_config_overrides_are_applied(self):
        cfg = composition.config_from_dict(
            {"min_score": 0.5, "verify_citations": False}
        )
        assert cfg.min_score == 0.5
        assert cfg.verify_citations is False
