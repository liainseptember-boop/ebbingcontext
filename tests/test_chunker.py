"""Tests for message chunker."""

from ebbingcontext.core.chunker import MessageChunker


class TestMessageChunker:
    def setup_method(self):
        self.chunker = MessageChunker()

    def test_empty_content(self):
        assert self.chunker.chunk("", "user") == []
        assert self.chunker.chunk("   ", "user") == []

    def test_short_content_not_split(self):
        chunks = self.chunker.chunk("Hello!", "user")
        assert len(chunks) == 1
        assert chunks[0].content == "Hello!"

    # --- Sentence splitting (user messages) ---

    def test_user_sentence_split(self):
        text = "First sentence. Second sentence. Third sentence here."
        chunks = self.chunker.chunk(text, "user")
        assert len(chunks) >= 2
        # All content should be preserved
        combined = " ".join(c.content for c in chunks)
        assert "First" in combined
        assert "Third" in combined

    def test_user_question_split(self):
        text = "What is Python? It is a programming language. How does it work?"
        chunks = self.chunker.chunk(text, "user")
        assert len(chunks) >= 2

    # --- Paragraph splitting (agent messages) ---

    def test_agent_paragraph_split(self):
        text = "First paragraph with some content here.\n\nSecond paragraph with different content."
        chunks = self.chunker.chunk(text, "agent")
        assert len(chunks) == 2
        assert chunks[0].source_type == "agent"

    def test_agent_single_paragraph(self):
        text = "Just one paragraph with enough content to not be short."
        chunks = self.chunker.chunk(text, "agent")
        assert len(chunks) == 1

    # --- Tool output splitting ---

    def test_tool_json_split(self):
        import json

        data = {"name": "Alice", "age": 30, "role": "developer"}
        text = json.dumps(data)
        chunks = self.chunker.chunk(text, "tool")
        assert len(chunks) == 3
        assert chunks[0].metadata.get("field_name") == "name"
        assert "Alice" in chunks[0].content

    def test_tool_invalid_json_falls_back_to_lines(self):
        text = "Line one output\nLine two output\nLine three output"
        chunks = self.chunker.chunk(text, "tool")
        assert len(chunks) >= 1
        combined = " ".join(c.content for c in chunks)
        assert "Line one" in combined

    def test_tool_non_dict_json(self):
        text = '["item1", "item2", "item3"]'
        chunks = self.chunker.chunk(text, "tool")
        assert len(chunks) == 1  # Non-dict JSON returned as single chunk

    # --- Merge short fragments ---

    def test_short_fragments_merged(self):
        text = "Hi. Ok. Yes. This is a longer sentence that should stand alone."
        chunks = self.chunker.chunk(text, "user")
        # Short fragments should be merged
        for chunk in chunks:
            assert len(chunk.content) >= 10 or chunk == chunks[-1]

    # --- Source type preserved ---

    def test_source_type_preserved(self):
        chunks = self.chunker.chunk("Some user message content here.", "user")
        assert all(c.source_type == "user" for c in chunks)

        chunks = self.chunker.chunk("Some tool output content here.", "tool")
        assert all(c.source_type == "tool" for c in chunks)
