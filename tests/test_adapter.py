"""Tests for adapter components."""

from ebbingcontext.adapter.token_counter import TokenCounter


class TestTokenCounter:
    def setup_method(self):
        self.counter = TokenCounter()

    def test_empty_string(self):
        assert self.counter.count("") == 0

    def test_english_text(self):
        # "Hello world" ~ 2 tokens, but heuristic may differ slightly
        count = self.counter.count("Hello world")
        assert 1 <= count <= 5

    def test_longer_english(self):
        text = "The quick brown fox jumps over the lazy dog"
        count = self.counter.count(text)
        # ~9 tokens in reality, heuristic should be in range
        assert 5 <= count <= 20

    def test_cjk_text(self):
        text = "你好世界"
        count = self.counter.count(text)
        # 4 CJK chars, ~2-3 tokens
        assert 1 <= count <= 6

    def test_mixed_content(self):
        text = "Hello 你好 World 世界"
        count = self.counter.count(text)
        assert count > 0

    def test_code_text(self):
        text = "def hello_world():\n    print('Hello, World!')"
        count = self.counter.count(text)
        assert 5 <= count <= 30

    def test_always_at_least_one(self):
        assert self.counter.count("a") >= 1
