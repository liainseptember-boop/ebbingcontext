"""Tests for the dual-label classifier."""

from ebbingcontext.core.classifier import Classifier, RuleClassifier
from ebbingcontext.models import DecayStrategy, SensitivityLevel


class TestRuleClassifier:
    def setup_method(self):
        self.clf = RuleClassifier()

    def test_system_source_is_pinned(self):
        strategy, rule = self.clf.classify_decay_strategy("anything", "system")
        assert strategy == DecayStrategy.PIN
        assert rule == "source:system"

    def test_system_prompt_keyword_is_pinned(self):
        strategy, _ = self.clf.classify_decay_strategy("This is the system prompt", "user")
        assert strategy == DecayStrategy.PIN

    def test_debug_output_is_irreversible(self):
        strategy, _ = self.clf.classify_decay_strategy("debug log output here", "tool")
        assert strategy == DecayStrategy.DECAY_IRREVERSIBLE

    def test_normal_text_is_recoverable(self):
        strategy, rule = self.clf.classify_decay_strategy("The weather is nice today", "user")
        assert strategy == DecayStrategy.DECAY_RECOVERABLE
        assert rule is None

    def test_api_key_is_sensitive(self):
        level, _ = self.clf.classify_sensitivity("my api_key is abc123")
        assert level == SensitivityLevel.SENSITIVE

    def test_password_is_sensitive(self):
        level, _ = self.clf.classify_sensitivity("password: hunter2")
        assert level == SensitivityLevel.SENSITIVE

    def test_internal_keyword(self):
        level, _ = self.clf.classify_sensitivity("this is internal documentation")
        assert level == SensitivityLevel.INTERNAL

    def test_normal_text_is_public(self):
        level, rule = self.clf.classify_sensitivity("Hello world, how are you?")
        assert level == SensitivityLevel.PUBLIC
        assert rule is None

    def test_importance_system_higher(self):
        score_sys = self.clf.estimate_importance("test", "system")
        score_user = self.clf.estimate_importance("test", "user")
        assert score_sys > score_user

    def test_importance_bounded(self):
        score = self.clf.estimate_importance("x" * 5000, "tool")
        assert 0.0 <= score <= 1.0


class TestClassifier:
    def setup_method(self):
        self.clf = Classifier()

    def test_full_classification(self):
        result = self.clf.classify("Store my api_key: sk-12345", "user")
        assert result.sensitivity == SensitivityLevel.SENSITIVE
        assert result.importance > 0
        assert result.confidence > 0

    def test_system_prompt_classification(self):
        result = self.clf.classify("You are a helpful assistant", "system")
        assert result.decay_strategy == DecayStrategy.PIN
        assert result.importance >= 0.8

    def test_confidence_higher_with_rule_match(self):
        result_matched = self.clf.classify("debug trace output", "tool")
        result_generic = self.clf.classify("Hello there", "user")
        assert result_matched.confidence > result_generic.confidence

    def test_rule_matched_field(self):
        result = self.clf.classify("You are a helpful assistant", "system")
        assert result.rule_matched is not None
        assert "system" in result.rule_matched
