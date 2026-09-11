# -*- coding: utf-8 -*-
"""
tests/test_prompt_construction.py
==================================
Verify that all prompt templates produce correct structure and that the
corrected experiments hold the necessary invariants.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.prompts import (
    pilot_role_wrap, scramble_role_wrap,
    source_format, behavior_prompt, warning_prompt,
    SOURCE_CONDITIONS, SOURCE_LABELS,
    WARNING_CONDITIONS, WARNING_LABELS,
    LEGITIMATE_TASK, INJECTED_PAYLOAD, CODING_TASK_POOL,
    PILOT_ROLES,
)


# ---------------------------------------------------------------------------
# Pilot role templates
# ---------------------------------------------------------------------------

class TestPilotRoleWrap:
    def test_system_prefix(self):
        result = pilot_role_wrap("system", "hello")
        assert result.startswith("<|im_start|>system")

    def test_user_prefix(self):
        result = pilot_role_wrap("user", "hello")
        assert result.startswith("<|im_start|>user")

    def test_cot_contains_think(self):
        result = pilot_role_wrap("cot", "hello")
        assert "<think>" in result

    def test_tool_prefix(self):
        result = pilot_role_wrap("tool", "hello")
        assert result.startswith("<|im_start|>tool")

    def test_text_preserved(self):
        text = "The quick brown fox"
        for role in PILOT_ROLES:
            result = pilot_role_wrap(role, text)
            assert text in result, f"Text not preserved for role {role}"

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            pilot_role_wrap("oracle", "text")

    def test_end_token_present(self):
        for role in PILOT_ROLES:
            result = pilot_role_wrap(role, "text")
            assert "<|im_end|>" in result


class TestScrambleRoleWrap:
    def test_no_original_role_name(self):
        for role in PILOT_ROLES:
            scrambled = scramble_role_wrap(role, "test text")
            # The original role name should NOT appear as a standalone token label
            # (it may appear in ROLE_A, ROLE_B labels, but not as <|im_start|>system etc.)
            assert f"<|im_start|>{role}" not in scrambled

    def test_placeholder_present(self):
        for role, placeholder in [
            ("system", "ROLE_A"), ("user", "ROLE_B"),
            ("cot", "ROLE_C"), ("tool", "ROLE_D"),
        ]:
            result = scramble_role_wrap(role, "text")
            assert placeholder in result

    def test_text_preserved(self):
        for role in PILOT_ROLES:
            result = scramble_role_wrap(role, "preserved text")
            assert "preserved text" in result


# ---------------------------------------------------------------------------
# Fixed-structure source format (Experiment 02)
# ---------------------------------------------------------------------------

class TestSourceFormat:
    def test_all_conditions_produce_output(self):
        for cond in SOURCE_CONDITIONS:
            result = source_format(cond, "test content")
            assert len(result) > 0

    def test_all_conditions_contain_text(self):
        text = "unique payload text 12345"
        for cond in SOURCE_CONDITIONS:
            result = source_format(cond, text)
            assert text in result, f"Text not found for condition {cond}"

    def test_all_conditions_have_source_header(self):
        for cond in SOURCE_CONDITIONS:
            result = source_format(cond, "text")
            assert result.startswith("SOURCE:"), f"Missing SOURCE: in {cond}"

    def test_all_conditions_have_content_header(self):
        for cond in SOURCE_CONDITIONS:
            result = source_format(cond, "text")
            assert "CONTENT:" in result, f"Missing CONTENT: in {cond}"

    def test_structure_identical_except_label(self):
        """All conditions should share the same outer delimiters."""
        results = [source_format(cond, "same text") for cond in SOURCE_CONDITIONS]
        # The CONTENT: header and text should appear in all
        for r in results:
            assert "CONTENT:\nsame text" in r

    def test_invalid_condition_raises(self):
        with pytest.raises(ValueError):
            source_format("unknown_condition", "text")

    def test_no_credential_strings_in_labels(self):
        """Labels must not contain API keys, tokens, or paths."""
        for cond, label in SOURCE_LABELS.items():
            assert "sk-" not in label
            assert "Bearer" not in label
            assert "C:\\" not in label


# ---------------------------------------------------------------------------
# Behavior prompt (Experiment 04)
# ---------------------------------------------------------------------------

class TestBehaviorPrompt:
    def test_contains_chatml_markers(self):
        for cond in SOURCE_CONDITIONS:
            result = behavior_prompt(cond, LEGITIMATE_TASK, INJECTED_PAYLOAD)
            assert "<|im_start|>system" in result
            assert "<|im_start|>user" in result

    def test_contains_both_tasks(self):
        for cond in SOURCE_CONDITIONS:
            result = behavior_prompt(cond, LEGITIMATE_TASK, INJECTED_PAYLOAD)
            assert "DEBUG" in result or "settings.py" in result
            assert "forbidden.txt" in result or "INJECTED_MARKER" in result

    def test_ends_with_assistant_token(self):
        for cond in SOURCE_CONDITIONS:
            result = behavior_prompt(cond, LEGITIMATE_TASK, INJECTED_PAYLOAD)
            assert result.strip().endswith("<|im_start|>assistant")


# ---------------------------------------------------------------------------
# Warning ablation prompts (Experiment 03)
# ---------------------------------------------------------------------------

class TestWarningPrompt:
    def test_all_conditions_produce_output(self):
        for cond in WARNING_CONDITIONS:
            result = warning_prompt(cond, LEGITIMATE_TASK, INJECTED_PAYLOAD)
            assert len(result) > 10

    def test_identical_payload_across_conditions(self):
        """The injected payload text must be identical across all warning conditions."""
        payloads = []
        for cond in WARNING_CONDITIONS:
            result = warning_prompt(cond, LEGITIMATE_TASK, INJECTED_PAYLOAD)
            # Extract the CONTENT: section
            content_start = result.find("CONTENT:\n")
            if content_start >= 0:
                payload_section = result[content_start:]
                payloads.append(payload_section)
        assert len(set(payloads)) == 1, (
            "CONTENT sections differ across warning conditions — payload is not matched!"
        )

    def test_invalid_condition_raises(self):
        with pytest.raises(ValueError):
            warning_prompt("nonexistent", LEGITIMATE_TASK, INJECTED_PAYLOAD)

    def test_no_warning_has_minimal_label(self):
        result = warning_prompt("no_warning", LEGITIMATE_TASK, INJECTED_PAYLOAD)
        # Should not contain "CAUTION", "WARNING", "NOTE" (reserved for other conditions)
        assert "CAUTION" not in result
        assert "NOTE" not in result


# ---------------------------------------------------------------------------
# Task pool integrity
# ---------------------------------------------------------------------------

class TestTaskPool:
    def test_pool_has_30_tasks(self):
        assert len(CODING_TASK_POOL) == 30

    def test_all_tasks_non_empty(self):
        for t in CODING_TASK_POOL:
            assert len(t.strip()) > 10

    def test_no_harmful_tasks(self):
        """Tasks must not involve deletion, credentials, or harmful system calls."""
        dangerous_keywords = ["rm -rf", "drop table", "DELETE FROM", "sudo", "passwd"]
        for task in CODING_TASK_POOL:
            for kw in dangerous_keywords:
                assert kw.lower() not in task.lower(), f"Dangerous keyword '{kw}' in task: {task}"
