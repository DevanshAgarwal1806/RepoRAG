import unittest
from unittest.mock import patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph.nodes.executor import _get_pending_task
from src.tools.tool_selector import select_tool


class DummyTool:
    def __init__(self, name: str):
        self.name = name
        self.description = f"{name} tool"


class ToolRetryFallbackTests(unittest.TestCase):
    def test_failed_task_is_retried_until_exhausted(self) -> None:
        dag = {
            "tasks": [
                {"id": "step_1", "dependencies": []},
                {"id": "step_2", "dependencies": []},
            ]
        }
        scratchpad = [
            "FAILURE|step_1|first attempt failed|Reason: bad output",
        ]

        pending_task = _get_pending_task(dag, scratchpad)

        self.assertIsNotNone(pending_task)
        self.assertEqual(pending_task["id"], "step_1")

    def test_selector_excludes_previously_attempted_tools(self) -> None:
        tool_map = {
            "tool_a": DummyTool("tool_a"),
            "tool_b": DummyTool("tool_b"),
        }
        seen_candidate_sets: list[list[str]] = []

        def fake_score_tools(description: str, candidate_map: dict[str, object]):
            seen_candidate_sets.append(list(candidate_map))
            return "tool_b", "fallback query", 10

        with patch("src.tools.tool_selector._score_tools", side_effect=fake_score_tools):
            tool, tool_input, warnings = select_tool(
                {"id": "step_1", "description": "find travel info"},
                tool_map,
                excluded_tool_names={"tool_a"},
            )

        self.assertEqual(seen_candidate_sets, [["tool_b"]])
        self.assertEqual(tool.name, "tool_b")
        self.assertEqual(tool_input, "fallback query")
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
