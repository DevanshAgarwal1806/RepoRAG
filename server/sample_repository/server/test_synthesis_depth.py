import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph.nodes.synthesizer import (
    _collect_latest_raw_results,
    _parse_success_payload,
)


class SynthesisDepthTests(unittest.TestCase):
    def test_parse_success_payload_prefers_structured_details(self) -> None:
        payload = (
            '{"headline": "Best season is October to March.", '
            '"detailed_summary": "Pleasant weather, festival season, and easier sightseeing.", '
            '"reason": "Relevant and specific."}'
        )

        parsed = _parse_success_payload(payload)

        self.assertEqual(parsed["headline"], "Best season is October to March.")
        self.assertIn("festival season", parsed["detailed_summary"])
        self.assertEqual(parsed["reason"], "Relevant and specific.")

    def test_collect_latest_raw_results_keeps_latest_entry_per_task(self) -> None:
        scratchpad = [
            "RAW_RESULT|step_1|older result",
            "RAW_RESULT|step_2|different task result",
            "RAW_RESULT|step_1|latest result with more detail",
        ]

        raw_results = _collect_latest_raw_results(scratchpad)

        self.assertEqual(raw_results["step_1"], "latest result with more detail")
        self.assertEqual(raw_results["step_2"], "different task result")


if __name__ == "__main__":
    unittest.main()
