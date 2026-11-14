from html.parser import HTMLParser
from pathlib import Path
import unittest


class Inspector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.external_scripts = []

    def handle_starttag(self, tag, attributes):
        props = dict(attributes)
        if "id" in props:
            self.ids.add(props["id"])
        if tag == "script" and "src" in props:
            self.external_scripts.append(props["src"])


class RobustnessDashboardTests(unittest.TestCase):
    def test_static_viewer_has_expected_contract(self):
        root = Path(__file__).resolve().parents[1]
        page = (root / "dashboard" / "robustness.html").read_text(encoding="utf-8")
        parser = Inspector()
        parser.feed(page)
        self.assertEqual(parser.external_scripts, [])
        self.assertTrue({"file", "reload", "status", "noise", "chart", "legend", "means", "costs", "runs"} <= parser.ids)
        self.assertIn('schema_version!=="4.0"', page)
        self.assertIn("population_stdev", page)
        self.assertIn("dense_recurrent_pair_scans", page)


if __name__ == "__main__":
    unittest.main()
