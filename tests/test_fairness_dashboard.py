from html.parser import HTMLParser
from pathlib import Path
import unittest


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.external_scripts = []

    def handle_starttag(self, tag, attrs):
        props = dict(attrs)
        if "id" in props:
            self.ids.add(props["id"])
        if tag == "script" and "src" in props:
            self.external_scripts.append(props["src"])


class FairnessDashboardTests(unittest.TestCase):
    def test_offline_viewer_and_versioned_contract(self):
        source = (Path(__file__).resolve().parents[1] / "dashboard" / "fairness.html").read_text(encoding="utf-8")
        parser = Parser()
        parser.feed(source)
        self.assertEqual(parser.external_scripts, [])
        self.assertTrue({"file", "status", "signal", "noise", "delay", "seed",
                         "chart", "results", "selected", "profiles"} <= parser.ids)
        self.assertIn('schema_version!=="8.0"', source)
        self.assertIn("median_elapsed_ns", source)


if __name__ == "__main__":
    unittest.main()
