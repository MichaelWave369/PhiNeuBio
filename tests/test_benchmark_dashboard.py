from html.parser import HTMLParser
from pathlib import Path
import unittest


class Analyzer(HTMLParser):
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


class BenchmarkDashboardTests(unittest.TestCase):
    def test_dashboard_is_self_contained_and_has_report_contract(self):
        source = (Path(__file__).resolve().parents[1] / "dashboard" / "benchmark.html").read_text(encoding="utf-8")
        parser = Analyzer()
        parser.feed(source)
        self.assertTrue({"file", "load", "plot", "summary", "runs", "legend"} <= parser.ids)
        self.assertEqual(parser.external_scripts, [])
        self.assertIn('schema_version!=="3.0"', source)
        self.assertIn("population_stdev", source)


if __name__ == "__main__":
    unittest.main()
