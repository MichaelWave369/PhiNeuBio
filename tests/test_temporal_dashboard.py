from html.parser import HTMLParser
from pathlib import Path
import unittest


class SourceHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.external_scripts = []

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if "id" in data:
            self.ids.add(data["id"])
        if tag == "script" and "src" in data:
            self.external_scripts.append(data["src"])


class TemporalDashboardTests(unittest.TestCase):
    def test_phase5_viewer_has_expected_data_contract(self):
        page = (Path(__file__).resolve().parents[1] / "dashboard" / "temporal.html").read_text(encoding="utf-8")
        parser = SourceHTML()
        parser.feed(page)
        self.assertFalse(parser.external_scripts)
        self.assertTrue({"file", "reload", "status", "noise", "delay", "accuracy", "metrics", "profile"} <= parser.ids)
        self.assertIn('schema_version!=="5.0"', page)
        self.assertIn("peak_traced_python_bytes", page)
        self.assertIn("tanh_reservoir", page)


if __name__ == "__main__":
    unittest.main()
