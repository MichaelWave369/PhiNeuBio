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


class ValidationDashboardTests(unittest.TestCase):
    def test_phase7_contract(self):
        source = (Path(__file__).resolve().parents[1] / "dashboard" / "validation.html").read_text(encoding="utf-8")
        p = Parser()
        p.feed(source)
        self.assertFalse(p.external_scripts)
        self.assertTrue({"file", "status", "noise", "delay", "seed",
                         "accuracy", "results", "selection", "parameter-table"} <= p.ids)
        self.assertIn('schema_version!=="7.0"', source)
        self.assertIn("validation_cross_entropy", source)


if __name__ == "__main__":
    unittest.main()
