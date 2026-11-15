"""Phase 14 offline dashboard version and structure contract."""
from html.parser import HTMLParser
from pathlib import Path
import unittest


class Inspector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.scripts = []
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if "id" in d:
            self.ids.add(d["id"])
        if tag == "script" and "src" in d:
            self.scripts.append(d["src"])


class WorkBudgetDashboardTests(unittest.TestCase):
    def test_offline_viewer(self):
        page = (Path(__file__).resolve().parents[1] / "dashboard" /
                "workbudget.html").read_text(encoding="utf-8")
        p = Inspector()
        p.feed(page)
        self.assertFalse(p.scripts)
        self.assertTrue({"file", "load", "status", "signal", "noise",
                         "delay", "cap", "seed", "chart", "results", "ledger"} <= p.ids)
        self.assertIn('schema_version!=="14.0"', page)
        self.assertIn("unused_cap", page)
        self.assertIn("optimizer_updates", page)


if __name__ == "__main__":
    unittest.main()
