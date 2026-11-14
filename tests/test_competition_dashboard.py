from html.parser import HTMLParser
from pathlib import Path
import unittest


class Tags(HTMLParser):
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


class CompetitionDashboardTests(unittest.TestCase):
    def test_offline_page_and_schema_contract(self):
        html = (Path(__file__).resolve().parents[1] / "dashboard" / "competition.html").read_text(encoding="utf-8")
        parsed = Tags()
        parsed.feed(html)
        self.assertFalse(parsed.external_scripts)
        self.assertTrue({"file", "status", "noise", "delay", "seed", "accuracy", "loss", "results", "profiles"} <= parsed.ids)
        self.assertIn('schema_version!=="6.0"', html)
        self.assertIn("training_loss_by_epoch", html)


if __name__ == "__main__":
    unittest.main()
