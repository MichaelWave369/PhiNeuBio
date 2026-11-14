from html.parser import HTMLParser
from pathlib import Path
import unittest


class DashboardParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.canvases = 0
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag == "canvas":
            self.canvases += 1
        if tag == "script":
            self.scripts.append(attributes)


class DashboardTests(unittest.TestCase):
    def test_self_contained_activity_viewer(self):
        document = (Path(__file__).resolve().parents[1] / "dashboard" / "index.html").read_text(encoding="utf-8")
        parser = DashboardParser()
        parser.feed(document)
        self.assertEqual(parser.canvases, 2)
        self.assertTrue({"raster", "rewards", "file", "play", "pause", "scrub", "speed"} <= parser.ids)
        self.assertTrue(parser.scripts)
        self.assertTrue(all("src" not in script for script in parser.scripts))
        self.assertIn("software_simulation", document)
        self.assertIn("schema_version", document)


if __name__ == "__main__":
    unittest.main()
