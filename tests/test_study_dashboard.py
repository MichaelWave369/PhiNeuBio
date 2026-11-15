from html.parser import HTMLParser
from pathlib import Path
import unittest


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids=set()
        self.external_scripts=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag=="script" and "src" in attrs:
            self.external_scripts.append(attrs["src"])


class StudyDashboardTests(unittest.TestCase):
    def test_local_dashboard_is_self_contained(self):
        page=(Path(__file__).resolve().parents[1]/"dashboard"/"study.html").read_text(encoding="utf-8")
        parser=Parser();parser.feed(page)
        self.assertFalse(parser.external_scripts)
        self.assertTrue({"file","status","signal","noise","delay","pair","chart","contrasts","accuracy","fingerprint"}<=parser.ids)
        self.assertIn('schema_version!=="10.0"',page)
        self.assertIn("holm_adjusted_sign_p",page)


if __name__=="__main__":
    unittest.main()
