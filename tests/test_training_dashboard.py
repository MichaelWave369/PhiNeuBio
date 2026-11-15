from html.parser import HTMLParser
from pathlib import Path
import unittest

class Inspector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids=set()
        self.external=[]
    def handle_starttag(self,tag,attrs):
        data=dict(attrs)
        if "id" in data:self.ids.add(data["id"])
        if tag=="script" and "src" in data:self.external.append(data["src"])

class TrainingDashboardTests(unittest.TestCase):
    def test_offline_dashboard_contract(self):
        text=(Path(__file__).resolve().parents[1]/"dashboard"/"training.html").read_text(encoding="utf-8")
        parser=Inspector();parser.feed(text)
        self.assertFalse(parser.external)
        self.assertTrue({"file","status","signal","noise","delay","budget","seed","chart","accuracy","effort"}<=parser.ids)
        self.assertIn('schema_version!=="13.0"',text)
        self.assertIn("target_presentations",text)
        self.assertIn("optimizer_updates",text)

if __name__=="__main__":
    unittest.main()
