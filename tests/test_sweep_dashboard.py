from html.parser import HTMLParser
from pathlib import Path
import unittest


class Inspect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids=set()
        self.external=[]
    def handle_starttag(self,tag,attrs):
        props=dict(attrs)
        if "id" in props:self.ids.add(props["id"])
        if tag=="script" and "src" in props:self.external.append(props["src"])


class SweepDashboardTests(unittest.TestCase):
    def test_standalone_html_and_report_contract(self):
        s=(Path(__file__).resolve().parents[1]/"dashboard"/"sweep.html").read_text(encoding="utf-8")
        p=Inspect();p.feed(s)
        self.assertEqual(p.external,[])
        self.assertTrue({"file","status","signal","noise","delay","budget","seed",
                         "chart","accuracy","ledger"}<=p.ids)
        self.assertIn('schema_version!=="12.0"',s)
        self.assertIn("lif_active_weight_addends",s)


if __name__=="__main__":
    unittest.main()
