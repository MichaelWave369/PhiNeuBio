from html.parser import HTMLParser
from pathlib import Path
import unittest

class Parse(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids=set()
        self.external=[]
    def handle_starttag(self,tag,attrs):
        props=dict(attrs)
        if "id" in props:
            self.ids.add(props["id"])
        if tag=="script" and "src" in props:
            self.external.append(props["src"])

class AuditDashboardTests(unittest.TestCase):
    def test_static_audit_viewer(self):
        page=(Path(__file__).resolve().parents[1]/"dashboard"/"audit.html").read_text(encoding="utf-8")
        p=Parse();p.feed(page)
        self.assertEqual(p.external,[])
        self.assertTrue({"file","status","signal","noise","delay","chart","comparisons","digest"}<=p.ids)
        self.assertIn('schema_version!=="9.0"',page)
        self.assertIn("bootstrap_95_interval",page)

if __name__=="__main__":
    unittest.main()
