"""Phase 11 offline compute dashboard contract."""
from html.parser import HTMLParser
from pathlib import Path
import unittest


class Inspector(HTMLParser):
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


class ComputeDashboardTests(unittest.TestCase):
    def test_offline_viewer_renders_structural_and_timing_contract(self):
        page=(Path(__file__).resolve().parents[1]/"dashboard"/"compute.html").read_text(encoding="utf-8")
        obj=Inspector()
        obj.feed(page)
        self.assertEqual(obj.external,[])
        self.assertTrue({"file","load","status","budget","signal","noise","delay","seed","chart","costs","results","rounds"}<=obj.ids)
        self.assertIn('schema_version!=="11.0"',page)
        self.assertIn("median_elapsed_ns",page)
        self.assertIn("recurrent_pair_visits_per_step",page)


if __name__=="__main__":
    unittest.main()
