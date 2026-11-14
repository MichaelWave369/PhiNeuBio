import contextlib
import io
import json
import unittest

from phineubio.cli import main


class CLITests(unittest.TestCase):
    def test_json_report(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = main([
                "--neurons", "8", "--train-sequences", "3",
                "--test-sequences", "2", "--sequence-length", "12",
                "--burn-in", "2", "--epochs", "2"
            ])
        self.assertEqual(result, 0)
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["samples"]["features"], 16)
        self.assertIn("test_accuracy", report["metrics"])


if __name__ == "__main__":
    unittest.main()
