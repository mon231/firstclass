import contextlib
import io
import runpy
import unittest
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "example.py"


class Example(unittest.TestCase):
    def test_example_runs(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(EXAMPLE), run_name="__main__")
        self.assertIn("struct Telemetry {", out.getvalue())
        self.assertIn("rejected:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
