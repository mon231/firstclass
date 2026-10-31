"""The docs, packaging metadata and CI config must stay in step with the code."""
import re
import sys
import unittest
from pathlib import Path

import firstclass
from firstclass import firstclass as fc, uint8_t

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


class Readme(unittest.TestCase):
    def test_first_example_runs_and_its_size_comment_is_true(self):
        block = re.search(r"```python\n(.*?)```", README, re.S).group(1)
        scope = {}
        exec(compile(block, "README example", "exec"), scope)
        claimed = int(re.search(r"sizeof\(\)\s*#\s*(\d+)", block).group(1))
        self.assertEqual(scope["Packet"].sizeof(), claimed)
        self.assertEqual(scope["Packet"].from_bytes(scope["raw"]), scope["p"])

    def test_every_public_name_is_documented(self):
        for name in firstclass.__all__:
            self.assertIn(name, README, f"{name} is exported but not mentioned in the README")

    def test_method_table_matches_the_generated_methods(self):
        @fc
        class Sample:
            a: uint8_t
        table = README[README.index("## Methods added"):README.index("## Import C/C++ headers")]
        names = set(re.findall(r"(?:obj|Cls)\.(\w+)", table))
        self.assertTrue(names, "no methods found in the README table")
        for name in names:
            self.assertTrue(hasattr(Sample, name), f"README documents {name}() but classes do not have it")

    def test_every_generated_method_is_in_the_readme(self):
        from firstclass._core import _METHODS
        for name in _METHODS:
            self.assertIn(name, README, f"{name} exists but is not in the README")

    def test_type_table_covers_every_type_marker(self):
        table = README[README.index("| Marker |"):README.index("Values are checked")]
        for name in ("uint8_t", "int64_t", "float32_t", "float64_t", "bool_t", "char_t", "char_array", "wchar_t",
                     "wchar_array", "bytes_t", "array_t", "enum_t", "bitfield"):
            self.assertIn(name, table)

    def test_example_files_exist_and_are_referenced(self):
        for path in ("examples/example.py", "examples/protocol.h"):
            self.assertTrue((ROOT / path).exists(), path)
            self.assertIn(path.split("/")[-1], README)

    def test_public_functions_have_docstrings(self):
        for fn in (firstclass.from_header, firstclass.from_rust, firstclass.firstclass):
            self.assertTrue((fn.__doc__ or "").strip(), fn.__name__)


@unittest.skipIf(sys.version_info < (3, 11), "tomllib needs Python 3.11+")
class Packaging(unittest.TestCase):
    def setUp(self):
        import tomllib
        self.meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def test_metadata(self):
        project = self.meta["project"]
        self.assertEqual(project["name"], "firstclass")
        self.assertEqual(project["dependencies"], [])  # stdlib only
        self.assertEqual(project["requires-python"], ">=3.10")
        self.assertTrue((ROOT / project["readme"]).exists())
        self.assertRegex(project["version"], r"^\d+\.\d+\.\d+$")

    def test_package_is_listed(self):
        self.assertEqual(self.meta["tool"]["setuptools"]["packages"], ["firstclass"])

    def test_no_third_party_imports(self):
        stdlib = set(sys.stdlib_module_names) | {"firstclass"}
        for path in (ROOT / "firstclass").glob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*(?:from|import)\s+([A-Za-z_][\w]*)", line)
                if m and not line.lstrip().startswith("from ."):
                    self.assertIn(m.group(1), stdlib, f"{path.name}: {line.strip()}")


class Workflows(unittest.TestCase):
    def workflows(self):
        files = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
        self.assertTrue(files)
        return {f.name: f.read_text(encoding="utf-8") for f in files}

    def test_ci_runs_on_ubuntu_only(self):
        for name, text in self.workflows().items():
            self.assertNotRegex(text, r"(?i)windows|macos", f"{name} must stay ubuntu-only")
            self.assertIn("ubuntu-latest", text, name)

    def test_publish_uses_the_tag_and_the_pypi_secret(self):
        text = self.workflows()["publish.yml"]
        self.assertIn("tags:", text)
        self.assertIn("GITHUB_REF_NAME", text)
        self.assertIn("secrets.PYPI_API_TOKEN", text)

    def test_ci_covers_every_supported_python(self):
        text = self.workflows()["ci.yml"]
        for version in ("3.10", "3.11", "3.12", "3.13", "3.14"):
            self.assertIn(f'"{version}"', text)

    def test_tests_never_depend_on_the_gitignored_plan(self):
        for path in (ROOT / "tests").glob("*.py"):
            if path.name != "test_docs.py":
                self.assertNotIn("PLAN.md", path.read_text(encoding="utf-8"), path.name)


if __name__ == "__main__":
    unittest.main()
