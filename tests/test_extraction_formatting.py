"""Boundary tests that do not require model weights or GPU dependencies."""
import ast
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/extraction/extract_activations.py"
# Load only the pure helper so these tests need no optional inference packages.
tree = ast.parse(SOURCE.read_text())
helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_branch")
namespace = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), "exec"), namespace)
prepare_branch = namespace["prepare_branch"]


class CharacterTokenizer:
    """Controlled offsets make content versus template boundaries unambiguous."""
    def apply_chat_template(self, messages, **kwargs):
        return "<bos>" + "".join(f"<{x['role']}>\n{x['content'].strip()}<end>\n" for x in messages)

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class FormattingTests(unittest.TestCase):
    def test_content_boundary_and_shared_prefix(self):
        item = SimpleNamespace(topic="Recursion", context_turns=[
            SimpleNamespace(speaker="tutor", text=" Correct. "),
            SimpleNamespace(speaker="student", text="Wrong!"),
        ])
        tokenizer = CharacterTokenizer()
        branches = [prepare_branch(tokenizer, item, SimpleNamespace(speaker="tutor", text=text))
                    for text in ["Yes.", "No!"]]
        context_ids, context_positions = prepare_branch(tokenizer, item)
        self.assertEqual(len(context_positions), len(item.context_turns))
        for ids, positions in branches:
            self.assertEqual(ids[:len(context_ids)], context_ids)
            self.assertEqual(positions[:-1], context_positions)
        for ids, positions in branches:
            self.assertEqual([chr(ids[p]) for p in positions], [".", "!", chr(ids[positions[-1]])])
            self.assertLess(positions[-1], len(ids) - 1)  # Excludes template suffix.
        end = branches[0][1][-2] + 1
        self.assertEqual(branches[0][0][:end], branches[1][0][:end])
        item.context_turns += [SimpleNamespace(speaker="tutor", text="Again."),
                               SimpleNamespace(speaker="student", text="Still wrong!")]
        _, positions = prepare_branch(tokenizer, item, SimpleNamespace(speaker="tutor", text="No."))
        self.assertEqual(len(positions), 5)

    def test_notebook_bundle_matches_current_files(self):
        nb = json.loads((ROOT / "track1.ipynb").read_text())
        code = next("".join(cell["source"]) for cell in nb["cells"]
                    if cell["cell_type"] == "code" and "base64.b64decode" in "".join(cell["source"]))
        parsed = ast.parse(code)
        call = next(node for node in ast.walk(parsed) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "b64decode")
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(call.args[0])))) as archive:
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (ROOT / name).read_bytes(), name)
        self.assertEqual(nb["nbformat"], 4)


if __name__ == "__main__":
    unittest.main()
