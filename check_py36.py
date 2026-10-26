"""Verify every file that runs on the Jetson imports on Python 3.6.

    python check_py36.py

JetPack 4.6 is the final release for the original Jetson Nano and ships
Python 3.6.9. Installing a newer Python there costs more than it buys: the
apt OpenCV (with CUDA and GStreamer, which the RTSP capture needs) and
NVIDIA's onnxruntime-gpu wheel are both built for 3.6.

So 3.6 is a standing constraint, and it is the kind that breaks silently -
a walrus operator or a lowercase generic added on a laptop will import fine
there and fail on the device. This script fails loudly instead.

It checks syntax and standard-library usage, not third-party APIs. Run it
before pushing anything that will go on the board.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# Everything that runs on the device. Add new device-side files here.
DEVICE_PATHS = [
    "config.py", "main_pipeline.py", "node_agent.py", "enroll.py",
    "benchmark.py", "bench_denoise.py", "denoise.py", "check_device.py",
    "check_models.py", "arcface_faiss.py", "gpfgan.py", "yolocropping.py",
    "pipeline", "hub",
    # The suites run on the board too - they are how you tell a broken
    # install from a broken camera, and they need no model files.
    "tests",
]

# (module, attribute) pairs that do not exist in 3.6.
LATE_STDLIB = {
    ("math", "prod"): "3.8",
    ("functools", "cache"): "3.9",
    ("functools", "cached_property"): "3.8",
    ("contextlib", "nullcontext"): "3.7",
    ("http.server", "ThreadingHTTPServer"): "3.7",
    ("datetime", "fromisoformat"): "3.7",
    ("importlib", "metadata"): "3.8",
    ("typing", "Protocol"): "3.8",
    ("typing", "Literal"): "3.8",
    ("typing", "Final"): "3.8",
    ("typing", "TypedDict"): "3.8",
    ("shutil", "copytree"): None,          # only the dirs_exist_ok kwarg
}

# Keyword arguments added after 3.6, keyed by the function name they appear on.
LATE_KWARGS = {
    "run": {"text": "3.7", "capture_output": "3.7"},
    "Popen": {"text": "3.7"},
    "check_output": {"text": "3.7"},
    "unlink": {"missing_ok": "3.8"},
    "copytree": {"dirs_exist_ok": "3.8"},
    "dump": {"sort_keys": None},
}

LATE_METHODS = {
    "removeprefix": "3.9",
    "removesuffix": "3.9",
    "is_relative_to": "3.9",
    "readlink": "3.9",
}

BUILTIN_GENERICS = {"list", "dict", "tuple", "set", "frozenset", "type"}


class Checker(ast.NodeVisitor):
    def __init__(self, path, source):
        self.path = path
        self.source = source
        self.problems = []

    def flag(self, node, message):
        self.problems.append((getattr(node, "lineno", 0), message))

    # --- syntax introduced after 3.6 --------------------------------------
    def visit_NamedExpr(self, node):            # 3.8
        self.flag(node, "walrus operator := (3.8+)")
        self.generic_visit(node)

    def visit_Match(self, node):                # 3.10
        self.flag(node, "match statement (3.10+)")
        self.generic_visit(node)

    def visit_arguments(self, node):            # 3.8
        if getattr(node, "posonlyargs", None):
            self.flag(node, "positional-only parameters / (3.8+)")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "__future__":
            for alias in node.names:
                if alias.name == "annotations":
                    self.flag(node, "from __future__ import annotations (3.7+)")
        if node.module == "dataclasses":
            self.flag(node, "dataclasses needs the backport on 3.6 "
                            "(pip install dataclasses) - not an error, a reminder")
        # `from module import name` must be checked too, not just
        # `module.name(...)` calls - `from http.server import
        # ThreadingHTTPServer` passed this check and failed on the Nano.
        for alias in node.names:
            key = (node.module, alias.name)
            if key in LATE_STDLIB and LATE_STDLIB[key]:
                self.flag(node, "from %s import %s (%s+)"
                          % (key[0], key[1], LATE_STDLIB[key]))
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "dataclasses":
                self.flag(node, "dataclasses needs the backport on 3.6 - reminder")
        self.generic_visit(node)

    # --- annotations ------------------------------------------------------
    def _check_annotation(self, node):
        """Lowercase generics (3.9) and X | Y unions (3.10)."""
        if node is None:
            return
        for sub in ast.walk(node):
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                if sub.value.id in BUILTIN_GENERICS:
                    self.flag(sub, "builtin generic %s[...] (3.9+); "
                                   "use typing.%s" % (sub.value.id, sub.value.id.title()))
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                self.flag(sub, "X | Y union in an annotation (3.10+); "
                               "use typing.Optional/Union")

    def visit_FunctionDef(self, node):
        for arg in list(node.args.args) + list(node.args.kwonlyargs):
            self._check_annotation(arg.annotation)
        self._check_annotation(node.returns)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._check_annotation(node.annotation)
        self.generic_visit(node)

    # --- calls ------------------------------------------------------------
    def visit_Call(self, node):
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                key = (node.func.value.id, node.func.attr)
                if key in LATE_STDLIB and LATE_STDLIB[key]:
                    self.flag(node, "%s.%s (%s+)"
                              % (key[0], key[1], LATE_STDLIB[key]))
            if name in LATE_METHODS:
                self.flag(node, "str/Path.%s (%s+)" % (name, LATE_METHODS[name]))
        elif isinstance(node.func, ast.Name):
            name = node.func.id

        if name in LATE_KWARGS:
            for kw in node.keywords:
                version = LATE_KWARGS[name].get(kw.arg)
                if version:
                    self.flag(node, "%s(%s=...) (%s+)" % (name, kw.arg, version))
        self.generic_visit(node)


def check_fstring_equals(source):
    """f'{x=}' debugging specifier, 3.8+. Regex, since the AST hides it."""
    found = []
    pattern = re.compile(r"""f["'][^"']*\{[^{}]+=\}""")
    for i, line in enumerate(source.splitlines(), 1):
        if pattern.search(line):
            found.append((i, "f-string = specifier (3.8+)"))
    return found


def collect_files():
    files = []
    for entry in DEVICE_PATHS:
        path = ROOT / entry
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.exists():
            files.append(path)
    return [f for f in files if "__pycache__" not in str(f)]


def main():
    files = collect_files()
    total_problems = 0
    reminders = 0

    print("Checking %d device-side files against Python 3.6\n" % len(files))

    for path in files:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            print("  %s: SyntaxError: %s" % (path.relative_to(ROOT), exc))
            total_problems += 1
            continue

        checker = Checker(path, source)
        checker.visit(tree)
        problems = checker.problems + check_fstring_equals(source)

        for lineno, message in sorted(problems):
            tag = "note " if "reminder" in message else "FAIL "
            if "reminder" in message:
                reminders += 1
            else:
                total_problems += 1
            print("  %s%s:%d  %s" % (tag, path.relative_to(ROOT), lineno, message))

    print()
    if total_problems:
        print("%d incompatibility(ies) found - these will fail on the Nano."
              % total_problems)
    else:
        print("No 3.6 incompatibilities. Every device-side file should import "
              "on JetPack 4.6.")
    if reminders:
        print("%d reminder(s): dataclasses is stdlib from 3.7, so the device "
              "needs `pip3 install dataclasses`." % reminders)
    print("\nThis checks syntax and the standard library, not third-party APIs.")
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(main())
