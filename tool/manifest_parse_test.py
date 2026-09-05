#!/usr/bin/env python3
"""Tests for manifest_parse's `[lints]` -> rustc flag translation.

`cargo buckal migrate` never reads a manifest's `[lints]` /
`[workspace.lints]` table, so the generated targets used to build with no lint
policy at all: a declared `deny` did not deny. The translation lives here now,
and what is pinned below is the part that is easy to get subtly wrong and
worse than not implementing at all -- cargo's `priority` ordering, which is
what keeps a group-level `deny` from swallowing the specific overrides that
follow it, and workspace inheritance, which is how every member opts in.

Run directly: `python3 tool/manifest_parse_test.py`
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest_parse as mp  # noqa: E402


def flags(package_toml, workspace_toml=None):
    """Translate a parsed member manifest, with an optional workspace root."""
    return mp.lint_flags(package_toml, lambda: workspace_toml)


WORKSPACE = {
    "workspace": {
        "lints": {
            "rust": {"unsafe_code": "deny", "dead_code": "warn"},
            "clippy": {"collapsible_if": "warn"},
        }
    }
}


class TestInheritance(unittest.TestCase):
    def test_workspace_true_resolves_against_workspace_lints(self):
        self.assertEqual(
            flags({"lints": {"workspace": True}}, WORKSPACE),
            ["--deny=unsafe_code", "--warn=dead_code", "--warn=clippy::collapsible_if"],
        )

    def test_own_table_is_used_without_inheritance(self):
        # A member declaring its own table never consults the workspace, even
        # when one is available.
        self.assertEqual(
            flags({"lints": {"rust": {"unsafe_code": "forbid"}}}, WORKSPACE),
            ["--forbid=unsafe_code"],
        )

    def test_opted_out_member_emits_nothing(self):
        self.assertEqual(flags({}, WORKSPACE), [])
        self.assertEqual(flags({"lints": {"workspace": False}}, WORKSPACE), [])

    def test_no_workspace_manifest_skips_rather_than_raises(self):
        # `cargo_manifest` omits --workspace for every vendored third-party
        # target. Raising here would break each one that ever opts in.
        self.assertEqual(flags({"lints": {"workspace": True}}, None), [])

    def test_workspace_without_a_lints_table_skips(self):
        self.assertEqual(flags({"lints": {"workspace": True}}, {"workspace": {}}), [])


class TestOrdering(unittest.TestCase):
    def test_lower_priority_is_emitted_first(self):
        # The shape the ordering exists for: a group-level deny that specific
        # entries are meant to override. Emitted the other way round, the deny
        # wins and the overrides are dead.
        self.assertEqual(
            flags({"lints": {"clippy": {
                "correctness": {"level": "deny", "priority": -1},
                "eq_op": "allow",
            }}}),
            ["--deny=clippy::correctness", "--allow=clippy::eq_op"],
        )

    def test_ties_break_on_descending_lint_name(self):
        # cargo's `Reverse(name)` tiebreak, which exists so the `all` group
        # lands last among same-priority entries rather than first.
        self.assertEqual(
            flags({"lints": {"clippy": {"all": "warn", "eq_op": "deny"}}}),
            ["--deny=clippy::eq_op", "--warn=clippy::all"],
        )

    def test_priority_orders_across_tools(self):
        self.assertEqual(
            flags({"lints": {
                "clippy": {"correctness": {"level": "deny", "priority": -1}},
                "rust": {"unsafe_code": "deny"},
            }}),
            ["--deny=clippy::correctness", "--deny=unsafe_code"],
        )


class TestCargoParity(unittest.TestCase):
    def test_matches_cargo_on_a_mixed_table(self):
        # The expectation is not derived -- it is the flag sequence `cargo
        # build -v` emitted for this exact manifest (cargo 1.95.0), captured
        # so a future refactor of the sort key has something to fail against.
        self.assertEqual(
            flags({"lints": {"workspace": True}}, {"workspace": {"lints": {
                "rust": {"unsafe_code": "deny", "dead_code": "warn"},
                "clippy": {
                    "correctness": {"level": "deny", "priority": -1},
                    "collapsible_if": "warn",
                    "all": "warn",
                    "eq_op": "allow",
                },
                "rustdoc": {"broken_intra_doc_links": "deny"},
            }}}),
            ["--deny=clippy::correctness",
             "--deny=unsafe_code",
             "--allow=clippy::eq_op",
             "--warn=dead_code",
             "--warn=clippy::collapsible_if",
             "--deny=rustdoc::broken_intra_doc_links",
             "--warn=clippy::all"],
        )


class TestNamespacing(unittest.TestCase):
    def test_rust_table_is_unprefixed_and_tools_are_namespaced(self):
        self.assertEqual(
            flags({"lints": {
                "rust": {"unsafe_code": "deny"},
                "clippy": {"eq_op": "deny"},
                "rustdoc": {"broken_intra_doc_links": "deny"},
            }}),
            # Ordering is the shared `Reverse(name)` tiebreak, not the tool --
            # the namespace prefix is not part of the sort key.
            [
                "--deny=unsafe_code",
                "--deny=clippy::eq_op",
                "--deny=rustdoc::broken_intra_doc_links",
            ],
        )

    def test_every_level_maps_to_its_flag(self):
        self.assertEqual(
            flags({"lints": {"rust": {
                "a_lint": "allow",
                "b_lint": "warn",
                "c_lint": "force-warn",
                "d_lint": "deny",
                "e_lint": "forbid",
            }}}),
            [
                "--forbid=e_lint",
                "--deny=d_lint",
                "--force-warn=c_lint",
                "--warn=b_lint",
                "--allow=a_lint",
            ],
        )


class TestRejects(unittest.TestCase):
    def test_unknown_level_raises(self):
        with self.assertRaises(ValueError):
            flags({"lints": {"rust": {"unsafe_code": "nope"}}})

    def test_non_integer_priority_raises(self):
        with self.assertRaises(ValueError):
            flags({"lints": {"rust": {"unsafe_code": {"level": "deny", "priority": "hi"}}}})


class TestResponseFileGeneration(unittest.TestCase):
    """End-to-end over a fixture workspace: what lands in the response file."""

    def test_fixture_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text(textwrap.dedent("""\
                [workspace]
                members = ["crates/foo"]

                [workspace.package]
                version = "1.2.3"

                [workspace.lints.rust]
                unsafe_code = "deny"

                [workspace.lints.clippy]
                correctness = { level = "deny", priority = -1 }
                collapsible_if = "warn"
                """), encoding="utf-8")
            member = root / "crates" / "foo"
            member.mkdir(parents=True)
            (member / "Cargo.toml").write_text(textwrap.dedent("""\
                [package]
                name = "foo"
                version = { workspace = true }

                [lints]
                workspace = true
                """), encoding="utf-8")

            out_flags = root / "flags.txt"
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parent / "manifest_parse.py"),
                 f"--vendor={member}",
                 f"--workspace={root / 'Cargo.toml'}",
                 f"--out-dict={root / 'dict.json'}",
                 f"--out-flags={out_flags}"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

            lines = out_flags.read_text(encoding="utf-8").splitlines()
            # The pre-existing env surface still comes first and is unchanged.
            self.assertTrue(lines[0].startswith("--env-set=CARGO_MANIFEST_DIR="))
            self.assertIn("--env-set=CARGO_PKG_VERSION=1.2.3", lines)
            # And the lint flags now follow it, in priority order.
            self.assertEqual(
                [line for line in lines if not line.startswith("--env-set=")],
                ["--deny=clippy::correctness",
                 "--deny=unsafe_code",
                 "--warn=clippy::collapsible_if"],
            )


class TestCheckoutRootIndependence(unittest.TestCase):
    """The emitted artifacts must not name the checkout that produced them.

    The action's digest does not include the buck2 project root, so two
    checkouts on one host mint the same digest for these outputs. That is only
    correct if the outputs are genuinely identical -- when they carried an
    absolute `CARGO_MANIFEST_DIR`, whichever checkout populated a shared cache
    first served its own tree's paths to the other, silently, for every crate
    reading `env!("CARGO_MANIFEST_DIR")`. Measured at 218 of 967 artifacts on a
    four-checkout box before this fix.
    """

    VENDOR_REL = Path("buck-out") / "vendor"

    MANIFEST = textwrap.dedent(
        """
        [package]
        name = "demo"
        version = "1.2.3"
        readme = "README.md"
        license-file = "LICENSE.txt"
        """
    ).strip()

    def _emit(self, root: Path):
        """Run the tool with `root` as cwd; return (flags_text, dict_text)."""
        vendor = root / self.VENDOR_REL
        vendor.mkdir(parents=True)
        (vendor / "Cargo.toml").write_text(self.MANIFEST, encoding="utf-8")
        (vendor / "README.md").write_text("readme", encoding="utf-8")
        (vendor / "LICENSE.txt").write_text("license", encoding="utf-8")

        out_flags = root / "ENV_FLAGS"
        out_dict = root / "ENV_DICT"
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "manifest_parse.py"),
                f"--vendor={self.VENDOR_REL}",
                f"--out-dict={out_dict}",
                f"--out-flags={out_flags}",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return (
            out_flags.read_text(encoding="utf-8"),
            out_dict.read_text(encoding="utf-8"),
        )

    def test_two_checkouts_emit_identical_bytes(self):
        # The property the cache key actually rests on.
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            flags_a, dict_a = self._emit(Path(a))
            flags_b, dict_b = self._emit(Path(b))
        self.assertEqual(flags_a, flags_b)
        self.assertEqual(dict_a, dict_b)

    def test_no_absolute_path_is_emitted(self):
        with tempfile.TemporaryDirectory() as root:
            flags_text, dict_text = self._emit(Path(root))
        # The temp root is what a real checkout root stands in for here.
        for blob, name in ((flags_text, "ENV_FLAGS"), (dict_text, "ENV_DICT")):
            self.assertNotIn(str(Path(root)), blob, f"{name} names the producing root")

    def test_path_vars_are_wrapped_for_the_consuming_action(self):
        # rustc still needs an absolute path; `rustc_action.py` expands
        # `$(abspath ...)` against the *consuming* checkout's root.
        with tempfile.TemporaryDirectory() as root:
            flags_text, _ = self._emit(Path(root))
        for key in ("CARGO_MANIFEST_DIR", "CARGO_MANIFEST_PATH",
                    "CARGO_PKG_README", "CARGO_PKG_LICENSE_FILE"):
            self.assertIn(f"--env-set={key}=$(abspath ", flags_text)

    def test_non_path_vars_are_left_unwrapped(self):
        with tempfile.TemporaryDirectory() as root:
            flags_text, _ = self._emit(Path(root))
        self.assertIn("--env-set=CARGO_PKG_VERSION=1.2.3\n", flags_text)
        self.assertNotIn("CARGO_PKG_VERSION=$(abspath", flags_text)

    def test_dict_carries_the_bare_relative_path(self):
        # buildscript_run.py absolutizes these; `$(abspath ...)` is a
        # rustc_action mechanism and would arrive there as a literal string.
        with tempfile.TemporaryDirectory() as root:
            _, dict_text = self._emit(Path(root))
        parsed = json.loads(dict_text)
        self.assertNotIn("$(abspath", dict_text)
        for key in ("CARGO_MANIFEST_DIR", "CARGO_PKG_README"):
            self.assertFalse(
                os.path.isabs(parsed[key]), f"{key} is absolute in ENV_DICT"
            )


if __name__ == "__main__":
    unittest.main()
