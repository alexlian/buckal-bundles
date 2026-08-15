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


if __name__ == "__main__":
    unittest.main()
