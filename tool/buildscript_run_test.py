#!/usr/bin/env python3
"""Tests for buildscript_run's `links` metadata path encoding.

A build script's `DEP_*` metadata lands in the METADATA action output, which
is cached. An absolute path stored there verbatim is replayed into whichever
checkout gets the cache hit, so a dependent build script in checkout B is
pointed at checkout A's tree (e.g. an include dir that no longer exists).
What is pinned below: the cached bytes are identical across checkout roots,
the consumer resolves them against its own root, text that is not a
project-local path is untouched, and malformed values are refused.

Run directly: `python3 tool/buildscript_run_test.py`
"""

import io
import json
import ntpath
import os
import posixpath
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import buildscript_run as runner  # noqa: E402

TAG = runner.PROJECT_RELATIVE_PATH_TAG


class TestEncodeDecode(unittest.TestCase):
    def test_cached_value_is_root_independent_and_consumer_resolves_it(self):
        for paths, producer, consumer in (
            (ntpath, "D:\\producer", "C:\\consumer"),
            (posixpath, "/producer", "/consumer"),
        ):
            with self.subTest(paths=paths.__name__), patch.object(
                runner.os, "path", paths
            ):
                suffix = paths.join("buck-out", "bzip2", "include")
                encoded = runner.encode_metadata_value(
                    paths.join(producer, suffix), producer
                )
                self.assertEqual(
                    encoded,
                    runner.encode_metadata_value(
                        paths.join(consumer, suffix), consumer
                    ),
                )
                self.assertNotIn(producer, json.dumps(encoded))
                self.assertEqual(
                    runner.decode_metadata_value(
                        json.loads(json.dumps(encoded)), consumer
                    ),
                    paths.join(consumer, suffix),
                )

    def test_root_with_trailing_separator(self):
        # TOOL_CWD is spelled with a trailing separator.
        with patch.object(runner.os, "path", posixpath):
            self.assertEqual(
                runner.encode_metadata_value("/producer/out", "/producer/"),
                {TAG: "out"},
            )

    def test_windows_containment_ignores_case(self):
        with patch.object(runner.os, "path", ntpath):
            self.assertEqual(
                runner.encode_metadata_value("d:\\PRODUCER\\out", "D:\\producer\\"),
                {TAG: "out"},
            )

    def test_non_project_values_are_unchanged(self):
        with patch.object(runner.os, "path", posixpath):
            for value in (
                "1.2.3",
                "include",
                "",
                "/external/include",
                # A string-prefix match would wrongly claim this sibling.
                "/producer-sibling/include",
                # Lists and embedded paths are not rewritten.
                "/producer/a:/producer/b",
                "-I/producer/include",
            ):
                with self.subTest(value=value):
                    self.assertEqual(
                        runner.encode_metadata_value(value, "/producer"), value
                    )
                    self.assertEqual(
                        runner.decode_metadata_value(value, "/consumer"), value
                    )
        with patch.object(runner.os, "path", ntpath):
            for value in ("E:\\include", "D:\\producer\\a;D:\\producer\\b"):
                with self.subTest(value=value):
                    self.assertEqual(
                        runner.encode_metadata_value(value, "D:\\producer"), value
                    )

    def test_malformed_values_are_refused(self):
        with patch.object(runner.os, "path", posixpath):
            for value in (
                {TAG: "../outside"},
                {TAG: "out/../../outside"},
                {TAG: "/outside"},
                {TAG: 1},
                {TAG: "out", "extra": "x"},
                {"unexpected": "value"},
                ["list"],
                None,
            ):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    runner.decode_metadata_value(value, "/consumer")


class TestMain(unittest.TestCase):
    def run_main(self, root, extra_env, script_output, on_env=lambda env: None):
        outenv = io.StringIO()
        args = SimpleNamespace(
            extra_env=[str(p) for p in extra_env],
            rustc_cfg=None,
            create_cwd=None,
            manifest_dir=None,
            buildscript="fake",
            outfile=io.StringIO(),
            outenv=outenv,
        )

        def run_buildscript(*_args, env, **_kwargs):
            on_env(env)
            return script_output

        with patch.object(runner, "arg_parse", return_value=args), patch.object(
            runner, "TOOL_CWD", os.path.join(str(root), "")
        ), patch.object(runner, "cfg_env", return_value={}), patch.object(
            runner, "create_cwd", return_value=root
        ), patch.object(runner, "ensure_rustc_available"), patch.object(
            runner, "run_buildscript", side_effect=run_buildscript
        ), patch.dict(
            os.environ,
            {
                "OUT_DIR": str(root / "out"),
                "TARGET": "test",
                "CARGO_MANIFEST_LINKS": "bzip2",
            },
        ):
            runner.main()
        return json.loads(outenv.getvalue()), outenv.getvalue()

    def test_both_syntaxes_are_encoded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = root / "buck-out" / "bzip2" / "include"
            values, raw = self.run_main(
                root,
                [],
                f"cargo:include={include}\n"
                f"cargo::metadata=root={root / 'buck-out'}\n"
                "cargo::metadata=version=1.0.8\n",
            )
            self.assertEqual(
                values["DEP_BZIP2_INCLUDE"],
                {TAG: os.path.join("buck-out", "bzip2", "include")},
            )
            self.assertEqual(values["DEP_BZIP2_ROOT"], {TAG: "buck-out"})
            self.assertEqual(values["DEP_BZIP2_VERSION"], "1.0.8")
            self.assertNotIn(json.dumps(str(root))[1:-1], raw)

    def test_consumer_reads_its_own_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = root / "buck-out" / "bzip2" / "include"
            include.mkdir(parents=True)
            (include / "bzlib.h").write_text("header", encoding="utf-8")
            incoming = root / "METADATA"
            # As a producer in another checkout would have written it.
            incoming.write_text(
                json.dumps(
                    {
                        "DEP_BZIP2_INCLUDE": {
                            TAG: os.path.join("buck-out", "bzip2", "include")
                        },
                        "DEP_BZIP2_VERSION": "1.0.8",
                    }
                ),
                encoding="utf-8",
            )
            seen = {}

            def on_env(env):
                seen.update(env)

            self.run_main(root, [incoming], "", on_env)
            self.assertEqual(seen["DEP_BZIP2_INCLUDE"], str(include))
            self.assertTrue((Path(seen["DEP_BZIP2_INCLUDE"]) / "bzlib.h").is_file())
            self.assertEqual(seen["DEP_BZIP2_VERSION"], "1.0.8")

    def test_malformed_metadata_fails_the_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming = root / "METADATA"
            incoming.write_text(
                json.dumps({"DEP_BZIP2_INCLUDE": {TAG: "../outside"}}),
                encoding="utf-8",
            )
            with patch.object(runner, "eprint") as eprint, self.assertRaises(
                SystemExit
            ) as exit_:
                self.run_main(root, [incoming], "")
            self.assertEqual(exit_.exception.code, 1)
            self.assertIn("DEP_BZIP2_INCLUDE", eprint.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
