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

    def test_stored_bytes_do_not_depend_on_the_producing_os(self):
        stored = []
        for paths, root in ((ntpath, "D:\\producer"), (posixpath, "/producer")):
            with patch.object(runner.os, "path", paths):
                value = paths.join(root, "buck-out", "bzip2", "include")
                stored.append(json.dumps(runner.encode_metadata_value(value, root)))
        self.assertEqual(stored[0], stored[1])
        self.assertEqual(json.loads(stored[0]), {TAG: "buck-out/bzip2/include"})

    def test_project_relative_is_the_same_on_every_os(self):
        for paths, root in ((ntpath, "D:\\producer\\"), (posixpath, "/producer/")):
            with self.subTest(paths=paths.__name__), patch.object(
                runner.os, "path", paths
            ):
                self.assertEqual(
                    runner.project_relative(paths.join(root, "out", "lib"), root),
                    "out/lib",
                )
                self.assertIsNone(
                    runner.project_relative(
                        paths.join(root.rstrip("\\/") + "_old", "lib"), root
                    )
                )
                self.assertIsNone(runner.project_relative("out", root))
        with patch.object(runner.os, "path", ntpath):
            for spelled in ("D:/producer/out/lib", "d:\\PRODUCER\\out\\lib"):
                with self.subTest(spelled=spelled):
                    self.assertEqual(
                        runner.project_relative(spelled, "D:\\producer\\"), "out/lib"
                    )

    def test_windows_consumer_gets_a_native_path(self):
        with patch.object(runner.os, "path", ntpath):
            self.assertEqual(
                runner.decode_metadata_value(
                    {TAG: "buck-out/bzip2/include"}, "C:\\consumer\\"
                ),
                "C:\\consumer\\buck-out\\bzip2\\include",
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
    def run_main(self, root, extra_env, script_output, on_env=lambda env: None, target="test"):
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
                "TARGET": target,
                "CARGO_MANIFEST_LINKS": "bzip2",
            },
        ):
            runner.main()
        self.flags = args.outfile.getvalue()
        return json.loads(outenv.getvalue()), outenv.getvalue()

    def test_flag_paths_are_project_relative_with_forward_slashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "out" / "gen.rs"
            lib = root / "out" / "lib"
            self.run_main(
                root,
                [],
                f"cargo:rustc-env=GENERATED={generated}\n"
                f"cargo::rustc-link-search=native={lib}\n"
                f"cargo:rustc-link-search={lib}\n",
            )
            self.assertIn("--env-set=GENERATED=$(abspath out/gen.rs)\n", self.flags)
            self.assertIn("-Lnative=out/lib\n", self.flags)
            self.assertIn("-Lout/lib\n", self.flags)
            self.assertNotIn(str(root), self.flags)
            # Expanded against the consuming root, as rustc_action.py does.
            self.assertEqual(
                os.path.abspath(os.path.join(root, "out/gen.rs")), str(generated)
            )

    def test_root_spelled_with_forward_slashes_is_inside(self):
        # Native build tools on Windows often print `D:/...`; a string prefix
        # match against `D:\...` missed it and dropped the search path.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spelled = str(root).replace("\\", "/") + "/out/lib"
            self.run_main(root, [], f"cargo:rustc-link-search=native={spelled}\n")
            self.assertEqual(self.flags, "-Lnative=out/lib\n")

    def test_sibling_with_root_as_string_prefix_is_outside(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sibling = Path(f"{root}_old")
            self.run_main(
                root,
                [],
                f"cargo:rustc-env=GENERATED={sibling / 'gen.rs'}\n"
                f"cargo:rustc-link-search=native={sibling / 'lib'}\n",
            )
            self.assertIn(f"--env-set=GENERATED={sibling / 'gen.rs'}\n", self.flags)
            self.assertNotIn("$(abspath", self.flags)
            self.assertNotIn("-L", self.flags)

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
                values["DEP_BZIP2_INCLUDE"], {TAG: "buck-out/bzip2/include"}
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
                        # "/" on every OS; the consumer gets a native path.
                        "DEP_BZIP2_INCLUDE": {TAG: "buck-out/bzip2/include"},
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

    def test_msvc_build_script_sees_the_root_trim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seen = {}
            with patch.dict(os.environ, {}, clear=False):
                for name in ("CFLAGS", "CXXFLAGS"):
                    os.environ.pop(name, None)
                self.run_main(
                    root, [], "", on_env=seen.update, target="x86_64-pc-windows-msvc"
                )
            flag = "/d1trimfile:" + os.path.join(str(root), "")
            self.assertEqual(seen["CFLAGS"], flag)
            self.assertEqual(seen["CXXFLAGS"], flag)


class TestTrimMsvcSourceRoot(unittest.TestCase):
    ROOT = "D:\\ws\\"
    FLAG = "/d1trimfile:D:\\ws\\"
    TARGET = "x86_64-pc-windows-msvc"

    def test_plain_names_are_set_when_nothing_is(self):
        env = {}
        runner.trim_msvc_source_root(env, self.TARGET, self.ROOT)
        self.assertEqual(env, {"CFLAGS": self.FLAG, "CXXFLAGS": self.FLAG})

    def test_every_set_name_keeps_its_value_and_gains_the_flag(self):
        # `cc` reads the most specific name set; each one must carry the flag.
        env = {
            "CFLAGS_x86_64-pc-windows-msvc": "/W3",
            "CFLAGS_x86_64_pc_windows_msvc": "/W3",
            "TARGET_CFLAGS": "/W3",
            "HOST_CFLAGS": "/W3",
            "CFLAGS": "/W4",
        }
        runner.trim_msvc_source_root(env, self.TARGET, self.ROOT)
        for name, value in env.items():
            if name.startswith(("CXX",)):
                continue
            self.assertTrue(value.endswith(" " + self.FLAG), name)
            self.assertTrue(value.startswith("/W"), name)

    def test_non_msvc_targets_are_untouched(self):
        for target in ("x86_64-unknown-linux-gnu", "x86_64-pc-windows-gnu", "aarch64-apple-darwin"):
            env = {"CFLAGS": "-O2"}
            runner.trim_msvc_source_root(env, target, self.ROOT)
            self.assertEqual(env, {"CFLAGS": "-O2"}, target)

    def test_root_with_whitespace_is_left_untrimmed(self):
        env = {}
        with patch.object(runner, "eprint") as eprint:
            runner.trim_msvc_source_root(env, self.TARGET, "C:\\My Work\\ws\\")
        self.assertEqual(env, {})
        self.assertIn("whitespace", eprint.call_args[0][0])

    def test_root_without_trailing_separator_gets_one(self):
        env = {}
        runner.trim_msvc_source_root(env, self.TARGET, os.path.join("D:", "ws"))
        self.assertTrue(env["CFLAGS"].endswith(os.sep))


if __name__ == "__main__":
    unittest.main()
