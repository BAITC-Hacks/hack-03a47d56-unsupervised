"""Public command-line entrypoint checks, with no external services."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MAIN = Path(__file__).resolve().parents[1] / "main.py"


class EntrypointTests(unittest.TestCase):
    def run_command(self, *arguments):
        return subprocess.run(
            [sys.executable, "-X", "utf8", str(MAIN), *arguments],
            cwd=tempfile.gettempdir(),
            env={**os.environ, "OPENAI_API_KEY": ""},
            capture_output=True, encoding="utf-8", timeout=10,
        )

    def test_help_and_subcommand_help_work_outside_project_directory(self):
        for arguments in (("--help",), ("serve", "--help"),
                          ("recommend", "--help"), ("filter", "--help")):
            with self.subTest(arguments=arguments):
                completed = self.run_command(*arguments)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout.lower())
                self.assertNotIn("Traceback", completed.stderr)
        root_help = self.run_command("--help").stdout
        for command in ("serve", "recommend", "filter"):
            self.assertIn(command, root_help)

    def test_unknown_command_fails_with_usage_instead_of_starting_server(self):
        completed = self.run_command("unknown-command")
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("unknown-command", completed.stderr)
        self.assertIn("usage:", completed.stderr.lower())
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
