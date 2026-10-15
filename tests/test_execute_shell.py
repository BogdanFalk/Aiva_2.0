"""execute_shell runs PowerShell (it used to be cmd.exe via shell=True, where every `$var` failed)."""
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell only")


def test_powershell_syntax_runs():
    from modules.tools import execute_shell
    r = execute_shell("$p = 'hello'; if (-not (Test-Path -LiteralPath 'Z:\\no\\such')) { Write-Output $p }")
    assert r["success"], r
    assert r["stdout"].strip() == "hello"


def test_failure_carries_the_error_text():
    from modules.tools import execute_shell
    r = execute_shell("throw 'Exact folder not found: X'")
    assert not r["success"] and r["returncode"] != 0
    assert "Exact folder not found" in r["stderr"]
