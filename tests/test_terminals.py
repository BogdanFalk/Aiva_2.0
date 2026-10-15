"""Offline tests for the terminal readback fix: the deterministic tee-to-file + done-marker capture
that replaced buffered Start-Transcript. No real window — only the pure capture logic."""
import modules.terminals as term
from modules.terminals import TerminalManager


def tm():
    return TerminalManager()


def test_wrap_dot_sources_and_tees_with_marker():
    w = tm()._wrap("git status; ls", "C:\\tmp\\cmd-x-1.out", "AIVADONE1X000001")
    # dot-source (not &) so cd/$vars persist; the marker is emitted last; output is teed to the file
    assert w.startswith(". { git status; ls ; 'AIVADONE1X000001' }")
    assert "2>&1 | Tee-Object -FilePath 'C:\\tmp\\cmd-x-1.out'" in w
    assert "& {" not in w                                   # never a child scope


def test_read_capture_decodes_utf16_and_utf8(tmp_path):
    t = tm()
    u16 = tmp_path / "a.out"
    u16.write_bytes(b"\xff\xfe" + "git ok\nAIVADONE1X1".encode("utf-16-le"))
    assert "git ok" in t._read_capture(str(u16)) and "AIVADONE1X1" in t._read_capture(str(u16))

    u8 = tmp_path / "b.out"
    u8.write_text("plain utf8 output\nAIVADONE2X2", encoding="utf-8")
    assert "plain utf8 output" in t._read_capture(str(u8))

    bom8 = tmp_path / "c.out"
    bom8.write_bytes(b"\xef\xbb\xbf" + "bom utf8".encode("utf-8"))
    assert t._read_capture(str(bom8)).strip() == "bom utf8"    # BOM stripped, not shown as text


def test_read_capture_tail_of_utf16_still_finds_marker(tmp_path):
    # a big UTF-16 file read only at the tail must still decode cleanly and see the end marker
    t = tm()
    big = tmp_path / "big.out"
    body = ("x" * 5000) + "\nTAILMARKER99"
    big.write_bytes(b"\xff\xfe" + body.encode("utf-16-le"))
    out = t._read_capture(str(big), tail_bytes=64)             # force the tail path
    assert "TAILMARKER99" in out and "\x00" not in out         # aligned decode, no stray nulls


def test_finish_capture_strips_marker_and_prompt_noise():
    t = tm()
    raw = "PS C:\\repo> ignored prompt\nreal output line\nAIVADONE3X3"
    cleaned = t._finish_capture(raw, "AIVADONE3X3")
    assert "real output line" in cleaned
    assert "AIVADONE3X3" not in cleaned and "ignored prompt" not in cleaned


async def test_await_capture_returns_done_when_marker_present(tmp_path):
    t = tm()
    out = tmp_path / "done.out"
    out.write_bytes(b"\xff\xfe" + "the answer is 42\nDONEMARK".encode("utf-16-le"))
    text, done = await t._await_capture(str(out), "DONEMARK")
    assert done is True and "the answer is 42" in text


async def test_await_capture_times_out_without_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(term, "CAPTURE_TIMEOUT", 0.4)          # don't wait the real 20s
    t = tm()
    out = tmp_path / "running.out"
    out.write_bytes(b"\xff\xfe" + "still going".encode("utf-16-le"))   # no marker yet
    text, done = await t._await_capture(str(out), "NEVERCOMES")
    assert done is False and "still going" in text            # returns the partial, flagged not-done


def test_read_session_reports_running_vs_done(tmp_path):
    t = tm()
    t.sessions["s"] = {"cwd": "C:\\repo", "history": [], "log_path": str(tmp_path / "sess.log"),
                       "hwnd": None}
    # nothing run yet
    assert "nothing run" in t.read_session("s")["recent_output"]
    # a finished command
    done = tmp_path / "cmd-s-1.out"
    done.write_bytes(b"\xff\xfe" + "branch created\nMARKA".encode("utf-16-le"))
    t.sessions["s"].update({"last_out": str(done), "last_marker": "MARKA"})
    r = t.read_session("s")
    assert "branch created" in r["recent_output"] and "still running" not in r["recent_output"]
    # a still-running command (no marker in the file)
    run = tmp_path / "cmd-s-2.out"
    run.write_bytes(b"\xff\xfe" + "working...".encode("utf-16-le"))
    t.sessions["s"].update({"last_out": str(run), "last_marker": "MARKB"})
    assert "still running" in t.read_session("s")["recent_output"]


def test_a_job_that_changes_code_runs_in_auto_mode_unless_set_otherwise(monkeypatch):
    monkeypatch.delenv("AIVA_CLAUDE_PERMISSION_MODE", raising=False)
    assert term.execute_permission_mode() == "auto"
    monkeypatch.setenv("AIVA_CLAUDE_PERMISSION_MODE", "acceptEdits")
    assert term.execute_permission_mode() == "acceptEdits"
    for unchecked in ("bypassPermissions", "dontAsk", "nonsense"):
        monkeypatch.setenv("AIVA_CLAUDE_PERMISSION_MODE", unchecked)
        assert term.execute_permission_mode() == "auto"      # never a mode without the safety check


async def test_start_claude_passes_the_mode_and_refuses_his_whole_home(tmp_path, monkeypatch):
    monkeypatch.delenv("AIVA_CLAUDE_PERMISSION_MODE", raising=False)
    launched = []

    class Proc:
        class stdin:
            write = staticmethod(lambda data: None)
            close = staticmethod(lambda: None)

    async def fake_exec(*args, **kw):
        launched.append(args)
        return Proc()

    monkeypatch.setattr(term.asyncio, "create_subprocess_exec", fake_exec)
    t = tm()
    monkeypatch.setattr(t, "_track_job", lambda *a, **kw: None)
    for where in (".", "~", ""):
        r = await t.start_claude(job_name="j", project_dir=where, prompt="fix it", mode="execute")
        assert r["success"] is False and "home" in r["error"]
    assert launched == []
    await t.start_claude(job_name="e", project_dir=str(tmp_path), prompt="fix it", mode="execute")
    await t.start_claude(job_name="p", project_dir=str(tmp_path), prompt="look", mode="plan")
    modes = [a[a.index("--permission-mode") + 1] for a in launched]
    assert modes == ["auto", "plan"]
