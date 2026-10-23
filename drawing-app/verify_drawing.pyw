"""The clean-launch check with no terminal: double-click this file, or open it from code
(os.startfile, Aiva's open_file tool). Windows runs .pyw files with pyw.exe, which never
opens a console.

It runs `drawapp.verify` (launch() + DrawClient, so application code all the way) and leaves
the drawing window open, without focus, showing an Aiva stroke and a Saga stroke. The JSON
report goes to verify-report.json next to this file, and "ok": true means every check passed.
Close the window when you've seen it.

pyw.exe may pick an interpreter without PyQt5. This script only needs the stdlib, and the
window itself runs on the repo's .venv. Arguments, if any, replace the defaults
(e.g. `--offscreen --report out.json`).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "verify-report.json")


def main() -> int:
    sys.path.insert(0, HERE)
    from drawapp import launcher, verify

    args = sys.argv[1:] or ["--keep-open", "--report", REPORT]
    python = launcher.project_python()
    if python and "--python" not in args:
        args += ["--python", python]
    return verify.main(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # pyw has nowhere to show a traceback: leave it in the report
        import json
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": f"{type(e).__name__}: {e}"}, f)
        sys.exit(1)
