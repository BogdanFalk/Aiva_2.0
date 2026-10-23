import os

# Must be set before any QApplication exists: widgets render into memory and
# no window ever reaches the screen.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(["drawapp-tests"])


@pytest.fixture(autouse=True)
def _no_real_pictures(tmp_path_factory, monkeypatch):
    """Every window saves under `~` by default (storage.default_save_dir); in tests `~` is a temp
    folder, for this process and any child it launches, so nothing lands in the real Pictures."""
    home = str(tmp_path_factory.mktemp("home"))
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.setenv("HOME", home)
