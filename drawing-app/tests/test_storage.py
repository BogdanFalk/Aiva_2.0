"""Saving and getting drawings back (storage.py): the file format, autosave + restore, the owner
lock, clear/open keeping what they replace, and the wire commands on top. Pure Python except the
window tests at the end (offscreen)."""
import json
import os

import pytest

from drawapp import ops, storage
from drawapp.commands import CommandRunner, parse
from drawapp.model import Document
from drawapp.storage import EXT, Session, StorageError, load_into, read_drawing, write_drawing
from drawapp.tools import BrushTool


def draw(doc, author="aiva", points=((10, 10), (60, 40), (90, 20))):
    tool = BrushTool(doc, author)
    tool.press(*points[0])
    for p in points[1:]:
        tool.move(*p)
    tool.release(*points[-1])
    return doc.strokes[-1].id


def shape(doc):
    return [(s.id, s.author, s.brush.color, s.brush.width, s.points) for s in doc.strokes]


def session(tmp_path, author="aiva", doc=None, **kwargs):
    s = Session(doc or Document(), author, str(tmp_path / "drawings"), **kwargs)
    s.start()
    return s


# --- the format ---------------------------------------------------------------------------------

def test_a_drawing_round_trips_exactly(tmp_path):
    doc = Document(300, 200)
    draw(doc, "aiva")
    draw(doc, "saga", ((100, 100),))                  # a dot is one point
    path = str(tmp_path / "x.aivadraw")
    write_drawing(path, doc)
    copy = Document(300, 200)
    load_into(copy, read_drawing(path))
    assert shape(copy) == shape(doc) and all(s.done for s in copy.strokes)
    assert not os.path.exists(path + ".tmp")


@pytest.mark.parametrize("content, problem", [
    ("{not json", "damaged"),
    ('{"format": "png"}', "not a drawapp drawing"),
    ('{"format": "aivadraw", "version": 99, "width": 10, "height": 10, "strokes": []}', "newer drawapp"),
    ('{"format": "aivadraw", "version": 1, "width": 0, "height": 10, "strokes": []}', "canvas size"),
    ('{"format": "aivadraw", "version": 1, "width": 10, "height": 10, "strokes": [{"id": "a", '
     '"author": "aiva", "color": "red", "width": 4, "points": [[1, 1]]}]}', "color"),
    ('{"format": "aivadraw", "version": 1, "width": 10, "height": 10, "strokes": [{"id": "a", '
     '"author": "aiva", "color": "#000000", "width": 4, "points": [[1, "x"]]}]}', "has a point that isn't"),
])
def test_bad_files_are_refused_by_name(tmp_path, content, problem):
    path = tmp_path / "bad.aivadraw"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(StorageError, match=problem):
        read_drawing(str(path))


def test_duplicate_stroke_ids_and_missing_files_are_refused(tmp_path):
    doc = Document()
    draw(doc)
    data = storage.to_dict(doc)
    data["strokes"].append(dict(data["strokes"][0]))
    with pytest.raises(StorageError, match="duplicate id"):
        storage.from_dict(data)
    with pytest.raises(StorageError, match="no drawing at"):
        read_drawing(str(tmp_path / "gone.aivadraw"))


def test_a_failed_write_leaves_the_old_file_whole(tmp_path, monkeypatch):
    doc = Document()
    draw(doc)
    path = str(tmp_path / "keep.aivadraw")
    write_drawing(path, doc)
    before = open(path, encoding="utf-8").read()
    draw(doc, "saga")

    def disk_full(*_a):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(storage.os, "replace", disk_full)
    with pytest.raises(StorageError, match="No space left"):
        write_drawing(path, doc)
    assert open(path, encoding="utf-8").read() == before and not os.path.exists(path + ".tmp")


def test_a_canvas_size_mismatch_changes_nothing(tmp_path):
    small = Document(100, 100)
    draw(small)
    path = str(tmp_path / "small.aivadraw")
    write_drawing(path, small)
    big = Document()
    draw(big)
    rev = big.revision
    with pytest.raises(StorageError, match="100x100"):
        load_into(big, read_drawing(path))
    assert big.revision == rev and len(big.strokes) == 1


# --- saving -------------------------------------------------------------------------------------

def test_save_writes_a_new_drawing_then_updates_it_keeping_a_backup(tmp_path):
    pngs = []
    s = session(tmp_path, png_writer=lambda doc, p: pngs.append(p) or open(p, "wb").close() or True)
    with pytest.raises(StorageError, match="empty"):
        s.save()
    draw(s.document)
    first = s.save()
    assert os.path.basename(first["path"]).startswith("aiva-") and first["path"].endswith(EXT)
    assert first["png"] == first["path"][:-len(EXT)] + ".png" and pngs == [first["png"]]
    assert s.current_path == first["path"] and not s.dirty and "Saved as" in s.describe()
    draw(s.document, "aiva", ((5, 5), (8, 8)))
    assert s.dirty and "Not saved yet" in s.describe()
    again = s.save()
    assert again["path"] == first["path"] and again["strokes"] == 2         # its own file, updated
    assert len(read_drawing(first["path"] + ".bak").strokes) == 1           # the old version kept


def test_a_named_save_never_overwrites_another_drawing(tmp_path):
    s = session(tmp_path)
    draw(s.document)
    cat = s.save("cat")
    assert os.path.basename(cat["path"]) == "cat.aivadraw"
    other = session(tmp_path, "saga")
    draw(other.document, "saga")
    second = other.save("cat")
    assert os.path.basename(second["path"]) == "cat (2).aivadraw"
    assert read_drawing(cat["path"]).strokes[0].author == "aiva"            # hers is untouched
    assert os.path.basename(s.save("../../evil\\name")["path"]) == "evil name.aivadraw"


def test_a_save_failure_is_a_storage_error_and_keeps_the_canvas(tmp_path, monkeypatch):
    s = session(tmp_path)
    draw(s.document)

    def denied(*_a, **_k):
        raise StorageError("couldn't save to X: Access is denied")

    monkeypatch.setattr(storage, "write_drawing", denied)
    with pytest.raises(StorageError, match="Access is denied"):
        s.save()
    assert len(s.document.strokes) == 1 and s.dirty and s.current_path is None


# --- the existing canvas: autosave, restore, one owner -------------------------------------------

def test_every_finished_stroke_is_autosaved_and_the_next_window_restores_it(tmp_path):
    s = session(tmp_path)
    assert s.owner and s.autosave_path.endswith(os.path.join("recovery", "aiva-autosave" + EXT))
    draw(s.document)
    draw(s.document, "aiva", ((200, 200), (220, 260)))
    assert len(read_drawing(s.autosave_path).strokes) == 2                 # no save, no close needed
    original = shape(s.document)
    s.close()

    back = Session(Document(), "aiva", s.save_dir)
    notice = back.start()
    assert "Restored your last drawing (2 strokes)" == notice and back.restored == 2
    assert shape(back.document) == original and back.dirty and back.current_path is None


def test_a_restored_canvas_remembers_its_file(tmp_path):
    s = session(tmp_path)
    draw(s.document)
    saved = s.save("tree")
    s.close()
    back = session(tmp_path)
    assert back.current_path == saved["path"] and not back.dirty
    draw(back.document)
    assert back.save()["path"] == saved["path"]                             # Ctrl+S goes back to it


def test_a_window_ended_without_close_still_restores(tmp_path):
    """TerminateProcess (launcher.DrawApp.close) runs no cleanup: the autosave is already there."""
    s = session(tmp_path)
    draw(s.document)
    storage._unlock(s._lock)                 # what the OS does when the process dies
    s._lock = None
    back = session(tmp_path)
    assert back.owner and back.restored == 1


def test_a_second_window_for_the_same_author_never_overwrites_the_first(tmp_path):
    first = session(tmp_path)
    draw(first.document)
    second = session(tmp_path)
    assert first.owner and not second.owner and second.document.strokes == []
    assert second.autosave_path.endswith(f"aiva-autosave-{os.getpid()}{EXT}")
    draw(second.document, "aiva", ((1, 1), (9, 9)))
    draw(second.document, "aiva", ((2, 2), (8, 8)))
    assert len(read_drawing(first.autosave_path).strokes) == 1
    assert len(read_drawing(second.autosave_path).strokes) == 2
    second.close()
    assert os.path.exists(second.autosave_path)                             # unsaved: kept for Open
    first.close()
    assert session(tmp_path).owner                                          # the slot is free again


def test_close_is_idempotent_and_keeps_the_owners_autosave(tmp_path):
    s = session(tmp_path)
    draw(s.document)
    s.save()
    s.close()
    s.close()
    assert os.path.exists(s.autosave_path)


def test_a_damaged_autosave_is_moved_aside_not_overwritten(tmp_path):
    s = session(tmp_path)
    s.close()
    with open(s.autosave_path, "w", encoding="utf-8") as f:
        f.write('{"format": "aivadraw", "version": 1, "width": 1024, "height": 768, "strokes": [')
    back = Session(Document(), "aiva", s.save_dir)
    notice = back.start()
    assert "Couldn't restore the last drawing" in notice and back.document.strokes == []
    [aside] = [n for n in os.listdir(back.recovery_dir) if ".unrecovered-" in n]
    with open(os.path.join(back.recovery_dir, aside), encoding="utf-8") as f:
        assert f.read().endswith('"strokes": [')                            # byte for byte
    draw(back.document)                                                     # the new autosave
    assert len(read_drawing(back.autosave_path).strokes) == 1


def test_an_autosave_failure_never_stops_the_drawing(tmp_path, monkeypatch):
    s = session(tmp_path)
    changes = []
    s.on_change = lambda sess: changes.append(sess.describe())

    def broken(*_a, **_k):
        raise StorageError("couldn't save to X: disk full")

    monkeypatch.setattr(storage, "write_drawing", broken)
    draw(s.document)
    assert len(s.document.strokes) == 1 and "autosave failed" in s.last_error
    assert changes[-1].startswith("Autosave failed")
    monkeypatch.undo()
    draw(s.document)
    assert s.last_error is None                                             # recovers by itself


def test_an_unusable_save_folder_turns_saving_off_but_drawing_works(tmp_path):
    blocker = tmp_path / "drawings"
    blocker.write_text("a file where the folder should be")
    s = Session(Document(), "aiva", str(blocker))
    assert s.start().startswith("Saving is off") and not s.owner
    draw(s.document)
    assert len(s.document.strokes) == 1


# --- clear and open keep what they replace --------------------------------------------------------

def test_clear_keeps_a_copy_that_open_brings_back(tmp_path):
    s = session(tmp_path)
    draw(s.document)
    draw(s.document, "saga", ((300, 300), (320, 330)))
    before = shape(s.document)
    s.document.apply(ops.ClearCanvas("saga"))
    assert s.document.strokes == [] and read_drawing(s.autosave_path).strokes == ()
    path = s.resolve("recovery/aiva-before-clear")
    out = s.open(path)
    assert shape(s.document) == before and out["strokes"] == 2 and out["saved_previous"] is None
    assert s.current_path is None and s.dirty                               # rescued work, unsaved


def test_open_saves_unsaved_work_first(tmp_path):
    s = session(tmp_path)
    draw(s.document)
    tree = s.save("tree")["path"]
    s.document.apply(ops.ClearCanvas("aiva"))
    draw(s.document, "aiva", ((400, 400), (450, 450)))
    sketch = shape(s.document)
    out = s.open(tree)
    assert out["saved_previous"] and out["saved_previous"] != tree
    copy = Document()
    load_into(copy, read_drawing(out["saved_previous"]))
    assert shape(copy) == sketch                                            # nothing was lost
    assert s.current_path == tree and not s.dirty and len(s.document.strokes) == 1


def test_a_bad_file_opened_over_the_canvas_changes_nothing(tmp_path):
    s = session(tmp_path)
    draw(s.document)
    bad = os.path.join(s.save_dir, "broken" + EXT)
    with open(bad, "w", encoding="utf-8") as f:
        f.write("{oops")
    rev = s.document.revision
    with pytest.raises(StorageError, match="damaged"):
        s.open(bad)
    assert s.document.revision == rev and s.dirty and len(s.document.strokes) == 1
    assert not [n for n in os.listdir(s.save_dir) if n.endswith(EXT) and n != "broken" + EXT]


def test_drawings_are_listed_newest_first_and_resolved_by_name_only(tmp_path):
    s = session(tmp_path)
    with pytest.raises(StorageError, match="no saved drawings"):
        s.resolve(None)
    draw(s.document)
    a = s.save("first")["path"]
    os.utime(a, (1, 1))                                                     # make it clearly older
    b = s.save("second")["path"]
    listed = s.drawings()
    assert [d["name"] for d in listed["drawings"]] == ["second", "first"]
    assert "recovery/aiva-autosave" in [d["name"] for d in listed["recovery"]]
    assert s.resolve(None) == b and s.resolve("FIRST.aivadraw") == a
    with pytest.raises(StorageError, match="no saved drawing called '../x'.*second, first"):
        s.resolve("../x")


# --- the wire commands --------------------------------------------------------------------------

def test_save_open_drawings_and_status_commands(tmp_path):
    s = session(tmp_path)
    runner = CommandRunner(s.document, s)
    assert runner.handle({"cmd": "stroke", "author": "aiva", "points": [[10, 10], [50, 50]]})["ok"]
    saved = runner.handle({"cmd": "save", "author": "aiva", "name": "wave"})
    assert saved["ok"] and saved["name"] == "wave" and os.path.isfile(saved["path"])
    status = runner.handle({"cmd": "status"})
    assert status == {"ok": True, "pid": os.getpid(), "author": "aiva", "revision": s.document.revision,
                      "strokes": 1, "save_dir": s.save_dir, "current": saved["path"], "unsaved": False}
    runner.handle({"cmd": "clear", "author": "aiva"})
    opened = runner.handle({"cmd": "open", "author": "aiva", "name": "wave"})
    assert opened["ok"] and opened["strokes"] == 1 and len(s.document.strokes) == 1
    listed = runner.handle({"cmd": "drawings"})
    assert listed["ok"] and [d["name"] for d in listed["drawings"]] == ["wave"]
    missing = runner.handle({"cmd": "open", "name": "nope"})
    assert missing == {"ok": False, "code": "storage",
                       "error": "there is no saved drawing called 'nope' (saved: wave)"}
    assert runner.handle({"cmd": "save", "name": ""})["code"] == "invalid"
    assert runner.handle({"cmd": "save", "name": "a\nb"})["code"] == "invalid"
    assert runner.handle({"cmd": "drawings", "author": "aiva"})["code"] == "invalid"


def test_without_a_session_saving_is_unsupported_but_status_answers():
    runner = CommandRunner(Document())
    assert runner.handle({"cmd": "save"})["code"] == "unsupported"
    assert runner.handle({"cmd": "status"})["author"] is None
    assert parse({"cmd": "open"}).name is None


# --- the window (offscreen) ---------------------------------------------------------------------

def test_the_window_saves_from_its_toolbar_and_shows_where(qapp, tmp_path):
    from drawapp.app import MainWindow, save_png

    doc = Document(200, 120)
    s = Session(doc, "saga", str(tmp_path / "drawings"), png_writer=save_png)
    window = MainWindow(doc, author="saga", session=s)
    window.show_notice(s.start())
    try:
        assert window.save_action.shortcut().toString() == "Ctrl+S"
        assert str(tmp_path / "drawings") in window.save_action.toolTip()
        assert "Drawings save to" in window.save_label.text()
        draw(doc, "saga", ((20, 20), (180, 100)))
        assert "Not saved yet" in window.save_label.text()
        result = window.save_drawing()
        assert result and os.path.isfile(result["path"]) and os.path.isfile(result["png"])
        assert window.save_label.text() == f"Saved {result['path']}"
        from PyQt5.QtGui import QImage
        img = QImage(result["png"])
        assert (img.width(), img.height()) == (200, 120)
        assert img.pixelColor(100, 60).name() != "#ffffff"                 # the stroke is in it
    finally:
        window.close()
        window.deleteLater()
    assert not s.owner                                                     # close gave up the slot


def test_a_failed_save_in_the_window_warns_and_keeps_the_canvas(qapp, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    from drawapp.app import MainWindow

    doc = Document(200, 120)
    s = Session(doc, "aiva", str(tmp_path / "drawings"))
    window = MainWindow(doc, author="aiva", session=s)
    s.start()
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warned.append(a[2]))
    try:
        assert window.save_drawing() is None and "empty" in warned[-1]
        draw(doc)
        monkeypatch.setattr(storage, "write_drawing",
                            lambda *a, **k: (_ for _ in ()).throw(StorageError("disk full")))
        assert window.save_drawing() is None and warned[-1] == "disk full"
        assert len(doc.strokes) == 1
    finally:
        window.close()
        window.deleteLater()


def test_a_window_without_a_session_draws_as_before(qapp):
    from drawapp.app import MainWindow

    window = MainWindow(Document(50, 50))
    try:
        assert not hasattr(window, "save_action") and window.session is None
    finally:
        window.deleteLater()


def test_the_file_is_plain_json_people_can_read(tmp_path):
    doc = Document(64, 48)
    draw(doc, "aiva", ((1, 2), (30, 40)))
    path = tmp_path / "x.aivadraw"
    write_drawing(str(path), doc)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "aivadraw" and data["version"] == 1
    assert data["strokes"][0]["points"] == [[1.0, 2.0], [30.0, 40.0]]


def test_an_undo_is_autosaved_at_once_and_the_next_window_restores_without_it(tmp_path):
    first = session(tmp_path)
    kept = draw(first.document)
    draw(first.document, points=((100, 100), (150, 120)))
    reply = CommandRunner(first.document, first).handle({"cmd": "undo", "author": "aiva"})
    assert reply["ok"] and reply["remaining"] == 1 and first.dirty
    first.close()
    second = session(tmp_path)
    assert [s.id for s in second.document.strokes] == [kept]
    second.close()


def test_the_window_undoes_its_authors_last_stroke_from_the_toolbar(qapp, tmp_path):
    from drawapp.app import MainWindow

    doc = Document(200, 120)
    s = Session(doc, "aiva", str(tmp_path / "drawings"))
    window = MainWindow(doc, author="aiva", session=s)
    s.start()
    try:
        assert window.undo_action.shortcut().toString() == "Ctrl+Z"
        mine = draw(doc, "aiva")
        sisters = draw(doc, "saga", ((20, 20), (90, 90)))
        window.undo_action.trigger()
        assert [st.id for st in doc.strokes] == [sisters]          # only the window author's
        assert window.undo() is None and "Nothing of aiva's to undo" in window.save_label.text()
        assert mine not in {st.id for st in doc.strokes}
    finally:
        window.close()
        window.deleteLater()


# --- a shared canvas: one slot for several painters, taking in their old canvases once -----------

def old_canvas(tmp_path, name, *points):
    doc = Document()
    for p in points:
        draw(doc, name, p)
    path = str(tmp_path / "drawings" / storage.RECOVERY / f"{name}-autosave{EXT}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_drawing(path, doc)
    return doc


def test_a_shared_canvas_takes_in_the_old_ones_once_keeping_their_authors(tmp_path):
    aiva = old_canvas(tmp_path, "aiva", ((10, 10), (20, 20)))
    saga = old_canvas(tmp_path, "saga", ((30, 30),), ((40, 40), (50, 50)))
    doc = Document()
    s = Session(doc, "shared", str(tmp_path / "drawings"), absorb=("aiva", "saga", "shared"))
    notice = s.start()
    assert shape(doc) == shape(aiva) + shape(saga) and s.restored == 3
    assert notice == "Added aiva's (1) and saga's (2) strokes from the old canvases"
    assert s.dirty and s.current_path is None                       # unsaved work, not a file yet
    files = os.listdir(s.recovery_dir)
    assert "aiva-autosave.aivadraw" not in files and "saga-autosave.aivadraw" not in files
    assert sorted(f.split(".merged-")[0] for f in files if ".merged-" in f) == ["aiva-autosave",
                                                                             "saga-autosave"]
    assert shape(_restored(s.autosave_path)) == shape(doc)          # on disk before they moved
    s.close()
    again = session(tmp_path, "shared", absorb=("aiva", "saga"))     # nothing left to take twice
    assert len(again.document.strokes) == 3 and again.restored == 3
    again.close()


def _restored(path):
    doc = Document()
    load_into(doc, read_drawing(path))
    return doc


def test_a_canvas_still_open_in_its_own_window_is_left_alone_until_it_closes(tmp_path):
    old_canvas(tmp_path, "aiva", ((10, 10), (20, 20)))
    owner = session(tmp_path, "aiva")                                # Aiva's old window, still up
    first = Session(Document(), "shared", str(tmp_path / "drawings"), absorb=("aiva",))
    assert first.start() == "Aiva's canvas is still open in its own window"
    assert not first.document.strokes and os.path.exists(owner.autosave_path)
    first.close()
    draw(owner.document, "aiva", ((60, 60),))                         # still drawing over there
    owner.close()
    later = session(tmp_path, "shared", absorb=("aiva",))
    assert [s.author for s in later.document.strokes] == ["aiva", "aiva"]
    later.close()


def test_an_old_canvas_that_doesnt_fit_stays_where_it_is(tmp_path):
    path = str(tmp_path / "drawings" / storage.RECOVERY / f"aiva-autosave{EXT}")
    os.makedirs(os.path.dirname(path))
    small = Document(300, 200)
    draw(small, "aiva")
    write_drawing(path, small)
    s = Session(Document(), "shared", str(tmp_path / "drawings"), absorb=("aiva",))
    notice = s.start()
    assert notice.startswith("Couldn't add aiva's canvas (it is 300x200 and this canvas is 1024x768)")
    assert not s.document.strokes and os.path.exists(path)          # never moved, never lost
    s.close()


def test_the_shared_window_names_its_canvas_and_the_mouse_draws_as_its_author(qapp, tmp_path):
    from drawapp.app import MainWindow

    s = session(tmp_path, "shared")
    window = MainWindow(s.document, author="user", session=s)
    try:
        assert window.windowTitle() == "Aiva & Saga — Drawing (shared canvas)"
        assert window.canvas.tool.author == "user"                    # his strokes are his own
        runner = CommandRunner(s.document, s)
        assert runner.run(parse({"cmd": "status"}))["author"] == "shared"
    finally:
        window.close()
        window.deleteLater()
