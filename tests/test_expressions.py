import json

from modules.avatar_renderer import Expressions


class FakeModel:
    def __init__(self):
        self.params = {}

    def SetParameterValue(self, pid, value, *a, **k):
        self.params[pid] = value


def _write_exp(d, name, param_id):
    (d / f"{name}.exp3.json").write_text(
        json.dumps({"Parameters": [{"Id": param_id, "Value": 1.0, "Blend": "Add"}]}), encoding="utf-8")


def test_toggle_off_returns_param_to_zero(tmp_path):
    _write_exp(tmp_path, "hood", "ParamHood")
    _write_exp(tmp_path, "hearts", "ParamHeart")
    exp = Expressions(str(tmp_path))
    m = FakeModel()

    # hood on
    assert exp.toggle(exp.match("hood")) is True
    exp.apply(m)
    assert m.params["ParamHood"] == 1.0 and m.params["ParamHeart"] == 0.0

    # hood off — the regression: it must go back to 0, not stay frozen at 1
    assert exp.toggle(exp.match("hood")) is False
    exp.apply(m)
    assert m.params["ParamHood"] == 0.0

    # two at once, then drop one
    exp.toggle(exp.match("hood"))
    exp.toggle(exp.match("hearts"))
    exp.apply(m)
    assert m.params["ParamHood"] == 1.0 and m.params["ParamHeart"] == 1.0
    exp.toggle(exp.match("hearts"))
    exp.apply(m)
    assert m.params["ParamHood"] == 1.0 and m.params["ParamHeart"] == 0.0
