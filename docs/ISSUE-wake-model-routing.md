# Issue: the wake-word model name never routes a turn (`hey_saga` ≠ `saga`)

Status: fixed in the working tree 2026-10-26 (`wake.wake_slugs` at `house.py` `on_wake`, `tests/test_wake.py`; not yet committed, not yet seen live) · found 2026-10-18 by read-only inspection · version 0.10.0 working tree

## Summary

The house is meant to send a turn to the sister whose wake word you said ("hey Saga, what time
is it" → Saga) even when the rest of the line doesn't name her. That rule exists in
`addressee_decision`, but it never fires. The wake listener reports the **model basename**
(`hey_aiva`, `hey_saga`). The router compares the hint against sister **slugs** (`aiva`, `saga`).
Nothing maps one to the other, so `wake_hint in slugs` is always false and the `wake_word`
branch is dead code in production.

## Where the value comes from and where it's checked

1. `aiva/modules/wake.py:21-24`: `model_name()` returns the basename without its extension, so
   `models/hey_saga.onnx` becomes `"hey_saga"`. openWakeWord keys its scores by that name.
2. `aiva/modules/wake.py:99-100,108`: the listener picks the top-scoring key from
   `oww.predict()` and passes it unchanged as `name` to `on_wake(score, preroll, name)`.
3. `aiva/house.py:521-526`: `on_wake` forwards the name as `mic.wake(..., verify_wakes, name)`
   (the `hint` parameter).
4. `aiva/modules/ear.py:98-99`: `if hint: self.wake_hint = hint`, which stores `"hey_saga"`.
5. `aiva/house.py:320-325`: `_on_user_turn` reads `mic.wake_hint` and passes it unchanged to
   `floor.submit_user_turn(text, ..., wake_hint=hint)`.
6. `aiva/modules/house_floor.py:306`: passes it unchanged to `addressee_decision(..., wake_hint=wake_hint)`.
7. `aiva/modules/sisters.py:175,183-184`:
   ```python
   slugs = {ns.slug for ns in namesets}          # {"aiva", "saga"}
   ...
   if wake_hint in slugs:                        # "hey_saga" in {"aiva","saga"} -> False
       return Decision(wake_hint, "wake_word", True)
   ```

No translation step exists. A search for `removeprefix`, `"hey_"`, or any model-name-to-slug map
in `aiva/*.py` and `aiva/modules/*.py` finds nothing.

`aiva/modules/wake.py:6` documents the intended contract: "`name` is which model fired
(e.g. "hey_aiva" / "hey_saga") so the house knows who was called". The consumer was written
against a different vocabulary.

## Why the tests don't catch it

Each side is tested on its own, and each uses a different string:

| Test | Value used |
|---|---|
| `tests/test_ear.py:14-22` (`test_wake_hint_is_kept_and_cleared_on_reject`) | `"hey_saga"`, `"hey_aiva"` (what the listener really produces) |
| `tests/test_sisters.py:86-90` (`test_wake_hint_counts_when_no_name_spoken`) | `"saga"` |
| `tests/test_house_floor.py:119-122` (`test_wake_hint_routes_when_no_name_spoken`) | `"saga"` |
| `tests/test_house_proto.py:23` | `"saga"` |

No test runs the real wake name through the ear and the floor to the decision.

## Evidence from the live logs (`aiva/logs/house-*.log`, 2026-09-11 → 2026-10-18)

- **1,935** `floor: addressee=` routing decisions are logged, and **none** has the reason `wake_word`:

  | reason | count |
  |---|---|
  | sticky | 1510 |
  | follow_up | 253 |
  | named | 72 |
  | fallback:last_speaker | 51 |
  | group_word | 22 |
  | both_named | 7 |
  | sticky_last_speaker | 2 |
  | fallback:default | 2 |
  | **wake_word** | **0** |

- During the same period the listener logged **374** `wake word detected (hey_aiva, …)` and
  **19** `wake word detected (hey_saga, …)` lines. The model names arrive with the `hey_`
  prefix every time.
- Every `hey_saga` wake that reached the floor (2026-09-11 20:13:04, 2026-09-13 15:52:43,
  2026-09-13 17:42:50) was routed with reason `named`. That means the transcript happened to
  contain "Saga", not that the wake word decided the route. The other `hey_saga` wakes were
  either `wake rejected` or `wake ignored — mic muted (F9)` (for example 2026-09-24 18:32:17
  and 2026-10-12 01:09:44).
- `.env` sets `AIVA_WAKE_MODEL=models/hey_aiva.onnx`. `SAGA_WAKE_MODEL` is commented out, so
  Saga uses the default `aiva/models/hey_saga.onnx` (`aiva/modules/character.py:156,185`).
  Both files exist in `aiva/models/`. The basenames are therefore exactly `hey_aiva` and `hey_saga`.

## Impact

- **Default config (`AIVA_WAKE_VERIFY` unset → `1`): mostly hidden, but the rule is dead.**
  A wake is confirmed only when the transcript names a sister: `_name_check` →
  `which_names` (`aiva/house.py:327-328`). That is the same matcher the router uses for
  `named`, so the confirmed line is almost always routed as `named` before the `wake_word`
  check runs. The `wake_word` rule, its tests and the CHANGELOG/README promise that the wake
  word picks the sister add nothing in practice.
- **With `AIVA_WAKE_VERIFY=0`, it misroutes.** No name is needed to confirm the wake, so
  "hey Saga" followed by an unnamed or garbled line skips `wake_word` and falls through to
  `follow_up`, `sticky`, the arbiter or `fallback:last_speaker`. Within the 90 s sticky window
  after talking to Aiva, the turn goes to **Aiva** even though you woke Saga. Outside it, the
  turn costs an arbiter LLM call that a correct hint would have avoided.
- **Mis-transcribed names.** If the wake-word model heard the name correctly but Deepgram's
  transcript didn't (with verification off), the wake hint is the only signal of who was
  called, and that signal is thrown away.
- **Wire noise.** The raw `"hey_saga"` string still travels to the sisters in
  `TurnMsg.wake_hint` (`aiva/modules/house_proto.py:195`, `house_floor.py:353,361,848`),
  while `test_house_proto.py` documents it as a slug.
- **Fallback model.** With no custom model, the house arms `hey_jarvis`
  (`aiva/house.py:521`), which maps to no sister at all. Whatever fix is chosen should cope
  with an unmapped name.

## Suggested fix (not applied)

Translate at one boundary, either in `house.py`'s `on_wake` or in `_wake_models()`. Build a
`{model_name(ch.wake_model): ch.slug}` map from the same `Character.from_env` loop that
already collects the models, and pass `map.get(name)` as the hint. Unknown names such as
`hey_jarvis` would become `None`. Then add one end-to-end test that feeds `"hey_saga"`
through `MicController.wake` → `submit_user_turn` and asserts `reason == "wake_word"` when the
line names nobody.
