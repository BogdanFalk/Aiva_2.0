"""Pure decisions in modules.sisters: name sets (truth table incl. the disjoint loose shapes),
addressee routing, sister-exchange stops, the floor priority/hold rules, text builders and the
tolerant arbiter parser. Everything takes `now` as epoch seconds."""
from datetime import datetime

import pytest

from modules.sisters import (AIVA_NAMES, ARBITER_PROMPT, BOTH, DEFAULT_NAMESETS, SAGA_NAMES, AddressState,
                             Decision, ExchangeConfig, ExchangeState, FloorState, Turn, absent_cue,
                             addressee_decision, arbiter_messages, both_second_cue, chat_cue,
                             exchange_should_continue, floor_decision, has_group_word, is_follow_up,
                             mentions_name, name_order, overheard_reply_line, overheard_user_line,
                             parse_arbiter_reply, prune_stale, sister_cue, which_names)

T0 = datetime(2026, 9, 11, 15, 0, 0).timestamp()      # 15:00 local, outside quiet hours
NIGHT = datetime(2026, 9, 11, 3, 0, 0).timestamp()    # 03:00 local, inside quiet hours


# --- names -------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Aiva, what's the weather", "hey ava", "eyva can you hear me",
                                  "Ayvah stop", "hey va open chrome", "so, eva?"])
def test_aiva_names_hit(text):
    assert mentions_name(text, AIVA_NAMES)


@pytest.mark.parametrize("text", ["avocado toast", "I have a van", "the eve of the war", "nova",
                                  "saga, are you there", "sage advice"])
def test_aiva_names_miss(text):
    assert not mentions_name(text, AIVA_NAMES)


@pytest.mark.parametrize("text", ["Saga, come here", "hey saga", "sarga what do you think", "sagah?",
                                  "sa ga please", "sega are you there", "seiga", "sager"])
def test_saga_names_hit(text):
    assert mentions_name(text, SAGA_NAMES)


@pytest.mark.parametrize("text", ["sage advice", "pass the sugar", "sarah called", "the stage is set",
                                  "segway", "aiva what time is it"])
def test_saga_names_miss(text):
    assert not mentions_name(text, SAGA_NAMES)


def test_mentions_name_accepts_bare_variant_tuple_like_main_py():
    assert mentions_name("hey ava", ("aiva", "ava"))
    assert mentions_name("eyva", ("aiva",))            # loose shape still applies


def test_which_names_and_order():
    assert which_names("Saga, ask Aiva", DEFAULT_NAMESETS) == {"aiva", "saga"}
    assert which_names("nobody here", DEFAULT_NAMESETS) == set()
    assert name_order("Saga, ask Aiva") == ["saga", "aiva"]
    assert name_order("Aiva and Saga") == ["aiva", "saga"]


def test_group_words():
    assert has_group_word("what do you two think")
    assert has_group_word("Both of you, listen")
    assert has_group_word("girls, dinner")
    assert not has_group_word("you too")
    assert not has_group_word("the girlish laugh")


# --- addressing ----------------------------------------------------------------------------

def st(**over):
    s = AddressState()
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_both_names_and_group_word_route_to_both():
    d = addressee_decision("Aiva, Saga, come here", st(), now=T0)
    assert d.addressee == BOTH and d.reason == "both_named" and d.confident
    d = addressee_decision("what do you two think", st(), now=T0)
    assert d.addressee == BOTH and d.reason == "group_word"


def test_one_name_routes_to_her():
    assert addressee_decision("Saga, what's up", st(), now=T0) == Decision("saga", "named", True)
    assert addressee_decision("hey ava", st(), now=T0).addressee == "aiva"


def test_wake_hint_counts_when_no_name_spoken():
    d = addressee_decision("what's the weather", st(), wake_hint="saga", now=T0)
    assert d.addressee == "saga" and d.reason == "wake_word"
    # a spoken name beats the wake hint
    assert addressee_decision("Aiva what's the weather", st(), wake_hint="saga", now=T0).addressee == "aiva"


def test_short_follow_up_goes_to_the_sister_who_asked():
    s = st(last_speaker="saga", last_speaker_at=T0 - 5, last_reply_was_question=True,
           last_addressee="aiva", last_addressee_at=T0 - 60)
    d = addressee_decision("yes please", s, now=T0)
    assert d.addressee == "saga" and d.reason == "follow_up"
    # a long new sentence is not a follow-up → sticky to the last addressee instead
    d = addressee_decision("can you open the terminal and run the tests for me now", s, now=T0)
    assert d.addressee == "aiva" and d.reason == "sticky" and not d.confident
    # the question was too long ago
    s.last_speaker_at = T0 - 600
    s.last_addressee_at = T0 - 600
    assert addressee_decision("yes", s, now=T0) is None


def test_sticky_window_then_none():
    s = st(last_addressee="saga", last_addressee_at=T0 - 30)
    assert addressee_decision("and then what", s, now=T0).addressee == "saga"
    assert addressee_decision("and then what", s, now=T0 + 100) is None


def test_sticky_after_both_goes_to_last_speaker():
    s = st(last_addressee=BOTH, last_addressee_at=T0 - 10, last_speaker="aiva")
    d = addressee_decision("tell me more about that", s, now=T0)
    assert d.addressee == "aiva" and d.reason == "sticky_last_speaker"


def test_is_follow_up_shapes():
    assert is_follow_up("yes")
    assert is_follow_up("because it's late")
    assert is_follow_up("the weather")
    assert not is_follow_up("I want you to look up the weather for tomorrow in Bucharest")


# --- exchange ------------------------------------------------------------------------------

def turn(mind="aiva", kind="user", addressee="him"):
    return Turn(id=1, mind=mind, kind=kind, audible=True, addressee=addressee, created_at=T0)


def test_exchange_continues_on_name_plus_question():
    ex = ExchangeState()
    cfg = ExchangeConfig()
    assert exchange_should_continue("Saga, what do you think?", turn(), AddressState(), ex, cfg, T0) == "saga"
    assert exchange_should_continue("I think so too, Saga.", turn(), AddressState(), ex, cfg, T0) is None
    assert exchange_should_continue("What do you think?", turn(), AddressState(), ex, cfg, T0) is None


def test_exchange_continues_on_sister_turn_question():
    t = turn(mind="saga", kind="sister", addressee="aiva")
    assert exchange_should_continue("Why would you say that?", t, AddressState(), ExchangeState(), ExchangeConfig(), T0) == "aiva"
    assert exchange_should_continue("Fair enough.", t, AddressState(), ExchangeState(), ExchangeConfig(), T0) is None


def test_a_question_she_relays_to_him_is_not_one_for_her_sister():
    # 2026-10-14: Saga's task asked him "…search the Aiva repo…; should I let it run?" and Aiva got a
    # sister round and said "Yes, let it run"
    t = turn(mind="saga", kind="announce", addressee="him")
    q = "The task wants to search the Aiva repo for witness notes; should I let it run?"
    assert exchange_should_continue(q, t, AddressState(), ExchangeState(), ExchangeConfig(), T0) is None
    # opening by addressing her sister still reaches her
    assert exchange_should_continue("Aiva, did you see this?", t, AddressState(), ExchangeState(),
                                    ExchangeConfig(), T0) == "aiva"


def test_exchange_hard_stops():
    cfg = ExchangeConfig()
    reply = "Saga, what do you think?"
    assert exchange_should_continue(reply, turn(), AddressState(), ExchangeState(user_speaking=True), cfg, T0) is None
    assert exchange_should_continue(reply, turn(), AddressState(), ExchangeState(awake=False), cfg, T0) is None
    assert exchange_should_continue(reply, turn(), AddressState(), ExchangeState(rounds=cfg.max_rounds), cfg, T0) is None
    assert exchange_should_continue(reply, turn(), AddressState(), ExchangeState(backoff_until=T0 + 10), cfg, T0) is None
    marks = [T0 - i * 60 for i in range(cfg.max_per_hour)]
    assert exchange_should_continue(reply, turn(), AddressState(), ExchangeState(hour_marks=marks), cfg, T0) is None
    assert exchange_should_continue(reply, turn(), AddressState(), ExchangeState(hour_marks=[T0 - 7200] * 20), cfg, T0) == "saga"
    assert exchange_should_continue(reply, turn(), AddressState(), ExchangeState(), cfg, NIGHT) is None


def test_exchange_config_from_env(monkeypatch):
    monkeypatch.setenv("AIVA_SISTER_MAX_ROUNDS", "5")
    monkeypatch.setenv("AIVA_SISTER_BACKOFF_S", "30")
    monkeypatch.setenv("AIVA_SISTER_MAX_PER_HOUR", "2")
    monkeypatch.setenv("AIVA_SISTER_CHATS_PER_DAY", "1")
    monkeypatch.setenv("AIVA_ADDRESS_STICKY_S", "45")
    cfg = ExchangeConfig.from_env()
    assert (cfg.max_rounds, cfg.backoff_s, cfg.max_per_hour, cfg.chats_per_day, cfg.sticky_s) == (5, 30.0, 2, 1, 45.0)
    assert cfg.quiet_hours == (1, 8)


# --- floor decision --------------------------------------------------------------------------

def q(id, mind, kind, created):
    return Turn(id=id, mind=mind, kind=kind, audible=True, created_at=created)


def test_floor_holds_while_speaking_or_turn_open_or_user_speaking():
    queue = [q(1, "aiva", "user", T0)]
    assert floor_decision(queue, FloorState(speaking=True), T0) == (None, "hold:speaking")
    assert floor_decision(queue, FloorState(turn_id=7), T0) == (None, "hold:turn_open")
    assert floor_decision(queue, FloorState(user_speaking=True), T0) == (None, "hold:user_speaking")
    assert floor_decision([], FloorState(), T0) == (None, "idle")


def test_floor_priority_and_fifo():
    queue = [q(1, "aiva", "initiative", T0 - 5), q(2, "saga", "sister", T0 - 4), q(3, "aiva", "announce", T0 - 3),
             q(4, "saga", "both_second", T0 - 2), q(5, "aiva", "user", T0 - 1), q(6, "saga", "user", T0 - 0.5)]
    fs = FloorState(last_user_speech_at=T0 - 100)
    picked, reason = floor_decision(queue, fs, T0)
    assert picked.id == 5 and reason == "grant:user"
    queue.remove(picked)
    assert floor_decision(queue, fs, T0)[0].id == 6          # FIFO within a kind
    queue.pop(queue.index(next(t for t in queue if t.id == 6)))
    assert floor_decision(queue, fs, T0)[0].id == 4
    queue.remove(queue[-1])
    assert floor_decision(queue, fs, T0)[0].id == 3
    queue.remove(queue[-1])
    assert floor_decision(queue, fs, T0)[0].id == 2
    queue.remove(queue[-1])
    assert floor_decision(queue, fs, T0)[0].id == 1


def test_floor_drops_stale_sister_turns_when_user_spoke():
    queue = [q(1, "saga", "sister", T0 - 100), q(2, "aiva", "announce", T0 - 100)]
    fs = FloorState(last_user_speech_at=T0 - 50)
    dropped = prune_stale(list(queue), fs)
    assert [t.id for t in dropped] == [1]
    picked, reason = floor_decision(queue, fs, T0)
    assert picked.id == 2 and [t.id for t in queue] == [2]


def test_announces_wait_for_a_quiet_moment():
    queue = [q(1, "aiva", "announce", T0 - 10)]
    fs = FloorState(last_user_speech_at=T0 - 0.5)
    assert floor_decision(queue, fs, T0) == (None, "hold:announce_delay")
    assert floor_decision(queue, fs, T0 + 10.0) == (None, "hold:announce_delay")   # still mid-chat
    assert floor_decision(queue, fs, T0 + 46.0)[0].id == 1                        # quiet now
    # a sister's reply resets the quiet clock too — news never lands mid-exchange
    fs2 = FloorState(last_user_speech_at=T0 - 100, last_said_at=T0 - 5)
    assert floor_decision(list(queue), fs2, T0) == (None, "hold:announce_delay")
    # ...and a queued sister turn keeps the exchange going while the news waits
    queue2 = [q(1, "aiva", "announce", T0 - 10), q(3, "saga", "sister", T0 - 2)]
    assert floor_decision(queue2, fs2, T0)[0].id == 3
    # a user turn is never delayed
    queue.append(q(2, "saga", "user", T0))
    assert floor_decision(queue, fs, T0)[0].id == 2


# --- text ------------------------------------------------------------------------------------

def test_text_builders():
    assert overheard_user_line("Saga", "  open   the door ") == '[He said to Saga: "open the door"]'
    assert overheard_reply_line("Saga", "him", "sure") == '[Saga said to him: "sure"]'
    assert overheard_reply_line("Saga", "you", "hey Aiva") == '[Saga said to you: "hey Aiva"]'
    assert sister_cue("Saga", True).startswith("[Saga just spoke to you") and "private" in sister_cue("Saga", True)
    assert "hear" in sister_cue("Saga", False)
    assert "Aiva answered first" in both_second_cue("Aiva")
    assert "Saga isn't running right now" in absent_cue("Saga")
    assert "just you" in absent_cue("Saga", both=True)
    assert "Saga" in chat_cue("Saga") and "not to him" in chat_cue("Saga")


# --- arbiter -----------------------------------------------------------------------------------

def test_arbiter_messages_shape():
    msgs = arbiter_messages("what about tomorrow", [{"speaker": "user", "addressee": "saga", "text": "weather?"},
                                                    {"speaker": "saga", "addressee": "him", "text": "sunny."}],
                            {"aiva": "cheeky, tools", "saga": "new, forming"})
    assert msgs[0] == {"role": "system", "content": ARBITER_PROMPT}
    body = msgs[1]["content"]
    assert "- aiva: cheeky, tools" in body and "- saga: new, forming" in body
    assert 'He -> saga: "weather?"' in body and 'Saga -> him: "sunny."' in body
    assert body.rstrip().endswith('His new line: "what about tomorrow"')


@pytest.mark.parametrize("raw,expected", [
    ('{"to": "saga", "why": "she asked"}', "saga"),
    ('```json\n{"to": "aiva", "why": "thread"}\n```', "aiva"),
    ('Sure! {"to":"SAGA"} there you go', "saga"),
    ("saga", "saga"),
    ('{"addressee": "aiva"}', "aiva"),
    ('{"to": "bob"}', None),
    ("no idea", None),
    ("", None),
    (None, None),
])
def test_parse_arbiter_reply_is_tolerant(raw, expected):
    d = parse_arbiter_reply(raw)
    if expected is None:
        assert d is None
    else:
        assert d.addressee == expected and d.reason.startswith("arbiter") and d.confident is False


def test_parse_arbiter_reply_accepts_reply_objects():
    class R:
        text = '{"to": "saga"}'
    assert parse_arbiter_reply(R()).addressee == "saga"


def test_private_exchange_may_continue_while_the_house_sleeps():
    from modules.sisters import AddressState, ExchangeConfig, ExchangeState, Turn, exchange_should_continue
    cfg = ExchangeConfig(quiet_hours=(0, 0))
    t = Turn(id=1, mind="saga", kind="sister", audible=False, addressee="aiva", created_at=0.0)
    reply = "Aiva, do you ever get bored at night?"
    now = 1_700_000_000.0 + 12 * 3600      # noon-ish for any zone
    assert exchange_should_continue(reply, t, AddressState(), ExchangeState(awake=False), cfg, now,
                                    audible=True) is None
    assert exchange_should_continue(reply, t, AddressState(), ExchangeState(awake=False), cfg, now,
                                    audible=False) == "aiva"


def test_asks_something_and_vocative_helpers():
    from modules.sisters import DEFAULT_NAMESETS, asks_something, vocative
    assert asks_something("Do you like rain?")
    assert asks_something("I keep wondering whether devotion can protect a place.")
    assert asks_something("Saga, I'd like to know whether you think love can accept limits.")
    assert not asks_something("I think it can, but only if the obligation stays responsive.")
    others = [ns for ns in DEFAULT_NAMESETS if ns.slug != "aiva"]
    assert vocative("Hey, Saga. Fancy joining us?", others) == "saga"
    assert vocative("Saga, what do you make of this?", others) == "saga"
    assert vocative("Pizza. Saga, you in?", others) is None
    assert vocative("I was reading with Saga earlier.", others) is None


def test_chat_openers_and_indirect_questions_get_an_answer():
    from modules.sisters import (AddressState, ExchangeConfig, ExchangeState, Turn,
                                 exchange_should_continue)
    cfg = ExchangeConfig(quiet_hours=(0, 0))
    now = 1_700_000_000.0 + 12 * 3600
    # a chat Saga opened with Aiva, phrased as a statement → Aiva still answers once
    opener = Turn(id=1, mind="saga", kind="initiative", audible=False, addressee="aiva", created_at=0.0)
    assert exchange_should_continue("Aiva, I think protection stays clean only when it accepts limits.",
                                    opener, AddressState(), ExchangeState(), cfg, now, audible=False) == "aiva"
    # inside the exchange: a plain statement ends it, an indirect question continues it
    reply = Turn(id=2, mind="aiva", kind="sister", audible=False, addressee="saga", created_at=1.0, reply_to=1)
    assert exchange_should_continue("I think it can, but only if the obligation stays responsive.",
                                    reply, AddressState(), ExchangeState(rounds=1), cfg, now, audible=False) is None
    assert exchange_should_continue("Saga, I keep wondering whether devotion can protect a place.",
                                    reply, AddressState(), ExchangeState(rounds=1), cfg, now, audible=False) == "saga"
    # on his turn: opening with her name hands her the floor even without a question mark
    user = Turn(id=3, mind="saga", kind="user", audible=True, addressee="him", created_at=2.0)
    assert exchange_should_continue("Aiva, Frank wants to know what you think about gratitude.",
                                    user, AddressState(), ExchangeState(), cfg, now) == "aiva"
    # …but merely mentioning her mid-sentence without asking does not
    assert exchange_should_continue("I think so too, Aiva.", user, AddressState(), ExchangeState(), cfg, now) is None


def test_calls_errands_and_reports_2026_10_14():
    from modules.sisters import called_in, errand_to, vocative, wants_report
    others = [SAGA_NAMES]
    # about her is not to her
    assert vocative("Saga's already here, and I've got her set to audible.", others) is None
    assert vocative("Saga is starting up now.", others) is None
    assert vocative("Saga!", others) == "saga" and vocative("Saga what do you think?", others) == "saga"
    # a call anywhere in the reply
    assert called_in("Yeah, that's fair. Saga, describe the actual screen appearance.", others) == "saga"
    assert called_in("I think so too, Saga.", others) is None
    assert exchange_should_continue("Yeah, that's fair. Saga, describe the actual look.", turn(),
                                    AddressState(), ExchangeState(), ExchangeConfig(), T0) == "saga"
    # his errands
    for line in ("Ok she's up ask her what I told you and let me know", "ask her",
                 "Ok bring saga in here and ask her if she wants to change anything", "tell Saga I said hi",
                 "check with your sister", "find out from her what she needs"):
        assert errand_to(line, SAGA_NAMES), line
    for line in ("tell me a joke", "Open a terminal in Unseen Media work folder", "what did she say earlier",
                 "ask me tomorrow", "Saga lets start with you retry doing the text file I asked you"):
        assert not errand_to(line, SAGA_NAMES), line
    assert wants_report("look into it with her and let me know", SAGA_NAMES)
    assert wants_report("ask her what she wants changed", SAGA_NAMES)
    assert not wants_report("Saga, ask your sister something.", AIVA_NAMES)      # just "go talk to her"
    assert wants_report("ask your sister something and let me know", AIVA_NAMES)
    # her report to him never restarts the sisters, whatever it says
    rep = turn(kind="report")
    assert exchange_should_continue("Saga, want me to make that change?", rep, AddressState(), ExchangeState(),
                                    ExchangeConfig(), T0) is None


def test_filler_openers_dashes_and_spoken_to_2026_10_21():
    from modules.sisters import BOTH, spoken_to, vocative
    others = [SAGA_NAMES]
    assert vocative("Yep, Saga, you’re looking at it—the shared drawing canvas.", others) == "saga"
    assert vocative("Sure, Saga, go ahead.", others) == "saga"
    assert vocative("Oh, hey—Saga, look at this.", others) == "saga"
    assert vocative("Saga—look at this.", others) == "saga"
    # still about her, not to her
    assert vocative("Wow—Saga's going to love it.", others) is None
    assert vocative("Nice, Saga is starting up.", others) is None
    assert spoken_to("Yep, Saga, you’re looking at it.", others) == "saga"
    assert spoken_to("Oh, hey—you’re back. Saga, Frank wants you to see my app.", others) == BOTH
    assert spoken_to("Pizza. Saga, you in?", others) == BOTH
    assert spoken_to("Sunny all day, and Saga agrees.", others) is None
