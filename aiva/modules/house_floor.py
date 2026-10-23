"""The Floor: the house's single authority on who speaks, when, and who hears what.

One `Floor` per house process. It owns the turn queue, the current grant, the mic-gate flag and
the sister-exchange state; it decides with the pure functions in `sisters.py` and remembers through
`house_memory.HouseMemory`. It knows NOTHING about sockets or pipecat: each connected sister is a
`ClientLike` — an object with `.info` (`ClientInfo`: slug, name, blurb, has_voice) and
`.send(msg)` taking a `house_proto` dataclass — injected by the server (tests use fakes). Time comes
from the injected `now()` (epoch seconds) and every timer is driven from `tick(now)`, so nothing
here sleeps.

Lifecycle of a turn (rules from the plan, Part A3/A5):
  queued → granted (holder=slug, gate=turn.audible, `turn` sent) → `response start` (ack; no ack
  in 10 s → release) → tools ±1 (a response that called tools re-runs; wait for its new start) →
  `response end` → release on `speaking off` | grace (1.5 s with no speech) | immediately for
  private turns | 120 s cap | holder death (+3 s gate tail) | interrupt (+2 s fallback).
  Release waits up to 1 s for her `said` before the next grant so the other sister always gets
  the overheard line BEFORE her own turn runs. `said` → transcript row, overheard to the other,
  AddressState, maybe a `sister` continuation (audible only when the mode is not private).

Private mode (default): sister-to-sister turns are text-only (`audible=False`) and never raise the
gate, so the user can talk over a private exchange; lines to him are always audible.
"""

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Optional

from modules import house_proto as proto
from modules.sisters import (BOTH, DEFAULT_NAMESETS, HIM, AddressState, Decision, ExchangeConfig,
                             ExchangeState, FloorState, NameSet, Turn, absent_cue, addressee_decision,
                             arbiter_messages, arrival_cue, both_second_cue, chat_cue, errand_cue,
                             errand_to, exchange_should_continue, floor_decision, is_question, name_order,
                             overheard_reply_line, overheard_user_line, parse_arbiter_reply, prune_stale,
                             report_cue, sister_cue, spoken_to, wants_report, which_names)

# a line he meant for a sister who wasn't running is picked up when she arrives — if she arrives
# within this long (booting takes ~10-30 s; a minute later it's a different conversation)
ERRAND_TTL_S = 180.0


@dataclass
class ClientInfo:
    slug: str
    name: str
    blurb: str = ""
    has_voice: bool = True
    nameset: Optional[NameSet] = None      # from her `hello`; default set by slug when None


class Floor:
    def __init__(self, memory, *, private=True, log=print, now=time.time, clients=None,
                 on_grant=None, on_gate_change=None, arbiter=None, cfg=None, namesets=None,
                 fallback_slug="aiva", arbiter_timeout_s=4.0, turn_cap_s=120.0, ack_timeout_s=10.0,
                 death_tail_s=3.0, interrupt_fallback_s=2.0, said_wait_s=1.0, announce_quiet_s=None):
        self.memory = memory
        self.log = log
        self._now = now
        self.clients = clients if clients is not None else {}
        self.on_grant = on_grant
        self.on_gate_change = on_gate_change
        self.arbiter = arbiter
        self.cfg = cfg or ExchangeConfig()
        self._default_namesets = tuple(namesets or DEFAULT_NAMESETS)
        self.fallback_slug = fallback_slug
        self.arbiter_timeout_s = arbiter_timeout_s
        self.turn_cap_s = turn_cap_s
        self.ack_timeout_s = ack_timeout_s
        self.death_tail_s = death_tail_s
        self.interrupt_fallback_s = interrupt_fallback_s
        self.said_wait_s = said_wait_s
        self.announce_quiet_s = (announce_quiet_s if announce_quiet_s is not None
                                 else getattr(self.cfg, "announce_quiet_s", 45.0))

        stored = memory.kv_get("sister_talk")
        if stored in ("private", "audible"):
            self.private = stored == "private"
        else:
            self.private = bool(private)
            memory.kv_set("sister_talk", "private" if self.private else "audible")

        # sisters whose VOICE is muted from the dashboard: their turns are forced non-audible
        # (text-only, like private mode) so they still think/log but make no sound and spend no TTS.
        self._muted = set(s for s in (memory.kv_get("muted_slugs", "") or "").split(",") if s)

        self._queue = []
        self._turns = {}                      # id → Turn, recent only
        self._waiting = {}                    # absent slug → his line for her, until she arrives
        self._reported = set()                # user turn ids whose errand was already reported back
        self._fs = FloorState()
        self._addr = AddressState()
        self._ex = ExchangeState()
        self._current: Optional[Turn] = None
        self._gate = False
        self._seq = int(memory.kv_get("turn_seq", 0) or 0)
        self._msg_seq = 0
        # per-grant flags
        self._response_started = False
        self._response_open = False
        self._speaking_seen = False
        self._said_seen = False
        self._interrupted = False
        self._tools_open = 0
        self._tool_calls_seen = 0
        # timers (epoch deadlines, None = off)
        self._ack_deadline = None
        self._grace_deadline = None
        self._cap_deadline = None
        self._interrupt_deadline = None
        self._closing = None                  # turn id released but awaiting its `said`
        self._closing_deadline = None
        self._tail_deadline = None

    # --- read-only views --------------------------------------------------------------------

    @property
    def gate_active(self) -> bool:
        """True while the mic must be zero-filled: from the grant of an AUDIBLE turn to its release
        (plus the death tail)."""
        return self._gate

    @property
    def busy(self) -> bool:
        return self._fs.turn_id is not None or self._fs.speaking

    @property
    def mode(self) -> str:
        return "private" if self.private else "audible"

    @property
    def present(self) -> list:
        return sorted(self.clients.keys())

    @property
    def holder(self) -> Optional[str]:
        return self._fs.holder

    @property
    def current_turn(self) -> Optional[Turn]:
        return self._current

    @property
    def queue(self) -> list:
        return list(self._queue)

    @property
    def address_state(self) -> AddressState:
        return self._addr

    @property
    def exchange_state(self) -> ExchangeState:
        return self._ex

    # --- presence ---------------------------------------------------------------------------

    def add_client(self, slug, client):
        self.clients[slug] = client
        self._pick_up_errand(slug)

    def _pick_up_errand(self, slug):
        """She just arrived and he had asked for her while she was off: the sister who took his line
        gets a turn to put it to her (2026-10-14: "bring Saga in and ask her…" — Saga booted, and
        nothing happened until he said "ok she's up, ask her what I told you")."""
        w = self._waiting.pop(slug, None)
        if not w:
            return
        now = self._now()
        if now - w["at"] > ERRAND_TTL_S or w["by"] == slug or w["by"] not in self.clients:
            return
        nt = self._new_turn(now, mind=w["by"], kind="sister", audible=True, addressee=slug,
                            cue=arrival_cue(self._name(slug), w["text"]), source_text=w["text"],
                            reply_to=w["turn"])
        self._queue.append(nt)
        self.log(f"floor: {self._name(slug)} arrived; {w['by']} picks up his line for her (turn {w['turn']})")
        self._kick(now)

    def _namesets(self):
        out = []
        for ns in self._default_namesets:
            c = self.clients.get(ns.slug)
            over = getattr(getattr(c, "info", None), "nameset", None) if c is not None else None
            out.append(over or ns)
        known = {ns.slug for ns in out}
        for slug, c in self.clients.items():
            if slug not in known:
                over = getattr(c.info, "nameset", None)
                out.append(over or NameSet(slug, c.info.name, (slug,), ""))
        return tuple(out)

    def _slugs(self):
        return {ns.slug for ns in self._namesets()}

    def _name(self, slug):
        c = self.clients.get(slug)
        if c is not None:
            return c.info.name
        for ns in self._default_namesets:
            if ns.slug == slug:
                return ns.name
        return str(slug).capitalize()

    def _present(self):
        return list(self.clients.keys())

    # --- plumbing -----------------------------------------------------------------------------

    def _stamp(self, msg, now):
        self._msg_seq += 1
        msg.id = self._msg_seq
        msg.ts = now
        return msg

    def _send(self, slug, msg, now=None) -> bool:
        client = self.clients.get(slug)
        if client is None:
            return False
        self._stamp(msg, now if now is not None else self._now())
        try:
            client.send(msg)
            return True
        except Exception as e:
            self.log(f"floor: send to {slug} failed: {e!r}")
            return False

    def _broadcast(self, msg, now=None):
        for slug in list(self.clients):
            self._send(slug, msg, now)

    def _set_gate(self, on: bool):
        on = bool(on)
        if on == self._gate:
            return
        self._gate = on
        if self.on_gate_change is not None:
            try:
                self.on_gate_change(on)
            except Exception as e:
                self.log(f"floor: on_gate_change failed: {e!r}")

    def _new_turn(self, now, **kw) -> Turn:
        self._seq += 1
        self.memory.kv_set("turn_seq", self._seq, now=now)
        turn = Turn(id=self._seq, created_at=now, **kw)
        self._remember(turn)
        return turn

    def _remember(self, turn):
        self._turns[turn.id] = turn
        if len(self._turns) > 400:
            for k in sorted(self._turns)[:200]:
                del self._turns[k]

    def _drop_queued(self, pred, why):
        dropped = [t for t in self._queue if pred(t)]
        if dropped:
            self._queue[:] = [t for t in self._queue if t not in dropped]
            self.log(f"floor: dropped {len(dropped)} queued turn(s) ({why}): "
                     + ", ".join(f"{t.mind}/{t.kind}#{t.id}" for t in dropped))
        return dropped

    def _is_current(self, slug, turn_id):
        return self._current is not None and self._current.id == turn_id and self._current.mind == slug

    def _rooted_in_user(self, turn) -> bool:
        """True when this exchange grew out of something HE said (or a both-of-you turn): he is in
        the room and part of it, so the sisters speak aloud even in private mode. Sister-initiated
        chats and exchanges rooted in a sister's own announce/initiative stay under the mode."""
        seen = 0
        t = turn
        while t is not None and seen < 20:
            if t.kind in ("user", "both_second"):
                return True
            if t.kind in ("announce", "initiative", "intro"):
                return False
            t = self._turns.get(t.reply_to) if t.reply_to else None
            seen += 1
        return False

    def _sister_audible(self, turn) -> bool:
        return (not self.private) or self._rooted_in_user(turn)

    def _spoken_to(self, slug, text):
        """Her sister's slug when a reply OPENS by calling her ('Hey, Saga, …'), BOTH when it turns
        to her after answering him ('Pizza. Saga, you in?'), else None (`sisters.spoken_to`)."""
        return spoken_to(text, [ns for ns in self._namesets() if ns.slug != slug])

    # --- the user spoke ------------------------------------------------------------------------

    async def submit_user_turn(self, text, *, speaker_vec=None, affect=None, wake_hint=None):
        """One final user transcript → decide who he meant, log it, queue her turn (two for
        'both'), tell the other sister what she overheard, and grant if the floor is free.
        Returns the Decision (None when the line was empty)."""
        text = (text or "").strip()
        if not text:
            return None
        now = self._now()
        self._fs.user_speaking = False
        self._ex.user_speaking = False
        self._fs.last_user_speech_at = now
        self._addr.last_user_turn_at = now
        self._ex.rounds = 0
        prune_stale(self._queue, self._fs)
        namesets = self._namesets()

        d = addressee_decision(text, self._addr, namesets, wake_hint=wake_hint, now=now,
                               sticky_secs=self.cfg.sticky_s)
        if d is None and self.arbiter is not None and len(self._present()) >= 2:
            d = await self._ask_arbiter(text)
        if d is None:
            if self._addr.last_speaker in self._slugs():
                d = Decision(self._addr.last_speaker, "fallback:last_speaker", False)
            else:
                d = Decision(self.fallback_slug, "fallback:default", False)

        present = self._present()
        minds, cue, absent_slug, both = [], None, None, False
        if d.addressee == BOTH:
            order = [s for s in name_order(text, namesets)]
            wanted = order + [s for s in sorted(self._slugs()) if s not in order]
            avail = [s for s in wanted if s in present]
            missing = [s for s in wanted if s not in present]
            if len(avail) >= 2:
                minds, both = avail[:2], True
            elif len(avail) == 1:
                minds = avail
                if missing:
                    absent_slug = missing[0]
                    cue = absent_cue(self._name(absent_slug), both=True)
        elif d.addressee in present:
            minds = [d.addressee]
        else:
            absent_slug = d.addressee
            if present:
                minds = [self._addr.last_speaker] if self._addr.last_speaker in present else [present[0]]
                cue = absent_cue(self._name(absent_slug))

        if not minds:
            self.memory.log_user_line(None, text, d.addressee, kind=f"{d.reason}|unrouted", now=now)
            self.log(f"floor: nobody home for '{text[:60]}' (addressee={d.addressee}, {d.reason})")
            return d

        errand = None
        if not both and cue is None:
            # "ask her what she wants changed": she asks in THIS reply, and reports back afterwards
            other = next((ns for ns in namesets if ns.slug != minds[0] and ns.slug in present), None)
            if other is not None and errand_to(text, other):
                errand = other.slug
                cue = errand_cue(self._name(other.slug))

        first = self._new_turn(now, mind=minds[0], kind="user", audible=True, addressee=HIM, text=text,
                               cue=cue, source_text=text, speaker_vec=speaker_vec, affect=affect,
                               wake_hint=wake_hint)
        if absent_slug is not None:
            self._waiting[absent_slug] = {"by": minds[0], "text": text, "turn": first.id, "at": now}
        turns = [first]
        if both:
            turns.append(self._new_turn(now, mind=minds[1], kind="both_second", audible=True, addressee=HIM,
                                        cue=both_second_cue(self._name(minds[0])), source_text=text,
                                        reply_to=first.id, speaker_vec=speaker_vec, affect=affect,
                                        wake_hint=wake_hint))
        label = BOTH if both else minds[0]
        kind = d.reason + (f"|absent:{absent_slug}" if absent_slug is not None else "")
        self.memory.log_user_line(first.id, text, label, kind=kind, now=now)
        self._addr.last_addressee = label
        self._addr.last_addressee_at = now

        if absent_slug is not None:
            absent_name = self._name(absent_slug)
            self._send(minds[0], proto.Absent(slug=absent_slug, name=absent_name), now)
            self.memory.add_event(absent_slug, "absent",
                                  f"{absent_name} isn't running; his line went to {self._name(minds[0])}",
                                  relayed_to=minds[0], now=now)
            self.log(f"floor: {absent_name} absent, routing to {minds[0]}")

        line = overheard_user_line("both of you" if both else self._name(minds[0]), text)
        for slug in present:
            if slug != minds[0]:
                self._send(slug, proto.Overheard(turn=first.id, line=line, speaker_vec=speaker_vec,
                                                 affect=affect), now)

        self._queue.extend(turns)
        self.log(f"floor: addressee={d.addressee} ({d.reason}) -> " + ", ".join(f"{t.mind}/{t.kind}" for t in turns)
                 + (f" (errand to {errand})" if errand else ""))
        self._kick(now)
        return d

    async def _ask_arbiter(self, text) -> Optional[Decision]:
        tail = list(reversed(self.memory.recent_transcript(limit=8)))
        blurbs = {slug: c.info.blurb for slug, c in self.clients.items()}
        msgs = arbiter_messages(text, tail, blurbs)
        try:
            res = self.arbiter(msgs)
            if inspect.isawaitable(res):
                res = await asyncio.wait_for(res, self.arbiter_timeout_s)
        except asyncio.TimeoutError:
            self.log("floor: arbiter timed out")
            return None
        except Exception as e:
            self.log(f"floor: arbiter failed: {e!r}")
            return None
        d = parse_arbiter_reply(res, slugs=tuple(self.clients.keys()))
        if d is None:
            self.log(f"floor: arbiter reply unusable: {str(getattr(res, 'text', res))[:80]!r}")
        return d

    # --- a sister asks for the floor --------------------------------------------------------------

    def want_floor(self, slug, kind="announce", to=None) -> Optional[Turn]:
        """Her `want_floor` message → a Turn (announce to him; initiative to him or to her sister,
        the latter text-only in private mode and capped per day). None when refused."""
        if slug not in self.clients:
            self.log(f"floor: want_floor from unknown {slug}")
            return None
        now = self._now()
        kind = kind if kind in ("announce", "initiative") else "announce"
        to = to or HIM
        if kind == "announce":
            turn = self._new_turn(now, mind=slug, kind="announce", audible=True, addressee=HIM)
        elif to in self._slugs() and to != slug:
            turn = self._new_turn(now, mind=slug, kind="initiative", audible=not self.private, addressee=to,
                                  cue=chat_cue(self._name(to)))
        else:
            turn = self._new_turn(now, mind=slug, kind="initiative", audible=True, addressee=HIM)
        return turn if self.request(turn) else None

    def request(self, turn: Turn) -> bool:
        """Queue an announce/initiative turn (or any pre-built Turn). Enforces the house-wide
        AIVA_SISTER_CHATS_PER_DAY cap on sister-addressed initiatives. False = refused."""
        now = self._now()
        if turn.mind not in self.clients:
            self.log(f"floor: request refused, {turn.mind} not connected")
            return False
        if turn.kind == "initiative" and turn.addressee != HIM:
            if turn.addressee not in self.clients:
                self.log(f"floor: chat refused, {turn.addressee} not connected")
                return False
            used = self.memory.day_count("sister_chats", now)
            if used >= self.cfg.chats_per_day:
                self.log(f"floor: chat refused, daily cap {used}/{self.cfg.chats_per_day}")
                return False
            self.memory.bump_day_count("sister_chats", now)
        queued = next((q for q in self._queue if q.kind == "announce" and q.mind == turn.mind), None)
        if turn.kind == "announce" and queued is not None:
            # her notes are already in her context; one run says them all — a second queued
            # announce would make her speak twice in a row about the same news. The queued one now
            # carries THIS newer note too, so it is as new as this one: a reply already being
            # written when it came didn't say it (2026-10-14: a task question was folded into an
            # older announce, which was then dropped as "spoken already" — she waited in silence)
            queued.created_at = max(queued.created_at, now)
            self.log(f"floor: collapsed a second queued announce from {turn.mind}")
            return False                          # nothing new queued (her note still gets said)
        if not turn.id or turn.id <= 0:
            self._seq += 1
            self.memory.kv_set("turn_seq", self._seq, now=now)
            turn.id = self._seq
        if not turn.created_at:
            turn.created_at = now
        self._remember(turn)
        self._queue.append(turn)
        self.log(f"floor: queued {turn.mind}/{turn.kind}#{turn.id} to={turn.addressee}")
        self._kick(now)
        return True

    # --- reports from the holder ------------------------------------------------------------------

    def on_response(self, slug, turn_id, phase):
        if not self._is_current(slug, turn_id):
            self.log(f"floor: stray response {phase} from {slug} turn {turn_id}")
            return
        now = self._now()
        if phase == "start":
            self._response_started = True
            self._response_open = True
            self._ack_deadline = None
            self._grace_deadline = None
            self._tool_calls_seen = 0
        elif phase == "end":
            self._response_open = False
            self._after_response_end(now)

    def _after_response_end(self, now):
        if self._tools_open > 0:
            return                              # tools still running; their -1 comes next
        if self._tool_calls_seen > 0:           # this response called tools → the LLM re-runs
            self._tool_calls_seen = 0
            self._ack_deadline = now + self.ack_timeout_s
            return
        turn = self._current
        if not turn.audible:
            self._release("end:private", now)
        elif self._interrupted:
            if not self._fs.speaking:
                self._release("end:interrupted", now)
        elif self._speaking_seen:
            if not self._fs.speaking:
                self._release("end:spoken", now)
        else:
            self._grace_deadline = now + self.cfg.speech_grace_s

    def on_tool(self, slug, turn_id, delta):
        if not self._is_current(slug, turn_id):
            return
        now = self._now()
        self._tools_open = max(0, self._tools_open + int(delta))
        if delta > 0:
            self._tool_calls_seen += 1
            self._ack_deadline = None
        elif self._tools_open == 0 and not self._response_open:
            self._tool_calls_seen = 0
            self._ack_deadline = now + self.ack_timeout_s     # expect the re-run's `response start`

    def on_speaking(self, slug, turn_id, on):
        if not self._is_current(slug, turn_id):
            # the holder's own report with a missing/stale id still counts — a lost `speaking off`
            # would otherwise pin the mic gate up until the 120 s cap
            if self._current is not None and self._current.mind == slug and self._fs.holder == slug:
                self.log(f"floor: {slug} speaking {'on' if on else 'off'} with turn {turn_id!r}; "
                         f"taking it for the current turn {self._current.id}")
                turn_id = self._current.id
            else:
                if on:
                    self.log(f"floor: {slug} speaking on stray turn {turn_id}")
                return
        now = self._now()
        self._fs.speaking = bool(on)
        if on:
            self._speaking_seen = True
            self._grace_deadline = None
            if self._current.audible:
                self._set_gate(True)
            return
        if self._interrupted:
            self._release("speaking_off:interrupted", now)
        elif (not self._response_open and self._tools_open == 0 and self._tool_calls_seen == 0
              and self._ack_deadline is None):
            self._release("speaking_off", now)

    def on_said(self, slug, turn_id, text, addressee=None, private=None, interrupted=False):
        now = self._now()
        text = (text or "").strip()
        turn = self._turns.get(turn_id)
        speaker_name = self._name(slug)
        addressee = addressee or (turn.addressee if turn else HIM)
        if addressee == HIM and text and not (turn and turn.kind == "report"):
            # "Hey, Saga. Fancy joining us?" is to her sister even on a turn he started, and "Pizza.
            # Saga, you in?" is to both of them — label the transcript and the overheard line by whom
            # she actually spoke to. A report is his, whatever it says (exchange_should_continue).
            to = self._spoken_to(slug, text)
            if to == BOTH or to in self._slugs():
                addressee = to
        audible = (not private) if private is not None else (turn.audible if turn else True)
        # her news was in her context when she spoke: an announce of hers queued BEFORE this turn
        # was granted is delivered. One queued mid-turn (a task question raised by the tool call
        # she just made) arrived after her reply was already being written — it keeps its own turn.
        cutoff = turn.granted_at if turn is not None and turn.granted_at else now
        self._drop_queued(lambda t: t.kind == "announce" and t.mind == slug and t.id != turn_id
                          and t.created_at <= cutoff, "spoken already")
        if text or interrupted:
            self.memory.log_said(turn_id, slug, text, addressee, audible, interrupted,
                                 kind=turn.kind if turn else "", now=now)
        if text:
            self._fs.last_said_at = now          # background news waits for quiet after this
        if text:
            for other in self._present():
                if other == slug:
                    continue
                to = ("you" if addressee == other else "him" if addressee == HIM
                      else "you and him" if addressee == BOTH else self._name(addressee))
                self._send(other, proto.Overheard(turn=turn_id, line=overheard_reply_line(speaker_name, to, text)), now)
        self._addr.last_speaker = slug
        self._addr.last_speaker_at = now
        self._addr.last_reply_was_question = is_question(text)

        if self._current is not None and self._current.id == turn_id:
            self._said_seen = True
        if self._closing == turn_id:
            self._close_turn()

        if turn is not None and text and not interrupted:
            self._maybe_continue(turn, slug, speaker_name, text, audible, now)
        self._kick(now)

    def _maybe_continue(self, turn, slug, speaker_name, text, audible, now):
        if self._fs.last_user_speech_at > turn.created_at:
            return                                  # he spoke since; the exchange is over
        if any(q.kind in ("user", "both_second") for q in self._queue):
            return
        rooted = self._rooted_in_user(turn)
        will_be_audible = (not self.private) or rooted
        other = exchange_should_continue(text, turn, self._addr, self._ex, self.cfg, now,
                                         namesets=self._namesets(), audible=will_be_audible)
        if (other and other in self.clients
                and not any(q.mind == other and q.kind in ("sister", "initiative") for q in self._queue)):
            self._ex.rounds += 1
            self._ex.hour_marks = [m for m in self._ex.hour_marks if now - m < 3600.0] + [now]
            nt = self._new_turn(now, mind=other, kind="sister", audible=will_be_audible, addressee=slug,
                                cue=sister_cue(speaker_name, not will_be_audible), source_text=text,
                                reply_to=turn.id)
            self._queue.append(nt)
            self.log(f"floor: sister round {self._ex.rounds}/{self.cfg.max_rounds}: {slug} -> {other}"
                     f" ({'audible' if will_be_audible else 'private'}"
                     f"{', he started it' if rooted and self.private else ''})")
            return
        if other is None and turn.kind in ("sister", "initiative") and self._ex.rounds >= self.cfg.max_rounds:
            self._ex.backoff_until = now + self.cfg.backoff_s
            self._ex.rounds = 0
            self.log(f"floor: exchange hit {self.cfg.max_rounds} rounds; backoff {int(self.cfg.backoff_s)} s")
        if other is None or other not in self.clients:          # the exchange is over
            self._maybe_report(turn, now)

    def _root_user_turn(self, turn) -> Optional[Turn]:
        t, seen = turn, 0
        while t is not None and seen < 20:
            if t.kind == "user":
                return t
            if t.kind in ("announce", "initiative", "intro", "both_second"):
                return None
            t = self._turns.get(t.reply_to) if t.reply_to else None
            seen += 1
        return None

    def _maybe_report(self, turn, now):
        """The sisters' exchange is over and it was HIS errand ("ask her … and let me know"): the
        sister he sent gets one turn to tell him how it went (2026-10-14: Saga answered Aiva with
        exact sizes and colours, and the loop ended there — he had played middle man all along)."""
        if turn.kind != "sister" or turn.addressee == HIM:
            return
        root = self._root_user_turn(turn)
        if root is None or root.id in self._reported or root.mind not in self.clients:
            return
        other = next((ns for ns in self._namesets() if ns.slug != root.mind), None)
        line = root.text or root.source_text or ""
        if other is None or not wants_report(line, other):
            return
        if any(q.kind == "report" for q in self._queue):
            return
        self._reported.add(root.id)
        nt = self._new_turn(now, mind=root.mind, kind="report", audible=True, addressee=HIM,
                            cue=report_cue(self._name(other.slug), line), source_text=line, reply_to=turn.id)
        self._queue.append(nt)
        self.log(f"floor: {root.mind} reports back to him (his errand, turn {root.id})")

    # --- release paths --------------------------------------------------------------------------------

    def _release(self, reason, now, tail=False):
        turn = self._current
        if turn is None:
            return
        self.log(f"floor: release {turn.mind} turn {turn.id} ({reason})")
        self._fs.holder = None
        self._fs.speaking = False
        self._ack_deadline = self._grace_deadline = self._cap_deadline = self._interrupt_deadline = None
        self._tools_open = 0
        self._tool_calls_seen = 0
        self._interrupted = False
        self._current = None
        if self._said_seen:
            self._fs.turn_id = None
        else:
            self._closing = turn.id
            self._closing_deadline = now + self.said_wait_s
        if tail and turn.audible:
            self._tail_deadline = now + self.death_tail_s          # gate stays up for the tail
        elif self._tail_deadline is None:
            self._set_gate(False)
        self._kick(now)

    def _close_turn(self):
        self._closing = None
        self._closing_deadline = None
        self._fs.turn_id = None

    def on_client_lost(self, slug):
        """She disconnected: drop her queued turns (re-route queued user turns to a present sister
        with the absent cue), release the floor if she held it (gate tail), close her pending said."""
        now = self._now()
        self.clients.pop(slug, None)
        name = self._name(slug)
        kept = []
        for t in self._queue:
            if t.mind != slug:
                kept.append(t)
            elif t.kind == "user" and self._reroute(t, slug, now):
                kept.append(t)
            else:
                self.log(f"floor: dropped {t.kind}#{t.id} for departed {slug}")
        self._queue[:] = kept
        if self._closing is not None:
            ct = self._turns.get(self._closing)
            if ct is not None and ct.mind == slug:
                self._close_turn()
        if self._fs.holder == slug:
            cur = self._current
            started = self._response_started      # read before _release() kicks the next grant
            self._said_seen = True                  # her said can't arrive any more
            self._release("client_lost", now, tail=True)
            if cur.kind == "user" and not started:
                nt = Turn(id=0, mind=slug, kind="user", audible=True, addressee=HIM, text=cur.text,
                          source_text=cur.source_text, created_at=now, speaker_vec=cur.speaker_vec,
                          affect=cur.affect)
                if self._reroute(nt, slug, now):
                    self._seq += 1
                    self.memory.kv_set("turn_seq", self._seq, now=now)
                    nt.id = self._seq
                    self._remember(nt)
                    self._queue.append(nt)
        self._kick(now)

    def _reroute(self, turn, absent_slug, now) -> bool:
        present = self._present()
        if not present:
            return False
        target = present[0]
        turn.mind = target
        turn.cue = absent_cue(self._name(absent_slug))
        self._send(target, proto.Absent(slug=absent_slug, name=self._name(absent_slug)), now)
        self.memory.add_event(absent_slug, "absent",
                              f"{self._name(absent_slug)} went away; his line went to {self._name(target)}",
                              relayed_to=target, now=now)
        self.log(f"floor: re-routed user turn to {target} ({absent_slug} absent)")
        return True

    def interrupt(self) -> bool:
        """F10: tell the holder (only) to stop; drop queued sister/initiative turns."""
        now = self._now()
        self._drop_queued(lambda t: t.kind in ("sister", "initiative"), "interrupt")
        if self._current is None:
            return False
        self._interrupted = True
        self._send(self._current.mind, proto.Interrupt(turn=self._current.id), now)
        self._interrupt_deadline = now + self.interrupt_fallback_s
        self.log(f"floor: interrupt -> {self._current.mind} turn {self._current.id}")
        return True

    # --- mode / user activity ------------------------------------------------------------------------

    def set_private(self, private: bool) -> bool:
        """Flip sister talk between text-only and spoken; persisted, broadcast as `mode`.
        Refuses 'audible' while no connected sister has a voice."""
        private = bool(private)
        if not private and not any(getattr(c.info, "has_voice", True) for c in self.clients.values()):
            self.log("floor: audible sister talk refused — no connected sister has a voice")
            return False
        now = self._now()
        self.private = private
        self.memory.kv_set("sister_talk", self.mode, now=now)
        for t in self._queue:
            if t.kind == "sister" or (t.kind == "initiative" and t.addressee != HIM):
                t.audible = self._sister_audible(t)
        self._broadcast(proto.Mode(sister_talk=self.mode), now)
        self.log(f"floor: sister talk -> {self.mode}")
        return True

    def is_muted(self, slug) -> bool:
        return (slug or "").strip().lower() in self._muted

    def set_muted(self, slug, muted: bool) -> bool:
        """Mute/unmute a sister's VOICE. A muted sister still gets her turns, thinks, and logs — but
        every turn is forced non-audible (text-only), so she makes no sound and spends no TTS, exactly
        like a private sister exchange. Persisted in house_kv; applies from her next granted turn."""
        slug = (slug or "").strip().lower()
        if not slug:
            return False
        muted = bool(muted)
        (self._muted.add if muted else self._muted.discard)(slug)
        self.memory.kv_set("muted_slugs", ",".join(sorted(self._muted)), now=self._now())
        self.log(f"floor: {slug} voice {'muted' if muted else 'unmuted'}")
        return True

    def note_user_speaking(self, on: bool):
        now = self._now()
        on = bool(on)
        self._fs.user_speaking = on
        self._ex.user_speaking = on
        self._fs.last_user_speech_at = now
        prune_stale(self._queue, self._fs)
        if not on:
            self._kick(now)

    def set_awake(self, awake: bool):
        self._ex.awake = bool(awake)

    # --- timers ----------------------------------------------------------------------------------------

    def tick(self, now=None):
        """Drive every deadline. Call often (the server: every ~100 ms); tests: with a fake clock."""
        now = now if now is not None else self._now()
        if self._ack_deadline is not None and now >= self._ack_deadline:
            self._said_seen = True                  # she never started; no said will come
            self._release("no_response_start", now)
        if self._grace_deadline is not None and now >= self._grace_deadline:
            self._release("grace", now)
        if self._cap_deadline is not None and now >= self._cap_deadline and self._current is not None:
            self._send(self._current.mind, proto.Interrupt(turn=self._current.id), now)
            self._release("cap", now)
        if self._interrupt_deadline is not None and now >= self._interrupt_deadline:
            self._release("interrupt_fallback", now)
        if self._closing_deadline is not None and now >= self._closing_deadline:
            self.log(f"floor: no said for turn {self._closing}; closing")
            self._close_turn()
        if self._tail_deadline is not None and now >= self._tail_deadline:
            self._tail_deadline = None
            if not (self._current is not None and self._current.audible):
                self._set_gate(False)
        self._kick(now)

    # --- granting ------------------------------------------------------------------------------------------

    def _kick(self, now=None):
        if self._fs.turn_id is not None:
            return
        now = now if now is not None else self._now()
        while True:
            turn, reason = floor_decision(self._queue, self._fs, now, announce_quiet_s=self.announce_quiet_s)
            if turn is None:
                return
            self._queue.remove(turn)
            if turn.mind not in self.clients:
                self.log(f"floor: dropped {turn.kind}#{turn.id}, {turn.mind} not connected")
                continue
            self._grant(turn, now)
            return

    def _grant(self, turn, now):
        if turn.kind == "sister" or (turn.kind == "initiative" and turn.addressee != HIM):
            turn.audible = self._sister_audible(turn)
        if turn.mind in self._muted:            # a muted sister is silent on every turn (text-only)
            turn.audible = False
        turn.granted_at = now
        self._current = turn
        self._fs.holder = turn.mind
        self._fs.turn_id = turn.id
        self._fs.speaking = False
        self._response_started = self._response_open = False
        self._speaking_seen = self._said_seen = self._interrupted = False
        self._tools_open = self._tool_calls_seen = 0
        self._ack_deadline = now + self.ack_timeout_s
        self._cap_deadline = now + self.turn_cap_s
        self._grace_deadline = self._interrupt_deadline = None
        self._set_gate(turn.audible or self._tail_deadline is not None)
        if self.on_grant is not None:
            try:
                self.on_grant(turn)
            except Exception as e:
                self.log(f"floor: on_grant failed: {e!r}")
        self._send(turn.mind, proto.TurnMsg(turn=turn.id, kind=turn.kind, audible=turn.audible,
                                             addressee=turn.addressee, text=turn.text, cue=turn.cue,
                                             speaker_vec=turn.speaker_vec, affect=turn.affect,
                                             wake_hint=turn.wake_hint, reply_to=turn.reply_to), now)
        self.log(f"floor: grant {turn.mind} turn {turn.id} kind={turn.kind} audible={turn.audible} to={turn.addressee}")
