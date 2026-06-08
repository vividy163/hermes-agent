"""Gateway-side clarify primitive (blocking event-based queue).

The ``clarify`` tool needs to ask the user a question and block the agent
thread until they respond.  In CLI mode this is trivial — ``input()`` is
synchronous.  In gateway mode the agent runs on a worker thread while the
event loop handles the user's reply, so we need a thread-safe primitive
that:

  * stores a pending clarify request (with a generated ``clarify_id``),
  * blocks the agent thread on an ``Event``,
  * resolves the wait when the gateway's button-callback or text-intercept
    fires ``resolve_gateway_clarify(clarify_id, response)``,
  * supports timeouts so a user who never responds does NOT hang the agent
    thread forever (which would also pin the gateway's running-agent guard).

State is module-level (same shape as ``tools.approval``) so platform
adapters can call ``resolve_gateway_clarify`` without holding a back-
reference to the ``GatewayRunner`` instance.

Two delivery paths from the adapter:

  1. **Button UI** — adapters override ``send_clarify`` to render inline
     buttons (e.g. Telegram ``InlineKeyboardMarkup``).  The button
     callback resolves with the chosen string.  A final "Other (type
     answer)" button enters text-capture mode for free-form responses.

  2. **Text fallback** — adapters without rich UI render a numbered list.
     The user replies with a number ("2") or with free text; the gateway's
     ``_handle_message`` intercepts the reply and resolves directly.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# Module-level state
# =========================================================================

@dataclass
class _ClarifyEntry:
    """One pending clarify request inside a gateway session."""
    clarify_id: str
    session_key: str
    question: str
    choices: Optional[List[str]]
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[str] = None
    awaiting_text: bool = False  # set when user picked "Other" or clarify is open-ended

    def signature(self) -> Dict[str, object]:
        return {
            "clarify_id": self.clarify_id,
            "session_key": self.session_key,
            "question": self.question,
            "choices": list(self.choices) if self.choices else None,
        }


_lock = threading.RLock()
# clarify_id → _ClarifyEntry  (primary lookup for button callbacks)
_entries: Dict[str, _ClarifyEntry] = {}
# session_key → list[clarify_id]  (FIFO; for text-fallback intercept and session cleanup)
_session_index: Dict[str, List[str]] = {}


# =========================================================================
# Public API — agent-thread side
# =========================================================================

def register(
    clarify_id: str,
    session_key: str,
    question: str,
    choices: Optional[List[str]],
) -> _ClarifyEntry:
    """Register a pending clarify request and return the entry.

    The caller (gateway clarify_callback) will then send the prompt to the
    user and block on ``wait_for_response(clarify_id, timeout)``.
    """
    entry = _ClarifyEntry(
        clarify_id=clarify_id,
        session_key=session_key,
        question=question,
        choices=list(choices) if choices else None,
        # Open-ended (no choices) → next message IS the response, no buttons needed.
        awaiting_text=not bool(choices),
    )
    with _lock:
        _entries[clarify_id] = entry
        _session_index.setdefault(session_key, []).append(clarify_id)
    return entry


def wait_for_response(clarify_id: str, timeout: float) -> Optional[str]:
    """Block on the entry's event until resolved or timeout fires.

    Polls in 1-second slices so the agent's inactivity heartbeat keeps
    firing — without this, ``Event.wait(timeout=600)`` blocks the thread
    for 10 minutes with zero activity touches and the gateway's inactivity
    watchdog kills the agent while the user is still typing.

    Returns the resolved response string, or ``None`` on timeout.
    """
    with _lock:
        entry = _entries.get(clarify_id)
    if entry is None:
        return None

    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - optional
        touch_activity_if_due = None

    deadline = time.monotonic() + max(timeout, 0.0)
    activity_state = {"last_touch": time.monotonic(), "start": time.monotonic()}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if entry.event.wait(timeout=min(1.0, remaining)):
            break
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for user clarify response")

    with _lock:
        # Remove from indices regardless of resolution outcome.
        _entries.pop(clarify_id, None)
        ids = _session_index.get(entry.session_key)
        if ids and clarify_id in ids:
            ids.remove(clarify_id)
            if not ids:
                _session_index.pop(entry.session_key, None)

    return entry.response


# =========================================================================
# Public API — gateway / adapter side
# =========================================================================

def resolve_gateway_clarify(clarify_id: str, response: str, *, source: str = "text") -> bool:
    """Unblock the agent thread waiting on ``clarify_id``.

    Args:
        clarify_id: The pending entry to resolve.
        response: The user's chosen text (or free-form reply).
        source: Where the resolve came from. One of:
            ``"button"`` — user clicked a UI button (rich adapter, e.g.
                Telegram ``InlineKeyboardButton`` or Feishu card action).
                After-resolve hooks registered with ``fire_on=("button",)``
                fire; those registered for text-only (the Feishu PATCH
                hook) are skipped, since the button path already updated
                the message in place via the click-ack response.
            ``"text"`` — user typed a free-form reply captured by the
                gateway's text-intercept. Default. Fires all registered
                hooks whose ``fire_on`` includes ``"text"``; this is the
                only path that needs a server-side PATCH to update the
                sent card (no click-ack was generated).
            ``"auto"`` — explicit "fire unconditionally" sentinel for
                tests and edge cases (timeout, session reset). Fires all
                registered hooks regardless of ``fire_on``.

    Returns True if an entry was found and resolved, False otherwise
    (already resolved, expired, or never existed).
    """
    if source not in ("button", "text", "auto"):
        # Defensive: an unknown source means a caller is using the API
        # wrong. Fall back to "text" (the historically-safe behavior,
        # which is also the only path that exists for non-rich adapters).
        logger.warning(
            "[clarify_gateway] resolve_gateway_clarify got unknown source=%r; "
            "treating as 'text'",
            source,
        )
        source = "text"
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None:
            return False
    entry.response = str(response) if response is not None else ""
    entry.event.set()
    # Fire-and-forget after-resolve side effects (e.g. card PATCH).
    # Released the lock before invoking so the hook can re-enter
    # clarify_gateway APIs (e.g. clear_session) without deadlocking.
    _fire_after_resolve(clarify_id, entry.response, source=source)
    return True


def get_pending_for_session(session_key: str) -> Optional[_ClarifyEntry]:
    """Return the OLDEST pending clarify entry for a session, or None.

    Used by the text-fallback intercept in ``_handle_message`` — when a
    clarify is awaiting a free-form text response, the next user message
    in that session is captured as the answer.
    """
    with _lock:
        ids = _session_index.get(session_key) or []
        for cid in ids:
            entry = _entries.get(cid)
            if entry is None:
                continue
            if entry.awaiting_text:
                return entry
        return None


def mark_awaiting_text(clarify_id: str) -> bool:
    """Flip an entry into text-capture mode (user picked the 'Other' button).

    Returns True if the entry exists and was flipped, False otherwise.
    """
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None:
            return False
        entry.awaiting_text = True
        return True


# Per-clarify after-resolve hooks.
#
# Adapters that need to run side effects when a clarify entry resolves
# (e.g. Feishu PATCHing a sent card to "received" state) can register a
# one-shot callback keyed by ``clarify_id``.  When ``resolve_gateway_clarify``
# actually fires the resolve, the hook is invoked with the user's chosen
# text and then consumed.  Exceptions raised by the hook are logged and
# swallowed so a buggy adapter can never block the agent thread.
#
# This sits next to the existing ``register_notify`` pattern but at a
# finer granularity (per-clarify rather than per-session) so that the
# hook can carry the message_id or other call-site context that
# ``register_notify`` doesn't have access to.
#
# Each entry stores ``(callback, fire_on)`` where ``fire_on`` is the set
# of resolve sources the hook wants to fire on.  The default
# ``("text",)`` matches the historical "fire on every resolve" behavior
# closely enough that existing tests pass — but a hook that wants to
# distinguish "I just got a click" from "I just got a typed reply" can
# opt into ``fire_on=("button",)`` (or both, or ``("auto",)`` for the
# timeout/clear_session path).
_AfterResolveCB = Tuple[Callable[[str], None], Tuple[str, ...]]
_after_resolve_cbs: Dict[str, _AfterResolveCB] = {}


def register_after_resolve(
    clarify_id: str,
    callback: Callable[[str], None],
    *,
    fire_on: Tuple[str, ...] = ("text",),
) -> bool:
    """Register a one-shot callback fired when ``clarify_id`` resolves.

    The callback receives the resolved choice text as its sole argument.
    Returns True if registered against an existing entry, False otherwise
    (the entry may have already been resolved/cleared).

    Args:
        clarify_id: The pending entry to hook onto.
        callback: Function called with the resolved text.
        fire_on: Tuple of resolve sources this hook cares about. The
            callback fires only when ``resolve_gateway_clarify`` is
            called with a matching ``source`` (or with ``source="auto"``,
            which fires every hook regardless of ``fire_on``).
            Default ``("text",)`` preserves the historical "fire on
            resolve" behavior for callers that don't care about the
            source.  Use ``("button",)`` for side effects that should
            only run on UI button clicks (e.g. the Feishu PATCH hook,
            which the click-ack already covered), or ``("button",
            "text")`` for hooks that need both.
    """
    # Normalize and validate fire_on.
    normalized = tuple(s for s in fire_on if s in ("button", "text", "auto"))
    if not normalized:
        # Empty fire_on means "never fire" — almost certainly a caller
        # bug. Surface it loudly and refuse the registration so the
        # adapter code can fail fast in tests.
        logger.warning(
            "[clarify_gateway] register_after_resolve: fire_on=%r is empty after "
            "normalization; refusing to register hook for clarify_id=%s",
            fire_on,
            clarify_id,
        )
        return False
    with _lock:
        if _entries.get(clarify_id) is None:
            return False
        _after_resolve_cbs[clarify_id] = (callback, normalized)
    return True


def _fire_after_resolve(clarify_id: str, choice_text: str, *, source: str) -> None:
    """Pop and invoke the registered hook if its ``fire_on`` matches ``source``.

    Errors raised by the hook are swallowed (logged). A hook registered
    with a non-matching ``fire_on`` is consumed silently (popped from the
    dict) so it does not leak across clarifies. ``source="auto"`` matches
    every registered hook (the timeout/clear_session path).
    """
    with _lock:
        entry = _after_resolve_cbs.pop(clarify_id, None)
        total = len(_after_resolve_cbs)
    if entry is None:
        logger.debug(
            "[clarify_gateway] _fire_after_resolve: no hook for clarify_id=%s "
            "(remaining_hooks=%d, choice_text=%r, source=%r)",
            clarify_id,
            total,
            choice_text,
            source,
        )
        return
    callback, fire_on = entry
    if source != "auto" and source not in fire_on:
        logger.debug(
            "[clarify_gateway] _fire_after_resolve: hook for clarify_id=%s "
            "skipped (source=%r not in fire_on=%r, choice_text=%r, "
            "remaining_hooks=%d)",
            clarify_id,
            source,
            fire_on,
            choice_text,
            total,
        )
        return
    logger.debug(
        "[clarify_gateway] _fire_after_resolve: firing hook for clarify_id=%s "
        "(choice_text=%r, source=%r, fire_on=%r, remaining_hooks=%d)",
        clarify_id,
        choice_text,
        source,
        fire_on,
        total,
    )
    try:
        callback(choice_text)
    except Exception:
        logger.warning(
            "[clarify_gateway] after_resolve hook for %s raised; swallowed",
            clarify_id,
            exc_info=True,
        )


def has_pending(session_key: str) -> bool:
    """Return True when this session has at least one pending clarify entry."""
    with _lock:
        ids = _session_index.get(session_key) or []
        return any(_entries.get(cid) is not None for cid in ids)


def clear_session(session_key: str) -> int:
    """Resolve and drop every pending clarify for a session.

    Used by session-boundary cleanup (e.g. ``/new``, gateway shutdown,
    cached-agent eviction) so blocked agent threads don't hang past the
    end of their session.  Returns the number of entries cancelled.
    """
    with _lock:
        ids = list(_session_index.pop(session_key, []) or [])
        entries = [_entries.pop(cid, None) for cid in ids]
    cancelled = 0
    for entry in entries:
        if entry is None:
            continue
        # Empty string sentinel — agent code can distinguish from a real
        # response by inspecting the wait_for_response return value
        # alongside its own timeout deadline.  Most callers just treat any
        # falsy result as "user did not respond".
        entry.response = ""
        entry.event.set()
        cancelled += 1
    return cancelled


# =========================================================================
# Config
# =========================================================================

def get_clarify_timeout() -> int:
    """Read the clarify response timeout (seconds) from config.

    Defaults to 600 (10 minutes) — long enough for the user to type a
    thoughtful response, short enough that an abandoned prompt eventually
    unblocks the agent thread instead of pinning the running-agent guard
    forever.

    Reads ``agent.clarify_timeout`` from config.yaml.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        agent_cfg = cfg.get("agent", {}) or {}
        return int(agent_cfg.get("clarify_timeout", 600))
    except Exception:
        return 600


# =========================================================================
# Per-session notify hook (gateway → adapter bridge)
# =========================================================================
# Mirrors tools.approval's _gateway_notify_cbs: the gateway registers a
# per-session callback that sends the clarify prompt to the user.  The
# callback bridges sync→async (runs on the agent thread; schedules the
# adapter ``send_clarify`` call on the event loop).

_notify_cbs: Dict[str, Callable[[_ClarifyEntry], None]] = {}


def register_notify(session_key: str, cb: Callable[[_ClarifyEntry], None]) -> None:
    """Register a per-session notify callback used by ``clarify_callback``."""
    with _lock:
        _notify_cbs[session_key] = cb


def unregister_notify(session_key: str) -> None:
    """Drop the per-session notify callback and cancel any pending clarify entries."""
    with _lock:
        _notify_cbs.pop(session_key, None)
    # Cancel any pending entries so blocked threads unwind when the run
    # ends (interrupt, completion, gateway shutdown).
    clear_session(session_key)


def get_notify(session_key: str) -> Optional[Callable[[_ClarifyEntry], None]]:
    with _lock:
        return _notify_cbs.get(session_key)
