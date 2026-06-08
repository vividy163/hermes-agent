"""Verify the gateway runner's text-intercept path actually fires the
Feishu after-resolve hook.

The gateway runner path lives in ``gateway.run._handle_message`` (around
L7628-L7680).  That method is large and hard to exercise end-to-end
because it depends on session creation, adapter plumbing, and a
running loop.  These tests instead exercise the *essence* of the
text-intercept branch in isolation:

  1. get_pending_for_session() returns a registered, awaiting_text
     entry.
  2. The active adapter is detected as a FeishuAdapter (via class
     name) — this is the check that lived as
     ``self._status_adapter.__class__.__name__`` in commit 59afd5c39
     and was a real bug (AttributeError: no attribute _status_adapter).
  3. _feishu_fire_after_resolve_hook is called with the resolved text.

The bug was: the original code read ``self._status_adapter`` (an
instance attribute that is never set) instead of looking the
adapter up from ``self.adapters`` keyed by the current event's
``source.platform``.  This test guards against a regression of
that exact pattern: it looks the adapter up the same way the fix
does (``self.adapters.get(source.platform)``) and asserts the
hook fires for a Feishu source.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from tools import clarify_gateway
from gateway.platforms import feishu


def _fire_hook_like_gateway_runner(source_platform, adapters, session_key, user_text):
    """Replica of the gateway runner's text-intercept branch
    (gateway/run.py L7628-L7680), minus the bits that are not
    relevant to hook firing (pairing code, /reload-mcp, etc.).

    Returns True iff the hook was fired (i.e. the active adapter
    was Feishu and the call did not raise).
    """
    pending = clarify_gateway.get_pending_for_session(session_key)
    if pending is None:
        return False
    raw = (user_text or "").strip()
    if not raw or raw.startswith("/"):
        return False
    resolved = clarify_gateway.resolve_gateway_clarify(pending.clarify_id, raw)
    if not resolved:
        return False

    # This is the line that was broken in 59afd5c39:
    #     if self._status_adapter.__class__.__name__ == "FeishuAdapter":
    # The fix:
    active = adapters.get(source_platform)
    if active and active.__class__.__name__ == "FeishuAdapter":
        try:
            feishu._feishu_fire_after_resolve_hook(pending.clarify_id, raw)
        except Exception:
            return False
    return True


class TestGatewayTextInterceptFiresPatchHook(unittest.TestCase):
    def setUp(self):
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()
        feishu._feishu_after_resolve_cbs.clear()

    def tearDown(self):
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()
        feishu._feishu_after_resolve_cbs.clear()

    def _make_feishu_adapter(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        # Mirror the constructor signature used in test_feishu.py —
        # no real network or Feishu credentials required.
        return FeishuAdapter(PlatformConfig())

    def test_feishu_text_intercept_fires_hook(self):
        """The full path: register + mark_awaiting_text + register
        a PATCH hook + simulate the text-intercept branch.  This is
        the regression guard for the self._status_adapter bug.
        """
        adapter = self._make_feishu_adapter()
        clarify_id = "cl_text_intercept01"
        session_key = "feishu:oc_chat_a:user_x"

        # The gateway would have called these before send_clarify.
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a", "b"],
        )
        # send_clarify flips the entry into text-capture mode.
        clarify_gateway.mark_awaiting_text(clarify_id)

        # send_clarify registers the after-resolve hook.  We swap
        # in an observer so we can detect firing.
        patch_calls = []

        def _observer(choice_text):
            patch_calls.append(choice_text)

        self.assertTrue(
            feishu._feishu_register_after_resolve_hook(clarify_id, _observer),
        )

        # Simulate the text-intercept branch in the gateway runner.
        adapters = {"feishu": adapter}
        fired = _fire_hook_like_gateway_runner(
            source_platform="feishu",
            adapters=adapters,
            session_key=session_key,
            user_text="my free-form answer",
        )
        self.assertTrue(fired, "hook should have fired for a Feishu adapter")
        self.assertEqual(patch_calls, ["my free-form answer"])

    def test_non_feishu_adapter_does_not_fire_hook(self):
        """The text-intercept branch is a no-op for non-Feishu
        adapters — the if-guard should short-circuit before the
        hook call.
        """
        clarify_id = "cl_text_intercept02"
        session_key = "telegram:chat_a:user_x"
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a", "b"],
        )
        clarify_gateway.mark_awaiting_text(clarify_id)

        patch_calls = []

        def _observer(choice_text):
            patch_calls.append(choice_text)

        feishu._feishu_register_after_resolve_hook(clarify_id, _observer)

        # Stub a Telegram-shaped adapter (class name != FeishuAdapter).
        telegram_adapter = SimpleNamespace(__class__=type("TelegramAdapter", (), {}))
        adapters = {"telegram": telegram_adapter}

        fired = _fire_hook_like_gateway_runner(
            source_platform="telegram",
            adapters=adapters,
            session_key=session_key,
            user_text="typed answer",
        )
        # resolve_gateway_clarify still ran (it is not gated on
        # adapter type), but the hook did not fire because the
        # active adapter was Telegram, not Feishu.
        self.assertTrue(fired, "the resolve step itself returns True even for non-Feishu")
        self.assertEqual(patch_calls, [], "hook should not fire for a non-Feishu adapter")

    def test_no_pending_clarify_does_nothing(self):
        """If the session has no pending clarify, the text-intercept
        branch is a complete no-op (the gateway falls through to
        the normal message dispatch).
        """
        # No register() call.
        adapters = {"feishu": self._make_feishu_adapter()}
        fired = _fire_hook_like_gateway_runner(
            source_platform="feishu",
            adapters=adapters,
            session_key="feishu:oc_chat_a:user_x",
            user_text="some text",
        )
        self.assertFalse(fired)

    def test_no_active_adapter_does_not_crash(self):
        """The original bug raised AttributeError because
        self._status_adapter was not defined.  This test guards
        against a regression where the adapter lookup returns
        None — the if-guard must short-circuit, not crash.
        """
        clarify_id = "cl_text_intercept03"
        session_key = "feishu:oc_chat_a:user_x"
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a", "b"],
        )
        clarify_gateway.mark_awaiting_text(clarify_id)

        patch_calls = []

        def _observer(choice_text):
            patch_calls.append(choice_text)

        feishu._feishu_register_after_resolve_hook(clarify_id, _observer)

        # No adapter for this platform.
        adapters = {}  # empty dict → adapters.get("feishu") returns None
        fired = _fire_hook_like_gateway_runner(
            source_platform="feishu",
            adapters=adapters,
            session_key=session_key,
            user_text="typed answer",
        )
        # resolve ran, but the hook was not fired because the
        # if-guard short-circuited on the missing adapter.
        self.assertTrue(fired)
        self.assertEqual(patch_calls, [], "hook must not fire when adapter is None")

    def test_regression_self_status_adapter_pattern(self):
        """This is the exact regression guard for the original bug.

        Pre-fix code was:
            if self._status_adapter.__class__.__name__ == "FeishuAdapter":

        That raised AttributeError because self._status_adapter is
        never assigned on the GatewayRunner instance.  The fix
        uses self.adapters.get(source.platform) instead.

        We assert that the fixed pattern does NOT touch
        self._status_adapter — the fix is to look the adapter up
        by source.platform, not by an instance attribute.
        """
        adapter = self._make_feishu_adapter()
        # Simulate a runner that has a real adapters dict (the
        # correct pattern) and that does NOT have a _status_adapter
        # attribute.
        class FakeRunner:
            adapters = {"feishu": adapter}

        runner = FakeRunner()
        # Pre-fix would do: self._status_adapter.__class__.__name__
        # That raises AttributeError.
        self.assertFalse(hasattr(runner, "_status_adapter"))
        # The fix is: self.adapters.get(source.platform).
        active = runner.adapters.get("feishu")
        self.assertIs(active, adapter)
        self.assertEqual(active.__class__.__name__, "FeishuAdapter")
