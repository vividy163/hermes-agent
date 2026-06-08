"""Tests for the Feishu-specific after-resolve hook.

The hook lives in ``gateway.platforms.feishu`` (as
``_feishu_register_after_resolve_hook`` /
``_feishu_fire_after_resolve_hook``) because only the Feishu adapter
needs to PATCH a sent card to the "received" state on a typed reply
— Telegram edits the message at click time and the TUI never goes
through the gateway.

These tests cover the hook semantics: one-shot, exception-swallowing,
"unknown id" rejection, and that a second resolve does not re-fire.
"""

import unittest

from tools import clarify_gateway
from gateway.platforms import feishu


class TestFeishuAfterResolveHook(unittest.TestCase):
    """Verify the Feishu-only after-resolve hook semantics."""

    def setUp(self):
        feishu._feishu_after_resolve_cbs.clear()
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()

    def tearDown(self):
        feishu._feishu_after_resolve_cbs.clear()
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()

    def test_register_returns_true_for_known_id(self):
        clarify_id = "clt_feishu_hook_register"
        session_key = "feishu:oc_a:user_x"
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a", "b"],
        )
        try:
            result = feishu._feishu_register_after_resolve_hook(
                clarify_id, lambda text: None,
            )
            self.assertTrue(result)
        finally:
            clarify_gateway.clear_session(session_key)

    def test_register_returns_false_for_unknown_id(self):
        result = feishu._feishu_register_after_resolve_hook(
            "clt_feishu_unknown", lambda text: None,
        )
        self.assertFalse(result)

    def test_fire_invokes_hook_with_choice_text(self):
        clarify_id = "clt_feishu_hook_fire"
        session_key = "feishu:oc_a:user_x"
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a", "b"],
        )
        captured = []

        def _hook(choice_text: str) -> None:
            captured.append(choice_text)

        try:
            self.assertTrue(
                feishu._feishu_register_after_resolve_hook(clarify_id, _hook),
            )
            feishu._feishu_fire_after_resolve_hook(clarify_id, "a")
            self.assertEqual(captured, ["a"])
        finally:
            clarify_gateway.clear_session(session_key)

    def test_hook_consumed_after_fire(self):
        """Once fired, the hook is removed so a stray second fire
        does not re-trigger the side effect.
        """
        clarify_id = "clt_feishu_hook_consumed"
        session_key = "feishu:oc_a:user_x"
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a"],
        )
        fired = []

        def _hook(choice_text: str) -> None:
            fired.append(choice_text)

        try:
            feishu._feishu_register_after_resolve_hook(clarify_id, _hook)
            feishu._feishu_fire_after_resolve_hook(clarify_id, "a")
            # Re-fire (no entry in cbs after the first fire).
            feishu._feishu_fire_after_resolve_hook(clarify_id, "b")
            self.assertEqual(fired, ["a"])
        finally:
            clarify_gateway.clear_session(session_key)

    def test_hook_exception_is_swallowed(self):
        clarify_id = "clt_feishu_hook_exc"
        session_key = "feishu:oc_a:user_x"
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a"],
        )

        def _exploding_hook(choice_text: str) -> None:
            raise RuntimeError("boom")

        try:
            feishu._feishu_register_after_resolve_hook(clarify_id, _exploding_hook)
            # Should not raise.
            feishu._feishu_fire_after_resolve_hook(clarify_id, "a")
        finally:
            clarify_gateway.clear_session(session_key)

    def test_fire_with_no_hook_is_noop(self):
        """Firing a clarify_id that has no registered hook must not raise.
        """
        # No registration.
        feishu._feishu_fire_after_resolve_hook("clt_feishu_no_hook", "x")

    def test_clear_drops_hook_without_firing(self):
        """The gateway's clear_session path uses this to drop hooks
        for clarifies that never got answered.
        """
        clarify_id = "clt_feishu_hook_clear"
        session_key = "feishu:oc_a:user_x"
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a"],
        )
        fired = []

        def _hook(choice_text: str) -> None:
            fired.append(choice_text)

        try:
            feishu._feishu_register_after_resolve_hook(clarify_id, _hook)
            feishu._feishu_clear_after_resolve_hook(clarify_id)
            feishu._feishu_fire_after_resolve_hook(clarify_id, "a")
            self.assertEqual(fired, [])
        finally:
            clarify_gateway.clear_session(session_key)

    def test_concurrent_register_and_fire(self):
        """A race between register and the text-intercept path firing
        the hook should not deadlock; if the entry vanished between
        register-check and store, the register returns False and no
        fire happens.
        """
        import threading
        from tools.clarify_gateway import _lock  # type: ignore[attr-defined]

        clarify_id = "clt_feishu_hook_race"
        session_key = "feishu:oc_a:user_x"
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a"],
        )

        captured = []

        def _hook(choice_text: str) -> None:
            captured.append(choice_text)

        # Drop the entry under the lock to simulate timeout/clear.
        with _lock:
            clarify_gateway._entries.pop(clarify_id, None)
        result = feishu._feishu_register_after_resolve_hook(clarify_id, _hook)
        self.assertFalse(result)
        feishu._feishu_fire_after_resolve_hook(clarify_id, "a")
        self.assertEqual(captured, [])

        clarify_gateway.clear_session(session_key)
