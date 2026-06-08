"""Tests for tools.clarify_gateway after-resolve hook API.

The hook is used by Feishu adapter (and any other adapter that needs
post-resolve side effects, e.g. PATCHing a sent card to "received" state)
to register a callback that fires once when ``resolve_gateway_clarify``
is called for a given ``clarify_id``.
"""

import os
import unittest
from unittest.mock import patch

from tools import clarify_gateway


class TestAfterResolveHook(unittest.TestCase):
    """Verify register_after_resolve + _fire_after_resolve semantics."""

    def setUp(self):
        clarify_gateway._after_resolve_cbs.clear()
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()

    def tearDown(self):
        clarify_gateway._after_resolve_cbs.clear()
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()

    def test_register_after_resolve_returns_true_for_known_id(self):
        clarify_id = "clt_hook_register"
        session_key = "feishu:oc_a:user_x"
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a", "b"],
        )
        try:
            result = clarify_gateway.register_after_resolve(clarify_id, lambda text: None)
            self.assertTrue(result)
        finally:
            clarify_gateway.clear_session(session_key)

    def test_register_after_resolve_returns_false_for_unknown_id(self):
        result = clarify_gateway.register_after_resolve("clt_unknown", lambda text: None)
        self.assertFalse(result)

    def test_resolve_gateway_clarify_fires_after_resolve_hook_with_choice_text(self):
        clarify_id = "clt_hook_fire"
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
            self.assertTrue(clarify_gateway.register_after_resolve(clarify_id, _hook))
            self.assertTrue(clarify_gateway.resolve_gateway_clarify(clarify_id, "a"))
            self.assertEqual(captured, ["a"])
        finally:
            clarify_gateway.clear_session(session_key)

    def test_after_resolve_hook_consumed_after_fire(self):
        """Once fired, the hook should be removed so a stray second resolve
        does not re-trigger the side effect.
        """
        clarify_id = "clt_hook_consumed"
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
            clarify_gateway.register_after_resolve(clarify_id, _hook)
            clarify_gateway.resolve_gateway_clarify(clarify_id, "a")
            # Re-resolve (no-op for already-resolved entry) — hook should not fire again.
            clarify_gateway.resolve_gateway_clarify(clarify_id, "b")
            self.assertEqual(fired, ["a"])
        finally:
            clarify_gateway.clear_session(session_key)

    def test_after_resolve_hook_exception_is_swallowed(self):
        clarify_id = "clt_hook_exc"
        session_key = "feishu:oc_a:user_x"
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a"],
        )
        try:
            def _bad_hook(_text: str) -> None:
                raise RuntimeError("hook intentionally failed")

            clarify_gateway.register_after_resolve(clarify_id, _bad_hook)
            # Should not raise, should still return True (resolved happened).
            resolved = clarify_gateway.resolve_gateway_clarify(clarify_id, "a")
            self.assertTrue(resolved)
        finally:
            clarify_gateway.clear_session(session_key)


class TestAfterResolveFireOn(unittest.TestCase):
    """Verify the fire_on / source filtering added to avoid the
    double-update race on platforms with a click-ack (Feishu) and a
    follow-up PATCH hook.  See
    ``feishu-clarify-card-debugging`` SKILL (Class on adapter-race).
    """

    def setUp(self):
        clarify_gateway._after_resolve_cbs.clear()
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()

    def tearDown(self):
        clarify_gateway._after_resolve_cbs.clear()
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()

    def _register(self, clarify_id: str, session_key: str) -> None:
        clarify_gateway.register(
            clarify_id=clarify_id,
            session_key=session_key,
            question="Which?",
            choices=["a", "b"],
        )

    def test_default_fire_on_is_text_only(self):
        """A hook registered with no fire_on fires only on text (default)."""
        fired: list = []
        try:
            # Button click: skipped (default fire_on=("text",) excludes "button").
            clarify_id_btn = "clt_fireon_default_btn"
            self._register(clarify_id_btn, "feishu:oc_a:user_d")
            clarify_gateway.register_after_resolve(
                clarify_id_btn, lambda t: fired.append(f"btn:{t}"),
            )
            clarify_gateway.resolve_gateway_clarify(
                clarify_id_btn, "a", source="button",
            )
            self.assertEqual(
                fired, [],
                "button resolve should NOT fire a default (text-only) hook",
            )
            # Text reply: fires (independent entry, same default hook config).
            clarify_id_txt = "clt_fireon_default_txt"
            self._register(clarify_id_txt, "feishu:oc_a:user_e")
            clarify_gateway.register_after_resolve(
                clarify_id_txt, lambda t: fired.append(f"txt:{t}"),
            )
            clarify_gateway.resolve_gateway_clarify(
                clarify_id_txt, "a", source="text",
            )
            self.assertIn("txt:a", fired, "text resolve should fire the default hook")
        finally:
            clarify_gateway.clear_session("feishu:oc_a:user_d")
            clarify_gateway.clear_session("feishu:oc_a:user_e")

    def test_fire_on_button_skips_text_resolve(self):
        """A hook registered for "button" only fires on click, not on text."""
        clarify_id = "clt_fireon_button"
        session_key = "feishu:oc_a:user_x"
        self._register(clarify_id, session_key)
        fired: list = []

        try:
            clarify_gateway.register_after_resolve(
                clarify_id, lambda t: fired.append(t), fire_on=("button",),
            )
            clarify_gateway.resolve_gateway_clarify(clarify_id, "a", source="text")
            self.assertEqual(fired, [], "text resolve should NOT fire a button-only hook")
        finally:
            clarify_gateway.clear_session(session_key)

    def test_fire_on_text_and_button_fires_on_either(self):
        """Explicit "fire on both" registration fires on either source."""
        fired: list = []
        try:
            # First entry: button path
            clarify_id_btn = "clt_fireon_both_btn"
            self._register(clarify_id_btn, "feishu:oc_a:user_b")
            clarify_gateway.register_after_resolve(
                clarify_id_btn, lambda t: fired.append(f"button:{t}"),
                fire_on=("button", "text"),
            )
            clarify_gateway.resolve_gateway_clarify(
                clarify_id_btn, "a", source="button",
            )
            # Second entry: text path (independent entry, same hook config)
            clarify_id_txt = "clt_fireon_both_txt"
            self._register(clarify_id_txt, "feishu:oc_a:user_c")
            clarify_gateway.register_after_resolve(
                clarify_id_txt, lambda t: fired.append(f"text:{t}"),
                fire_on=("button", "text"),
            )
            clarify_gateway.resolve_gateway_clarify(
                clarify_id_txt, "b", source="text",
            )
            self.assertIn("button:a", fired, "button resolve should have fired the hook")
            self.assertIn("text:b", fired, "text resolve should have fired the hook")
        finally:
            clarify_gateway.clear_session("feishu:oc_a:user_b")
            clarify_gateway.clear_session("feishu:oc_a:user_c")

    def test_source_auto_fires_every_hook_regardless_of_fire_on(self):
        """``source="auto"`` is the timeout/clear_session sentinel:
        it fires every registered hook regardless of fire_on.
        """
        clarify_id = "clt_fireon_auto"
        session_key = "feishu:oc_a:user_x"
        self._register(clarify_id, session_key)
        fired: list = []

        try:
            clarify_gateway.register_after_resolve(
                clarify_id, lambda t: fired.append(t), fire_on=("button",),
            )
            clarify_gateway.resolve_gateway_clarify(clarify_id, "a", source="auto")
            self.assertEqual(fired, ["a"], "auto should fire button-only hooks too")
        finally:
            clarify_gateway.clear_session(session_key)

    def test_fire_on_empty_after_normalization_refuses_registration(self):
        """``fire_on=("garbage",)`` normalizes to empty, and we refuse to
        register such a hook (it would never fire)."""
        clarify_id = "clt_fireon_garbage"
        session_key = "feishu:oc_a:user_x"
        self._register(clarify_id, session_key)
        try:
            ok = clarify_gateway.register_after_resolve(
                clarify_id, lambda t: None, fire_on=("garbage",),
            )
            self.assertFalse(ok, "garbage fire_on should be refused")
            # No hook should be registered.
            self.assertNotIn(clarify_id, clarify_gateway._after_resolve_cbs)
        finally:
            clarify_gateway.clear_session(session_key)

    def test_unknown_source_falls_back_to_text(self):
        """A typo'd source (e.g. ``"botton"``) should not silently break the
        resolve — fall back to ``"text"`` so historical behavior holds."""
        clarify_id = "clt_fireon_typo"
        session_key = "feishu:oc_a:user_x"
        self._register(clarify_id, session_key)
        fired: list = []

        try:
            clarify_gateway.register_after_resolve(
                clarify_id, lambda t: fired.append(t),
            )
            clarify_gateway.resolve_gateway_clarify(clarify_id, "a", source="botton")  # typo
            self.assertEqual(fired, ["a"], "typo source should fall back to text")
        finally:
            clarify_gateway.clear_session(session_key)

    def test_hook_consumed_even_when_fire_on_does_not_match(self):
        """A non-matching fire_on still pops the hook from the registry,
        so a stray second resolve does not re-fire it."""
        clarify_id = "clt_fireon_consumed"
        session_key = "feishu:oc_a:user_x"
        self._register(clarify_id, session_key)
        fired: list = []

        try:
            clarify_gateway.register_after_resolve(
                clarify_id, lambda t: fired.append(t), fire_on=("button",),
            )
            # text resolve does NOT fire (button-only).
            clarify_gateway.resolve_gateway_clarify(clarify_id, "a", source="text")
            # But the hook should be consumed.
            self.assertNotIn(clarify_id, clarify_gateway._after_resolve_cbs)
            # A second resolve (any source) is a no-op for the entry,
            # so fired stays empty.
            clarify_gateway.resolve_gateway_clarify(clarify_id, "b", source="auto")
            self.assertEqual(fired, [])
        finally:
            clarify_gateway.clear_session(session_key)


if __name__ == "__main__":
    unittest.main()
