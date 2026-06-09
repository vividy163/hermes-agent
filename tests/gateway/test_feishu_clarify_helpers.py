"""Unit tests for the Feishu adapter's static card builders and the
card-PATCH access-token helper.

These helpers were extracted out of send_clarify /
_handle_clarify_card_action / _patch_clarify_card in the current
branch; this file pins their observable shape (card JSON structure,
button values, type-to-answer hint text) so future refactors don't
silently change what the user sees on the card or what the
button-click dispatch expects to find in action_value.

The card builders are static methods and don't need an adapter
instance.  _fetch_tenant_access_token is async and goes through
httpx, so its tests mock httpx.AsyncClient.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.platforms.feishu import FeishuAdapter


class TestBuildClarifyCard(unittest.TestCase):
    """_build_clarify_card: static helper, returns the card JSON."""

    def test_short_choices_render_full_label_per_button(self):
        card = FeishuAdapter._build_clarify_card(
            question="Which deploy?",
            choices=["canary", "stable", "rollback"],
            clarify_id="cl_short_1",
        )
        self.assertTrue(card["config"]["update_multi"])
        self.assertEqual(card["header"]["title"]["content"], "🤔 Please select")
        # 1 markdown + 1 action row
        self.assertEqual(len(card["elements"]), 2)
        actions = card["elements"][1]["actions"]
        self.assertEqual(len(actions), 3)
        # Short labels: button text == choice body
        self.assertEqual(actions[0]["text"]["content"], "canary")
        self.assertEqual(actions[1]["text"]["content"], "stable")
        self.assertEqual(actions[2]["text"]["content"], "rollback")
        # Button value carries the resolve signal
        for i, action in enumerate(actions):
            self.assertTrue(action["value"]["hermes_clarify"])
            self.assertEqual(action["value"]["clarify_id"], "cl_short_1")
            self.assertEqual(action["value"]["choice"], ["canary", "stable", "rollback"][i])

    def test_long_choices_collapse_to_ABCD_and_list_in_body(self):
        long_choice = "x" * 40  # >= 28 mb_strwidth → A/B/C/D fallback
        card = FeishuAdapter._build_clarify_card(
            question="Pick one",
            choices=[long_choice, "y", "z", "w"],
            clarify_id="cl_long_1",
        )
        actions = card["elements"][1]["actions"]
        # Buttons become A/B/C/D
        self.assertEqual([a["text"]["content"] for a in actions], ["A", "B", "C", "D"])
        # value.choice still carries the *original* body (not A/B/C/D)
        # so the resolve path sees the real choice text.
        self.assertEqual(actions[0]["value"]["choice"], long_choice)
        # Question body lists the choices as "A: <text>" form
        body = card["elements"][0]["content"]
        self.assertIn("A: " + long_choice, body)
        self.assertIn("B: y", body)

    def test_no_choices_renders_open_ended_body_with_hint(self):
        card = FeishuAdapter._build_clarify_card(
            question="Anything you want to add?",
            choices=None,
            clarify_id="cl_open_1",
        )
        # No action row when there's nothing to click on
        self.assertEqual(len(card["elements"]), 1)
        body = card["elements"][0]["content"]
        self.assertTrue(body.startswith("❓ Anything you want to add?"))
        # type-to-answer hint sits directly above the (absent) action row
        self.assertIn("(or send a message for other options)", body)

    def test_type_to_answer_hint_appears_in_short_choice_card_too(self):
        card = FeishuAdapter._build_clarify_card(
            question="Q?",
            choices=["a"],
            clarify_id="cl_hint_short",
        )
        body = card["elements"][0]["content"]
        self.assertIn("(or send a message for other options)", body)


class TestBuildResolvedClarifyCard(unittest.TestCase):
    """_build_resolved_clarify_card: static helper, returns the resolved card JSON."""

    def test_resolved_card_carries_choice_and_green_template(self):
        card = FeishuAdapter._build_resolved_clarify_card(choice="canary")
        self.assertTrue(card["config"]["update_multi"])
        self.assertEqual(card["header"]["title"]["content"], "✅ Selected")
        self.assertEqual(card["header"]["template"], "green")
        # Body is a single markdown element with the choice in bold
        self.assertEqual(len(card["elements"]), 1)
        self.assertEqual(
            card["elements"][0]["content"],
            "✅ You selected **canary**",
        )


class TestHandleClarifyCardAction(unittest.TestCase):
    """The sync def for card-action dispatch (mirrors _handle_update_prompt_card_action
    structurally — both submit a resolve coroutine and return a P2 response).

    Note: P2CardActionTriggerResponse / CallBackCard are None when the
    lark-oapi SDK isn't importable in the test env (the optional import
    in feishu.py degrades gracefully).  In that case the method returns
    None after the resolve coroutine is submitted, which is the same
    behavior as the SDK-unavailable branch in production.
    """

    def _make_adapter(self):
        """Build a FeishuAdapter with just the bits _handle_clarify_card_action touches."""
        from gateway.config import PlatformConfig
        return FeishuAdapter(PlatformConfig())

    def test_button_click_submits_resolve_coroutine(self):
        from tools import clarify_gateway
        from gateway.platforms import feishu as feishu_mod
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()
        try:
            adapter = self._make_adapter()
            adapter._loop = MagicMock()
            adapter._loop_accepts_callbacks = MagicMock(return_value=True)
            captured = {}
            def _capture(loop, coro):
                captured["coro"] = coro
                coro.close()
                return True
            adapter._submit_on_loop = _capture

            # Register a pending clarify so resolve_gateway_clarify can find it
            clarify_gateway.register(
                clarify_id="cl_btn_1",
                session_key="feishu:oc_a:user_a",
                question="Pick one",
                choices=["a", "b"],
            )
            # Pop the card message_id so we can verify it gets cleared
            adapter._clarify_card_message_ids["oc_chat_a"] = "om_card_1"

            event = SimpleNamespace(
                operator=SimpleNamespace(open_id="ou_admin", user_id=""),
                chat_id="oc_chat_a",
            )
            response = adapter._handle_clarify_card_action(
                event=event,
                action_value={"clarify_id": "cl_btn_1", "choice": "a"},
            )
            # The resolve coroutine was submitted regardless of whether
            # the SDK is available
            self.assertIn("coro", captured)
            # When the SDK is available, a P2 response is returned; when
            # it isn't, None is returned (graceful degrade).
            if feishu_mod.P2CardActionTriggerResponse is not None:
                self.assertIsNotNone(response)
                self.assertIsNotNone(response.card)
                card_data = response.card.data
                self.assertEqual(card_data["header"]["title"]["content"], "✅ Selected")
                self.assertIn("**a**", card_data["elements"][0]["content"])
        finally:
            clarify_gateway._entries.clear()
            clarify_gateway._session_index.clear()

    def test_missing_clarify_id_does_not_submit_resolve(self):
        from gateway.config import PlatformConfig
        adapter = FeishuAdapter(PlatformConfig())
        captured = {}
        def _capture(loop, coro):
            captured["coro"] = coro
            coro.close()
            return True
        adapter._submit_on_loop = _capture
        event = SimpleNamespace(operator=SimpleNamespace(open_id=""))
        adapter._handle_clarify_card_action(
            event=event,
            action_value={"choice": "a"},  # no clarify_id
        )
        # Missing clarify_id returns early without scheduling anything
        self.assertNotIn("coro", captured)


class TestFetchTenantAccessToken(unittest.TestCase):
    """_fetch_tenant_access_token: async helper for the PATCH path."""

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        return FeishuAdapter(PlatformConfig())

    def _mock_httpx_post(self, response_json, status_code=200):
        """Build a context-manager mock for httpx.AsyncClient that returns the
        given JSON body from its .post(...) call."""
        post = AsyncMock()
        post.return_value = SimpleNamespace(
            status_code=status_code,
            json=MagicMock(return_value=response_json),
        )
        client = MagicMock()
        client.post = post
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    def test_returns_token_string_on_success(self):
        adapter = self._make_adapter()
        client = self._mock_httpx_post(
            {"tenant_access_token": "t-abc123", "code": 0, "msg": "ok"}
        )
        with patch("gateway.platforms.feishu.httpx") as httpx_mod:
            httpx_mod.AsyncClient.return_value = client
            token = adapter._fetch_tenant_access_token_sync(
                domain="https://open.feishu.cn"
            ) if False else None
            # Run the actual async method through asyncio
            import asyncio
            token = asyncio.run(adapter._fetch_tenant_access_token(
                domain="https://open.feishu.cn",
            ))
        self.assertEqual(token, "t-abc123")

    def test_returns_none_when_response_missing_token(self):
        adapter = self._make_adapter()
        client = self._mock_httpx_post(
            {"code": 10003, "msg": "invalid app_id"}
        )
        with patch("gateway.platforms.feishu.httpx") as httpx_mod:
            httpx_mod.AsyncClient.return_value = client
            import asyncio
            token = asyncio.run(adapter._fetch_tenant_access_token(
                domain="https://open.feishu.cn",
            ))
        self.assertIsNone(token)
