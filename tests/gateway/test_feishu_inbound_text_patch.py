"""Verify the Feishu adapter's _handle_message_event_data path fires
the PATCH hook for text replies to clarify cards.

Architecture note: pre-restructuring, the PATCH hook was registered
via a cross-platform register_after_resolve API in
tools.clarify_gateway, and the gateway runner's text-intercept path
called _fire_after_resolve_hook with a __class__.__name__ string
check.  Both layers are gone now — the hook lives entirely inside
the Feishu adapter.  The text reply arrives in
_handle_message_event_data; if the message is text, the chat has
a pending clarify card, and the loop is up, _patch_clarify_card
is submitted to fire on the loop.

The button-click path is separate (L2300+); it updates the card
in place via P2CardActionTriggerResponse and is tested by
TestFeishuClarifyCard.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools import clarify_gateway
from gateway.platforms import feishu


def _make_feishu_adapter():
    from gateway.config import PlatformConfig
    from gateway.platforms.feishu import FeishuAdapter
    return FeishuAdapter(PlatformConfig())


def _make_text_message(chat_id="oc_chat_a", text="my free-form answer"):
    return SimpleNamespace(
        message_id="om_inbound_text_1",
        message_type="text",
        chat_id=chat_id,
        content=json.dumps({"text": text}),
        chat_type="p2p",
    )


def _make_sender():
    return SimpleNamespace(
        sender_id=SimpleNamespace(open_id="ou_user_a", user_id=None, union_id=None),
    )


def _make_event(message, sender):
    return SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))


class TestFeishuInboundTextPatch(unittest.TestCase):
    def setUp(self):
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()
        # Per-adapter _clarify_card_message_ids is reset in each test
        # by creating a fresh adapter.
        self.adapter = _make_feishu_adapter()
        # Mark the loop "ready" so _submit_on_loop does not drop the
        # PATCH.  We patch _loop_accepts_callbacks and _submit_on_loop
        # so we can capture the coroutine without running a real loop.
        self.adapter._loop = MagicMock()
        self.adapter._loop_accepts_callbacks = MagicMock(return_value=True)
        self._captured = []

        def _capture_submission(loop, coro):
            self._captured.append(coro)
            # Close the coroutine to silence the "never awaited" warning.
            coro.close()
            return None

        self.adapter._submit_on_loop = _capture_submission

    def tearDown(self):
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()

    def test_text_reply_with_pending_card_schedules_patch(self):
        chat_id = "oc_chat_a"
        # Simulate send_clarify having populated the card map.
        self.adapter._clarify_card_message_ids[chat_id] = "om_card_clarify_1"

        # Register a pending clarify so the entry exists.
        clarify_gateway.register(
            clarify_id="cl_pending_text_1",
            session_key="feishu:oc_chat_a:user_a",
            question="Which?",
            choices=["a", "b"],
        )
        clarify_gateway.mark_awaiting_text("cl_pending_text_1")

        message = _make_text_message(chat_id=chat_id, text="my answer")
        sender = _make_sender()
        event = _make_event(message, sender)

        asyncio.run(self.adapter._handle_message_event_data(event))

        # _clarify_card_message_ids is consumed on text reply.
        self.assertNotIn(chat_id, self.adapter._clarify_card_message_ids)
        # The PATCH coroutine was submitted.
        self.assertEqual(len(self._captured), 1)
        # The PATCH target is the card message_id that was stored.
        # We can't easily inspect the coroutine's bound args without
        # running it, but the submission itself is what we want to
        # verify here — the coroutine body is the same code path as
        # the existing _patch_clarify_card tests.

    def test_text_reply_with_no_pending_card_does_not_patch(self):
        chat_id = "oc_chat_no_pending"
        # _clarify_card_message_ids is empty for this chat.
        message = _make_text_message(chat_id=chat_id)
        sender = _make_sender()
        event = _make_event(message, sender)

        asyncio.run(self.adapter._handle_message_event_data(event))

        self.assertEqual(self._captured, [])

    def test_button_click_does_not_go_through_text_path(self):
        # Button clicks arrive as a different message_type.  This
        # test is here to guard the explicit check in
        # _is_text_answering_pending_clarify.
        chat_id = "oc_chat_a"
        self.adapter._clarify_card_message_ids[chat_id] = "om_card_clarify_1"

        message = SimpleNamespace(
            message_id="om_inbound_btn_1",
            message_type="interactive",  # not "text"
            chat_id=chat_id,
            content=json.dumps({"text": ""}),
            chat_type="p2p",
        )
        sender = _make_sender()
        event = _make_event(message, sender)

        asyncio.run(self.adapter._handle_message_event_data(event))

        # Card message_id is NOT consumed — button clicks are
        # handled by _handle_clarify_card_action, not the inbound
        # text path.
        self.assertIn(chat_id, self.adapter._clarify_card_message_ids)
        self.assertEqual(self._captured, [])

    def test_loop_not_ready_drops_patch(self):
        chat_id = "oc_chat_a"
        self.adapter._clarify_card_message_ids[chat_id] = "om_card_clarify_1"
        self.adapter._loop_accepts_callbacks = MagicMock(return_value=False)

        message = _make_text_message(chat_id=chat_id)
        sender = _make_sender()
        event = _make_event(message, sender)

        asyncio.run(self.adapter._handle_message_event_data(event))

        # PATCH is dropped because the loop is not ready.  The card
        # message_id is still consumed (no point holding it for a
        # loop that never started).
        self.assertEqual(self._captured, [])

    def test_message_type_string_check_uses_text(self):
        """Regression guard: the message_type check must be the
        string ``"text"``.  If someone refactors the type to an
        enum and forgets to update this, the PATCH silently
        breaks.  This test pins the contract.
        """
        # Pre-populate the card map so the chat_id membership check
        # does not mask the message_type check we actually want to
        # exercise here.
        self.adapter._clarify_card_message_ids["oc_chat_a"] = "om_card_x"

        self.assertTrue(
            self.adapter._is_text_answering_pending_clarify(
                _make_text_message(),
            ),
        )
        # Non-text message types return False.
        self.assertFalse(
            self.adapter._is_text_answering_pending_clarify(
                SimpleNamespace(
                    message_id="om_x",
                    message_type="interactive",
                    chat_id="oc_chat_a",
                    content=json.dumps({"text": ""}),
                ),
            ),
        )
        # No chat_id returns False.
        self.assertFalse(
            self.adapter._is_text_answering_pending_clarify(
                SimpleNamespace(
                    message_id="om_x",
                    message_type="text",
                    chat_id="",
                    content=json.dumps({"text": ""}),
                ),
            ),
        )
