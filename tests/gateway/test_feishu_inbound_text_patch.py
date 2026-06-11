"""Negative-path regression guards for the Feishu inbound-text PATCH hook.

The positive path (text reply -> card PATCH scheduled) lives in
``tests/gateway/test_feishu.py::test_text_message_after_send_clarify_triggers_patch``.
This file pins the two negative paths so the dispatch guard in
``_is_text_answering_pending_clarify`` (message_type == "text" AND
chat_id in pending map) doesn't silently regress.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from tools import clarify_gateway


def _make_message(chat_id, *, message_type, text=""):
    return SimpleNamespace(
        message_id="om_inbound_1",
        message_type=message_type,
        chat_id=chat_id,
        content=json.dumps({"text": text}),
        chat_type="p2p",
    )


def _make_sender():
    return SimpleNamespace(
        sender_id=SimpleNamespace(open_id="ou_user_a", user_id=None, union_id=None),
    )


class TestFeishuInboundTextPatchNegative(unittest.TestCase):
    def setUp(self):
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter
        self.adapter = FeishuAdapter(PlatformConfig())
        # Dedup state is persisted to disk from previous tests; reset
        # in-memory so our message_ids aren't silently dropped as dupes.
        self.adapter._seen_message_ids.clear()
        self.adapter._loop = MagicMock()
        self.adapter._loop_accepts_callbacks = MagicMock(return_value=True)
        self.captured = []

        def _capture(_loop, coro):
            self.captured.append(coro)
            coro.close()
            return None

        self.adapter._submit_on_loop = _capture

    def tearDown(self):
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()

    def _run(self, message):
        return asyncio.run(self.adapter._handle_message_event_data(
            SimpleNamespace(event=SimpleNamespace(message=message, sender=_make_sender())),
        ))

    def test_no_pending_card_or_non_text_message_does_not_fire_patch(self):
        """The two guard clauses in _is_text_answering_pending_clarify
        must both prevent the PATCH hook from firing."""
        chat_id = "oc_chat_a"

        with self.subTest(case="no_pending_card"):
            self.adapter._clarify_card_message_ids.pop(chat_id, None)
            self._run(_make_message(chat_id=chat_id, message_type="text", text="hi"))
            self.assertEqual(self.captured, [])

        with self.subTest(case="non_text_message_with_pending_card"):
            self.adapter._clarify_card_message_ids[chat_id] = "om_card_clarify_1"
            self._run(_make_message(chat_id=chat_id, message_type="interactive"))
            # Card message_id stays untouched; the dispatch fired into
            # the interactive (button-click) path, not the text-patch path.
            self.assertIn(chat_id, self.adapter._clarify_card_message_ids)
            self.assertEqual(self.captured, [])
