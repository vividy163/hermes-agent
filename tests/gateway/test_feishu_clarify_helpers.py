"""Unit tests for the Feishu adapter's static clarify card builders.

Card builders are static methods; their JSON shape is the user-visible
contract and the dispatch contract for card-action events.  These tests
pin it so future refactors don't silently change what the user sees on
the card or what the resolve path reads out of action_value.

End-to-end (send + click) coverage lives in
``tests/gateway/test_feishu.py::TestFeishuClarifyCard``.
"""

import unittest

from gateway.platforms.feishu import FeishuAdapter


def _buttons(card):
    return card["elements"][1]["actions"]


def _body(card):
    return card["elements"][0]["content"]


class TestBuildClarifyCard(unittest.TestCase):
    def test_three_layouts_render_expected_shape(self):
        """One parameterized check covering the three layouts
        (no-choices / short-choices / A-B-C-D fallback).  Each subTest
        pins the same JSON contract the legacy single-fixture tests did.
        """
        base = dict(clarify_id="cl_1")

        with self.subTest(case="no_choices"):
            card = FeishuAdapter._build_clarify_card(
                question="Anything?", choices=None, **base,
            )
            self.assertEqual(len(card["elements"]), 1)
            self.assertTrue(_body(card).startswith("❓ Anything?"))
            self.assertIn("(or send a message for other options)", _body(card))

        with self.subTest(case="short_choices"):
            card = FeishuAdapter._build_clarify_card(
                question="Which?", choices=["a", "b", "c"], **base,
            )
            self.assertTrue(card["config"]["update_multi"])
            self.assertEqual(
                card["header"]["title"]["content"], "🤔 Please select",
            )
            self.assertEqual(
                [b["text"]["content"] for b in _buttons(card)],
                ["a", "b", "c"],
            )
            for b in _buttons(card):
                self.assertTrue(b["value"]["hermes_clarify"])
                self.assertEqual(b["value"]["clarify_id"], "cl_1")
            self.assertIn("(or send a message for other options)", _body(card))

        with self.subTest(case="abcd_fallback"):
            long_choice = "x" * 40  # display width 40 >= 28 -> A/B/C/D
            card = FeishuAdapter._build_clarify_card(
                question="Pick one",
                choices=[long_choice, "y", "z", "w"],
                **base,
            )
            self.assertEqual(
                [b["text"]["content"] for b in _buttons(card)],
                ["A", "B", "C", "D"],
            )
            # value.choice still carries the original body so the resolve
            # path sees the real choice text (not the A/B/C/D label).
            self.assertEqual(_buttons(card)[0]["value"]["choice"], long_choice)
            self.assertIn("A: " + long_choice, _body(card))


class TestBuildResolvedClarifyCard(unittest.TestCase):
    def test_carries_choice_and_green_template(self):
        card = FeishuAdapter._build_resolved_clarify_card(choice="canary")
        self.assertTrue(card["config"]["update_multi"])
        self.assertEqual(card["header"]["template"], "green")
        self.assertEqual(card["elements"][0]["content"],
                         "✅ You selected **canary**")
