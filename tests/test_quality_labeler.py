import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))

from quality_labeler import (
    DIMENSIONS,
    FALLBACK_BETA,
    OUTPUT_SCHEMA,
    SAME_AS_START,
    agreement,
    batch_requests,
    claude_code_command,
    disagreements,
    parse_message,
    reply_rows,
    request_params,
    sample_conversations,
    score_reply,
    score_reply_claude_code,
    system_prompt,
    user_message,
)
from schema import Conversation, Speaker, Utterance

ROW = {
    "utterance_id": "abc123",
    "thread_start": "Taxation is theft.",
    "parent": "Consent is what matters.",
    "reply": "Would court-ordered damages be theft too?",
}


def message(payload=None, stop_reason="end_turn", text=None):
    """A stand-in for an API Message object."""
    body = text if text is not None else json.dumps(payload)
    return SimpleNamespace(
        stop_reason=stop_reason,
        model="claude-test",
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=body)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                              cache_read_input_tokens=None, cache_creation_input_tokens=None),
    )


GOOD = {"rationale": "Civil counterexample.", "civility": "2", "relevance": "2",
        "reasoning": "2", "constructiveness": "2"}


def utterance(uid, parent=None, text="hi", deleted=False, synthetic=False):
    return Utterance(
        id=f"reddit:x:utterance:{uid}", source_id=uid, conversation_id="reddit:x:conversation:c1",
        parent_id=f"reddit:x:utterance:{parent}" if parent else None, speaker_id="reddit:u/a",
        text=text, text_clean=text, created_at=None, depth=0 if parent is None else 1,
        is_deleted=deleted, metadata={"is_synthetic": synthetic},
    )


class PromptTests(unittest.TestCase):
    def test_system_prompt_includes_rubric_and_rules(self):
        prompt = system_prompt()
        for dim in ("Civility", "Relevance", "Reasoning", "Constructiveness"):
            self.assertIn(dim, prompt)
        self.assertIn("Rules for edge cases", prompt)

    def test_user_message_has_all_context(self):
        text = user_message(ROW)
        for key in ("thread_start", "parent", "reply"):
            self.assertIn(ROW[key], text)
        self.assertLess(text.index("<parent_comment>"), text.index("<reply>"))

    def test_request_params_constrain_output(self):
        params = request_params(ROW, "SYSTEM", model="m", effort="low")
        self.assertEqual(params["model"], "m")
        self.assertEqual(params["output_config"]["effort"], "low")
        self.assertEqual(params["output_config"]["format"]["schema"], OUTPUT_SCHEMA)
        self.assertEqual(params["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("thinking", params)  # adaptive by default; never disabled

    def test_schema_allows_na_only_for_reasoning(self):
        props = OUTPUT_SCHEMA["properties"]
        self.assertIn("N/A", props["reasoning"]["enum"])
        for dim in ("civility", "relevance", "constructiveness"):
            self.assertNotIn("N/A", props[dim]["enum"])

    def test_batch_requests_keyed_by_utterance(self):
        reqs = batch_requests([ROW, {**ROW, "utterance_id": "def456"}], "SYSTEM")
        self.assertEqual([r["custom_id"] for r in reqs], ["abc123", "def456"])


class ReplyRowTests(unittest.TestCase):
    def test_rows_skip_start_deleted_and_synthetic(self):
        conv = Conversation(
            id="reddit:x:conversation:c1", source_id="c1", source="reddit", created_at=None,
            utterances=[
                utterance("u0", text="start"),
                utterance("u1", parent="u0", text="first reply"),
                utterance("u2", parent="u1", text="second reply"),
                utterance("u3", parent="u1", text="[deleted]", deleted=True),
                utterance("u4", parent="u1", text="placeholder", synthetic=True),
            ],
            speakers={"reddit:u/a": Speaker(id="reddit:u/a", handle="a", source="reddit")},
            metadata={"source_metadata": {"split": "train", "has_removed_comment": True}},
        )
        rows = list(reply_rows([conv]))
        self.assertEqual([r["utterance_id"] for r in rows], ["u1", "u2"])
        self.assertEqual(rows[0]["parent"], SAME_AS_START)
        self.assertEqual(rows[1]["parent"], "first reply")
        self.assertEqual(rows[1]["thread_start"], "start")
        self.assertTrue(rows[0]["derailed"])
        self.assertEqual(rows[0]["split"], "train")


class ParseTests(unittest.TestCase):
    def test_valid_json(self):
        result = parse_message(message(GOOD))
        self.assertEqual(result["status"], "ok")
        self.assertEqual([result[d] for d in DIMENSIONS], ["2", "2", "2", "2"])

    def test_refusal_and_truncation_have_no_scores(self):
        self.assertEqual(parse_message(message(GOOD, stop_reason="refusal")), {"status": "refusal"})
        self.assertEqual(parse_message(message(GOOD, stop_reason="max_tokens")), {"status": "max_tokens"})

    def test_bad_json_or_out_of_range_is_invalid(self):
        self.assertEqual(parse_message(message(text="not json"))["status"], "invalid")
        self.assertEqual(parse_message(message({**GOOD, "civility": "N/A"}))["status"], "invalid")


class ScoreReplyTests(unittest.TestCase):
    def _client(self):
        calls = {}

        def create(**kwargs):
            calls.update(kwargs)
            return message(GOOD)

        api = SimpleNamespace(create=create)
        return SimpleNamespace(messages=api, beta=SimpleNamespace(messages=api)), calls

    def test_fallback_models_use_server_side_fallback(self):
        client, calls = self._client()
        result = score_reply(client, ROW, "SYSTEM", model="claude-opus-5-5")
        self.assertEqual(calls["betas"], [FALLBACK_BETA])
        self.assertEqual(calls["fallbacks"], "default")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["input_tokens"], 10)
        self.assertEqual(result["cache_read_input_tokens"], 0)

    def test_other_models_use_plain_call(self):
        client, calls = self._client()
        score_reply(client, ROW, "SYSTEM", model="claude-haiku-5-5")
        self.assertNotIn("fallbacks", calls)


class ClaudeCodeTests(unittest.TestCase):
    def _run(self, stdout, returncode=0):
        calls = {}

        def runner(cmd, **kwargs):
            calls["cmd"], calls["kwargs"] = cmd, kwargs
            return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)

        result = score_reply_claude_code(ROW, "SYSTEM", model="m", effort="low", workdir="/tmp/x", runner=runner)
        return result, calls

    def test_structured_output_is_parsed(self):
        out = json.dumps({"is_error": False, "structured_output": GOOD, "total_cost_usd": 0.01})
        result, calls = self._run(out)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["civility"], "2")
        self.assertEqual(result["list_price_usd"], 0.01)
        self.assertEqual(calls["kwargs"]["cwd"], "/tmp/x")

    def test_command_has_no_tools_and_uses_schema(self):
        cmd = claude_code_command(ROW, "SYSTEM", "m", "low")
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertEqual(json.loads(cmd[cmd.index("--json-schema") + 1]), OUTPUT_SCHEMA)
        self.assertEqual(cmd[cmd.index("--system-prompt") + 1], "SYSTEM")

    def test_errors_and_refusals_have_no_scores(self):
        self.assertEqual(self._run("not json")[0]["status"], "error")
        self.assertEqual(self._run(json.dumps({"is_error": True, "result": "limit"}))[0]["status"], "error")
        self.assertEqual(self._run(json.dumps({"safety_stops": 1}))[0]["status"], "refusal")
        self.assertEqual(self._run(json.dumps({"structured_output": {"civility": "9"}}))[0]["status"], "invalid")


class SampleTests(unittest.TestCase):
    def test_samples_whole_pairs(self):
        rows = pd.DataFrame({
            "utterance_id": [f"u{i}" for i in range(8)],
            "conversation_id": ["c1", "c1", "c2", "c3", "c3", "c4", "c5", "c6"],
            "pair_id": ["p1", "p1", "p1", "p2", "p2", "p2", "p3", "p3"],
        })
        sample = sample_conversations(rows, n=2, seed=1)
        self.assertEqual(sample["conversation_id"].nunique(), 2)
        self.assertEqual(sample["pair_id"].nunique(), 1)  # one complete pair
        full = rows[rows["pair_id"] == sample["pair_id"].iloc[0]]
        self.assertEqual(len(sample), len(full))  # every reply of those conversations


class AgreementTests(unittest.TestCase):
    ref = pd.DataFrame({
        "utterance_id": ["a", "b", "c", "d"],
        "civility": ["2", "1", "0", "2"],
        "relevance": ["2", "2", "2", "2"],
        "reasoning": ["2", "N/A", "0", "1"],
        "constructiveness": ["2", "1", "0", "2"],
    })

    def test_perfect_agreement(self):
        llm = self.ref.assign(status="ok")
        table = agreement(self.ref, llm).set_index("dimension")
        self.assertTrue((table["exact"] == 1.0).all())
        self.assertTrue((table["kappa"] == 1.0).all())

    def test_mixed_types_and_unscored_rows(self):
        llm = self.ref.copy()
        llm["civility"] = [2, 2, 0, 2]  # ints, one disagreement
        llm["status"] = ["ok", "ok", "ok", "refusal"]
        table = agreement(self.ref, llm).set_index("dimension")
        self.assertEqual(table.loc["civility", "n"], 3)
        self.assertEqual(table.loc["civility", "n_unscored"], 1)
        self.assertAlmostEqual(table.loc["civility", "exact"], 2 / 3)
        self.assertEqual(table.loc["civility", "within_one"], 1.0)

    def test_disagreements_lists_differing_rows(self):
        llm = self.ref.copy()
        llm.loc[2, "reasoning"] = "N/A"
        self.assertEqual(disagreements(self.ref, llm)["utterance_id"].tolist(), ["c"])


if __name__ == "__main__":
    unittest.main()
