import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, patch

from config import settings
from db.database import get_conn, init_tables
from training.collector import get_sft_pairs
from training.evaluator import evaluate_model_on_pairs, promotion_decision, reserve_evaluation_pairs
from training.orchestrator import pick_clone_winner
from training.state import finish_training_run, try_create_training_run


class PromotionDecisionTests(unittest.TestCase):
    def test_candidate_must_beat_baseline(self):
        result = promotion_decision(
            baseline_score=0.70,
            candidate_score=0.69,
            n_eval=8,
            min_pairs=3,
            min_score=0.25,
            margin=0.0,
        )
        self.assertFalse(result["promote"])
        self.assertEqual(result["reason"], "candidate_regressed")

    def test_insufficient_holdout_never_promotes(self):
        result = promotion_decision(
            baseline_score=0.10,
            candidate_score=0.90,
            n_eval=2,
            min_pairs=3,
        )
        self.assertFalse(result["promote"])
        self.assertEqual(result["reason"], "not_enough_held_out_pairs")

    def test_improved_candidate_promotes(self):
        result = promotion_decision(
            baseline_score=0.60,
            candidate_score=0.72,
            n_eval=5,
            min_pairs=3,
            min_score=0.25,
            margin=0.05,
        )
        self.assertTrue(result["promote"])
        self.assertAlmostEqual(result["delta"], 0.12)

    def test_tie_does_not_count_as_improvement_by_default(self):
        result = promotion_decision(
            baseline_score=0.60,
            candidate_score=0.60,
            n_eval=5,
        )
        self.assertFalse(result["promote"])
        self.assertEqual(result["reason"], "insufficient_improvement")


class SQLiteTrainingSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_sqlite_path = settings.sqlite_path
        settings.sqlite_path = str(Path(self.temp_dir.name) / "echo-test.db")
        await init_tables()

    async def asyncTearDown(self):
        settings.sqlite_path = self.previous_sqlite_path
        self.temp_dir.cleanup()

    async def test_only_one_global_training_run_can_start(self):
        first = await try_create_training_run("alice", "gemma4_e2b", 20)
        second = await try_create_training_run("bob", "gemma4_e2b", 20)
        self.assertIsNotNone(first)
        self.assertIsNone(second)

        await finish_training_run(first, "complete")
        third = await try_create_training_run("bob", "gemma4_e2b", 20)
        self.assertIsNotNone(third)
        await finish_training_run(third, "complete")

    async def test_holdout_selection_is_stable(self):
        async with get_conn() as db:
            for index in range(1, 6):
                await db.execute(
                    """
                    INSERT INTO training_pairs
                        (user_id, topic, user_msg, assistant_msg, model_used,
                         engagement_signal, perplexity)
                    VALUES (?, 'general', ?, ?, 'gemma4_e2b', 'thumbs_up', 1.0)
                    """,
                    (
                        "alice",
                        f"Question number {index} with enough text",
                        f"Reference answer number {index} with enough text",
                    ),
                )
            await db.commit()

        first = await reserve_evaluation_pairs("alice", 3)
        second = await reserve_evaluation_pairs("alice", 3)
        self.assertEqual([p["id"] for p in first], [5, 4, 3])
        self.assertEqual([p["id"] for p in first], [p["id"] for p in second])

        holdout_prompts = {p["user_msg"] for p in first}
        training_pairs = await get_sft_pairs(
            "alice",
            exclude_ids={p["id"] for p in first},
            exclude_prompts=holdout_prompts,
        )
        self.assertEqual({p["id"] for p in training_pairs}, {1, 2})
        self.assertTrue(holdout_prompts.isdisjoint({p["user_msg"] for p in training_pairs}))


class TournamentFairnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_clone_receives_the_same_frozen_exam(self):
        exam = [
            {"id": 7, "user_msg": "Question seven", "assistant_msg": "Answer seven"},
            {"id": 8, "user_msg": "Question eight", "assistant_msg": "Answer eight"},
            {"id": 9, "user_msg": "Question nine", "assistant_msg": "Answer nine"},
        ]

        async def fake_score(_user, path, **kwargs):
            self.assertIs(kwargs["evaluation_pairs"], exam)
            return {
                "score": 0.8 if path == "on-policy" else 0.6,
                "n_eval": 3,
            }

        with patch("training.orchestrator._score_adapter", side_effect=fake_score):
            result = await pick_clone_winner(
                "alice",
                {"seqkd": "seqkd", "on_policy": "on-policy"},
                evaluation_pairs=exam,
            )

        self.assertEqual(result["winner_name"], "on_policy")
        self.assertEqual(result["winner_path"], "on-policy")
        self.assertEqual(result["winner_score"], 0.8)


class _FakeVllmHandler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, _format, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.__class__.calls.append((self.path, payload))

        if self.path.endswith("/chat/completions"):
            prompt = payload["messages"][0]["content"]
            content = f"Reference for {prompt}" if payload["model"] == "good-adapter" else "unrelated"
            body = {"choices": [{"message": {"content": content}}]}
        else:
            body = {"ok": True}

        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class FakeVllmIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        _FakeVllmHandler.calls = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVllmHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    async def test_real_http_contract_scores_the_frozen_exam(self):
        pairs = [
            {
                "id": index,
                "user_msg": f"prompt-{index}",
                "assistant_msg": f"Reference for prompt-{index}",
            }
            for index in range(1, 4)
        ]
        result = await evaluate_model_on_pairs(
            "good-adapter",
            pairs,
            vllm_base_url=self.base_url,
        )
        self.assertEqual(result["n_eval"], 3)
        self.assertEqual(result["score"], 1.0)
        chat_calls = [call for call in _FakeVllmHandler.calls if call[0].endswith("/chat/completions")]
        self.assertEqual(len(chat_calls), 3)
        self.assertTrue(all(call[1]["temperature"] == 0.0 for call in chat_calls))


class CoordinatorPromotionTests(unittest.IsolatedAsyncioTestCase):
    def _patches(self, winner_score):
        holdout = [
            {"id": 1, "user_msg": "Question one long enough", "assistant_msg": "Answer one long enough"},
            {"id": 2, "user_msg": "Question two long enough", "assistant_msg": "Answer two long enough"},
            {"id": 3, "user_msg": "Question three long enough", "assistant_msg": "Answer three long enough"},
        ]
        return holdout, {
            "get_last_checkpoint": patch(
                "training.coordinator.get_last_checkpoint",
                AsyncMock(return_value="previous"),
            ),
            "reserve": patch(
                "training.coordinator.reserve_evaluation_pairs",
                AsyncMock(return_value=holdout),
            ),
            "ensure": patch(
                "training.coordinator.ensure_adapter_loaded",
                AsyncMock(return_value=True),
            ),
            "evaluate": patch(
                "training.coordinator.evaluate_model_on_pairs",
                AsyncMock(return_value={"score": 0.6, "n_eval": 3, "details": []}),
            ),
            "prepare": patch(
                "training.coordinator.prepare_clone_datasets",
                AsyncMock(return_value={
                    "seqkd": [{"id": 10}],
                    "self_critique": [],
                    "group_dpo": [],
                    "on_policy": [],
                    "base_sft": [{"id": 10}],
                    "base_dpo": [],
                    "training_floor": 1,
                }),
            ),
            "stop": patch(
                "training.coordinator.stop_gemma_vllm_for_training",
                AsyncMock(return_value=True),
            ),
            "start": patch(
                "training.coordinator.start_gemma_vllm_after_training",
                AsyncMock(return_value=True),
            ),
            "train": patch(
                "training.coordinator.run_clone_training",
                AsyncMock(return_value={"seqkd": "candidate"}),
            ),
            "winner": patch(
                "training.coordinator.pick_clone_winner",
                AsyncMock(return_value={
                    "winner_name": "seqkd",
                    "winner_path": "candidate",
                    "winner_score": winner_score,
                    "n_eval": 3,
                    "evaluations": {},
                }),
            ),
            "swap": patch(
                "training.coordinator.hot_swap_adapter",
                AsyncMock(return_value=True),
            ),
            "mark": patch(
                "training.coordinator.mark_used",
                AsyncMock(),
            ),
            "summary": patch(
                "training.coordinator.get_training_summary",
                AsyncMock(return_value={}),
            ),
        }

    async def test_regressing_candidate_is_never_checkpointed(self):
        from training.coordinator import run_training_cycle

        _holdout, patchers = self._patches(winner_score=0.5)
        started = {name: patcher.start() for name, patcher in patchers.items()}
        try:
            # Simulate the ids returned by the real trainer.
            async def train_with_ids(*_args, **kwargs):
                kwargs["used_ids_out"].add(10)
                return {"seqkd": "candidate"}

            started["train"].side_effect = train_with_ids
            result = await run_training_cycle("alice", "gemma4_e2b", "run-1")
        finally:
            for patcher in reversed(list(patchers.values())):
                patcher.stop()

        self.assertFalse(result["promoted"])
        self.assertEqual(result["promotion"]["reason"], "candidate_regressed")
        self.assertTrue(
            all(call.kwargs.get("record_checkpoint") is False for call in started["swap"].await_args_list)
        )
        started["mark"].assert_awaited_once_with("alice", {10})

    async def test_improving_candidate_is_checkpointed_once(self):
        from training.coordinator import run_training_cycle

        _holdout, patchers = self._patches(winner_score=0.8)
        started = {name: patcher.start() for name, patcher in patchers.items()}
        try:
            result = await run_training_cycle("alice", "gemma4_e2b", "run-2")
        finally:
            for patcher in reversed(list(patchers.values())):
                patcher.stop()

        self.assertTrue(result["promoted"])
        checkpoint_calls = [
            call for call in started["swap"].await_args_list
            if call.kwargs.get("record_checkpoint") is True
        ]
        self.assertEqual(len(checkpoint_calls), 1)
        self.assertEqual(checkpoint_calls[0].args[:2], ("alice", "candidate"))


if __name__ == "__main__":
    unittest.main()
