import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from src.sde.single_index.config import ConfigError, load_plan, render_shell
from src.sde.single_index.evaluate import select_document_terms
from src.sde.single_index.prepare_cache import validate_trace_coverage
from src.sde.single_index.trace import (
    clean_generated_query,
    iter_soft_alternatives,
    normalized_candidate_steps,
    rank1_trajectory_pieces,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/sde/sde_single_index_beir5.json"


def evidence(*probabilities: float) -> tuple[float, float, float, int, int]:
    return (
        sum(probabilities),
        sum(math.log1p(-probability) for probability in probabilities),
        max(probabilities),
        len(probabilities),
        1,
    )


def selection_args(**overrides) -> Namespace:
    values = {
        "mode": "sde",
        "position_mode": "all",
        "candidate_max_df_ratio": 1.0,
        "exclude_original_terms": False,
        "min_aggregate_probability": 0.01,
        "max_soft_terms": 4,
    }
    values.update(overrides)
    return Namespace(**values)


class SingleIndexSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = {
            "doc_ids": ["d1"],
            "original_counts": [{"original": 2}],
            "hard_counts": [{"hard": 1}],
            "base_df": {"original": 1, "hard": 1},
            "candidate_evidence": {
                "all": [
                    {
                        "hard": evidence(0.9),
                        "original": evidence(0.8),
                        "novel": evidence(0.2, 0.3),
                    }
                ]
            },
            "candidate_df": {
                "all": {"hard": 1, "original": 1, "novel": 1}
            },
        }

    def test_sde_keeps_hard_text_and_adds_each_soft_term_once(self) -> None:
        soft, stats = select_document_terms(self.cache, selection_args())
        self.assertNotIn("hard", soft[0])
        self.assertEqual(soft[0]["novel"], 1)
        self.assertEqual(soft[0]["original"], 1)
        self.assertTrue(all(frequency == 1 for frequency in soft[0].values()))
        self.assertEqual(stats["avg_soft_terms_per_doc"], 2.0)

    def test_original_term_filter_is_dataset_controlled(self) -> None:
        soft, _ = select_document_terms(
            self.cache,
            selection_args(exclude_original_terms=True),
        )
        self.assertEqual(soft, [{"novel": 1}])

    def test_hard_control_adds_no_soft_terms(self) -> None:
        soft, stats = select_document_terms(
            self.cache,
            selection_args(mode="hard"),
        )
        self.assertEqual(soft, [{}])
        self.assertEqual(stats["avg_soft_terms_per_doc"], 0.0)

    def test_candidate_document_frequency_filter(self) -> None:
        cache = {
            "doc_ids": ["d1", "d2"],
            "original_counts": [{}, {}],
            "hard_counts": [{}, {}],
            "base_df": {},
            "candidate_evidence": {
                "all": [
                    {"common": evidence(0.8), "rare": evidence(0.7)},
                    {"common": evidence(0.8)},
                ]
            },
            "candidate_df": {"all": {"common": 2, "rare": 1}},
        }
        soft, _ = select_document_terms(
            cache,
            selection_args(candidate_max_df_ratio=0.5),
        )
        self.assertEqual(soft, [{"rare": 1}, {}])


class TraceSemanticsTest(unittest.TestCase):
    def test_rank1_and_chosen_candidates_are_excluded(self) -> None:
        candidates = [
            {"decoded_token": "rank1", "chosen": False},
            {"decoded_token": "rank2", "chosen": False},
            {"decoded_token": "chosen", "chosen": True},
            {"decoded_token": "rank4", "chosen": False},
            {"decoded_token": "rank5", "chosen": False},
        ]
        retained = [
            candidate["decoded_token"]
            for _, candidate in iter_soft_alternatives(candidates, 5)
        ]
        self.assertEqual(retained, ["rank2", "rank4", "rank5"])

    def test_boundary_sequence_uses_rank1_trajectory(self) -> None:
        row = {
            "decoded_candidates": [
                [
                    {"decoded_token": " rank-one", "chosen": False},
                    {"decoded_token": " sampled", "chosen": True},
                ]
            ],
            "generated_tokens": [
                {"decoded_token": " sampled", "chosen": True}
            ],
        }
        self.assertEqual(rank1_trajectory_pieces(row, None), [" rank-one"])

    def test_token_id_trace_marks_sampled_candidate(self) -> None:
        class Tokenizer:
            def decode(self, token_ids, clean_up_tokenization_spaces=False):
                return f" token-{token_ids[0]}"

        row = {
            "indices": [[11, 12, 13]],
            "probs": [[0.5, 0.3, 0.2]],
            "generated_token_ids": [12],
        }
        steps, trace_format = normalized_candidate_steps(row, Tokenizer())
        self.assertEqual(trace_format, "token_ids")
        self.assertFalse(steps[0][0]["chosen"])
        self.assertTrue(steps[0][1]["chosen"])

    def test_token_id_trace_requires_sampled_token_identity(self) -> None:
        row = {"indices": [[11, 12]], "probs": [[0.6, 0.4]]}
        with self.assertRaisesRegex(ValueError, "generated_token_ids aligned"):
            normalized_candidate_steps(row, object())

    def test_decoded_trace_requires_chosen_metadata(self) -> None:
        row = {
            "decoded_candidates": [
                [{"decoded_token": " term", "prob": 0.5}]
            ]
        }
        with self.assertRaisesRegex(ValueError, "boolean chosen field"):
            normalized_candidate_steps(row, None)

    def test_generated_query_cleaning_matches_hard_text_path(self) -> None:
        self.assertEqual(
            clean_generated_query('  "1. useful generated query"  '),
            "useful generated query",
        )

    def test_complete_template_coverage_is_required(self) -> None:
        document_ids = ["d1", "d2"]
        prompt_ids = {"a", "b"}
        complete = {
            document_id: {(prompt_id, 0) for prompt_id in prompt_ids}
            for document_id in document_ids
        }
        validate_trace_coverage(document_ids, prompt_ids, complete, 2, 0)
        complete["d2"].remove(("b", 0))
        with self.assertRaisesRegex(RuntimeError, "complete retained"):
            validate_trace_coverage(document_ids, prompt_ids, complete, 2, 0)


class ReleasedConfigTest(unittest.TestCase):
    def test_released_beir5_parameters_render_to_commands(self) -> None:
        rendered = render_shell(load_plan(CONFIG_PATH))
        self.assertIn(
            "CONFIG_DATASETS=(nfcorpus scidocs fiqa-2018 arguana scifact)",
            rendered,
        )
        self.assertIn("CONFIG_POSITION_MODES=(all all all all boundary)", rendered)
        self.assertIn("CONFIG_MAX_SOFT_TERMS=(16 16 16 16 4)", rendered)
        self.assertIn("--candidate_topk 5", rendered)
        self.assertIn("--candidate_max_df_ratio 0.05", rendered)
        self.assertIn("--k1 0.9", rendered)
        self.assertIn("--b 0.4", rendered)

    def test_public_method_name_is_fixed(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["method"] = "alternate name"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "SDE, single index"):
                load_plan(path)

    def test_selection_protocol_is_validated(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["selection_protocol"]["test_tuning"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "test_tuning must be false"):
                load_plan(path)


if __name__ == "__main__":
    unittest.main()
