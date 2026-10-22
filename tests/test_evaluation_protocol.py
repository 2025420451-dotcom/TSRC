import logging
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from generators.data import FullEvalDataset
from generators.generator import Generator
from models.DualLLMSRS import LLMSemanticRerankerMixin
from trainers.sequence_trainer import SeqTrainer


class _RankingBackbone(nn.Module):
    def predict(self, seq, item_indices, positions, **kwargs):
        self.candidate_counts.append(item_indices.shape[-1])
        self.semantic_states.append(self.use_semantic_rerank)
        if item_indices.dim() == 1:
            return item_indices.float().unsqueeze(0).expand(seq.size(0), -1)
        return item_indices.float()


class _RankingModel(LLMSemanticRerankerMixin, _RankingBackbone):
    def __init__(self, args):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.item_num = 24
        self.candidate_counts = []
        self.semantic_states = []
        table = torch.rand((26, 4), generator=torch.Generator().manual_seed(7))
        table[0] = table[-1] = 0
        self._init_semantic_reranker(args, table.numpy())


class _RecordingStopper:
    best_epoch = 0
    early_stop = False

    def __init__(self):
        self.scores = []

    def __call__(self, indicator, epoch, model, optimizer, scheduler):
        self.scores.append(indicator)
        self.best_epoch = epoch


class _ProtocolTrainer(SeqTrainer):
    def _create_model(self):
        self.model = _RankingModel(self.args)

    def _set_stopper(self):
        self.stopper = _RecordingStopper()

    def _train_one_epoch(self, epoch):
        self.model.train()
        return 0.0

    def _load_eval_checkpoint(self):
        self.loaded_checkpoint_epoch = 0
        return 0


class EvaluationProtocolTest(unittest.TestCase):
    def make_trainer(self, semantic=False, mode=None, checkpoint_mode=None):
        main_path = Path(__file__).resolve().parents[1] / "main.py"
        cli_args = [str(main_path)]
        if mode is not None:
            cli_args += ["--eval_mode", mode]
        if checkpoint_mode is not None:
            cli_args += ["--checkpoint_eval_mode", checkpoint_mode]
        with patch.object(sys, "argv", cli_args):
            args = runpy.run_path(str(main_path), run_name="protocol_defaults")["args"]
        self.assertEqual(args.eval_mode, mode or "sampled")
        self.assertEqual(args.checkpoint_eval_mode, checkpoint_mode or "full")
        args.dataset = "fashion"
        args.num_train_epochs = 1
        args.test_neg = 2
        args.eval_batch_size = 2
        args.max_len = 4
        args.ts_user = 3
        args.ts_item = 4
        args.disable_semantic_rerank = not semantic
        args.semantic_weight_grid = "0,0.5,1"
        generator = Generator.__new__(Generator)
        generator.args = args
        generator.train = {1: [1, 2], 2: [3, 4]}
        generator.valid = {1: [3], 2: [5]}
        generator.test = {1: [4], 2: [6]}
        generator.item_num = 24
        generator.user_num = 2
        generator.num_workers = 0
        generator.bs = 2
        writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None)
        return _ProtocolTrainer(args, logging.getLogger(__name__), writer, torch.device("cpu"), generator)

    @patch("trainers.sequence_trainer.record_csv")
    def test_default_training_selects_on_full_catalog_and_tests_on_sampled_candidates(self, record):
        trainer = self.make_trainer()
        self.assertIsInstance(trainer.checkpoint_valid_loader.dataset, FullEvalDataset)
        self.assertNotIsInstance(trainer.test_loader.dataset, FullEvalDataset)
        result, epoch = trainer.train()
        # Increasing item scores put both full-catalog validation targets below
        # the top ten, while three sampled candidates always fit in the top ten.
        self.assertEqual(trainer.stopper.scores, [0.0])
        self.assertEqual(result["HR@10"], 1.0)
        self.assertEqual(trainer.model.candidate_counts, [24, 3])
        record.assert_called_once()

    @patch("trainers.sequence_trainer.record_csv")
    def test_test_time_calibration_preserves_the_original_sampled_protocol(self, record):
        trainer = self.make_trainer(semantic=True)
        trainer.test()
        # One validation batch for calibration, followed by one test batch.
        self.assertEqual(trainer.model.candidate_counts, [3, 3])
        record.assert_called_once()

    @patch("trainers.sequence_trainer.record_csv")
    def test_explicit_full_catalog_evaluation_remains_supported(self, record):
        trainer = self.make_trainer(mode="full")
        self.assertIs(trainer.checkpoint_valid_loader, trainer.valid_loader)
        self.assertIsInstance(trainer.test_loader.dataset, FullEvalDataset)
        trainer.test()
        self.assertEqual(trainer.model.candidate_counts, [24])
        record.assert_called_once()

    @patch("trainers.sequence_trainer.record_csv")
    def test_both_protocols_can_be_sampled(self, record):
        trainer = self.make_trainer(checkpoint_mode="sampled")
        self.assertIs(trainer.checkpoint_valid_loader, trainer.valid_loader)
        trainer.train()
        self.assertEqual(trainer.model.candidate_counts, [3, 3])
        self.assertGreater(trainer.stopper.scores[0], 0.0)
        record.assert_called_once()

    @patch("trainers.sequence_trainer.record_csv")
    def test_checkpoint_protocol_can_be_sampled_with_full_catalog_test(self, record):
        trainer = self.make_trainer(mode="full", checkpoint_mode="sampled")
        self.assertNotIsInstance(trainer.checkpoint_valid_loader.dataset, FullEvalDataset)
        self.assertIsInstance(trainer.test_loader.dataset, FullEvalDataset)
        trainer.train()
        self.assertEqual(trainer.model.candidate_counts, [3, 24])
        self.assertGreater(trainer.stopper.scores[0], 0.0)
        record.assert_called_once()

    @patch("trainers.sequence_trainer.record_csv")
    def test_checkpoint_selection_excludes_tsrc_but_final_test_restores_it(self, record):
        trainer = self.make_trainer(semantic=True)
        trainer.train()
        self.assertEqual(trainer.stopper.scores, [0.0])
        self.assertEqual(trainer.model.candidate_counts, [24, 3, 3])
        self.assertEqual(trainer.model.semantic_states, [False, True, True])
        self.assertTrue(trainer.model.use_semantic_rerank)
        record.assert_called_once()


if __name__ == "__main__":
    unittest.main()
