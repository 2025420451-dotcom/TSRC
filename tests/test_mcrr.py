import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DualLLMSRS import LLMSemanticRerankerMixin


class _BackboneStub(nn.Module):
    def predict(self, seq, item_indices, positions, **kwargs):
        del seq, positions, kwargs
        if item_indices.dim() == 1:
            return item_indices.float().unsqueeze(0)
        return item_indices.float()


class _MCRRStub(LLMSemanticRerankerMixin, _BackboneStub):
    def __init__(self, table, **overrides):
        nn.Module.__init__(self)
        self.item_num = table.size(0) - 2
        defaults = {
            "disable_semantic_rerank": False,
            "disable_semantic_centering": False,
            "semantic_pooling": "recency",
            "disable_candidate_standardization": False,
            "semantic_recency_decay": 0.2,
            "semantic_rerank_weight": 1.0,
        }
        defaults.update(overrides)
        self._init_semantic_reranker(
            SimpleNamespace(**defaults), table.numpy()
        )


class MCRRTest(unittest.TestCase):
    def setUp(self):
        # Row 0 is padding and the last row is the BERT mask token.
        self.table = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

    def test_catalog_center_excludes_padding_and_mask(self):
        model = _MCRRStub(self.table)
        expected = self.table[1:4].mean(dim=0)
        torch.testing.assert_close(model.semantic_item_center, expected)

    def test_centering_off_is_raw_unit_normalization(self):
        model = _MCRRStub(
            self.table, disable_semantic_centering=True
        )
        item_ids = torch.tensor([1, 2, 3])
        actual = model._semantic_vectors(item_ids)
        expected = F.normalize(self.table[item_ids], dim=-1)
        torch.testing.assert_close(actual, expected)

    def test_mean_pooling_equals_zero_decay_recency(self):
        mean_model = _MCRRStub(
            self.table,
            semantic_pooling="mean",
            disable_semantic_centering=True,
        )
        recency_model = _MCRRStub(
            self.table,
            semantic_pooling="recency",
            semantic_recency_decay=0.0,
            disable_semantic_centering=True,
        )
        seq = torch.tensor([[0, 1, 0, 2], [0, 0, 2, 3]])
        candidates = torch.tensor([1, 2, 3])
        torch.testing.assert_close(
            mean_model.semantic_rerank_scores(seq, candidates),
            recency_model.semantic_rerank_scores(seq, candidates),
        )

    def test_internal_gaps_do_not_change_valid_event_distance(self):
        model = _MCRRStub(
            self.table,
            semantic_pooling="recency",
            semantic_recency_decay=0.7,
            disable_semantic_centering=True,
        )
        compact = torch.tensor([[0, 0, 1, 2]])
        with_gap = torch.tensor([[0, 1, 0, 2]])
        candidates = torch.tensor([1, 2, 3])
        torch.testing.assert_close(
            model.semantic_rerank_scores(compact, candidates),
            model.semantic_rerank_scores(with_gap, candidates),
        )

    def test_no_standardization_is_raw_residual_fusion(self):
        model = _MCRRStub(
            self.table, disable_candidate_standardization=True
        )
        base = torch.tensor([[2.0, 4.0, 8.0]])
        semantic = torch.tensor([[0.1, 0.2, 0.3]])
        actual = model.fuse_semantic_rerank_scores(
            base, semantic, weight=3.0
        )
        torch.testing.assert_close(actual, base + 3.0 * semantic)

    def test_empty_history_has_zero_semantic_scores(self):
        model = _MCRRStub(self.table)
        seq = torch.zeros((2, 4), dtype=torch.long)
        scores = model.semantic_rerank_scores(
            seq, torch.tensor([1, 2, 3])
        )
        torch.testing.assert_close(scores, torch.zeros_like(scores))

    def test_disabling_mcrr_returns_backbone_scores(self):
        model = _MCRRStub(
            self.table, disable_semantic_rerank=True
        )
        seq = torch.tensor([[0, 0, 1, 2]])
        candidates = torch.tensor([[1, 2, 3]])
        positions = torch.arange(4).unsqueeze(0)
        expected = candidates.float()
        torch.testing.assert_close(
            model.predict(seq, candidates, positions), expected
        )


if __name__ == "__main__":
    unittest.main()
