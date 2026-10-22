import argparse
import os
from pathlib import Path
import tempfile
import unittest

import torch  # Match main.py: load Torch before pandas on Windows.

from utils.logger import Logger
from utils.utils import metric_pop_report, record_csv, record_group

import numpy as np
import pandas as pd


class PopularityMetricsTest(unittest.TestCase):
    def test_first_user_counts_in_either_item_group(self):
        for popularity, prefix in [(1, "Tail"), (9, "Popular")]:
            with self.subTest(group=prefix):
                actual = metric_pop_report(
                    np.array([0, 0]), np.array([0, popularity, popularity]),
                    np.array([1, 2]), args=argparse.Namespace(ts_item=4),
                )
                self.assertEqual(actual[f"{prefix} HR@10"], 1.0)
                self.assertEqual(actual[f"{prefix} NDCG@10"], 1.0)

    def test_group_scores_are_invariant_to_user_order(self):
        ranks = np.array([0, 9, 10, 2])
        targets = np.array([1, 2, 3, 4])
        popularity = np.array([0, 9, 1, 9, 1])
        args = argparse.Namespace(ts_item=4)
        expected = metric_pop_report(ranks, popularity, targets, args=args)
        permutation = np.array([2, 1, 3, 0])
        actual = metric_pop_report(
            ranks[permutation], popularity, targets[permutation], args=args,
        )
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key])
        self.assertEqual(expected["Popular HR@10"], 0.5)
        self.assertEqual(expected["Tail HR@10"], 1.0)
        self.assertAlmostEqual(expected["Popular NDCG@10"], 0.5)
        self.assertAlmostEqual(
            expected["Tail NDCG@10"], (1 / np.log2(11) + 1 / np.log2(4)) / 2,
        )


class LogOutputTest(unittest.TestCase):
    def test_text_and_csv_logs_use_requested_root(self):
        with tempfile.TemporaryDirectory() as directory:
            original_cwd = os.getcwd()
            os.chdir(directory)
            try:
                for timestamped in [True, False]:
                    with self.subTest(timestamped=timestamped):
                        root = Path(directory) / "log(重新评测)" / str(timestamped)
                        args = argparse.Namespace(
                            log_root=str(root), dataset="fashion",
                            model_name="llmesr_bert4rec", demo=False,
                            log=timestamped, tensorboard=False,
                            train_batch_size=128, lr=0.001,
                            experiment_variant="base", aug_file="",
                            now_str="test",
                        )
                        manager = Logger(args)
                        try:
                            manager.logger.info("reevaluation routing check")
                            manager.fh.flush()
                            log_path = Path(manager.fh.baseFilename)
                            self.assertTrue(log_path.is_relative_to(root))
                            self.assertIn(
                                "reevaluation routing check",
                                log_path.read_text(encoding="utf-8"),
                            )
                        finally:
                            manager.end_log()
                            manager.fh.close()
                            manager.ch.close()
                        record_csv(args, {"HR@10": 0.5})
                        record_group(args, {"HR@10": 0.75})
                        csv_path = root / "fashion" / "llmesr_bert4rec.csv"
                        self.assertEqual(pd.read_csv(csv_path)["HR@10"].tolist(), [0.5, 0.75])
                        explicit = root / "explicit"
                        record_csv(args, {"HR@10": 1.0}, path=str(explicit))
                        self.assertTrue((explicit / "fashion" / "llmesr_bert4rec.csv").is_file())
                self.assertFalse((Path(directory) / "log").exists())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
