import unittest
import numpy as np
import pandas as pd

from train import eval_metrics

class TrainTest(unittest.TestCase):


    def test_1_eval_metrics_ineffective_model(self):
        acc, recall, prec, f1, roc_auc = eval_metrics(pd.DataFrame({0: [0, 1]}), pd.DataFrame({0: [1, 0]}))
        self.assertEqual(acc, 0)
        self.assertEqual(recall, 0)
        self.assertEqual(prec, 0)
        self.assertEqual(f1, 0)
        self.assertEqual(roc_auc, 0)

    def test_2_eval_metrics_effective_model(self):
        acc, recall, prec, f1, roc_auc = eval_metrics(pd.DataFrame({0: [0, 1]}), pd.DataFrame({0: [0, 1]}))
        self.assertEqual(acc, 1)
        self.assertEqual(recall, 1)
        self.assertEqual(prec, 1)
        self.assertEqual(f1, 1)
        self.assertEqual(roc_auc, 1)


if __name__ == "__main__":
    unittest.main()
