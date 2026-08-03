import unittest
from unittest.mock import MagicMock

from src.local_augmenter import LocalAugmenter

class TestGenerateIntentDataBatch(unittest.TestCase):
    def setUp(self):
        self.augmenter = LocalAugmenter(device="cpu", lazy_load=True)

    def test_generate_batch_with_augmented_outputs(self):
        # Mock heavy model methods so test stays fast/offline.
        self.augmenter.back_translate = MagicMock(side_effect=lambda text, mid_lang: f"{text} ({mid_lang})")
        self.augmenter.paraphrase_t5 = MagicMock(return_value=[
            "what is my account balance",
            "could you show my balance",
            "show balance please",
            "how much money is in my account",
            "tell me my current balance",
        ])
        self.augmenter.quality_filter = MagicMock(side_effect=lambda original, candidates, **kwargs: [
            c for c in candidates if c.strip().lower() != original.strip().lower()
        ])

        data = self.augmenter.generate_intent_data_batch(
            intent_name="check_balance",
            description="Check account balance",
            seeds=["what is my balance", "balance please", "account funds"],
            total_count=25,
        )
        self.assertGreater(len(data), 0)
        self.assertLessEqual(len(data), 25)
        self.assertTrue(all(d["label"] == "check_balance" for d in data))
        self.assertTrue(all(isinstance(d["text"], str) and d["text"] for d in data))

    def test_generate_batch_handles_empty_seeds(self):
        self.augmenter.back_translate = MagicMock(return_value="")
        self.augmenter.paraphrase_t5 = MagicMock(return_value=[])
        self.augmenter.quality_filter = MagicMock(return_value=[])

        data = self.augmenter.generate_intent_data_batch(
            intent_name="transfer_funds",
            description="Transfer money to another account",
            seeds=[],
            total_count=10,
        )
        self.assertGreater(len(data), 0)
        self.assertLessEqual(len(data), 10)
        self.assertTrue(all(d["label"] == "transfer_funds" for d in data))
        self.assertTrue(all(isinstance(d["text"], str) and d["text"] for d in data))

if __name__ == "__main__":
    unittest.main()
