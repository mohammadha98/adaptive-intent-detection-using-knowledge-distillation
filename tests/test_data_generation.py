import sys
import os
import unittest
from collections import Counter
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.local_augmenter import LocalAugmenter

class TestDataGeneration(unittest.TestCase):
    def test_diverse_augmentation(self):
        print("\n=== TEST START: Offline LocalAugmenter Volume & Diversity ===")

        augmenter = LocalAugmenter(device="cpu", lazy_load=True)

        # Mock expensive model calls to keep unit test fast and deterministic.
        augmenter.back_translate = MagicMock(side_effect=lambda text, mid_lang: f"{text} ({mid_lang})")
        augmenter.paraphrase_t5 = MagicMock(return_value=[
            "could you check my balance",
            "please show my balance",
            "i need my account balance",
            "display available funds",
            "what is left in my account",
        ])
        augmenter.quality_filter = MagicMock(side_effect=lambda original, candidates, **kwargs: [
            c for c in candidates if c.strip().lower() != original.strip().lower()
        ])

        intents = [
            {
                "title": "balance",
                "description": "User wants to check balance",
                "seeds": ["check balance", "my money"]
            },
            {
                "title": "transfer",
                "description": "User wants to transfer funds",
                "seeds": ["send money", "transfer funds"]
            }
        ]

        print("Generating data using local offline augmenter...")
        data = augmenter.augment_seeds(intents)

        print(f"Total samples generated: {len(data)}")

        # Verify Labels
        labels = [d['label'] for d in data]
        counts = Counter(labels)
        print(f"Label distribution: {counts}")

        # Assertions
        self.assertIn("balance", counts, "Missing 'balance' intent")
        self.assertIn("transfer", counts, "Missing 'transfer' intent")

        # LocalAugmenter doesn't auto-generate OOS class; it augments provided intents only.
        self.assertNotIn("out_of_scope", counts)
        self.assertGreater(len(data), 10, "Expected reasonable amount of augmented data")

        print("PASS: Offline data generation logic is valid.")
        print("=== TEST END ===")

if __name__ == '__main__':
    unittest.main()
