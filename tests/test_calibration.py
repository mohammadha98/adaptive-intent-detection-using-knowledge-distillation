import sys
import os
import torch
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.student_model import BERTClassifier
from src.utils import set_seed

class TestCalibration(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.student = BERTClassifier()
        
        # Minimal training data
        self.data = [
            {"text": "check balance", "label": "balance"},
            {"text": "how much money", "label": "balance"},
            {"text": "account status", "label": "balance"},
            {"text": "send money", "label": "transfer"},
            {"text": "transfer funds", "label": "transfer"},
            {"text": "pay friend", "label": "transfer"}
        ]
        
        # Train the model
        print("\nTraining student model for calibration test...")
        self.student.train_on_data(self.data)

    def test_temperature_scaling(self):
        print("\n=== TEST START: Calibration Verification ===")
        
        # A query that should be somewhat confident but not 100%
        # "i want to pay" shares words with "pay friend" but is short.
        query = "i want to pay"
        
        # Step A: Standard Prediction (T=1.0)
        label_a, conf_a = self.student.predict(query, temperature=1.0)
        print(f"Prediction (T=1.0): Label={label_a}, Conf={conf_a:.4f}")
        
        # Step B: Calibrated Prediction (T=2.0)
        label_b, conf_b = self.student.predict(query, temperature=2.0)
        print(f"Prediction (T=2.0): Label={label_b}, Conf={conf_b:.4f}")
        
        # Assertions
        self.assertEqual(label_a, label_b, "Labels should match across temperatures")
        self.assertLess(conf_b, conf_a, "Confidence should decrease with higher temperature")
        
        # Check magnitude of drop
        drop = conf_a - conf_b
        print(f"Confidence Drop: {drop:.4f}")
        
        # We expect a significant drop because T=2.0 flattens the distribution
        # If the model was 0.99, it might drop to ~0.7-0.8
        self.assertGreater(drop, 0.05, "Expected at least 5% drop in confidence")
        
        print("PASS: Model is successfully calibrated (humbled).")
        print("=== TEST END ===")

if __name__ == '__main__':
    unittest.main()
