import sys
import os
import time
import json
import logging
import threading

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestrator import Orchestrator
from config import Config

def run_test():
    print("=== TEST START: Structured Intent Definition Verification ===")
    
    # 0. Setup: Clear feedback file
    feedback_file = "data/feedback_store.json"
    if os.path.exists(feedback_file):
        with open(feedback_file, 'w') as f:
            json.dump([], f)
    
    # 1. Initialize
    print("Step 1: Initializing Orchestrator...")
    orch = Orchestrator()
    
    # Define structured intents as per new requirement
    structured_intents = [
        {
            "title": "balance",
            "description": "User wants to check their current account balance or available funds.",
            "seeds": ["check balance", "how much money do I have", "show my account status"]
        },
        {
            "title": "transfer",
            "description": "User wants to send money to another person or account.",
            "seeds": ["send money", "transfer funds", "pay my friend"]
        }
    ]
    
    print("\nStep 2: Bootstrapping with Structured Intents...")
    try:
        count = orch.bootstrap_system(structured_intents)
        print(f"PASS: Bootstrapped successfully with {count} samples.")
    except Exception as e:
        print(f"FAIL: Bootstrap failed with error: {e}")
        return

    # 3. Verify Augmentation
    # Check if augmented data has correct labels
    labels = set(d['label'] for d in orch.current_data)
    print(f"Labels found in data: {labels}")
    
    if "balance" in labels and "transfer" in labels:
        print("PASS: Correct labels found in augmented data.")
    else:
        print("FAIL: Missing expected labels.")

    print("\n=== TEST END ===")

if __name__ == "__main__":
    run_test()
