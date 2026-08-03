import sys
import os
import json
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestrator import Orchestrator
from config import Config

# Setup logging
log_file = "tests/test_log.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')

def log(message):
    print(message)
    logging.info(message)

def run_test():
    log("=== TEST START: Active Learning Loop ===")
    
    # 0. Setup: Clear feedback file to ensure clean state
    feedback_file = "data/feedback_store.json"
    if os.path.exists(feedback_file):
        with open(feedback_file, 'w') as f:
            json.dump([], f)
    
    # 1. Initialize
    log("Step 1: Initializing Orchestrator & Bootstrapping...")
    orch = Orchestrator()
    
    # Minimal bootstrap to get student into a known state
    intents = {
        "balance": ["check balance"],
        "transfer": ["send money"]
    }
    orch.bootstrap_system(intents)
    
    # 2. Step A (Fail)
    # We need a query that is Out-Of-Distribution (OOD) so the student is uncertain.
    # Since we only trained on 'balance' and 'transfer', 'cooking' should be ambiguous.
    hard_query = "I want to cook a delicious dinner for my family" 
    log(f"\nStep 2: Sending Hard Query: '{hard_query}'")
    
    res1 = orch.handle_query(hard_query)
    log(f"Result 1: {res1}")
    
    # Verification
    if res1['source'] != 'teacher':
        log("FAIL: Expected fallback to teacher, but got student.")
        return
    log("PASS: System correctly fell back to Teacher.")
    
    # 3. Step B (Learn)
    log("\nStep 3: Triggering Retraining (Active Learning)...")
    orch.retrain_student()
    
    # 4. Step C (Succeed)
    log(f"\nStep 4: Sending Same Query Again: '{hard_query}'")
    res2 = orch.handle_query(hard_query)
    log(f"Result 2: {res2}")
    
    # Verification
    # Note: Since it's a mock training on very few samples, we hope it overfits enough to learn it.
    # With 1 epoch and small data, it usually learns exact matches.
    
    if res2['source'] == 'student':
        if res2['confidence'] >= Config.CONFIDENCE_THRESHOLD:
            log("PASS: Student successfully learned the new query!")
        else:
            log(f"PARTIAL: Student took it, but confidence {res2['confidence']} < threshold.")
    else:
        log("FAIL: Still fell back to teacher. Model didn't learn enough.")

    log("\n=== TEST END ===")

if __name__ == "__main__":
    run_test()
