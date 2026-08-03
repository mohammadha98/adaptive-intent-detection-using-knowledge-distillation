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

# Setup logging
log_file = "tests/test_async_concurrency.py.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')

def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")
    logging.info(message)

def run_test():
    log("=== TEST START: Async Active Learning & Concurrency ===")
    
    # 0. Setup: Clear feedback file
    feedback_file = "data/feedback_store.json"
    if os.path.exists(feedback_file):
        with open(feedback_file, 'w') as f:
            json.dump([], f)
    
    # 1. Initialize
    log("Step 1: Initializing Orchestrator & Bootstrapping...")
    orch = Orchestrator()
    
    intents = {
        "balance": ["check balance", "how much money"],
        "transfer": ["send money", "transfer funds"]
    }
    orch.bootstrap_system(intents)
    
    # 2. Trigger Training (Hard Query)
    # This should trigger the Teacher, save feedback, and START background training.
    # We use a query about 'investing' which is not in balance/transfer
    hard_query = "What is the capital of France?"
    log(f"\nStep 2: Sending Hard Query (Trigger): '{hard_query}'")
    
    res1 = orch.handle_query(hard_query)
    log(f"Result 1 (Hard): {res1['source']}")
    
    if res1['source'] != 'teacher':
        log("FAIL: Expected teacher fallback.")
        return

    # Verify training started
    time.sleep(0.5) # Give a moment for thread to start
    if orch.is_training:
        log("PASS: Background training flag is TRUE.")
    else:
        log("FAIL: Background training did not start.")
        return

    # 3. Concurrent Request (While Training)
    # Send a KNOWN query immediately. It should be fast and use the OLD model.
    known_query = "check balance"
    log(f"\nStep 3: Sending Concurrent Query: '{known_query}'")
    
    start_time = time.time()
    res2 = orch.handle_query(known_query)
    duration = time.time() - start_time
    
    log(f"Result 2 (Known): {res2['source']} (Confidence: {res2['confidence']})")
    log(f"Inference Duration: {duration:.4f}s")
    
    if duration > 1.0:
        log("FAIL: Inference took too long. Blocking occurred!")
    else:
        log("PASS: System responded instantly (Non-Blocking).")
        
    if res2['source'] != 'student':
        log("FAIL: Old model failed to answer known query.")

    # 4. Wait for Swap
    log("\nStep 4: Waiting for Training to Finish...")
    while orch.is_training:
        time.sleep(1)
        # log("Waiting...")
    
    log("Background training finished.")
    
    # 5. Assert Final Swap
    # The hard query should now be known by the NEW student.
    log(f"\nStep 5: Sending Hard Query Again: '{hard_query}'")
    res3 = orch.handle_query(hard_query)
    log(f"Result 3 (Hard, Retry): {res3['source']} (Confidence: {res3['confidence']})")
    
    if res3['source'] == 'student' and res3['confidence'] >= Config.CONFIDENCE_THRESHOLD:
        log("PASS: New model successfully swapped in and learned the query.")
    else:
        log(f"PARTIAL/FAIL: Model swapped but confidence {res3['confidence']} might be low.")

    log("\n=== TEST END ===")

if __name__ == "__main__":
    run_test()
