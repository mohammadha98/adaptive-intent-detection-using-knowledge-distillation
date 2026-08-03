import sys
import os
import time
import threading

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestrator import Orchestrator
from config import Config

def run_ewc_test():
    """
    اسکریپت تست برای اطمینان از عملکرد صحیح آموزش ناهمگام با EWC.
    """
    print("--- EWC Integration Test ---")
    
    # 1. مقداردهی اولیه Orchestrator
    # برای تست، مقادیر trigger را کوچک در نظر می‌گیریم
    Config.MIN_BATCH_SIZE = 4
    Config.MAX_WAIT_SECONDS = 10
    Config.INCREMENTAL_EPOCHS = 2
    Config.EWC_LAMBDA = 1.0 # مقدار لامبدا را برای مشاهده بهتر اثر، بالا می‌بریم
    
    orchestrator = Orchestrator()
    
    # 2. Bootstrap سیستم با چند اینتنت اولیه
    print("\n--- Step 1: Bootstrapping System ---")
    initial_intents = [
        {"title": "get_weather", "description": "Ask for the weather", "seeds": ["what's the weather like", "is it sunny"]},
        {"title": "play_music", "description": "Play a song", "seeds": ["play a song by artist", "I want to hear music"]}
    ]
    orchestrator.bootstrap_system(intents=initial_intents)
    print("Bootstrap complete. Initial Fisher matrix should be computed.")
    
    # 3. شبیه‌سازی دریافت داده‌های جدید برای تحریک آموزش
    print("\n--- Step 2: Simulating new data arrival ---")
    
    # این داده‌ها باید آموزش ناهمگام را فعال کنند
    new_samples = [
        {"text": "what's the temperature in London?", "label": "get_weather"},
        {"text": "tell me if it is raining", "label": "get_weather"},
        {"text": "play the new taylor swift song", "label": "play_music"},
        {"text": "I feel like rock music", "label": "play_music"},
        # یک نمونه از یک اینتنت کمی متفاوت
        {"text": "how about some jazz?", "label": "play_music"} 
    ]
    
    for sample in new_samples:
        print(f"Adding new sample: '{sample['text']}'")
        orchestrator.accumulator.add(sample['text'], sample['label'])
    
    # 4. بررسی وضعیت و انتظار برای اتمام آموزش
    print("\n--- Step 3: Waiting for async training to trigger and complete ---")
    
    # بررسی اینکه آیا آموزش فعال شده است یا نه
    if orchestrator.accumulator.should_trigger():
        print("Accumulator triggered training. Starting async worker...")
        # به صورت دستی تابع را فراخوانی می‌کنیم تا خروجی آن را ببینیم
        # در حالت واقعی، handle_query این کار را در یک ترد جدا انجام می‌دهد
        train_thread = threading.Thread(target=orchestrator._incremental_train_worker)
        train_thread.start()
        
        # منتظر می‌مانیم تا ترد تمام شود
        train_thread.join(timeout=120) # 120 ثانیه تایم‌اوت
        
        if train_thread.is_alive():
            print("TEST FAILED: Training thread timed out.")
        else:
            print("TEST PASSED: Training thread finished.")
    else:
        print("TEST FAILED: Accumulator did not trigger training.")

    print("\n--- Test Finished ---")
    print("Please check the console logs for '[Incremental Train]' messages to verify EWC loss calculation.")

if __name__ == "__main__":
    run_ewc_test()
