import threading
import time


class SmartBatchAccumulator:
    def __init__(self, min_batch=32, max_wait_seconds=300):
        self.pending_buffer = []
        self.min_batch = min_batch
        self.max_wait = max_wait_seconds
        self.last_train_time = time.time()
        self.lock = threading.Lock()
    
    def add(self, text, label):
        with self.lock:
            self.pending_buffer.append({"text": text, "label": label})
    
    def should_trigger(self):
        with self.lock:
            size_ready = len(self.pending_buffer) >= self.min_batch
            time_ready = (time.time() - self.last_train_time > self.max_wait 
                         and len(self.pending_buffer) > 0)
            return size_ready or time_ready
    
    def flush(self):
        """داده‌های pending رو برگردون و خالی کن"""
        with self.lock:
            data = self.pending_buffer.copy()
            self.pending_buffer = []
            self.last_train_time = time.time()
            return data
