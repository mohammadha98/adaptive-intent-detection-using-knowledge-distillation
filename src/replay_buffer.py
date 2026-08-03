
import random


class ReplayBuffer:
    def __init__(self, max_size=5000):
        self.buffer = {}  # {label: [samples]}
        self.max_size = max_size
    
    @property
    def size(self):
        return sum(len(v) for v in self.buffer.values())
    

    
    def add(self, text, label):
        if label not in self.buffer:
            self.buffer[label] = []
        self.buffer[label].append(text)
        self._enforce_limit()
    
    def sample(self, k, strategy="balanced"):
        """نمونه متوازن از هر کلاس"""
        samples = []
        labels = list(self.buffer.keys())
        per_class = max(1, k // len(labels))
        for label in labels:
            available = self.buffer[label]
            n = min(per_class, len(available))
            chosen = random.sample(available, n)
            samples.extend([{"text": t, "label": label} for t in chosen])
        return samples
    
    def _enforce_limit(self):
        total = sum(len(v) for v in self.buffer.values())
        while total > self.max_size:
            # حذف قدیمی‌ترین از بزرگ‌ترین کلاس
            largest = max(self.buffer, key=lambda k: len(self.buffer[k]))
            self.buffer[largest].pop(0)
            total -= 1
