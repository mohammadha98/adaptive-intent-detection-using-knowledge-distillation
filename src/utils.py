import random
import numpy as np
import torch
import logging

import json
import os
from config import Config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class FeedbackManager:
    def __init__(self, feedback_file="data/feedback_store.json"):
        self.feedback_file = feedback_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump([], f)

    def add_feedback(self, text, label):
        """
        Appends new feedback data. Handles duplicates by checking if text exists.
        """
        with open(self.feedback_file, 'r') as f:
            data = json.load(f)
        
        # Check for duplicates
        if any(d['text'] == text for d in data):
            print(f"FeedbackManager: Duplicate found for '{text}'. Skipping.")
            return

        data.append({"text": text, "label": label})
        
        with open(self.feedback_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"FeedbackManager: Added feedback for '{text}' as '{label}'.")

    def get_feedback_data(self):
        with open(self.feedback_file, 'r') as f:
            return json.load(f)

    def get_combined_data(self, original_data):
        """
        Merges Phase 0 data with feedback data.
        Prioritizes Feedback (Active Learning) if duplicates exist?
        For now, just simple merge with unique texts.
        """
        feedback_data = self.get_feedback_data()
        combined = original_data.copy()
        
        existing_texts = set(d['text'] for d in combined)
        
        for item in feedback_data:
            if item['text'] not in existing_texts:
                combined.append(item)
                existing_texts.add(item['text'])
        
        return combined
