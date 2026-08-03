import os
import torch
from dotenv import load_dotenv

load_dotenv()

class Config:
    CONFIDENCE_THRESHOLD = 0.8  # Raised slightly for conservative fallback
    TEMPERATURE = 1           # Calibration: Higher temp = lower confidence
    TEACHER_MODEL_NAME = "gpt-4o-mini"
    BASE_URL="https://api.gapgpt.app/v1"
    STUDENT_MODEL_NAME = "distilbert-base-uncased"
    # Detect device: cuda > mps > cpu
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Paths
    DATA_DIR = "data"
    MODEL_SAVE_PATH = "data/student_model.pt"

    TEACHER_CONFIDENCE_THRESHOLD = 0.7  # حداقل اطمینان Teacher
    
    # ─── Bootstrap ───
    BOOTSTRAP_SAMPLES_PER_INTENT = 50
    INTENTS = []  # Populated by user/API

    # ─── Offline Augmentation ───
    USE_LOCAL_AUGMENTATION = False  # Temporary switch: False => use LLM for augmentation
    AUGMENTER_DEVICE = "cpu"  # "cpu" (default) or "cuda" when available
    BT_LANGUAGES = ["de", "fr", "ru"]
    PARAPHRASE_COUNT = 5
    QUALITY_FILTER_LOW = 0.7
    QUALITY_FILTER_HIGH = 0.98
    AUGMENTER_SEED = 42
    
    # ─── Batch Accumulation ───
    MIN_BATCH_SIZE = 32
    MAX_WAIT_SECONDS = 300  # 5 minutes
    
    # ─── Replay Buffer ───
    REPLAY_BUFFER_SIZE = 5000
    REPLAY_RATIO = 0.4  # 40% replay, 60% new
    
    # ─── Continual Learning ───
    EWC_LAMBDA = 0.4
    INCREMENTAL_EPOCHS = 3
    INCREMENTAL_LR = 2e-5
    
    # ─── Validation ───
    MIN_VALIDATION_ACCURACY = 0.80  # حداقل دقت برای swap
