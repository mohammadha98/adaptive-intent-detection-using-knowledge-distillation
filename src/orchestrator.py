import threading
import copy
import os
import json
import time
import random
import torch
from src.student_model import BERTClassifier
from src.teacher_llm import TeacherAgent
from src.local_augmenter import LocalAugmenter
from src.replay_buffer import ReplayBuffer
from src.EWCRegularizer import EWCRegularizer
from src.SmartBatchAccumulator import SmartBatchAccumulator
from config import Config


class Orchestrator:
    def __init__(self):
        # Core Components
        self.student = BERTClassifier()
        self.teacher = TeacherAgent()
        self.augmenter = LocalAugmenter(device=Config.AUGMENTER_DEVICE)
        
        # AOCL Components (اضافه شده)
        self.replay_buffer = ReplayBuffer(max_size=Config.REPLAY_BUFFER_SIZE)
        self.ewc = EWCRegularizer(model=self.student, lambda_ewc=Config.EWC_LAMBDA)
        self.accumulator = SmartBatchAccumulator(
            min_batch=Config.MIN_BATCH_SIZE,
            max_wait_seconds=Config.MAX_WAIT_SECONDS
        )
        
        # Concurrency
        self.model_lock = threading.Lock()
        self.training_lock = threading.Lock()
        self.is_training = False
        
        # Metrics (اضافه شده)
        self.metrics = {
            "total_queries": 0,
            "student_answers": 0,
            "teacher_fallbacks": 0,
            "retrain_count": 0
        }

    # ─────────────────────────────────────────────
    #  BOOTSTRAP (Cold Start)
    # ─────────────────────────────────────────────
    @property
    def size(self):
       return sum(len(v) for v in self.buffer.values())

    def bootstrap_system(self, intents=None):
        if not intents and getattr(self.student, "is_trained", False):
            print("Orchestrator: Student already trained. Skipping bootstrap.")
            return 0

        print("Orchestrator: Cold start. Generating synthetic data...")
        all_data = []
        
        intents_source = self._resolve_intents(intents)
        self.teacher.intents = intents_source
        use_local_augmentation = getattr(Config, "USE_LOCAL_AUGMENTATION", False)
        
        for intent in intents_source:
            print(f"  Generating: {intent['title']}...")
            if use_local_augmentation:
                batch = self.augmenter.generate_intent_data_batch(
                    intent_name=intent["title"],
                    description=intent["description"],
                    seeds=intent.get("seeds", []),
                    total_count=Config.BOOTSTRAP_SAMPLES_PER_INTENT,
                )
            else:
                batch = self.teacher.generate_intent_data_batch(
                    intent_name=intent["title"],
                    description=intent["description"],
                    seeds=intent.get("seeds", []),
                    total_count=Config.BOOTSTRAP_SAMPLES_PER_INTENT,
                )
            all_data.extend(batch)

        # Save training data
        self._save_training_data(all_data)
        
        # Train (blocking for bootstrap)
        print(f"Orchestrator: Training on {len(all_data)} bootstrap samples...")
        with self.model_lock:
            self.student.train_on_data(all_data)
            self._save_model()
        
        # Initialize replay buffer with bootstrap data
        for item in all_data:
            self.replay_buffer.add(item["text"], item["label"])
        
        # Compute initial Fisher (for EWC)
        self.ewc.compute_fisher(self.student, all_data)
        
        print(f"Orchestrator: Bootstrap complete. {len(all_data)} samples.")
        return len(all_data)

    # ─────────────────────────────────────────────
    #  INFERENCE (Query Handling)
    # ─────────────────────────────────────────────
    
    def handle_query(self, text: str):
        self.metrics["total_queries"] += 1
        
        # 1. Student prediction (thread-safe)
        with self.model_lock:
            label, conf = self.student.predict(text)
        
        # 2. Confidence gate
        if conf >= Config.CONFIDENCE_THRESHOLD and label != "Untrained":
            self.metrics["student_answers"] += 1
            return {
                "label": label,
                "confidence": round(conf, 4),
                "source": "student",
                "status": "confident"
            }
        
        # 3. Teacher fallback
        self.metrics["teacher_fallbacks"] += 1
        teacher_result = self.teacher.get_fallback_prediction(text)
        
        # teacher_result باید dict باشه نه string
        teacher_label = (teacher_result if isinstance(teacher_result, str) 
                        else teacher_result.get("label", "unknown"))
        teacher_conf = (1.0 if isinstance(teacher_result, str) 
                       else teacher_result.get("confidence", 0.8))
        
        # 4. Add to accumulator (not immediate retrain!)
        if teacher_conf >= Config.TEACHER_CONFIDENCE_THRESHOLD:
            self.accumulator.add(text, teacher_label)
            self.replay_buffer.add(text, teacher_label)
        
        # 5. Check if should trigger training
        if self.accumulator.should_trigger():
            self._trigger_async_training()
        
        return {
            "label": teacher_label,
            "confidence": round(teacher_conf, 4),
            "source": "teacher",
            "status": "fallback"
        }

    # ─────────────────────────────────────────────
    #  ASYNC TRAINING (Incremental)
    # ─────────────────────────────────────────────
    
    def _trigger_async_training(self):
        with self.training_lock:
            if self.is_training:
                return
            self.is_training = True
        
        thread = threading.Thread(
            target=self._incremental_train_worker,
            daemon=True
        )
        thread.start()
    
    def _incremental_train_worker(self):
        try:
            print("Orchestrator: [Async] Starting incremental training...")
            
            # 1. Flush new data from accumulator
            new_data = self.accumulator.flush()
            if not new_data:
                return
            
            # 2. Sample replay data
            replay_count = int(len(new_data) * Config.REPLAY_RATIO)
            replay_data = self.replay_buffer.sample(
                k=replay_count, 
                strategy="balanced"
            )
            
            # 3. Combine batch
            training_batch = new_data + replay_data
            random.shuffle(training_batch)
            
            print(f"Orchestrator: [Async] Batch: "
                  f"{len(new_data)} new + {len(replay_data)} replay "
                  f"= {len(training_batch)} total")
            
            # 4. Create shadow model (copy of current)
            with self.model_lock:
                shadow = copy.deepcopy(self.student)
            
            # # 5. Incremental train with EWC
            # shadow.incremental_train(
            #     data=training_batch,
            #     ewc_regularizer=self.ewc,
            #     epochs=Config.INCREMENTAL_EPOCHS,
            #     lr=Config.INCREMENTAL_LR
            # )
            # بعد (کار میکنه ولی بدون EWC):
            shadow.train_on_data(training_batch)
            
            # 6. Validate before swap (optional but recommended)
            if self._validate_shadow(shadow):
                with self.model_lock:
                    self.student = shadow
                    self._save_model()
                
                # 7. Update EWC Fisher for new state
                self.ewc.compute_fisher(self.student, training_batch)
                
                self.metrics["retrain_count"] += 1
                print("Orchestrator: [Async] Model updated successfully.")
            else:
                print("Orchestrator: [Async] Validation failed. Keeping old model.")
                
        except Exception as e:
            print(f"Orchestrator: [Async] Error: {e}")
            import traceback
            traceback.print_exc()  # ← برای دیباگ بهتره
        finally:
            with self.training_lock:
                 self.is_training = False
            print(f"Orchestrator: [Async] Training flag released. is training: {self.is_training}")  # ← لاگ اضافه
    
    def _validate_shadow(self, shadow_model):
        """
        بررسی اینکه مدل جدید بدتر از قبلی نشده
        """
        validation_samples = self.replay_buffer.sample(
            k=min(50, self.replay_buffer.size),
            strategy="balanced"
        )
        if not validation_samples:
            return True  # اگه داده‌ای نیست، قبول کن
        
        correct = 0
        for item in validation_samples:
            label, conf = shadow_model.predict(item["text"])
            if label == item["label"]:
                correct += 1
        
        accuracy = correct / len(validation_samples)
        print(f"Orchestrator: [Async] Validation accuracy: {accuracy:.2%}")
        
        return accuracy >= Config.MIN_VALIDATION_ACCURACY

    # ─────────────────────────────────────────────
    #  UTILITY METHODS
    # ─────────────────────────────────────────────
    
    def _resolve_intents(self, intents):
        if intents:
            return [{"title": i.get("title", "unknown"),
                     "description": i.get("description", ""),
                     "seeds": i.get("seeds", [])} for i in intents]
        if hasattr(Config, "INTENTS") and Config.INTENTS:
            return [{"title": i.get("title", "unknown"),
                     "description": i.get("description", ""),
                     "seeds": i.get("seeds", [])} for i in Config.INTENTS]
        return []
    
    def _save_training_data(self, data):
        try:
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            path = os.path.join(Config.DATA_DIR, "training_data.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Orchestrator: Save error: {e}")
    
    def _save_model(self):
        try:
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(self.student.state_dict(), Config.MODEL_SAVE_PATH)
        except Exception as e:
            print(f"Orchestrator: Model save error: {e}")
    
    def get_metrics(self):
        fallback_rate = (self.metrics["teacher_fallbacks"] / 
                        max(1, self.metrics["total_queries"]))
        return {
            **self.metrics,
            "fallback_rate": round(fallback_rate, 4),
            "pending_samples": len(self.accumulator.pending_buffer),
            "replay_buffer_size": self.replay_buffer.size,
            "is_training": self.is_training
        }
