"""
ماژول ارزیابی سیستم AutoMTL با استفاده از دیتاست CLINC150
این ماژول قادر است عملکرد سیستم را در تشخیص OOD و برچسب‌گذاری LLM ارزیابی کند
"""

import json
import os
import time
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np


class CLINCDatasetLoader:
    """بارگذار داده‌های CLINC150"""
    
    def __init__(self, dataset_path: str = "datasets/clinic150/data/data_full.json"):
        self.dataset_path = dataset_path
        self.data = None
        self.intent_names = []
        self.oos_samples = []
        self.in_domain_samples = []
        
    def load_data(self) -> Dict[str, List[List[str]]]:
        """بارگذاری داده‌های CLINC150"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # جداسازی نمونه‌های OOS و In-Domain
            self._separate_samples()
            
            print(f"Dataset loaded successfully:")
            print(f"- Total intents: {len(self.intent_names)}")
            print(f"- OOS samples: {len(self.oos_samples)}")
            print(f"- In-domain samples: {len(self.in_domain_samples)}")
            
            return self.data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in dataset: {e}")
    
    def _separate_samples(self):
        """جداسازی نمونه‌های OOS و In-Domain"""
        if not self.data:
            return
            
        self.oos_samples = []
        self.in_domain_samples = []
        self.intent_names = []
        
        # داده‌های تست را برای ارزیابی استفاده می‌کنیم
        test_data = self.data.get('test', [])
        oos_test_data = self.data.get('oos_test', [])
        
        # پردازش نمونه‌های OOS
        for sample in oos_test_data:
            if isinstance(sample, list) and len(sample) == 2:
                text, label = sample
                self.oos_samples.append({
                    'text': text,
                    'label': 'oos',
                    'original_intent': label
                })
        
        # پردازش نمونه‌های In-Domain
        intent_counts = {}
        for sample in test_data:
            if isinstance(sample, list) and len(sample) == 2:
                text, label = sample
                if label not in intent_counts:
                    intent_counts[label] = 0
                    self.intent_names.append(label)
                intent_counts[label] += 1
                
                self.in_domain_samples.append({
                    'text': text,
                    'label': label,
                    'original_intent': label
                })
        
        print(f"Intent distribution in test set:")
        for intent, count in intent_counts.items():
            print(f"  {intent}: {count} samples")
    
    def get_test_samples(self, sample_size: int = None, include_oos: bool = True) -> List[Dict[str, Any]]:
        """دریافت نمونه‌های تست برای ارزیابی"""
        test_samples = []
        
        # افزودن نمونه‌های OOS
        if include_oos and self.oos_samples:
            oos_to_add = self.oos_samples
            if sample_size:
                # نمونه‌گیری متوازن از OOS و In-Domain
                oos_count = min(sample_size // 4, len(self.oos_samples))  # 25% OOS
                oos_to_add = np.random.choice(self.oos_samples, oos_count, replace=False).tolist()
            test_samples.extend(oos_to_add)
        
        # افزودن نمونه‌های In-Domain
        if self.in_domain_samples:
            in_domain_to_add = self.in_domain_samples
            if sample_size:
                in_domain_count = min(sample_size - len([s for s in test_samples if s['label'] == 'oos']), 
                                    len(self.in_domain_samples))
                in_domain_to_add = np.random.choice(self.in_domain_samples, in_domain_count, replace=False).tolist()
            test_samples.extend(in_domain_to_add)
        
        # مخلوط کردن نمونه‌ها
        np.random.shuffle(test_samples)
        
        return test_samples


class AutoMTLEvaluator:
    """ارزیاب‌ سیستم AutoMTL"""
    
    def __init__(self, orchestrator=None, dataset_path: str = None):
        self.orchestrator = orchestrator
        self.dataset_loader = CLINCDatasetLoader(dataset_path) if dataset_path else CLINCDatasetLoader()
        self.results = []
        self.metrics = {}
        
    def load_dataset(self) -> bool:
        """بارگذاری دیتاست CLINC150"""
        try:
            self.dataset_loader.load_data()
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def evaluate_system(self, sample_size: int = 1000, include_oos: bool = True) -> Dict[str, Any]:
        """ارزیابی کامل سیستم"""
        if not self.orchestrator:
            raise ValueError("Orchestrator not provided")
        
        if not self.dataset_loader.data:
            if not self.load_dataset():
                raise ValueError("Failed to load dataset")
        
        print("Starting system evaluation...")
        test_samples = self.dataset_loader.get_test_samples(sample_size, include_oos)
        
        self.results = []
        start_time = time.time()
        
        # پردازش هر نمونه
        for i, sample in enumerate(test_samples):
            if i % 100 == 0:
                print(f"Processing sample {i+1}/{len(test_samples)}")
            
            result = self._evaluate_sample(sample)
            self.results.append(result)
        
        evaluation_time = time.time() - start_time
        
        # محاسبه متریک‌ها
        self.metrics = self._calculate_metrics()
        self.metrics['evaluation_time'] = evaluation_time
        self.metrics['total_samples'] = len(test_samples)
        
        print(f"Evaluation completed in {evaluation_time:.2f} seconds")
        return self.metrics
    
    def _evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """ارزیابی یک نمونه خاص"""
        text = sample['text']
        true_label = sample['label']
        is_true_oos = true_label == 'oos'
        
        # ارسال به سیستم
        try:
            prediction = self.orchestrator.handle_query(text)
            
            result = {
                'text': text,
                'true_label': true_label,
                'is_true_oos': is_true_oos,
                'predicted_label': prediction['label'],
                'confidence': prediction['confidence'],
                'source': prediction['source'],
                'is_predicted_oos': prediction['label'] == 'oos' or prediction['confidence'] < 0.75,
                'is_fallback': prediction['source'] == 'teacher'
            }
            
            # بررسی صحت برچسب‌گذاری LLM
            if result['is_fallback']:
                result['llm_label_correct'] = self._is_llm_label_correct(
                    prediction['label'], true_label
                )
            else:
                result['llm_label_correct'] = None
                
        except Exception as e:
            result = {
                'text': text,
                'true_label': true_label,
                'is_true_oos': is_true_oos,
                'error': str(e),
                'predicted_label': 'error',
                'confidence': 0.0,
                'source': 'error',
                'is_predicted_oos': False,
                'is_fallback': False,
                'llm_label_correct': False
            }
        
        return result
    
    def _is_llm_label_correct(self, predicted_label: str, true_label: str) -> bool:
        """بررسی صحت برچسب‌گذاری LLM"""
        if true_label == 'oos':
            # برای نمونه‌های OOS، برچسب صحیح باید 'oos' باشد یا یک برچسب جدید
            return predicted_label == 'oos' or predicted_label not in self.dataset_loader.intent_names
        else:
            # برای نمونه‌های In-Domain، برچسب باید دقیقاً مطابقت داشته باشد
            return predicted_label == true_label
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """محاسبه متریک‌های ارزیابی"""
        if not self.results:
            return {}
        
        # تبدیل نتایج به لیست‌های جداگانه
        y_true_oos = [r['is_true_oos'] for r in self.results]
        y_pred_oos = [r['is_predicted_oos'] for r in self.results]
        
        # متریک‌های تشخیص OOD
        ood_accuracy = accuracy_score(y_true_oos, y_pred_oos)
        ood_precision, ood_recall, ood_f1, _ = precision_recall_fscore_support(
            y_true_oos, y_pred_oos, average='binary'
        )
        
        # متریک‌های برچسب‌گذاری LLM (فقط برای نمونه‌هایی که به LLM ارجاع شدند)
        llm_results = [r for r in self.results if r['is_fallback'] and r['llm_label_correct'] is not None]
        
        if llm_results:
            llm_accuracy = sum(r['llm_label_correct'] for r in llm_results) / len(llm_results)
            llm_precision, llm_recall, llm_f1, _ = precision_recall_fscore_support(
                [r['true_label'] == r['predicted_label'] for r in llm_results],
                [True] * len(llm_results),
                average='binary'
            )
        else:
            llm_accuracy = llm_precision = llm_recall = llm_f1 = 0.0
        
        # آمار کلی
        total_fallbacks = sum(1 for r in self.results if r['is_fallback'])
        total_oos_correct = sum(1 for r in self.results if r['is_true_oos'] and r['is_predicted_oos'])
        total_indomain_correct = sum(1 for r in self.results if not r['is_true_oos'] and not r['is_predicted_oos'])
        
        metrics = {
            'ood_detection': {
                'accuracy': float(ood_accuracy),
                'precision': float(ood_precision),
                'recall': float(ood_recall),
                'f1_score': float(ood_f1)
            },
            'llm_labeling': {
                'accuracy': float(llm_accuracy),
                'precision': float(llm_precision) if llm_precision else 0.0,
                'recall': float(llm_recall) if llm_recall else 0.0,
                'f1_score': float(llm_f1) if llm_f1 else 0.0,
                'total_fallbacks': total_fallbacks
            },
            'overall': {
                'total_samples': len(self.results),
                'fallback_rate': total_fallbacks / len(self.results),
                'oos_correct_detection': total_oos_correct,
                'indomain_correct_detection': total_indomain_correct,
                'oos_samples': sum(y_true_oos),
                'indomain_samples': len(y_true_oos) - sum(y_true_oos)
            }
        }
        
        return metrics
    
    def get_detailed_results(self) -> pd.DataFrame:
        """دریافت نتایج دقیق به صورت DataFrame"""
        return pd.DataFrame(self.results)
    
    def save_results(self, output_path: str = "evaluation_results.json"):
        """ذخیره نتایج ارزیابی"""
        output_data = {
            'metrics': self.metrics,
            'detailed_results': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_path}")


def run_evaluation(orchestrator, dataset_path: str = None, sample_size: int = 1000, 
                  include_oos: bool = True, output_path: str = None) -> Dict[str, Any]:
    """تابع کمکی برای اجرای ارزیابی"""
    
    evaluator = AutoMTLEvaluator(orchestrator, dataset_path)
    
    # اجرای ارزیابی
    metrics = evaluator.evaluate_system(sample_size, include_oos)
    
    # ذخیره نتایج (در صورت درخواست)
    if output_path:
        evaluator.save_results(output_path)
    
    return metrics


if __name__ == "__main__":
    # تست ماژول ارزیابی
    print("Testing evaluation module...")
    
    # بارگذاری دیتاست برای تست
    loader = CLINCDatasetLoader()
    try:
        loader.load_data()
        print("Dataset loaded successfully for testing")
        
        # نمایش اطلاعات آماری
        test_samples = loader.get_test_samples(100)
        print(f"Sample test data: {len(test_samples)} samples")
        
        oos_count = sum(1 for s in test_samples if s['label'] == 'oos')
        print(f"OOS samples: {oos_count}")
        print(f"In-domain samples: {len(test_samples) - oos_count}")
        
    except Exception as e:
        print(f"Error in testing: {e}")