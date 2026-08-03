import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class EWCRegularizer:
    def __init__(self, model, lambda_ewc=0.4):
        self.lambda_ewc = lambda_ewc
        self.saved_params = {}
        self.fisher = {}

    def compute_fisher(self, model, data, n_samples=200):
        """
        Fisher Information Matrix را محاسبه می‌کند.
        data: لیستی از {"text": ..., "label": ...}
        """
        if not data:
            return

        # 1. Tokenize data and build a DataLoader
        texts = [d["text"] for d in data]
        labels_str = [d["label"] for d in data]

        # Use model's own label2id (must be trained already)
        if not model.label2id:
            print("EWC: Model has no label2id. Skipping Fisher computation.")
            return

        labels = [model.label2id[l] for l in labels_str if l in model.label2id]
        if not labels:
            return

        # Filter texts to match valid labels
        valid_data = [(t, model.label2id[l]) for t, l in zip(texts, labels_str) if l in model.label2id]
        texts_valid = [t for t, _ in valid_data]
        labels_valid = [l for _, l in valid_data]

        encodings = model.tokenizer(texts_valid, truncation=True, padding=True, max_length=64)

        # Build simple dataset
        from src.student_model import SimpleDataset
        dataset = SimpleDataset(encodings, labels_valid)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        # 2. Compute Fisher
        fisher = {n: torch.zeros_like(p)
                  for n, p in model.named_parameters() if p.requires_grad}

        model.eval()
        count = 0
        for batch in loader:
            if count >= n_samples:
                break

            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            batch_labels = batch["labels"].to(model.device)

            model.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2

            count += len(batch_labels)

        # Normalize
        if count > 0:
            for n in fisher:
                fisher[n] /= count

        self.fisher = fisher
        self.saved_params = {n: p.data.clone()
                             for n, p in model.named_parameters()}

        print(f"EWC: Fisher computed on {count} samples.")

    def penalty(self, model):
        """جریمه EWC"""
        if not self.fisher:
            return 0.0

        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.saved_params[n]) ** 2).sum()
        return self.lambda_ewc * 0.5 * loss
