import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset
from config import Config
import random
import numpy as np
from torch.nn import functional as F










class SimpleDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class BERTClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(BERTClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(Config.STUDENT_MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.tokenizer = DistilBertTokenizer.from_pretrained(Config.STUDENT_MODEL_NAME)
        self.device = torch.device(Config.DEVICE)
        self.to(self.device)
        
        self.label_map = {} 
        self.label2id = {}  
        self.centroids = {} # New: Stores the "center" embedding of each class
        self.is_trained = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :] # This is the embedding vector
        logits = self.classifier(cls_output)
        return logits, cls_output # Return embeddings too

    def train_on_data(self, data):
        # 1. Setup Labels
        unique_labels = sorted(list(set(d['label'] for d in data)))
        self.label_map = {i: label for i, label in enumerate(unique_labels)}
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        
        # Reset Classifier Head
        num_labels = len(unique_labels)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels).to(self.device)
        
        # 2. Prepare Data
        texts = [d['text'] for d in data]
        labels = [self.label2id[d['label']] for d in data]
        
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=64)
        dataset = SimpleDataset(encodings, labels)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # 3. Training Loop
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.train()
        print("StudentModel: Starting training...")
        for epoch in range(5): # Keep epochs low to prevent overfitting on small data
            for batch in loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits, _ = self(input_ids, attention_mask) # We ignore embeddings during training backprop
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
            
        # 4. COMPUTE CENTROIDS (Metric Learning Step)
        self.eval()
        class_embeddings = {i: [] for i in range(num_labels)}
        
        with torch.no_grad():
            # Pass all data again to get clean embeddings
            # (In production, you might optimize this, but for small data it's fine)
            full_loader = DataLoader(dataset, batch_size=16) 
            for batch in full_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, embeddings = self(input_ids, attention_mask)
                
                for i, label_idx in enumerate(labels):
                    class_embeddings[label_idx.item()].append(embeddings[i].cpu())

        # Calculate mean embedding for each class
        self.centroids = {}
        for label_idx, emb_list in class_embeddings.items():
            if emb_list:
                # Stack and average -> Shape: (768,)
                mean_emb = torch.stack(emb_list).mean(dim=0)
                # Normalize centroid for Cosine Similarity
                self.centroids[label_idx] = F.normalize(mean_emb, p=2, dim=0).to(self.device)

        self.is_trained = True
        print("StudentModel: Training & Centroid Calculation finished.")

    def predict(self, text, temperature=1.0):
        if not self.is_trained:
            return ("Untrained", 0.0)

        self.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            logits, embedding = self(input_ids, attention_mask)
            
            # A. Softmax Confidence
            probs = torch.softmax(logits / temperature, dim=1)
            softmax_conf, predicted_id = torch.max(probs, dim=1)
            predicted_idx = predicted_id.item()
            
            # B. Cosine Similarity Confidence (The Sanity Check)
            # Normalize input embedding
            input_emb_norm = F.normalize(embedding[0], p=2, dim=0)
            # Get centroid of the PREDICTED class
            centroid = self.centroids.get(predicted_idx)
            
            if centroid is not None:
                # Compute Cosine Similarity: dot product of normalized vectors
                similarity = torch.dot(input_emb_norm, centroid).item()
            else:
                similarity = 0.0

            # C. Hybrid Score
            # We enforce that BOTH the classifier must be sure AND it must be similar to known examples.
            # If similarity is low (e.g. 0.5), it drags the final score down heavily.
            
            # Simple Weighted Average (You can tune this)
            # giving more weight to Similarity usually helps with OOD
            final_score = (softmax_conf.item() * 0.4) + (similarity * 0.6)
            
            # DEBUG LOG (Optional: remove in production)
            # print(f"DEBUG: Label: {self.label_map[predicted_idx]} | Softmax: {softmax_conf.item():.2f} | Similarity: {similarity:.2f} | Final: {final_score:.2f}")

            predicted_label = self.label_map[predicted_idx]

        return predicted_label, final_score

