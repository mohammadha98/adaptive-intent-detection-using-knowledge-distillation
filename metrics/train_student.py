"""
Standalone Student Model Training Script
=========================================
Fine-tunes DistilBERT on the CLINC150 intent-classification dataset
and saves the trained weights to ``data/student_model.pt``.

Run once before using the *real* benchmark mode:
    python metrics/train_student.py

The script also computes per-class centroids (metric-learning step) so
the AutoMTL OOD gate can use cosine-similarity confidence.
"""

import sys
import os
import argparse
import time

# ── project root on sys.path ──────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.student_model import BERTClassifier
from src.evaluate import CLINCDatasetLoader
from config import Config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DistilBERT student on CLINC150"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs (default: 5)."
    )
    parser.add_argument(
        "--output", type=str, default=Config.MODEL_SAVE_PATH,
        help="Where to save the trained model."
    )
    parser.add_argument(
        "--device", type=str, default=Config.DEVICE,
        help="Device for training (cpu / cuda / mps)."
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════╗")
    print("║   AutoMTL – Student Model Training           ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Model  : {Config.STUDENT_MODEL_NAME}")
    print(f"  Device : {args.device}")
    print(f"  Epochs : {args.epochs}")
    print()

    # ── 1. Load CLINC150 ────────────────────────────────────────────
    print("[1/4] Loading CLINC150 dataset …")
    loader = CLINCDatasetLoader()
    loader.load_data()

    # train_on_data expects a list of {text, label} dicts
    training_data = loader.in_domain_samples
    print(f"      {len(training_data)} training utterances")

    # ── 2. Instantiate model ────────────────────────────────────────
    print("\n[2/4] Initialising DistilBERT classifier …")
    model = BERTClassifier(num_labels=2)   # temporary head; train_on_data resizes it
    model.device = torch.device(args.device)
    model.to(model.device)

    # ── 3. Train ────────────────────────────────────────────────────
    print(f"\n[3/4] Fine-tuning for {args.epochs} epoch(s) …")
    t0 = time.time()

    model.train_on_data(training_data)

    elapsed = time.time() - t0
    print(f"      Training finished in {elapsed:.1f} s  "
          f"({elapsed / 60:.1f} min)")

    # ── 4. Save ─────────────────────────────────────────────────────
    print(f"\n[4/4] Saving model → {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)

    # Quick sanity check – can we load it back?
    dummy = BERTClassifier(num_labels=len(model.label_map))
    dummy.load_state_dict(torch.load(args.output, map_location=args.device))
    dummy.label_map = model.label_map
    dummy.label2id = model.label2id
    dummy.centroids = model.centroids
    dummy.is_trained = True

    test_text = "what is my balance"
    label, conf = dummy.predict(test_text)
    print(f"      Sanity check:  '{test_text}'  →  '{label}'  (conf={conf:.3f})")

    print("\n✅ All done.  Student model is ready for benchmarking.\n")


if __name__ == "__main__":
    main()