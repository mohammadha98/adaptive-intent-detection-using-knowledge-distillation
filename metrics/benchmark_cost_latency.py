"""
AutoMTL Cost & Latency Efficiency Benchmark
===========================================
Compares three routing strategies:
  1. LLM-Only (Baseline): every request → Teacher LLM
  2. Student-Only (Baseline): every request → Student model (no fallback)
  3. AutoMTL (Our Approach): Student handles ID; Teacher handles OOD via Gate

Run:
    python metrics/benchmark_cost_latency.py          # fast mock simulation
    python metrics/benchmark_cost_latency.py --real   # uses live models / real API delays
"""

import sys
import os
import time
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe for headless / CI
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# ── project root on sys.path ──────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate import CLINCDatasetLoader
from src.student_model import BERTClassifier
from config import Config


# ═══════════════════════════════════════════════════════════════════════
#  Configurable constants  (swap values here for different paper figures)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkConfig:
    """Central configuration – easy to override for real-API or paper tuning."""

    # ── costs (GPT‑4o‑mini public pricing) ──
    cost_per_1k_input_tokens:  float = 0.0001   # $0.0001  / 1K input tokens
    cost_per_1k_output_tokens: float = 0.0002   # $0.0002  / 1K output tokens

    # ── simulated latencies (seconds) ──
    student_latency_mean: float = 0.050   # ~50 ms  (typical DistilBERT on CPU)
    student_latency_std:  float = 0.015
    teacher_latency_mean: float = 1.200   # ~1.2 s  GPT‑4o-mini median
    teacher_latency_std:  float = 0.400

    # ── token estimates (used in mock mode) ──
    avg_input_tokens:  float = 12.0   # avg words per intent utterance
    min_output_tokens: int   = 3
    max_output_tokens: int   = 8

    # ── benchmark sizing ──
    total_requests: int = 1000
    ood_ratio:      float = 0.20      # 20 % OOD

    # ── mode ──
    use_real_sleep: bool = False      # when True, actually sleep to mimic real wall-clock

    # ── random seed ──
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
#  Mock / Simulator classes
# ═══════════════════════════════════════════════════════════════════════

class MockStudent:
    """Simulates a DistilBERT student model with OOD-gate behaviour."""

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        # Try loading the real classifier so the OOD gate behaves plausibly.
        # If the module-level BERTClassifier/torch are mocked (e.g. during unit
        # tests), this gracefully degrades to random OOD decisions.
        self.classifier: Optional[BERTClassifier] = None
        try:
            self.classifier = BERTClassifier()
            if os.path.exists(Config.MODEL_SAVE_PATH):
                state = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
                self.classifier.load_state_dict(state)
                self.classifier.to(Config.DEVICE)
                self.classifier.is_trained = True
                print(f"[MockStudent] Loaded trained student from {Config.MODEL_SAVE_PATH}")
            else:
                print("[MockStudent] No saved student model found – OOD gate will be random.")
        except Exception:
            self.classifier = None
            # Silently degrade to random OOD – expected in test / CI environments

    def predict(self, text: str) -> Dict:
        """Return dict with 'latency' and 'is_ood'."""
        # latency
        if self.cfg.use_real_sleep:
            t0 = time.perf_counter()
            # optionally run a real forward pass to get realistic timing
            if self.classifier and self.classifier.is_trained:
                _ = self.classifier.predict(text)
            elapsed = time.perf_counter() - t0
        else:
            elapsed = max(0.001, self.rng.gauss(self.cfg.student_latency_mean,
                                                self.cfg.student_latency_std))

        # OOD decision
        if self.classifier is not None and self.classifier.is_trained:
            _, confidence = self.classifier.predict(text)
            is_ood = confidence < Config.CONFIDENCE_THRESHOLD
        else:
            is_ood = self.rng.random() < self.cfg.ood_ratio

        return {"latency": elapsed, "is_ood": is_ood}


class MockTeacher:
    """Simulates a remote LLM API (GPT‑4o-mini)."""

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed + 1000)

    def get_fallback_prediction(self, text: str) -> Dict:
        """Return dict with latency, input_tokens, output_tokens."""
        if self.cfg.use_real_sleep:
            t0 = time.perf_counter()
            time.sleep(self.cfg.teacher_latency_mean)  # coarse but illustrative
            elapsed = time.perf_counter() - t0
        else:
            elapsed = max(0.001, self.rng.gauss(self.cfg.teacher_latency_mean,
                                                self.cfg.teacher_latency_std))

        input_tokens = len(text.split()) * 1.5
        output_tokens = self.rng.randint(self.cfg.min_output_tokens,
                                         self.cfg.max_output_tokens)

        return {"latency": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens}


# ═══════════════════════════════════════════════════════════════════════
#  Data-stream helpers
# ═══════════════════════════════════════════════════════════════════════

def get_simulated_data_stream(cfg: BenchmarkConfig) -> List[Dict]:
    """Create a shuffled stream of ID + OOD utterances from CLINC150."""
    print("Loading CLINC150 dataset …")
    loader = CLINCDatasetLoader()
    loader.load_data()

    ood_count = int(cfg.total_requests * cfg.ood_ratio)
    id_count = cfg.total_requests - ood_count

    rng = random.Random(cfg.seed)
    id_samples = rng.choices(loader.in_domain_samples, k=id_count)
    ood_samples = rng.choices(loader.oos_samples, k=ood_count)

    for s in id_samples:
        s["is_ood_ground_truth"] = False
    for s in ood_samples:
        s["is_ood_ground_truth"] = True

    stream = id_samples + ood_samples
    rng.shuffle(stream)

    print(f"Data stream ready: {len(stream)} requests  "
          f"({id_count} ID, {ood_count} OOD)")
    return stream


# ═══════════════════════════════════════════════════════════════════════
#  Cost functions
# ═══════════════════════════════════════════════════════════════════════

def calculate_llm_cost(input_tokens: float, output_tokens: float,
                       cfg: Optional[BenchmarkConfig] = None) -> float:
    """Calculate $ cost given token counts.

    Parameters
    ----------
    input_tokens : float
    output_tokens : float
    cfg : BenchmarkConfig | None
        If None, uses default BenchmarkConfig.

    Returns
    -------
    float
    """
    if cfg is None:
        cfg = BenchmarkConfig()
    input_cost = (input_tokens / 1000.0) * cfg.cost_per_1k_input_tokens
    output_cost = (output_tokens / 1000.0) * cfg.cost_per_1k_output_tokens
    return input_cost + output_cost


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark runner
# ═══════════════════════════════════════════════════════════════════════

def run_scenarios(data_stream: List[Dict], cfg: BenchmarkConfig) -> Dict:
    """Execute all three routing strategies over *data_stream*."""
    student = MockStudent(cfg)
    teacher = MockTeacher(cfg)

    results = {
        "llm_only":     {"total_latency": 0.0, "total_cost": 0.0, "total_requests": 0},
        "student_only": {"total_latency": 0.0, "total_cost": 0.0, "total_requests": 0},
        "automtl":      {"total_latency": 0.0, "total_cost": 0.0, "total_requests": 0,
                         "fallbacks": 0},
    }

    total = len(data_stream)
    for i, req in enumerate(data_stream):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"\r  Processing {i+1}/{total} …", end="", flush=True)
        text = req["text"]

        # ── Scenario 1: LLM-Only ──
        t_res = teacher.get_fallback_prediction(text)
        results["llm_only"]["total_latency"] += t_res["latency"]
        results["llm_only"]["total_cost"] += calculate_llm_cost(
            t_res["input_tokens"], t_res["output_tokens"], cfg
        )
        results["llm_only"]["total_requests"] += 1

        # ── Scenario 2: Student-Only ──
        s_res = student.predict(text)
        results["student_only"]["total_latency"] += s_res["latency"]
        results["student_only"]["total_requests"] += 1

        # ── Scenario 3: AutoMTL (hybrid) ──
        if s_res["is_ood"]:
            results["automtl"]["fallbacks"] += 1
            results["automtl"]["total_latency"] += s_res["latency"] + t_res["latency"]
            results["automtl"]["total_cost"] += calculate_llm_cost(
                t_res["input_tokens"], t_res["output_tokens"], cfg
            )
        else:
            results["automtl"]["total_latency"] += s_res["latency"]
        results["automtl"]["total_requests"] += 1

    print("\r  Benchmark complete.                    ")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════

def print_results_table(results: Dict, cfg: BenchmarkConfig) -> None:
    """Pretty-print an ASCII table with the three scenarios."""
    table = PrettyTable()
    table.title = "Benchmark Results: Cost & Latency Efficiency"
    table.field_names = [
        "Scenario",
        "Avg Latency (ms/req)",
        "Total Time (s)",
        "Total Cost ($)",
        "LLM Fallbacks",
    ]

    # 1. LLM-Only
    r = results["llm_only"]
    avg_ms = (r["total_latency"] / r["total_requests"]) * 1000.0
    table.add_row([
        "1. LLM-Only (Baseline)",
        f"{avg_ms:.2f}",
        f"{r['total_latency']:.2f}",
        f"{r['total_cost']:.4f}",
        "N/A",
    ])

    # 2. Student-Only
    r = results["student_only"]
    avg_ms = (r["total_latency"] / r["total_requests"]) * 1000.0
    table.add_row([
        "2. Student-Only (Baseline)",
        f"{avg_ms:.2f}",
        f"{r['total_latency']:.2f}",
        "0.0000",
        "0",
    ])

    # 3. AutoMTL
    r = results["automtl"]
    avg_ms = (r["total_latency"] / r["total_requests"]) * 1000.0
    fb = r["fallbacks"]
    fb_pct = fb / r["total_requests"]
    table.add_row([
        "3. AutoMTL (Our Approach)",
        f"{avg_ms:.2f}",
        f"{r['total_latency']:.2f}",
        f"{r['total_cost']:.4f}",
        f"{fb} / {r['total_requests']} ({fb_pct:.1%})",
    ])

    print("\n" + str(table) + "\n")

    # ── savings summary ──
    llm_cost = results["llm_only"]["total_cost"]
    automtl_cost = results["automtl"]["total_cost"]
    if llm_cost > 0:
        saving = (1.0 - automtl_cost / llm_cost) * 100.0
        print(f"💰 Cost saving vs LLM-Only:  {saving:.1f} %")

    llm_lat = results["llm_only"]["total_latency"]
    automtl_lat = results["automtl"]["total_latency"]
    if llm_lat > 0:
        speedup = llm_lat / automtl_lat if automtl_lat > 0 else float("inf")
        print(f"⚡ Latency speed-up vs LLM-Only:  {speedup:.1f}×")


def generate_chart(results: Dict, cfg: BenchmarkConfig,
                   output_path: str = "results/cost_latency_chart.png") -> None:
    """Save a side-by-side bar chart (cost + latency)."""
    scenarios = ["LLM-Only", "Student-Only", "AutoMTL"]

    costs = [
        results["llm_only"]["total_cost"],
        results["student_only"]["total_cost"],
        results["automtl"]["total_cost"],
    ]
    avg_latencies = [
        (results["llm_only"]["total_latency"] / cfg.total_requests) * 1000,
        (results["student_only"]["total_latency"] / cfg.total_requests) * 1000,
        (results["automtl"]["total_latency"] / cfg.total_requests) * 1000,
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("AutoMTL Framework: Cost & Latency Efficiency Benchmark",
                 fontsize=16, fontweight="bold")

    colors = ["#ff6961", "#77dd77", "#fdfd96"]

    # ── Cost subplot ──
    bars1 = ax1.bar(scenarios, costs, color=colors)
    ax1.set_ylabel("Total Cost ($) for 1 000 Requests")
    ax1.set_title("Cost Comparison")
    ax1.set_yscale("log")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval,
                 f"${yval:.4f}", va="bottom", ha="center", fontsize=9)

    # ── Latency subplot ──
    bars2 = ax2.bar(scenarios, avg_latencies, color=colors)
    ax2.set_ylabel("Average Latency (ms / request)")
    ax2.set_title("Latency Comparison")
    ax2.set_yscale("log")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval,
                 f"{yval:.2f} ms", va="bottom", ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"📊 Chart saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI entry
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoMTL Cost & Latency Benchmark"
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Use real time.sleep() delays (slower, but wall-clock realistic)."
    )
    parser.add_argument(
        "--requests", type=int, default=1000,
        help="Number of requests in the simulated stream."
    )
    parser.add_argument(
        "--ood-ratio", type=float, default=0.20,
        help="Fraction of OOD requests (0.0–1.0)."
    )
    parser.add_argument(
        "--output-chart", type=str, default="results/cost_latency_chart.png",
        help="Path for the output chart PNG."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    # Build config
    cfg = BenchmarkConfig(
        total_requests=args.requests,
        ood_ratio=args.ood_ratio,
        use_real_sleep=args.real,
        seed=args.seed,
    )

    print("╔══════════════════════════════════════════════╗")
    print("║   AutoMTL  Cost & Latency Benchmark         ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Requests : {cfg.total_requests}")
    print(f"  OOD ratio: {cfg.ood_ratio:.0%}")
    print(f"  Mode     : {'real-sleep' if cfg.use_real_sleep else 'instant (mock)'}")
    print(f"  Seed     : {cfg.seed}")
    print()

    # 1. Data
    data = get_simulated_data_stream(cfg)

    # 2. Run
    results = run_scenarios(data, cfg)

    # 3. Report
    print_results_table(results, cfg)
    generate_chart(results, cfg, output_path=args.output_chart)


if __name__ == "__main__":
    main()