from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from misinfo_detection.cli import run_claim

@dataclass
class EvalConfig:
    dataset: str = "eval_set"
    checkpoint_path: str = "evaluation/results/checkpoint.jsonl"
    report_path: str = "evaluation/results/report.json"
    batch_size: int = 10
    inter_batch_delay_seconds: float = 5.0
    claim_timeout_seconds: int = 300
    max_claims: int = 0
    random_seed: int = 42
    resume: bool = True


class RawResult(dict):
    """
    One record per claim run, stored as a line in the checkpoint JSONL.
    Inherits from dict so it works seamlessly with the metrics functions.

    Keys:
        claim_id, claim, ground_truth_label, final_verdict,
        evidence_pool_query_count, evidence_pool_total_items,
        debate_log, latest_negative_argument, latest_affirmative_argument,
        advisor_analysis, advisor_advice,
        verifier_evidence_query_count, verifier_evidence_total_items,
        final_report, elapsed_seconds, timed_out, error
    """
    pass

def load_checkpoint(path: str) -> dict:
    completed = {}
    p = Path(path)
    if not p.exists():
        return completed
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                result = json.loads(line)
                completed[result["claim_id"]] = result
    return completed


def append_checkpoint(path: str, result: RawResult) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")

def run_single_claim_timed(claim: str, timeout_seconds: int) -> tuple:
    start = time.time()
    timed_out = False
    error = None
    state = None

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(run_claim, claim)
    try:
        state = future.result(timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
    except Exception as e:
        error = str(e)
    finally:
        executor.shutdown(wait=False)

    elapsed = time.time() - start
    return state, elapsed, timed_out, error


def build_raw_result(
    claim_id: int,
    entry: dict,
    state: Optional[dict],
    elapsed: float,
    timed_out: bool,
    error: Optional[str],
) -> RawResult:
    evidence_pool = state.get("evidence_pool", {}) if state else {}
    verifier_evidence = state.get("verifier_evidence", {}) if state else {}

    return RawResult({
        "claim_id": claim_id,
        "claim": entry["statement"],
        "ground_truth_label": entry["label"],
        "final_verdict": state.get("final_verdict") if state else None,
        "evidence_pool_query_count": len(evidence_pool),
        "evidence_pool_total_items": sum(len(v) for v in evidence_pool.values()),
        "debate_log": state.get("debate_log", []) if state else [],
        "latest_negative_argument": state.get("latest_negative_argument") if state else None,
        "latest_affirmative_argument": state.get("latest_affirmative_argument") if state else None,
        "advisor_analysis": state.get("advisor_analysis") if state else None,
        "advisor_advice": state.get("advisor_advice") if state else None,
        "verifier_evidence_query_count": len(verifier_evidence),
        "verifier_evidence_total_items": sum(len(v) for v in verifier_evidence.values()),
        "final_report": state.get("final_report") if state else None,
        "elapsed_seconds": elapsed,
        "timed_out": timed_out,
        "error": error,
    })

def build_raw_result(
    claim_id: int,
    entry: dict,
    state: Optional[dict],
    elapsed: float,
    timed_out: bool,
    error: Optional[str],
) -> RawResult:
    evidence_pool = state.get("evidence_pool", {}) if state else {}
    verifier_evidence = state.get("verifier_evidence", {}) if state else {}

    return RawResult({
        "claim_id": claim_id,
        "claim": entry["statement"],
        "ground_truth_label": entry["label"],
        "final_verdict": state.get("final_verdict") if state else None,
        "evidence_pool_query_count": len(evidence_pool),
        "evidence_pool_total_items": sum(len(v) for v in evidence_pool.values()),
        "debate_log": state.get("debate_log", []) if state else [],
        "latest_negative_argument": state.get("latest_negative_argument") if state else None,
        "latest_affirmative_argument": state.get("latest_affirmative_argument") if state else None,
        "advisor_analysis": state.get("advisor_analysis") if state else None,
        "advisor_advice": state.get("advisor_advice") if state else None,
        "verifier_evidence_query_count": len(verifier_evidence),
        "verifier_evidence_total_items": sum(len(v) for v in verifier_evidence.values()),
        "final_report": state.get("final_report") if state else None,
        "elapsed_seconds": elapsed,
        "timed_out": timed_out,
        "error": error,
    })

def run_dataset(cfg: EvalConfig) -> list:
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "data" / f"{cfg.dataset}.json"
    dataset = json.loads(dataset_path.read_text())

    if cfg.max_claims > 0:
        random.seed(cfg.random_seed)
        dataset = random.sample(dataset, min(cfg.max_claims, len(dataset)))

    completed = load_checkpoint(cfg.checkpoint_path) if cfg.resume else {}
    remaining = [(i, entry) for i, entry in enumerate(dataset) if i not in completed]

    print(f"Dataset: {cfg.dataset} ({len(dataset)} claims)")
    print(f"Already completed: {len(completed)}, remaining: {len(remaining)}")

    batches = [remaining[i:i + cfg.batch_size] for i in range(0, len(remaining), cfg.batch_size)]

    for batch_num, batch in enumerate(batches):
        print(f"\nBatch {batch_num + 1}/{len(batches)}")
        for claim_id, entry in batch:
            print(f"  [{claim_id}] {entry['statement'][:60]}...")
            state, elapsed, timed_out, error = run_single_claim_timed(
                entry["statement"], cfg.claim_timeout_seconds
            )
            result = build_raw_result(claim_id, entry, state, elapsed, timed_out, error)
            append_checkpoint(cfg.checkpoint_path, result)
            completed[claim_id] = result
            status = "TIMEOUT" if timed_out else ("ERROR" if error else result.get("final_verdict", "?"))
            print(f"  → {status} ({elapsed:.1f}s)")

        if batch_num < len(batches) - 1:
            print(f"  Sleeping {cfg.inter_batch_delay_seconds}s...")
            time.sleep(cfg.inter_batch_delay_seconds)

    return list(completed.values())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="eval-runner")
    parser.add_argument("--dataset", default="eval_set")
    parser.add_argument("--max-claims", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--out-dir", default="evaluation/results")
    args = parser.parse_args()

    cfg = EvalConfig(
        dataset=args.dataset,
        checkpoint_path=f"{args.out_dir}/checkpoint.jsonl",
        report_path=f"{args.out_dir}/report.json",
        batch_size=args.batch_size,
        inter_batch_delay_seconds=args.delay,
        claim_timeout_seconds=args.timeout,
        max_claims=args.max_claims,
        resume=not args.no_resume,
    )

    results = run_dataset(cfg)

    from evaluation.report import generate_weakness_report
    report = generate_weakness_report(results)

    Path(cfg.report_path).write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {cfg.report_path}")
    print(f"Binary accuracy:    {report['summary']['binary_accuracy']:.3f}")
    print(f"Inclusive accuracy: {report['summary']['inclusive_accuracy']:.3f}")
    print(f"Insufficient rate:  {report['summary']['insufficient_rate']:.3f}")
