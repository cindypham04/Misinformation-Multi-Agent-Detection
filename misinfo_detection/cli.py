from __future__ import annotations

import argparse

from .config import load_config
from .graph.parent import build_parent_graph
from .schemas import ParentState


def run_claim(claim: str) -> ParentState:
    config = load_config()
    graph = build_parent_graph(config=config)

    initial: ParentState = ParentState(
        claim=claim,
        guidance="",
        current_round=0,
        max_rounds=config.max_rounds,
        evidence_pool={},
        debate_log=[],
        latest_negative_argument=None,
        latest_affirmative_argument=None,
        advisor_analysis=None,
        advisor_advice=None,
        verifier_evidence={},
        final_verdict=None,
        final_report=None,
    )

    return graph.invoke(initial)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="misinfo-detect")
    parser.add_argument("--claim", type=str, required=True, help="Claim to evaluate")
    args = parser.parse_args(argv)

    out = run_claim(args.claim)
    print(out.get("final_verdict"))
    print(out.get("final_report") or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

