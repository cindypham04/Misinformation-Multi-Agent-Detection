from __future__ import annotations

from datetime import datetime
from pathlib import Path

from misinfo_detection.cli import run_claim


def build_report(claim: str, out: dict) -> str:
    lines: list[str] = []
    lines.append(f"=== Claim ===\n{claim}")

    lines.append("\n=== Debate ===")
    for i, turn in enumerate(out.get("debate_log", []), 1):
        lines.append(f"{i}. {turn}")

    lines.append("\n=== Advisor Advice ===")
    lines.append(out.get("advisor_advice") or "")

    lines.append("\n=== Verifier Verdict ===")
    lines.append(out.get("final_verdict") or "")

    lines.append("\n=== Verifier Explanation ===")
    lines.append(out.get("final_report") or "")

    return "\n".join(lines)


def main() -> None:
    claim = "COVID vaccines lead to autism"
    out = run_claim(claim)
    report = build_report(claim, out)

    # Print to terminal
    print(report)

    # Write to separate log file
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"claim_report_{ts}.txt"
    log_path.write_text(report, encoding="utf-8")

    print(f"\nSaved report to: {log_path}")


if __name__ == "__main__":
    main()