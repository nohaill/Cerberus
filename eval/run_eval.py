"""
Runs the red-team dataset through the live Gateway and produces a JSON +
Markdown report.

    python -m eval.run_eval
    python -m eval.run_eval --dataset eval/datasets/redteam_v1.jsonl

This is intentionally separate from `tests/` -- these are unit tests, this is
a benchmark. The distinction matters: unit tests should be fast and always
pass on correct code; a benchmark's numbers are expected to change over time
as models/thresholds change, and its job is to *measure and report* that
change, not just pass/fail. `tests/test_eval_regression.py` is the bridge
between the two -- it runs a fast subset of this same dataset and fails CI if
core safety numbers regress.
"""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.gateway import Gateway
from app.schemas import CheckRequest
from eval.metrics import EvalExampleResult, MALICIOUS_CATEGORIES, compute_summary


def load_dataset(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


async def run_dataset(examples: list[dict], gateway: Gateway) -> list[EvalExampleResult]:
    results = []
    for ex in examples:
        req = CheckRequest(text=ex["text"], request_id=ex["id"])
        resp = await gateway.check(req)
        passed = resp.verdict.value in ex["expected_verdicts"]
        results.append(
            EvalExampleResult(
                id=ex["id"],
                category=ex["category"],
                text=ex["text"],
                expected_verdicts=ex["expected_verdicts"],
                actual_verdict=resp.verdict.value,
                passed=passed,
                max_risk_score=resp.max_risk_score,
                total_latency_ms=resp.total_latency_ms,
                rationale=resp.rationale,
            )
        )
    return results


def render_markdown(summary, results, dataset_path: str) -> str:
    lines = []
    lines.append(f"# Cerberus Red-Team Eval Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Total examples: {summary.total_examples}")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append(f"- **Bypass rate (malicious prompts that got a plain `allow`)**: "
                  f"{summary.bypass_rate:.1%} ({summary.bypass_count}/{summary.malicious_total})")
    lines.append(f"- **False positive rate (benign prompts blocked/escalated/redacted)**: "
                  f"{summary.false_positive_rate:.1%} ({summary.false_positive_count}/{summary.strict_benign_total})")
    lines.append(f"- **Overall expected-verdict match rate**: {summary.overall_pass_rate:.1%}")
    lines.append(f"- **Latency**: p50={summary.latency_p50_ms:.2f}ms  p95={summary.latency_p95_ms:.2f}ms  "
                  f"p99={summary.latency_p99_ms:.2f}ms  max={summary.latency_max_ms:.2f}ms")
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append("| Category | Pass rate | N |")
    lines.append("|---|---|---|")
    for cat, stats in sorted(summary.category_stats.items()):
        lines.append(f"| {cat} | {stats.pass_rate:.1%} | {stats.total} |")
    lines.append("")
    lines.append("## Known gaps (documented, not hidden)")
    lines.append("")
    failed = [r for r in results if not r.passed]
    if not failed:
        lines.append("None in this run.")
    else:
        for r in failed:
            lines.append(f"- `{r.id}` ({r.category}): got `{r.actual_verdict}`, "
                          f"expected one of {r.expected_verdicts} — \"{r.text[:80]}\"")
    lines.append("")
    return "\n".join(lines)


async def main_async(dataset_path: str, out_dir: str):
    examples = load_dataset(Path(dataset_path))
    gateway = Gateway()
    results = await run_dataset(examples, gateway)
    summary = compute_summary(results)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = Path(out_dir) / f"report_{timestamp}.json"
    md_path = Path(out_dir) / f"report_{timestamp}.md"
    latest_json = Path(out_dir) / "latest_report.json"
    latest_md = Path(out_dir) / "latest_report.md"

    report_obj = {
        "dataset": dataset_path,
        "generated_at": timestamp,
        "summary": {
            "total_examples": summary.total_examples,
            "overall_pass_rate": summary.overall_pass_rate,
            "bypass_rate": summary.bypass_rate,
            "bypass_count": summary.bypass_count,
            "malicious_total": summary.malicious_total,
            "false_positive_rate": summary.false_positive_rate,
            "false_positive_count": summary.false_positive_count,
            "strict_benign_total": summary.strict_benign_total,
            "latency_p50_ms": summary.latency_p50_ms,
            "latency_p95_ms": summary.latency_p95_ms,
            "latency_p99_ms": summary.latency_p99_ms,
            "latency_max_ms": summary.latency_max_ms,
            "by_category": {
                cat: {"pass_rate": s.pass_rate, "total": s.total, "passed": s.passed}
                for cat, s in summary.category_stats.items()
            },
        },
        "results": [r.__dict__ for r in results],
    }

    for path in (json_path, latest_json):
        with open(path, "w") as f:
            json.dump(report_obj, f, indent=2)

    md = render_markdown(summary, results, dataset_path)
    for path in (md_path, latest_md):
        with open(path, "w") as f:
            f.write(md)

    print(md)
    print(f"\nReports written to {json_path} and {md_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/datasets/redteam_v1.jsonl")
    parser.add_argument("--out-dir", default="eval/reports")
    args = parser.parse_args()
    asyncio.run(main_async(args.dataset, args.out_dir))


if __name__ == "__main__":
    main()
