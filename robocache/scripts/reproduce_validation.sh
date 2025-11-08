#!/usr/bin/env bash
# reproduce_validation.sh
#
# Regenerate the Markdown tables embedded in FINAL_VALIDATION_SUMMARY.md by
# consuming the structured benchmark/profiling exports under profiling/artifacts.
#
# Usage:
#   ./scripts/reproduce_validation.sh [output_file]
# If output_file is omitted the tables are printed to stdout.

set -euo pipefail

OUTPUT_PATH="${1:-}"
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

python3 - "$OUTPUT_PATH" "$REPO_ROOT" <<'PY'
import json
import sys
from pathlib import Path
from textwrap import dedent

if len(sys.argv) < 3:
    raise SystemExit("usage: reproduce_validation.py <output> <repo_root>")

output_arg = sys.argv[1]
repo_root = Path(sys.argv[2])
artifacts_dir = repo_root / "profiling" / "artifacts"
benchmarks_path = artifacts_dir / "benchmarks" / "real_world_benchmarks.json"
performance_path = artifacts_dir / "performance_snapshot.json"

if not benchmarks_path.exists():
    raise SystemExit(f"Benchmark snapshot missing: {benchmarks_path}")
if not performance_path.exists():
    raise SystemExit(f"Performance snapshot missing: {performance_path}")

benchmarks = json.loads(benchmarks_path.read_text(encoding="utf-8"))
performance = json.loads(performance_path.read_text(encoding="utf-8"))

perf_table_rows = [
    "| Metric | Target | H100 Actual | A100 Actual | Status |",
    "|--------|--------|-------------|-------------|--------|",
]
for row in performance["metrics"]:
    perf_table_rows.append(
        "| {metric} | {target} | {h100} | {a100} | {status} |".format(**row)
    )

rw_table_rows = [
    "| Dataset | Domain | H100 | A100 | Target | Status |",
    "|---------|--------|------|------|--------|--------|",
]
for entry in performance["real_world_gpu"]:
    rw_table_rows.append(
        "| {dataset} | {domain} | {h100} | {a100} | {target} | {status} |".format(**entry)
    )

content = dedent(
    """
    ### Industry-Leading Metrics

    {perf_table}

    ### Real-World Dataset Performance

    {rw_table}
    """
).strip().format(
    perf_table="\n".join(perf_table_rows),
    rw_table="\n".join(rw_table_rows),
)

if output_arg:
    output_path = Path(output_arg)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.write_text(content + "\n", encoding="utf-8")
else:
    print(content)
PY
