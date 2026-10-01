import argparse
import time
from pathlib import Path
from _common import analyze, report

parser = argparse.ArgumentParser(description="Mede latência da análise individual.")
parser.add_argument("--url", required=True)
parser.add_argument("--api-key", required=True)
parser.add_argument("--image", required=True, type=Path)
parser.add_argument("--requests", type=int, default=20)
parser.add_argument("--timeout", type=float, default=30)
args = parser.parse_args()
latencies, failures = [], []
started = time.perf_counter()
for _ in range(args.requests):
    latency, success, error = analyze(args.url, args.api_key, args.image, args.timeout)
    if success:
        latencies.append(latency)
    else:
        failures.append(error)
report(latencies, failures, time.perf_counter() - started)
