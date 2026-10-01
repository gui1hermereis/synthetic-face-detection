import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from _common import analyze, report

parser = argparse.ArgumentParser(description="Mede concorrência da API de análise.")
parser.add_argument("--url", required=True)
parser.add_argument("--api-key", required=True)
parser.add_argument("--image", required=True, type=Path)
parser.add_argument("--requests", type=int, default=50)
parser.add_argument("--concurrency", type=int, default=10)
parser.add_argument("--timeout", type=float, default=30)
args = parser.parse_args()
latencies, failures = [], []
started = time.perf_counter()
with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
    futures = [executor.submit(analyze, args.url, args.api_key, args.image, args.timeout) for _ in range(args.requests)]
    for future in as_completed(futures):
        latency, success, error = future.result()
        if success:
            latencies.append(latency)
        else:
            failures.append(error)
report(latencies, failures, time.perf_counter() - started)
