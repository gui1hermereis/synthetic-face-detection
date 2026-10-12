import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from _common import analyze, report, validate_benchmark_arguments

parser = argparse.ArgumentParser(description="Mede concorrência da API de análise.")
parser.add_argument("--url", required=True)
parser.add_argument("--api-key", required=True)
parser.add_argument("--image", required=True, type=Path)
parser.add_argument("--requests", type=int, default=50)
parser.add_argument("--concurrency", type=int, default=10)
parser.add_argument("--timeout", type=float, default=30)
args = parser.parse_args()
validate_benchmark_arguments(parser, args.image, args.requests, args.concurrency)
latencies, failures = [], []
started = time.perf_counter()
with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
    futures = [executor.submit(analyze, args.url, args.api_key, args.image, args.timeout) for _ in range(args.requests)]
    for future in as_completed(futures):
        try:
            latency, success, error = future.result()
        except Exception as error:  # Count unexpected worker failures as failed requests.
            latency, success = 0.0, False
            error = f"{type(error).__name__}: {error}"
        latencies.append(latency)
        if not success:
            failures.append(error)
report(latencies, failures, time.perf_counter() - started, requests=args.requests,
       concurrency=args.concurrency, url=args.url)
