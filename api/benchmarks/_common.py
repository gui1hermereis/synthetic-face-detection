import json
import math
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def analyze(url, api_key, image_path, timeout):
    content = image_path.read_bytes()
    boundary = "synthetic-face-benchmark-boundary"
    body = b"\r\n".join([
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"'.encode(),
        b"Content-Type: image/jpeg", b"", content, f"--{boundary}--".encode(), b"",
    ])
    request = Request(url.rstrip("/") + "/api/v1/images/analyze", data=body, method="POST", headers={
        "X-API-Key": api_key, "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
            return time.perf_counter() - started, 200 <= response.status < 300, None
    except HTTPError as error:
        return time.perf_counter() - started, False, f"HTTP {error.code}"
    except (URLError, TimeoutError, OSError) as error:
        return time.perf_counter() - started, False, str(error)


def report(latencies, failures, elapsed):
    payload = {
        "successes": len(latencies), "failures": len(failures),
        "minimum_seconds": min(latencies) if latencies else 0.0,
        "maximum_seconds": max(latencies) if latencies else 0.0,
        "mean_seconds": statistics.mean(latencies) if latencies else 0.0,
        "median_seconds": statistics.median(latencies) if latencies else 0.0,
        "p95_seconds": percentile(latencies, 0.95),
        "throughput_requests_per_second": len(latencies) / elapsed if elapsed else 0.0,
    }
    print(json.dumps(payload, indent=2))
    if failures:
        print("failure_samples:", failures[:5])
