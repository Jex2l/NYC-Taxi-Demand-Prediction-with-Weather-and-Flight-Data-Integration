#!/usr/bin/env python3
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor

# ▶ Change this if your FastAPI is on a different host/port
FASTAPI_URL = "http://localhost:8000/predict"

# ✔ A realistic payload generator matching your PredictRequest schema
def generate_payload():
    return {
        "location_id": random.randint(1, 100),
        "year": 2025,
        "month": random.randint(1, 12),
        "day": random.randint(1, 28),
        "hour": random.randint(0, 23),
        "minute": random.choice([0, 15, 30, 45]),
        "dow": random.randint(0, 6),
        "dep_now": round(random.uniform(0, 50), 2),
        "dep_next_30": round(random.uniform(0, 50), 2),
        "dep_next_60": round(random.uniform(0, 50), 2),
        "dep_next_90": round(random.uniform(0, 50), 2),
        "dep_next_120": round(random.uniform(0, 50), 2),
        "arr_now": round(random.uniform(0, 50), 2),
        "arr_next_30": round(random.uniform(0, 50), 2),
        "arr_next_60": round(random.uniform(0, 50), 2),
        "arr_next_90": round(random.uniform(0, 50), 2),
        "arr_next_120": round(random.uniform(0, 50), 2),
        "tmpf": round(random.uniform(10, 90), 1),
        "dwpf": round(random.uniform(10, 90), 1),
        "relh": round(random.uniform(20, 100), 1),
        "feel": round(random.uniform(10, 90), 1),
        "sknt": round(random.uniform(0, 30), 1),
    }

# ▶ Worker that sends requests at a given rate (requests/sec) for a duration
def send_load(duration_s: int, rps: int):
    interval = 1.0 / rps
    end_time = time.time() + duration_s
    while time.time() < end_time:
        payload = generate_payload()
        try:
            resp = requests.post(FASTAPI_URL, json=payload, timeout=5)
            # optional: check resp.status_code or resp.json()
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(interval)

# ▶ A simple “rush‑hour” pattern: (duration_seconds, requests_per_second)
LOAD_PATTERN = [
    (30, 1),    # light traffic
    (60, 5),    # moderate traffic
    (120, 10),  # peak traffic
    (60, 5),    # back to moderate
    (30, 1),    # taper off
]

def main():
    print("Starting load generator...")
    for duration, rps in LOAD_PATTERN:
        print(f"→ {rps} RPS for {duration}s")
        # Run each stage in parallel threads equal to rps
        with ThreadPoolExecutor(max_workers=rps) as exec:
            for _ in range(rps):
                exec.submit(send_load, duration, rps)
    print("Load generation complete.")

if __name__ == "__main__":
    main()
