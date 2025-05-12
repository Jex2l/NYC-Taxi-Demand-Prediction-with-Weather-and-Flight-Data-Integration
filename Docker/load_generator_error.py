#!/usr/bin/env python3
import time, random, requests
from concurrent.futures import ThreadPoolExecutor

FASTAPI_URL = "http://localhost:8000/predict"
ERROR_RATE = 0.1  # 10% of requests will be malformed

FEATURE_KEYS = [
    "location_id","year","month","day","hour","minute","dow",
    "dep_now","dep_next_30","dep_next_60","dep_next_90","dep_next_120",
    "arr_now","arr_next_30","arr_next_60","arr_next_90","arr_next_120",
    "tmpf","dwpf","relh","feel","sknt"
]

def generate_valid():
    return {
        "location_id": random.randint(1,100),
        "year": 2025,
        "month": random.randint(1,12),
        "day": random.randint(1,28),
        "hour": random.randint(0,23),
        "minute": random.choice([0,15,30,45]),
        "dow": random.randint(0,6),
        "dep_now": round(random.uniform(0,50),2),
        "dep_next_30": round(random.uniform(0,50),2),
        "dep_next_60": round(random.uniform(0,50),2),
        "dep_next_90": round(random.uniform(0,50),2),
        "dep_next_120": round(random.uniform(0,50),2),
        "arr_now": round(random.uniform(0,50),2),
        "arr_next_30": round(random.uniform(0,50),2),
        "arr_next_60": round(random.uniform(0,50),2),
        "arr_next_90": round(random.uniform(0,50),2),
        "arr_next_120": round(random.uniform(0,50),2),
        "tmpf": round(random.uniform(10,90),1),
        "dwpf": round(random.uniform(10,90),1),
        "relh": round(random.uniform(20,100),1),
        "feel": round(random.uniform(10,90),1),
        "sknt": round(random.uniform(0,30),1),
    }

def make_payload():
    payload = generate_valid()
    if random.random() < ERROR_RATE:
        # drop a random required key to force a 422
        bad_key = random.choice(FEATURE_KEYS)
        del payload[bad_key]
    return payload

def send_load(duration_s:int, rps:int):
    interval = 1.0/rps
    end = time.time() + duration_s
    while time.time() < end:
        payload = make_payload()
        try:
            resp = requests.post(FASTAPI_URL, json=payload, timeout=5)
            # optional: log status codes
            print(f"{resp.status_code}", end=" ")
        except Exception as e:
            print(f"[ERR]{e}", end=" ")
        time.sleep(interval)

LOAD_PATTERN = [(30,1),(60,5),(120,10),(60,5),(30,1)]

def main():
    print("Starting load with error injection…")
    for dur, rps in LOAD_PATTERN:
        print(f"\n→ {rps} RPS for {dur}s")
        with ThreadPoolExecutor(max_workers=rps) as ex:
            for _ in range(rps):
                ex.submit(send_load, dur, rps)
    print("\nDone.")

if __name__ == "__main__":
    main()
