#!/usr/bin/env python3
"""Collect timestamped snapshots directly from the ESP32-CAM for offline testing."""

import os
import time
import requests
from datetime import datetime

CAM_SNAPSHOT_URL = os.getenv("CAM_SNAPSHOT_URL", "http://192.168.x.x/")
CAM_STREAM_URL   = os.getenv("CAM_STREAM_URL",   "http://192.168.x.x:8080/")
INTERVAL         = int(os.getenv("COLLECTOR_INTERVAL", "10"))
TIMEOUT          = 15


def warm_camera():
    """Briefly open the MJPEG stream to wake the camera from idle."""
    try:
        r = requests.get(CAM_STREAM_URL, stream=True, timeout=3)
        r.close()
    except Exception:
        pass
    time.sleep(0.5)


def fetch_snapshot() -> bytes:
    warm_camera()
    for attempt in range(3):
        try:
            r = requests.get(CAM_SNAPSHOT_URL, timeout=TIMEOUT)
            r.raise_for_status()
            if r.content:
                return r.content
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(1)
    raise RuntimeError("Could not fetch snapshot after 3 attempts")


session_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "snapshots",
    datetime.now().strftime("%Y%m%d_%H%M%S"),
)
os.makedirs(session_dir, exist_ok=True)
print(f"Camera:    {CAM_SNAPSHOT_URL}", flush=True)
print(f"Interval:  {INTERVAL}s", flush=True)
print(f"Saving to: {session_dir}", flush=True)

count = 0
while True:
    try:
        data = fetch_snapshot()
        fname = os.path.join(session_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg")
        with open(fname, "wb") as f:
            f.write(data)
        count += 1
        print(f"[{count}] {os.path.basename(fname)}", flush=True)
    except Exception as e:
        print(f"[{count}] WARN: {e}", flush=True)
    time.sleep(INTERVAL)
