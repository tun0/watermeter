#!/usr/bin/env python3
"""Collect timestamped snapshots directly from the ESP32-CAM for offline testing."""

import os
import time
from datetime import datetime

import requests

CAM_SNAPSHOT_URL = os.getenv("CAM_SNAPSHOT_URL", "http://192.168.x.x/")
CAM_STREAM_URL   = os.getenv("CAM_STREAM_URL",   "http://192.168.x.x:8080/")
INTERVAL         = int(os.getenv("COLLECTOR_INTERVAL", "10"))
MAX_AGE_HOURS    = float(os.getenv("COLLECTOR_MAX_AGE_HOURS", "48"))
TIMEOUT          = 15

snapshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
os.makedirs(snapshots_dir, exist_ok=True)


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


def prune_old_snapshots():
    """Delete snapshot files older than MAX_AGE_HOURS."""
    if MAX_AGE_HOURS <= 0:
        return
    cutoff = time.time() - MAX_AGE_HOURS * 3600
    pruned = 0
    try:
        for entry in os.scandir(snapshots_dir):
            if entry.is_file() and entry.name.endswith(".jpg"):
                if entry.stat().st_mtime < cutoff:
                    os.unlink(entry.path)
                    pruned += 1
    except OSError:
        pass
    if pruned:
        print(f"pruned {pruned} snapshots older than {MAX_AGE_HOURS}h", flush=True)


print(f"Camera:    {CAM_SNAPSHOT_URL}", flush=True)
print(f"Interval:  {INTERVAL}s", flush=True)
print(f"Max age:   {MAX_AGE_HOURS}h", flush=True)
print(f"Saving to: {snapshots_dir}", flush=True)

count = 0
while True:
    t0 = time.monotonic()
    try:
        data = fetch_snapshot()
        fname = os.path.join(snapshots_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg")
        with open(fname, "wb") as f:
            f.write(data)
        count += 1
        print(f"[{count}] {os.path.basename(fname)}", flush=True)
        if count % 360 == 0:
            prune_old_snapshots()
    except Exception as e:
        print(f"[{count}] WARN: {e}", flush=True)
    elapsed = time.monotonic() - t0
    time.sleep(max(0, INTERVAL - elapsed))
