# watermeter

Reads a physical water meter using an ESP32-S3 camera and computer vision, then pushes the reading to Home Assistant.

## How it works

```
                    ┌→ collector → snapshots/   (timestamped, for offline review)
ESP32-S3 camera ────┤
                    └→ meter_reader → Home Assistant
```

**collector** (`collect_snapshots.py`) fetches JPEG snapshots from the ESP32-CAM on a fixed interval and saves them with a timestamp for offline review.

**meter_reader** (`meter_reader.py`) fetches a live snapshot directly from the camera, runs it through a two-stage OCR pipeline, and pushes the result to the Home Assistant REST API:

1. Rotate the image to align the meter face
2. OCR the 5-digit digital counter strip (Tesseract, PSM 7/6/8 with per-digit fallback)
3. Detect 4 analog dial needle angles via spoke sampling
4. Apply gear-lash correction and phase correction to the analog digits
5. Validate the assembled reading against the last accepted value (monotonic guard, max-step check)
6. Persist state and push `sensor.water_meter` (m³) + `sensor.water_meter_flow` (L/min) to HA

The camera is configured via ESPHome (`camera.yaml`). Secrets (WiFi credentials, API key) live in a local `secrets.yaml` that is never committed.

## Running

### Docker Compose (recommended)

Set the required environment variables (e.g. in a `.env` file or your shell), then:

```bash
docker compose up
```

### Kubernetes (k3s)

```bash
cp k8s/secret.yaml.example k8s/secret.yaml   # fill in real values
kubectl apply -k k8s/
```

### Standalone

```bash
pip install -r requirements.txt
python3 meter_reader.py                        # read live from camera
python3 meter_reader.py --image foo.jpg --debug   # offline + annotated image
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `CAM_SNAPSHOT_URL` | — | Camera snapshot endpoint |
| `HA_URL` | — | Home Assistant base URL |
| `HA_TOKEN` | — | HA long-lived access token |
| `STATE_FILE` | `.meter_state.json` | Path to persisted state |
| `INITIAL_VALUE` | — | Seed `last_reading` on first run (no state file yet) |
| `READING_INTERVAL` | `10` | Loop interval in seconds |
| `FLOW_MAX_AGE` | — | Skip flow rate if gap since last reading exceeds this (seconds) |

## Tests

Unit tests cover the pure-logic pipeline functions: reading assembly, gear-lash correction, rollover disambiguation, validate, and state persistence.

### Run locally via Docker (recommended)

Uses the same image as production, including the Tesseract binary:

```bash
docker compose --profile test run --rm test
```

### Run standalone

```bash
pip install -r requirements-dev.txt
pytest
```

### CI

GitHub Actions runs lint (`ruff`) and tests (`pytest`) on every push before building and pushing the Docker images. The build job only runs if lint and tests pass.

### Pre-push hook

A repo-local pre-push hook runs the test suite automatically before every `git push`. Activate it once after cloning:

```bash
git config core.hooksPath .githooks
```

The hook chains the global pre-push hook afterwards (if one exists), so other repo-level checks such as branch protection remain active.

## Tooling

| Tool | Purpose |
|---|---|
| `calibrate.py` | Interactive calibration of crop boxes and dial positions |
| `annotate_snapshots.py` | Batch-annotate stored snapshots for visual debugging |
| `digit_templates.py` | Experimental template-based digit matcher (not in pipeline) |
