# Claude Code instructions for watermeter

## kubectl context

Always use the `k3s-home` context for kubectl commands in this project:

```
kubectl --context k3s-home ...
```

Never assume the active context is correct — always pass `--context k3s-home` explicitly.

## Configmap location

The live Kubernetes configmap is at:
`/home/ruben/projects/tun0/k8s-home/config/watermeter/meter-reader/configmap.yaml`

The file at `k8s/configmap.yaml` inside this repo is a dummy placeholder — do not edit it.

## State file recovery

When the reader is stuck rejecting readings, update the state file directly:

```
kubectl --context k3s-home -n watermeter exec deploy/meter-reader -- python3 -c "
import json, pathlib
p = pathlib.Path('/app/data/.meter_state.json')
p.write_text(json.dumps({'last_reading': <VALUE>, 'dial_zero_offsets': [None, None, None, None]}, indent=2))
print(p.read_text())
"
```

`<VALUE>` = physical meter reading − 0.3705 (DIAL_PHASE_CORRECTION). Always reset dial_zero_offsets to all null.

## Rollover stuck state

If the reader is stuck with a consistent +1.0 jump (e.g. `303.9763 → 304.9891`), the OCR-ahead rollover bridge was missing — fixed as of 2026-06-17. Post-fix, no manual intervention needed at rollovers. Before the fix deploys, set `last_reading` to just below the computed value to unblock.

## Docker Compose — running locally

Use the Makefile, not `docker compose` directly:

```
make collector   # rebuild + restart collector
make up          # start all services
make logs        # follow collector logs
make test        # run test suite
```

The Makefile injects `UID=$(shell id -u)` and `GID=$(shell id -g)` automatically so files written to bind-mounted volumes are owned by the current user. Never hardcode UID/GID in Dockerfiles or `.env`.

## Recalibration workflow

After any physical camera remount:
1. `curl -s "http://192.168.178.57:8080/" -o /tmp/meter_now.jpg`
2. `python3 calibrate.py --image /tmp/meter_now.jpg --rotation <current_ROTATE_DEG> --grid` — verify boxes
3. `python3 calibrate.py --image /tmp/meter_now.jpg --rotation <best_deg> --interactive` — get new constants
4. Paste output into `meter_reader.py`, update state file with corrected reading, commit both repos
