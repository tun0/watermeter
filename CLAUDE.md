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

When the reader is stuck rejecting readings, use `--last-reading` to force a known-good baseline for a single run. The script computes the current reading, validates it against the provided baseline, and writes the result to state:

```
kubectl --context k3s-home -n watermeter exec deploy/meter-reader -- python3 meter_reader.py --last-reading <VALUE>
```

`<VALUE>` is the last known-good reading (e.g. from HA history). Do not read it from the physical meter display — the computed fractional part differs from the meter face due to the dial zero offsets.

If no prior reading is known at all, use `--no-guard` instead to skip validation entirely:

```
kubectl --context k3s-home -n watermeter exec deploy/meter-reader -- python3 meter_reader.py --no-guard
```

Add `--push` to any one-off command to also update HA in the same run.

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
4. Update the environment variable configuration (configmap, `.env`, etc.) with new constants, update the state file with the corrected reading, commit
