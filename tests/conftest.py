import os

# Set all required env vars before meter_reader is imported.
# Values match .env.dist defaults; site-specific vars get dummy values.
_ENV = {
    "CAM_SNAPSHOT_URL": "http://test.example.com/",
    "HA_URL": "",
    "HA_TOKEN": "",
    "ROTATE_DEG": "62.5",
    "DIGITAL_STRIP": "654,911,244,50",
    "DIGITAL_DIGITS": "654,911,48,50;702,911,48,50;750,911,48,50;798,911,48,50;846,911,48,50",
    "ANALOG_DIALS": "1038,1067,61;975,1216,61;826,1271,61;676,1203,65",
    "DIAL_ZERO_OFFSETS": "126,270,270,0",
    "DIAL_INFLUENCE_HIGH": "0.85",
    "DIAL_INFLUENCE_LOW": "0.15",
    "ROLLOVER_START": "0.9",
    "READING_INTERVAL": "10",
    "MAX_STEP": "0.05",
    "MAX_DELTA_CAP": "1.0",
    "JITTER_TOLERANCE": "0.010",
    "STATE_FILE": "/tmp/test_meter_state.json",
}

for k, v in _ENV.items():
    os.environ.setdefault(k, v)
