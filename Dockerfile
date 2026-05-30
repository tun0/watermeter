FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir requests

COPY collect_snapshots.py .

USER 1000

VOLUME ["/app/snapshots"]

ENV CAM_SNAPSHOT_URL=http://192.168.x.x/
ENV CAM_STREAM_URL=http://192.168.x.x:8080/
ENV COLLECTOR_INTERVAL=10

CMD ["python3", "-u", "collect_snapshots.py"]
