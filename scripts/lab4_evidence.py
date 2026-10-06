"""Lab 4 Task 6 — collect the evidence of the injected-drift exercise into reports/lab4/.

    python scripts/lab4_evidence.py --since 2026-10-06T13:30:00Z

Writes
  dashboard-shift.png   the dashboard's feature panel and request rate, from Prometheus
  prometheus.csv        the series behind it
  drift-metrics.json    every drift score the scheduled job wrote to Azure Monitor
  alerts.json           the alert instances Azure Monitor fired, with their times

Run on the machine where `make monitor` is up. Reading the alert and metric history is an
Azure call and goes through `az`, which is why this is a script and not part of the adapter:
it is evidence collection for one exercise, not something the service ever does.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("reports/lab4")
PROM = "http://localhost:9090/api/v1/query_range"
SERIES = {
    "temp_c rolling mean": 'feature_rolling_mean{feature="temp_c"}',
    "temp_c rolling sd": 'feature_rolling_std{feature="temp_c"}',
    "request rate": 'sum(rate(http_requests_total{path=~"/predict.*"}[1m]))',
}


def prometheus(since: datetime.datetime, until: datetime.datetime) -> dict[str, list]:
    out = {}
    for label, expr in SERIES.items():
        query = urllib.parse.urlencode({"query": expr, "start": since.timestamp(),
                                        "end": until.timestamp(), "step": "15"})
        with urllib.request.urlopen(f"{PROM}?{query}", timeout=30) as r:
            result = json.loads(r.read())["data"]["result"]
        out[label] = [(float(t), float(v)) for t, v in result[0]["values"]] if result else []
    return out


def az(*args: str) -> object:
    done = subprocess.run(["az", *args, "-o", "json"], capture_output=True, text=True, check=True)
    return json.loads(done.stdout or "null")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="UTC, e.g. 2026-10-06T13:30:00Z")
    ap.add_argument("--injection", default=None, help="UTC time of the first shifted request")
    ap.add_argument("--group", default="itcs355-6688067")
    ap.add_argument("--app", default="itcs355-staging")
    args = ap.parse_args()
    since = datetime.datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    until = datetime.datetime.now(datetime.timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)

    series = prometheus(since, until)
    with (OUT / "prometheus.csv").open("w", encoding="utf-8") as fh:
        fh.write("series,utc,value\n")
        for label, points in series.items():
            for t, v in points:
                stamp = datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
                fh.write(f"{label},{stamp:%Y-%m-%dT%H:%M:%SZ},{v}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    def times(points):
        return [datetime.datetime.fromtimestamp(t, datetime.timezone.utc) for t, _ in points]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for label in ("temp_c rolling mean", "temp_c rolling sd"):
        pts = series[label]
        top.plot(times(pts), [v for _, v in pts], label=label)
    top.set_ylabel("°C")
    top.set_title("temp_c over the last 500 inputs (Grafana panel 4, from Prometheus)")
    pts = series["request rate"]
    bottom.plot(times(pts), [v for _, v in pts], color="grey", label="request rate")
    bottom.set_ylabel("req/s")
    if args.injection:
        injected = datetime.datetime.fromisoformat(args.injection.replace("Z", "+00:00"))
        for ax in (top, bottom):
            ax.axvline(injected, color="red", linestyle="--", label="injection")
    top.legend(loc="upper left")
    bottom.legend(loc="upper left")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=datetime.timezone.utc))
    bottom.set_xlabel("UTC")
    fig.tight_layout()
    fig.savefig(OUT / "dashboard-shift.png", dpi=110)

    app = az("containerapp", "show", "-g", args.group, "-n", args.app, "--query", "id")
    metrics = ",".join(["drift_threshold_ratio", "drift_window_rows", "drift_psi_temp_c",
                        "drift_psi_load_pct"])
    span = f"{since:%Y-%m-%dT%H:%M:%SZ}/{until:%Y-%m-%dT%H:%M:%SZ}"
    raw = az("rest", "--method", "get", "--url",
             f"https://management.azure.com{app}/providers/microsoft.insights/metrics"
             f"?api-version=2023-10-01&metricnamespace=itcs355&metricnames={metrics}"
             f"&timespan={span}&interval=PT1M&aggregation=Maximum")
    scores = {v["name"]["value"]: [(d["timeStamp"], d["maximum"])
                                   for d in v["timeseries"][0]["data"] if d.get("maximum") is not None]
              for v in raw["value"] if v["timeseries"]}
    (OUT / "drift-metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

    sub = app.split("/")[2]
    alerts = az("rest", "--method", "get", "--url",
                f"https://management.azure.com/subscriptions/{sub}/providers/"
                "Microsoft.AlertsManagement/alerts?api-version=2019-05-05-preview&timeRange=1d")
    fired = [{"name": a["properties"]["essentials"]["alertRule"].split("/")[-1],
              "state": a["properties"]["essentials"]["monitorCondition"],
              "fired_utc": a["properties"]["essentials"]["startDateTime"],
              "severity": a["properties"]["essentials"]["severity"]}
             for a in alerts.get("value", [])
             if a["properties"]["essentials"]["alertRule"].endswith("/itcs355-drift-alert")]
    (OUT / "alerts.json").write_text(json.dumps(fired, indent=2), encoding="utf-8")

    print(json.dumps({"scores": scores.get("drift_threshold_ratio"), "alerts": fired}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
