// ITCS355 Lab 3 — load test.
//
//   k6 run -e TARGET=https://<endpoint>/predict -e VUS=10 loadtest/k6.js
//
// Run this at THREE concurrency levels (suggested 1, 10, 50) and record p50, p95, p99,
// throughput, and error rate for each. Commit the results in reports/lab3-load.md.
//
// An uncommitted load test is not evidence.

import http from 'k6/http';
import { check } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const latency = new Trend('predict_latency_ms');
const failures = new Rate('predict_failures');

// OUR TARGET, set before the first measurement (commit this before any results):
//   p95 under 300 ms for POST /predict at 10 concurrent users, error rate under 1%,
//   measured from a client in the same region as the endpoint, on a warm instance.
// Why 300 ms: the caller is a dispatch screen a person reads, not another service in a
// request chain; a third of a second is below what a person notices, and it leaves room
// for one Azure hop. Cold start is reported separately because it is a different event.
export const options = {
  vus: Number(__ENV.VUS || 10),
  duration: __ENV.DURATION || '60s',
  thresholds: {
    'predict_latency_ms': ['p(95)<300'],
    'predict_failures': ['rate<0.01'],
  },
};

// MODE=single  one row per request (the target above applies to this)
// MODE=batch   ROWS rows in one /predict/batch call, for the batch and payload questions
const MODE = __ENV.MODE || 'single';
const ROWS = Number(__ENV.ROWS || 100);

const row = {
  temp_c: 78.4,
  vibration_mm_s: 3.1,
  pressure_kpa: 315.2,
  hours_since_service: 4200,
  load_pct: 68.0,
  ambient_humidity: 55.0,
};
const payload = MODE === 'batch'
  ? JSON.stringify({ rows: Array.from({ length: ROWS }, () => row) })
  : JSON.stringify(row);

export default function () {
  const res = http.post(__ENV.TARGET, payload, {
    headers: { 'Content-Type': 'application/json' },
  });
  latency.add(res.timings.duration);
  failures.add(res.status !== 200);
  check(res, {
    'status is 200': (r) => r.status === 200,
    'probability present': (r) => r.status === 200 && (
      MODE === 'batch' ? r.json('probabilities') !== undefined : r.json('probability') !== undefined),
    'version reported': (r) => r.headers['X-Model-Version'] !== undefined,
  });
}
