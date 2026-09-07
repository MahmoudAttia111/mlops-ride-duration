
# Module 3 Report

## Orchestration

- DAG: ride_duration_retrain, schedule=@weekly, retries=2

## Load Test Comparison

| Server   | Throughput | p50 | p95 | p99 | avg_batch_size |

|----------|-----------|-----|-----|-----|-----------------|

| FastAPI  |           |     |     |     | 1.0             |

| BentoML  |           |     |     |     |                 |

## Infrastructure

- terraform apply → external_ip: ...

