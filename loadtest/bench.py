# loadtest/bench.py
import argparse, asyncio, statistics, time
import httpx

async def one(client, url, payload):
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload)
        return (time.perf_counter() - t0) * 1000 if r.status_code == 200 else None
    except Exception:
        return None

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("-c", "--concurrency", type=int, default=32)
    ap.add_argument("-n", "--requests", type=int, default=200)
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/predict"
    payload = {"distance_km": 12.5, "passengers": 2}
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async def guarded():
            async with sem:
                return await one(client, url, payload)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded() for _ in range(args.requests)))
        wall = time.perf_counter() - t0
        oks = sorted(r for r in results if r is not None)
        p = lambda q: oks[min(int(len(oks) * q), len(oks) - 1)]
        print(f"throughput: {len(oks)/wall:.1f} req/s | p50: {statistics.median(oks):.1f}ms "
              f"| p95: {p(0.95):.1f}ms | p99: {p(0.99):.1f}ms")

if __name__ == "__main__":
    asyncio.run(main())