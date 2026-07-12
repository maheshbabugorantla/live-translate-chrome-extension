# Submission Report — Assignment 1 — Live Translate

- **Student:** Mahesh Babu Gorantla
- **Video demo:** _(paste your 60–90s demo link)_
- **Backend target:** `http://localhost:8787`
- **Auto-graded score:** **70 / 70**  ·  manual portion: 30 pts (grader)

## Rubric

| Criterion | Type | Points | Result |
|---|---|---|---|
| Widget lights up (contract works end to end) | auto | 15/15 | translate + batch return valid shapes |
| Caching correctness (two-tier, provable, persistent) | auto | 20/20 | 2nd cached=True, faster=True, sqlite_persisted=True |
| Performance & SLA gate | auto | 15/15 | bench SLA gate PASS |
| Logging & observability | auto | 10/10 | stats_hit_rate=True, health_reports_ai=True, ai_log_file=True, trace_correlated=True |
| Service separation & correct status codes | auto | 10/10 | 400_on_bad_input=True, gateway_nests_ai_health=True |
| LLM & prompt quality (natural Mexican Spanish) | manual | —/20 | **grader** — see evidence + video |
| Deploy & docs | manual | —/10 | **grader** — see evidence + video |

## Evidence

- Sample translation (`Good morning, welcome!`): **Buenos días, ¡bienvenidos!**
- Cache latency: first `1536 ms` → second `0 ms`
- Trace correlation (one request across both logs): ✅ yes
- Benchmark: hit p95 `7 ms`, miss p95 `2239 ms`, hit rate `78%`, throughput `1884 rps`, SLA **PASS**
- Cost: `$0.000167`/miss; monthly savings from cache `$64.84`
- Deploy: _(no --deploy-url given; grader verifies your Fly.io URL from README/video)_

<details><summary>Benchmark output</summary>

```
    cost per MISS (avg)         $0.000167
    @ 500,000/mo, no cache      $83.67
    @ 500,000/mo, cached        $18.82
    monthly savings from cache  $64.84
── SLA GATE ────────────────────────────────────────
    PASS  cache_hit_p95_ms         7.3 <= 60
    PASS  cache_miss_p95_ms        2238.5 <= 3500
    PASS  min_cache_hit_rate_pct   77.5 >= 60
    PASS  max_error_rate_pct       0.0 <= 1.0
    PASS  min_throughput_rps       1884.5 >= 20

✅ ALL SLAs MET

Wrote /Users/maheshbabugorantla/Code/Hamza_Farooq/fde_translate_chrome_extension/eval/_bench.json
```
</details>
