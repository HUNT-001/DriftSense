"""COOL (Cloud Optimized OpenCV, AWS Graviton4) benchmark harness for the DriftSense v2 engine.

Modules (each runnable with ``python -m bench.cool.<name> --help``):

- ``common``           timing core, bootstrap CIs, env capture, ResultsFile writer
- ``pairs``            committed DEV pairs (outputs/dataset), sealed split refused
- ``ops``              the op table: every engine cv2 call site with its KleidiCV group
- ``census``           wraps engine cv2 calls during localize: counts, time shares, argument capture
- ``microbench``       replays each captured engine call at production parameters
- ``stagebench``       per-stage timings from ``LocalizeResult.timings``
- ``e2e``              end-to-end latency p50/p95 and process-pool throughput
- ``agent_bench``      agent episodes per hour on the VirtualSEM twin
- ``equality``         numeric-equality check between two arms
- ``collect_evidence`` build/host evidence pack (getBuildInformation, ldd, IMDS, ...)
- ``report``           speed-ups with bootstrap CIs, Amdahl check, cost per 1,000
"""
