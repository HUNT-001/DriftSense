# DriftSense Pilot — day-by-day schedule (IST)

A = Person A (engine, accuracy, virtual SEM, agent, sealed eval). B = Person B (cloud, service, COOL bench, UI, docs pipeline).

## Daily ritual (both machines)
- **09:30** — 10-min call. Each runs `/sync`, states today's branch, declares any shared-file change (goes first).
- **13:00** — shared-file window: `shared/*` PRs reviewed and merged.
- **18:00** — every active branch has an open PR (draft is fine).
- **21:00** — review the other person's PRs with `/review-pr`.
- **22:00** — merge what is approved and green. No branch lives past 48 hours.

## Schedule
| Date | Person A (branch → PR to main) | Person B (branch → PR to main) | Merge order / gate |
|---|---|---|---|
| Sun Oct 11 | S1–S5 on `shared/p0-bootstrap`; then `a/p1-profile` | Machine + AWS SSO; approve bootstrap by 16:00; start `b/p4-cdk-core` | bootstrap merged; apply rulesets; G0 |
| Mon Oct 12 | `a/p1-profile` PR (baseline_v1.json); start `a/p2-s0-s2` | `b/p4-cdk-core` PR (S3, DynamoDB, SQS+DLQ, HTTP API+arm64 Lambda, no NAT, Budgets) | both |
| Tue Oct 13 | `a/p2-s0-s2` PR (ingest/lattice/rectify) | `b/p4-service` (FastAPI+Mangum, SQS worker, storage, vs fake; moto) | a/p2-s0-s2. Contract additive-only |
| Wed Oct 14 | `a/p2-s3-s6` (roughness, score, cascade, contour) | `b/p4-service` PR; start `b/p9-docs-pipeline` | b/p4-service |
| Thu Oct 15 | `a/p2-s3-s6` PR; start `a/p3-dram` | `b/p9-docs-pipeline` PR; start `b/p4-cool-worker` | both |
| Fri Oct 16 (G2) | `a/p2-g2` PR by 15:00 → review 18:00 → tag `g2-pass` | review `a/p2-g2` 15:00–18:00; continue cool-worker | a/p2-g2 → `int/g2-default-v2` |
| Sat Oct 17 | `a/p3-dram` PR; help debug v2 on arm64 | `int/g2-v2-swap`; `b/p4-cool-worker` PR | all three |
| Sun Oct 18 | `a/p3-confidence` PR; start `a/p5-virtual-sem` | `b/p4-canary-dash`; start `b/p8-ui-skeleton` | a/p3-confidence before freeze |
| Mon Oct 19 (G3,G4) | 12:00 tag `v2-freeze`; `a/p5-virtual-sem` PR | `b/p4-canary-dash` PR → G4 → `g4-pass` | freeze first |
| Tue Oct 20 (G5) | `a/p5-agent-mcp` PR by 14:00 → go/no-go 18:00 → `g5-pass` | `b/p8-ui-skeleton` PR; `int/g5-agent-service`; start `b/p6-cool-bench` | ui, agent, then int |
| Wed Oct 21 | `a/p7-protocol` → tag `protocol-v1`; `eval/agent_eval` | `int/g5-agent-service` PR; `b/p6-cool-bench`; start `b/p8-ui-polish` | both |
| Thu Oct 22 | `a/p7-sealed-run` (once): results/sealed/*.json, stats, charts | `b/p6-cool-bench` PR (COOL evidence) | both |
| Fri Oct 23 (numbers lock) | review B's rendered numbers vs results | `docs/numbers-lock`: make docs + check_claims → tag `numbers-lock` | both approve after |
| Sat Oct 24 | `a/p9-report` (REPORT, virtual_sem, PRIOR_WORK); video script | `b/p8-ui-polish`; `b/p9-docs` (ARCHITECTURE, COOL_EVIDENCE, SUBMISSION, RESPONSIBLE_USE) | all P8/P9 by 22:00 |
| Sun Oct 25 | verify B's area; `fix/*`; record video | verify A's area; `fix/*` | 20:00 tag `rc1` |
| Mon Oct 26 | 09:00 fresh verifier on clean clone; 20:00 tag `submission` | endpoint/COOL live check; Devpost form | dry run by 18:00; submit 21:00 IST |
