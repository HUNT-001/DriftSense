# DriftSense active-perception agent (G5)

An agent that drives a single `VirtualSEM` and the `driftsense.v2` localizer in a
closed perception loop. It does not trust the commanded stage coordinate; it
*acquires*, *localizes*, reads the uniqueness **margin** that the OpenCV pipeline
reports, and lets that margin pick the next physical action until it can either
commit a corrective stage move with confidence or hand the decision to a human.

This is the "Agentic Vision" claim made concrete: the output of the vision stage
(the margin between the best and second-best candidate) changes what the
instrument does next. Every step is recorded as a contract `TraceEvent`, and the
commit step records *why* it fired (`because="margin X >= tau Y"`), so the trace
is self-describing evidence rather than a log to be interpreted after the fact.

## Pieces

| module | role |
| --- | --- |
| `tools.py` | the six schema-validated tools over the instrument + localizer |
| `policy_rules.py` | the offline default policy (a margin-driven ladder) |
| `policy_llm.py` | optional LLM planner, behind a flag; proposes only, never commits |
| `runner.py` | the orchestration loop and the hard safety gate |
| `trace.py` | the `TraceEvent` sink (`traces/<run_id>.jsonl`) |
| `mcp_server.py` | FastMCP server exposing the tools over stdio or streamable HTTP |

## The loop, in words

A high-dwell reference acquisition seeds a template. Each subsequent search frame
is localized against that template; the implied feature offset is the stage
drift, and the margin is the agent's confidence in it. The rules policy then
chooses: raise dwell when the frame is noise-limited, ask for more rectify
hypotheses or a contour pass when the lattice is ambiguous, widen the field of
view when the true match sits at the frame edge with a periodic alias one period
away, commit the corrective move once the margin clears the threshold, or
escalate to a human when the budget runs out first.

The important property is honesty about failure. The localizer is strong, so most
frames clear the threshold on the first look. The cases that do not are dominated
by genuine periodic-lattice degeneracy rather than noise — the roughness
fingerprint is already noise-robust — and the agent's job is to turn a fraction
of those near-threshold escalations into correct commits by re-imaging, at a
modest extra dose, without ever committing a low-confidence (and therefore
possibly wrong) move. The four-arm evaluation quantifies exactly this; all
numbers live in `results/agent/g5_episode.json` and are printed by
`eval/agent_eval.py`, never typed here.

## Safety (enforced by the runner, not the policy)

- never commit a stage move while the margin is below the threshold — a policy
  that tries is coerced into an escalation;
- a bounded number of decision steps **and** a dose budget;
- a per-tool wall-clock timeout;
- any exception aborts the episode safely and escalates — never a silent commit.

The hidden stage error is the evaluator's alone: it is reachable only through
`VirtualSEM.verify`, and no agent module calls it (there is a test that asserts
this). The evaluator reads it through hooks to grade the agent; the agent never
sees it.

## MCP server

`mcp_server.py` exposes the same six tools with FastMCP, over stdio (default) or
streamable HTTP. The FastMCP import is **guarded**: the agent, the rules policy and
the G5 episode all run without `mcp`, and the offline smoke path (`smoke()` /
`list_tools()`) exercises the real tool registry and one `localize` without a live
transport. `commit_stage_move` over MCP is behind the same hard gate as the runner:
it is refused (and an `escalate` event is traced) unless the last `localize`
decided `commit`.

```
python -m driftsense.agent.mcp_server                                    # stdio
python -m driftsense.agent.mcp_server --transport streamable-http --port 8001
#   -> MCP endpoint http://127.0.0.1:8001/mcp (Inspector: transport "Streamable HTTP")
npx @modelcontextprotocol/inspector python -m driftsense.agent.mcp_server
python scripts/mcp_demo_client.py                                        # scripted client
```

Inspector lists the six tools from their pydantic schemas; calling `acquire`
twice (a reference then a search frame) and then `localize` returns a full
`LocalizeResult`, and `get_trace` returns the episode trace so far.

## Running it

```
# four-arm evaluation (writes the results JSON + demo traces, prints the table)
python eval/agent_eval.py --n 24 --demos

# offline smoke (no mcp, no network)
python -c "from driftsense.agent.mcp_server import smoke; print(smoke())"
```
