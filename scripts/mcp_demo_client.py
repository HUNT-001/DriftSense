"""Drive the DriftSense MCP server like an external agent would: ``python scripts/mcp_demo_client.py``.

Starts ``python -m driftsense.agent.mcp_server`` over stdio, lists the six tools, acquires a reference
and a search frame from the virtual SEM, and runs one v2 ``localize``. Needs ``mcp`` (requirements.lock).
"""

from __future__ import annotations

import asyncio
import json
import sys


async def main() -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "driftsense.agent.mcp_server"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", ", ".join(t.name for t in tools.tools))
        for label, fov, dwell in (("reference", 260.0, 24.0), ("search", 360.0, 3.0)):
            res = await session.call_tool(
                "acquire", {"args": {"center_nm": [0.0, 0.0], "fov_nm": fov, "dwell_us": dwell}}
            )
            meta = json.loads(res.content[0].text)
            print(f"acquire ({label}): snr={meta.get('snr')} dose={meta.get('dose_this_acq')}")
        res = await session.call_tool("localize", {"args": {}})
        result = json.loads(res.content[0].text)
        m = result["margin"]
        print(
            f"localize: engine={result['engine']} margin={m['margin']:.3f} tau={m['tau']} -> {m['decision']}"
        )
        return 1 if res.isError else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
