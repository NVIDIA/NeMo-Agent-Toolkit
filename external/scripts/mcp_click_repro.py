#!/usr/bin/env python3
# Python 3.11/3.12
import asyncio, gc, logging, sys, time
import click

# ---- async core (canonical usage) ----
async def list_tools_streamable_http(url: str):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url=url) as ctx:
        if isinstance(ctx, tuple):
            read, write = ctx[0], ctx[1]
        else:
            read, write = ctx

        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()

            # tiny yields; keep it otherwise “clean”
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    return [t.name for t in resp.tools]

# ---- Click CLI that mimics a “busy” teardown ----
@click.command()
@click.option("--url", default="http://localhost:9901/mcp", show_default=True)
@click.option("--post-gc", is_flag=True, help="Force GC after asyncio.run to provoke finalizers")
@click.option("--extra-logging", is_flag=True, help="Emit logging after asyncio.run returns")
def main(url: str, post_gc: bool, extra_logging: bool):
    logging.basicConfig(level=logging.INFO)  # typical CLI init
    log = logging.getLogger("mcp-click-repro")

    tools = asyncio.run(list_tools_streamable_http(url))
    for t in tools:
        click.echo(t)

    # simulate extra CLI work after the event loop is gone
    if extra_logging:
        log.info("extra logging after asyncio.run (loop closed)")

    sys.stdout.flush(); sys.stderr.flush()
    time.sleep(0.01)  # let background finalizers/threads run

    if post_gc:
        gc.collect()            # <- often triggers the noisy finalizers
        time.sleep(0.02)
        sys.stdout.flush(); sys.stderr.flush()

    # keep Click cleanup in play
    click.echo("done")

if __name__ == "__main__":
    main()
