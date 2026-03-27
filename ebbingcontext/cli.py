"""CLI entry point for EbbingContext."""

from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ebbingcontext",
        description="EbbingContext — Ebbinghaus Forgetting Curve-Based Context Management Engine",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the MCP server")
    serve_parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()

    if args.command == "serve":
        from ebbingcontext.config import load_config
        from ebbingcontext.interface.mcp_server import run_server

        config = load_config(args.config)
        asyncio.run(run_server(config=config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
