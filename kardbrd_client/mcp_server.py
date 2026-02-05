import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .client import KardbrdClient
from .tools import TOOLS, ToolExecutor

logger = logging.getLogger(__name__)


class KardbrdMCPServer:
    def __init__(self, api_url: str, token: str):
        self.client = KardbrdClient(base_url=api_url, token=token)
        self.executor = ToolExecutor(self.client)
        self.server = Server("kardbrd-mcp")
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [Tool(name=t["name"], description=t["description"], inputSchema=t["input_schema"]) for t in TOOLS]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            try:
                result = self.executor.execute(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Kardbrd MCP Server")
    parser.add_argument("--api-url", help="Kardbrd API URL (or set KARDBRD_API_URL env var)")
    parser.add_argument("--token", help="API token (or set KARDBRD_TOKEN env var)")
    args = parser.parse_args()

    # Use CLI args, fall back to env vars
    api_url = args.api_url or os.environ.get("KARDBRD_API_URL")
    token = args.token or os.environ.get("KARDBRD_TOKEN")

    if not api_url:
        parser.error("--api-url is required (or set KARDBRD_API_URL)")
    if not token:
        parser.error("--token is required (or set KARDBRD_TOKEN)")

    logging.basicConfig(level=logging.INFO)
    server = KardbrdMCPServer(api_url, token)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
