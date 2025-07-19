import asyncio
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Initialize the server
server = Server("test-server")

# Store handlers separately
handlers = {}

def tool(name: str):
    """Decorator to register tool handlers"""
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

# Define your tools with the decorator
@tool("detect_bias")
async def handle_detect_bias(arguments: dict) -> dict:
    # Your actual implementation
    return {"bias_score": 0.85}

# MCP protocol handlers
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=f"{name} tool",
            inputSchema={"type": "object", "properties": {}}
        )
        for name in handlers.keys()
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> dict:
    if name in handlers:
        return await handlers[name](arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
