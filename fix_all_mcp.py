#!/usr/bin/env python3
import os
import re

# Fix pattern for all server files
def fix_server_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix imports
    content = re.sub(
        r'from mcp\.server import Server\n',
        'from mcp.server import Server\n',
        content
    )
    
    # Fix @server.tool() to proper MCP format
    content = re.sub(
        r'@server\.tool\(\)',
        '@server.list_tools()\nasync def handle_list_tools():\n    return server.tools\n\n@server.call_tool()',
        content, count=1
    )
    
    # Fix remaining @server.tool() decorators
    content = re.sub(
        r'@server\.tool\(\)',
        'server.add_tool',
        content
    )
    
    # Fix the tool registration pattern
    content = re.sub(
        r'async def (\w+)\((.+?)\):',
        r'server.add_tool("\1", "\1 tool", {}, \1)\n\nasync def \1(\2):',
        content
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

# Fix all server files
servers = ["bias-detector", "data-cleaner", "fairness-evaluator", "compliance-logger", "prediction-server"]
for server in servers:
    fix_server_file(f"mcp-servers/{server}/server.py")
