#!/bin/bash

# Fix all server files by removing the broken add_tool lines
for server in mcp-servers/*/server.py; do
    echo "Fixing $server..."
    
    # Remove the problematic server.add_tool lines
    sed -i '' '/^server\.add_tool$/d' "$server"
    
    # Fix the decorator pattern back to original but with proper syntax
    # This is a temporary fix to just comment out the decorators
    sed -i '' 's/@server\.list_tools()/#@server.list_tools()/g' "$server"
    sed -i '' 's/@server\.call_tool()/#@server.call_tool()/g' "$server"
    
    # Add a simple HTTP endpoint instead at the end of each file
    cat >> "$server" << 'PYTHONFIX'

# Temporary HTTP wrapper for MCP
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/mcp")
async def mcp_endpoint(request: dict):
    method = request.get("method", "").replace("tools/", "")
    params = request.get("params", {})
    
    # Route to appropriate function
    if method == "detect_bias" and 'detect_bias' in globals():
        result = await detect_bias(params)
    elif method == "clean_bias" and 'clean_bias' in globals():
        result = await clean_bias(params)
    elif method == "evaluate_fairness" and 'evaluate_fairness' in globals():
        result = await evaluate_fairness(params)
    elif method == "log_compliance_event" and 'log_compliance_event' in globals():
        result = await log_compliance_event(params)
    elif method == "train_model" and 'train_model' in globals():
        result = await train_model(params)
    elif method == "make_predictions" and 'make_predictions' in globals():
        result = await make_predictions(params)
    else:
        result = {"error": f"Unknown method: {method}"}
    
    return {"result": result}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run FastAPI instead of MCP
    uvicorn.run(app, host="0.0.0.0", port=8000)
PYTHONFIX
    
done

# Update requirements to include FastAPI
for dir in mcp-servers/*/; do
    echo "fastapi>=0.104.0" >> "$dir/requirements.txt"
    echo "uvicorn>=0.24.0" >> "$dir/requirements.txt"
done

echo "Fixed all servers!"
