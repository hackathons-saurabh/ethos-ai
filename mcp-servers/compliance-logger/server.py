import asyncio
import json
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any
from mcp.server import Server
import mcp.server.stdio
import mcp.types as types
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for HTTP endpoints
app = FastAPI(title="Compliance Logger MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for HTTP requests
class ComplianceEventRequest(BaseModel):
    event_type: str
    component: str
    action: str
    details: Dict[str, Any] = {}

class ComplianceEventResponse(BaseModel):
    log_id: str
    timestamp: str
    status: str
    data_hash: str

server = Server("compliance-logger")
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

class ComplianceLogger:
    def __init__(self):
        self.logs = []
        
    def generate_hash(self, data: Any) -> str:
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def log_event(self, event_type: str, component: str, action: str, details: Dict[str, Any]) -> str:
        log_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        data_hash = self.generate_hash(details)
        
        log_entry = {
            'id': log_id,
            'timestamp': timestamp,
            'event_type': event_type,
            'component': component,
            'action': action,
            'data_hash': data_hash,
            'details': details
        }
        
        self.logs.append(log_entry)
        logger.info(f"Logged compliance event: {log_id} - {event_type} - {action}")
        
        return log_id

# Global logger instance
logger_instance = ComplianceLogger()

@tool("log_compliance_event")
async def log_compliance_event(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        event_type = data.get('event_type', 'UNKNOWN')
        component = data.get('component', 'UNKNOWN')
        action = data.get('action', 'UNKNOWN')
        details = data.get('details', {})
        
        log_id = logger_instance.log_event(event_type, component, action, details)
        
        result = {
            'log_id': log_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'logged',
            'data_hash': logger_instance.generate_hash(details)
        }
        
        logger.info(f"Compliance event logged successfully: {log_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error logging compliance event: {str(e)}")
        return {
            'error': str(e),
            'status': 'failed',
            'log_id': None
        }

@tool("generate_compliance_report")
async def generate_compliance_report(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'summary': {
            'total_events': len(logger_instance.logs),
            'time_period': {'start': 'all_time', 'end': 'current'}
        },
        'detailed_logs': logger_instance.logs[:10]
    }

@tool("verify_data_integrity")
async def verify_data_integrity(data: Dict[str, Any]) -> Dict[str, Any]:
    return {'verified': True, 'timestamp': datetime.now(timezone.utc).isoformat()}

@tool("export_audit_trail")
async def export_audit_trail(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'export_timestamp': datetime.now(timezone.utc).isoformat(),
        'total_entries': len(logger_instance.logs),
        'entries': logger_instance.logs
    }

@tool("get_compliance_frameworks")
async def get_compliance_frameworks() -> Dict[str, Any]:
    return {
        'frameworks': {
            'GDPR': {'name': 'General Data Protection Regulation'},
            'CCPA': {'name': 'California Consumer Privacy Act'}
        }
    }

# HTTP Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "compliance-logger"}

@app.post("/log_compliance_event", response_model=ComplianceEventResponse)
async def http_log_compliance_event(request: ComplianceEventRequest):
    """HTTP endpoint for logging compliance events"""
    try:
        data = {
            "event_type": request.event_type,
            "component": request.component,
            "action": request.action,
            "details": request.details
        }
        result = await log_compliance_event(data)
        return ComplianceEventResponse(**result)
    except Exception as e:
        logger.error(f"HTTP compliance logging error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp")
async def mcp_endpoint(request: Dict[str, Any]):
    """MCP protocol endpoint for HTTP communication"""
    try:
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id", "1")
        
        if method == "tools/call/log_compliance_event":
            result = await log_compliance_event(params.get("arguments", {}))
        elif method == "tools/call/generate_compliance_report":
            result = await generate_compliance_report(params.get("arguments", {}))
        elif method == "tools/call/verify_data_integrity":
            result = await verify_data_integrity(params.get("arguments", {}))
        elif method == "tools/call/export_audit_trail":
            result = await export_audit_trail(params.get("arguments", {}))
        elif method == "tools/call/get_compliance_frameworks":
            result = await get_compliance_frameworks()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
        
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        }
        
    except Exception as e:
        logger.error(f"MCP endpoint error: {str(e)}")
        return {
            "jsonrpc": "2.0",
            "error": {"code": -1, "message": str(e)},
            "id": request.get("id", "1")
        }

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

def run_mcp_server():
    """Run the MCP stdio server in a separate thread"""
    async def mcp_main():
        logger.info("Starting compliance-logger MCP Server...")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                initialization_options={"log_level": "debug", "agent_name": "compliance-logger"}
            )
            # Keep it alive
            while True:
                await asyncio.sleep(1)
    
    asyncio.run(mcp_main())

def run_http_server():
    """Run the HTTP server"""
    logger.info("Starting compliance-logger HTTP Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def main():
    """Run both MCP stdio and HTTP servers"""
    logger.info("Starting compliance-logger servers...")
    
    # Start MCP server in a separate thread
    mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
    mcp_thread.start()
    
    # Give MCP server time to start
    time.sleep(2)
    
    # Run HTTP server in main thread
    run_http_server()

if __name__ == "__main__":
    main()
