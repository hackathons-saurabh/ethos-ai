import asyncio
import json
import numpy as np
import pandas as pd
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
app = FastAPI(title="Fairness Evaluator MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for HTTP requests
class FairnessEvaluationRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    predictions: List[float]
    actual: List[float] = []
    sensitive_attributes: List[str] = ["gender", "race", "age"]

class FairnessEvaluationResponse(BaseModel):
    overall_fairness_score: float
    is_fair: bool
    detailed_metrics: Dict[str, Any]
    violations: List[str]
    recommendations: List[str]

server = Server("fairness-evaluator")
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

@tool("evaluate_fairness")
async def evaluate_fairness(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(data['dataset'])
        predictions = np.array(data['predictions'])
        actual = np.array(data.get('actual', predictions))
        sensitive_attrs = data.get('sensitive_attributes', ['gender', 'race', 'age'])
        
        results = {
            'overall_fairness_score': 0.0,
            'is_fair': True,
            'detailed_metrics': {},
            'violations': [],
            'recommendations': []
        }
        
        fairness_scores = []
        
        for attr in sensitive_attrs:
            if attr not in df.columns:
                continue
                
            groups = df[attr].unique()
            positive_rates = {}
            
            for group in groups:
                mask = df[attr] == group
                positive_rate = predictions[mask].mean()
                positive_rates[str(group)] = float(positive_rate)
            
            max_rate = max(positive_rates.values())
            min_rate = min(positive_rates.values())
            disparity = max_rate - min_rate
            
            fairness_scores.append(disparity)
            
            if disparity > 0.1:
                results['violations'].append(f"Statistical parity violation for {attr}")
                results['recommendations'].append(f"Adjust predictions to equalize positive rates across {attr} groups")
                results['is_fair'] = False
        
        if fairness_scores:
            results['overall_fairness_score'] = float(np.mean(fairness_scores))
        
        logger.info(f"Fairness evaluation complete. Overall score: {results['overall_fairness_score']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in fairness evaluation: {str(e)}")
        return {
            'error': str(e),
            'overall_fairness_score': 1.0,
            'is_fair': False,
            'detailed_metrics': {},
            'violations': ['Evaluation error'],
            'recommendations': ['Fix evaluation errors']
        }

@tool("suggest_mitigation")
async def suggest_mitigation(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    return {'mitigation_strategies': [], 'priority_order': []}

@tool("get_fairness_metrics")
async def get_fairness_metrics() -> Dict[str, Any]:
    return {
        'metrics': {
            'statistical_parity': 'Equal positive prediction rates across groups',
            'equal_opportunity': 'Equal true positive rates across groups'
        }
    }

# HTTP Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "fairness-evaluator"}

@app.post("/evaluate_fairness", response_model=FairnessEvaluationResponse)
async def http_evaluate_fairness(request: FairnessEvaluationRequest):
    """HTTP endpoint for fairness evaluation"""
    try:
        data = {
            "dataset": request.dataset,
            "predictions": request.predictions,
            "actual": request.actual,
            "sensitive_attributes": request.sensitive_attributes
        }
        result = await evaluate_fairness(data)
        return FairnessEvaluationResponse(**result)
    except Exception as e:
        logger.error(f"HTTP fairness evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp")
async def mcp_endpoint(request: Dict[str, Any]):
    """MCP protocol endpoint for HTTP communication"""
    try:
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id", "1")
        
        if method == "tools/call/evaluate_fairness":
            result = await evaluate_fairness(params.get("arguments", {}))
        elif method == "tools/call/suggest_mitigation":
            result = await suggest_mitigation(params.get("arguments", {}))
        elif method == "tools/call/get_fairness_metrics":
            result = await get_fairness_metrics()
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
        logger.info("Starting fairness-evaluator MCP Server...")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                initialization_options={"log_level": "debug", "agent_name": "fairness-evaluator"}
            )
            # Keep it alive
            while True:
                await asyncio.sleep(1)
    
    asyncio.run(mcp_main())

def run_http_server():
    """Run the HTTP server"""
    logger.info("Starting fairness-evaluator HTTP Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def main():
    """Run both MCP stdio and HTTP servers"""
    logger.info("Starting fairness-evaluator servers...")
    
    # Start MCP server in a separate thread
    mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
    mcp_thread.start()
    
    # Give MCP server time to start
    time.sleep(2)
    
    # Run HTTP server in main thread
    run_http_server()

if __name__ == "__main__":
    main()
