import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
import logging
import mcp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for HTTP endpoints
app = FastAPI(title="Bias Detector MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for HTTP requests
class BiasDetectionRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    target_column: str = "target"

class BiasDetectionResponse(BaseModel):
    overall_bias_score: float
    bias_types_detected: List[str]
    detailed_results: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any]

server = Server("bias-detector")

# Tool handlers storage
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

class BiasDetector:
    def __init__(self):
        self.sensitive_attributes = ['gender', 'race', 'age', 'ethnicity', 'religion']
        self.bias_thresholds = {
            'demographic_parity': 0.1,
            'disparate_impact': 0.8,
            'statistical_parity': 0.1
        }
    
    def detect_demographic_parity(self, df: pd.DataFrame, target_col: str, sensitive_col: str) -> Dict:
        if sensitive_col not in df.columns or target_col not in df.columns:
            return {'bias_detected': False, 'score': 0.0}
        
        groups = df.groupby(sensitive_col)[target_col].agg(['mean', 'count'])
        max_rate = float(groups['mean'].max())
        min_rate = float(groups['mean'].min())
        parity_diff = max_rate - min_rate
        
        return {
            'bias_detected': bool(parity_diff > self.bias_thresholds['demographic_parity']),
            'score': float(parity_diff),
            'groups': {str(k): {'mean': float(v['mean']), 'count': int(v['count'])} for k, v in groups.to_dict('index').items()},
            'metric': 'demographic_parity'
        }
    
    def detect_disparate_impact(self, df: pd.DataFrame, target_col: str, sensitive_col: str) -> Dict:
        if sensitive_col not in df.columns or target_col not in df.columns:
            return {'bias_detected': False, 'score': 0.0}
        
        groups = df.groupby(sensitive_col)[target_col].mean()
        
        if len(groups) < 2:
            return {'bias_detected': False, 'score': 1.0}
        
        majority_rate = groups.max()
        minority_rate = groups.min()
        
        if majority_rate == 0:
            impact_ratio = 0
        else:
            impact_ratio = minority_rate / majority_rate
        
        return {
            'bias_detected': impact_ratio < self.bias_thresholds['disparate_impact'],
            'score': float(impact_ratio),
            'groups': groups.to_dict(),
            'metric': 'disparate_impact'
        }

@tool("detect_bias")
async def detect_bias(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(data['dataset'])
        target_col = data.get('target_column', 'target')
        
        detector = BiasDetector()
        results = {
            'overall_bias_score': 0.0,
            'bias_types_detected': [],
            'detailed_results': {},
            'recommendations': []
        }
        
        bias_scores = []
        
        for attr in detector.sensitive_attributes:
            if attr in df.columns:
                parity_result = detector.detect_demographic_parity(df, target_col, attr)
                if parity_result['bias_detected']:
                    results['bias_types_detected'].append(f'demographic_parity_{attr}')
                    results['detailed_results'][f'demographic_parity_{attr}'] = parity_result
                    bias_scores.append(parity_result['score'])
                    results['recommendations'].append(
                        f"Address demographic parity bias in {attr} (difference: {parity_result['score']:.2f})"
                    )
        
        if bias_scores:
            results['overall_bias_score'] = float(np.mean(bias_scores))
        
        results['metadata'] = {
            'rows_analyzed': int(len(df)),
            'columns_analyzed': list(df.columns),
            'sensitive_attributes_found': [attr for attr in detector.sensitive_attributes if attr in df.columns]
        }
        
        logger.info(f"Bias detection complete. Overall score: {results['overall_bias_score']:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in bias detection: {str(e)}")
        return {
            'error': str(e),
            'overall_bias_score': 0.0,
            'bias_types_detected': [],
            'detailed_results': {},
            'recommendations': []
        }

@tool("analyze_feature_importance")
async def analyze_feature_importance(data: Dict[str, Any]) -> Dict[str, Any]:
    return {'feature_importance': [], 'most_biased_features': []}

@tool("get_bias_metrics")
async def get_bias_metrics() -> Dict[str, Any]:
    return {
        'metrics': {
            'demographic_parity': 'Ensures equal positive outcome rates across groups',
            'disparate_impact': '80% rule - minority group should have at least 80% of majority rate'
        }
    }

# HTTP Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "bias-detector"}

@app.post("/detect_bias", response_model=BiasDetectionResponse)
async def http_detect_bias(request: BiasDetectionRequest):
    """HTTP endpoint for bias detection"""
    try:
        data = {
            "dataset": request.dataset,
            "target_column": request.target_column
        }
        result = await detect_bias(data)
        return BiasDetectionResponse(**result)
    except Exception as e:
        logger.error(f"HTTP bias detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp")
async def mcp_endpoint(request: Dict[str, Any]):
    """MCP protocol endpoint for HTTP communication"""
    try:
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id", "1")
        
        if method == "tools/call/detect_bias":
            result = await detect_bias(params.get("arguments", {}))
        elif method == "tools/call/analyze_feature_importance":
            result = await analyze_feature_importance(params.get("arguments", {}))
        elif method == "tools/call/get_bias_metrics":
            result = await get_bias_metrics()
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
        logger.info("Starting bias-detector MCP Server...")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                initialization_options={"log_level": "debug", "agent_name": "bias-detector"}
            )
            # Keep it alive
            while True:
                await asyncio.sleep(1)
    
    asyncio.run(mcp_main())

def run_http_server():
    """Run the HTTP server"""
    logger.info("Starting bias-detector HTTP Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def main():
    """Run both MCP stdio and HTTP servers"""
    logger.info("Starting bias-detector servers...")
    
    # Start MCP server in a separate thread
    mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
    mcp_thread.start()
    
    # Give MCP server time to start
    time.sleep(2)
    
    # Run HTTP server in main thread
    run_http_server()

if __name__ == "__main__":
    main()
