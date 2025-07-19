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
app = FastAPI(title="Data Cleaner MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for HTTP requests
class DataCleaningRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    bias_report: Dict[str, Any] = {}
    target_column: str = "target"

class DataCleaningResponse(BaseModel):
    cleaned_dataset: List[Dict[str, Any]]
    cleaning_report: Dict[str, Any]

server = Server("data-cleaner")
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

class DataCleaner:
    def __init__(self):
        self.sensitive_attributes = ['gender', 'race', 'age', 'ethnicity', 'religion']
    
    def mask_sensitive_values(self, df: pd.DataFrame, attributes: List[str]) -> pd.DataFrame:
        df_cleaned = df.copy()
        for attr in attributes:
            if attr in df_cleaned.columns:
                if attr == 'gender':
                    df_cleaned[attr] = 'Person'
                elif attr == 'race':
                    df_cleaned[attr] = 'Individual'
                else:
                    df_cleaned[attr] = 'Masked'
        return df_cleaned
    
    def reweight_samples(self, df: pd.DataFrame, target_col: str, sensitive_col: str) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        if sensitive_col not in df.columns or target_col not in df.columns:
            return df_cleaned
        
        group_rates = df.groupby(sensitive_col)[target_col].mean()
        overall_rate = df[target_col].mean()
        
        weights = []
        for _, row in df.iterrows():
            group = row[sensitive_col]
            current_rate = group_rates[group]
            
            if current_rate > 0:
                weight = overall_rate / current_rate
            else:
                weight = 1.0
            
            weights.append(weight)
        
        df_cleaned['sample_weight'] = weights
        logger.info(f"Applied reweighting for {sensitive_col}")
        
        return df_cleaned

@tool("clean_bias")
async def clean_bias(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(data['dataset'])
        bias_report = data.get('bias_report', {})
        target_col = data.get('target_column', 'target')
        
        cleaner = DataCleaner()
        original_shape = df.shape
        
        cleaning_log = []
        df_cleaned = df.copy()
        
        # Apply cleaning based on detected biases
        bias_types = bias_report.get('bias_types_detected', [])
        
        for bias_type in bias_types:
            if 'demographic_parity' in bias_type:
                sensitive_attr = bias_type.split('_')[-1]
                df_cleaned = cleaner.reweight_samples(df_cleaned, target_col, sensitive_attr)
                cleaning_log.append(f"Applied reweighting for {sensitive_attr}")
        
        # Mask some sensitive attributes
        sensitive_attrs = [attr for attr in cleaner.sensitive_attributes if attr in df_cleaned.columns]
        if sensitive_attrs:
            df_cleaned = cleaner.mask_sensitive_values(df_cleaned, sensitive_attrs[:2])
            cleaning_log.append(f"Masked sensitive attributes: {sensitive_attrs[:2]}")
        
        cleaned_dataset = df_cleaned.to_dict('records')
        
        result = {
            'cleaned_dataset': cleaned_dataset,
            'cleaning_report': {
                'original_shape': original_shape,
                'cleaned_shape': df_cleaned.shape,
                'cleaning_methods_applied': cleaning_log,
                'samples_added': len(df_cleaned) - len(df)
            }
        }
        
        logger.info(f"Data cleaning complete. Shape: {original_shape} -> {df_cleaned.shape}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        return {
            'error': str(e),
            'cleaned_dataset': data.get('dataset', []),
            'cleaning_report': {'error': str(e)}
        }

@tool("preview_cleaning")
async def preview_cleaning(data: Dict[str, Any]) -> Dict[str, Any]:
    return {'preview': 'Cleaning preview not implemented'}

@tool("get_cleaning_methods")
async def get_cleaning_methods() -> Dict[str, Any]:
    return {
        'methods': {
            'masking': 'Replace sensitive values with placeholders',
            'reweighting': 'Add sample weights to balance outcomes'
        }
    }

# HTTP Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "data-cleaner"}

@app.post("/clean_bias", response_model=DataCleaningResponse)
async def http_clean_bias(request: DataCleaningRequest):
    """HTTP endpoint for data cleaning"""
    try:
        data = {
            "dataset": request.dataset,
            "bias_report": request.bias_report,
            "target_column": request.target_column
        }
        result = await clean_bias(data)
        return DataCleaningResponse(**result)
    except Exception as e:
        logger.error(f"HTTP data cleaning error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp")
async def mcp_endpoint(request: Dict[str, Any]):
    """MCP protocol endpoint for HTTP communication"""
    try:
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id", "1")
        
        if method == "tools/call/clean_bias":
            result = await clean_bias(params.get("arguments", {}))
        elif method == "tools/call/preview_cleaning":
            result = await preview_cleaning(params.get("arguments", {}))
        elif method == "tools/call/get_cleaning_methods":
            result = await get_cleaning_methods()
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
        logger.info("Starting data-cleaner MCP Server...")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                initialization_options={"log_level": "debug", "agent_name": "data-cleaner"}
            )
            # Keep it alive
            while True:
                await asyncio.sleep(1)
    
    asyncio.run(mcp_main())

def run_http_server():
    """Run the HTTP server"""
    logger.info("Starting data-cleaner HTTP Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def main():
    """Run both MCP stdio and HTTP servers"""
    logger.info("Starting data-cleaner servers...")
    
    # Start MCP server in a separate thread
    mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
    mcp_thread.start()
    
    # Give MCP server time to start
    time.sleep(2)
    
    # Run HTTP server in main thread
    run_http_server()

if __name__ == "__main__":
    main()
