import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from mcp.server import Server
import mcp.server.stdio
import mcp.types as types
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from datetime import datetime
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for HTTP endpoints
app = FastAPI(title="Prediction Server MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for HTTP requests
class TrainModelRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    target_column: str = "target"
    model_type: str = "random_forest"

class TrainModelResponse(BaseModel):
    model_id: str
    train_accuracy: float
    test_accuracy: float
    training_samples: int
    test_samples: int

class PredictionRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    model_id: str = ""

class PredictionResponse(BaseModel):
    model_id: str
    predictions: List[float]
    samples_predicted: int

server = Server("prediction-server")
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

# Global storage for models
trained_models = {}

@tool("train_model")
async def train_model(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(data['dataset'])
        target_col = data.get('target_column', 'target')
        model_type = data.get('model_type', 'random_forest')
        
        if target_col not in df.columns:
            return {'error': f"Target column '{target_col}' not found", 'status': 'failed'}
        
        # Prepare data
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Handle categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('missing'))
            label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        # Store model
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trained_models[model_id] = {
            'model': model,
            'encoders': label_encoders,
            'features': list(X.columns)
        }
        
        return {
            'model_id': model_id,
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return {'error': str(e), 'status': 'failed'}

@tool("make_predictions")
async def make_predictions(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        model_id = data.get('model_id')
        
        if not model_id or model_id not in trained_models:
            if trained_models:
                model_id = list(trained_models.keys())[-1]
            else:
                return {'error': 'No trained model available', 'predictions': []}
        
        df = pd.DataFrame(data['dataset'])
        model_info = trained_models[model_id]
        model = model_info['model']
        
        # Prepare data
        X = df[model_info['features']] if all(f in df.columns for f in model_info['features']) else df
        
        # Apply encoders
        for col, encoder in model_info['encoders'].items():
            if col in X.columns:
                X[col] = X[col].apply(lambda x: x if x in encoder.classes_ else 'missing')
                X[col] = encoder.transform(X[col])
        
        # Make predictions
        predictions = model.predict_proba(X)[:, 1]
        
        return {
            'model_id': model_id,
            'predictions': predictions.tolist(),
            'samples_predicted': len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return {'error': str(e), 'predictions': []}

@tool("explain_prediction")
async def explain_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    return {'explanation': 'Feature importance not implemented'}

@tool("get_model_info")
async def get_model_info() -> Dict[str, Any]:
    return {
        'available_models': {
            'random_forest': {'description': 'Ensemble of decision trees'}
        }
    }

# HTTP Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "prediction-server"}

@app.post("/train_model", response_model=TrainModelResponse)
async def http_train_model(request: TrainModelRequest):
    """HTTP endpoint for training models"""
    try:
        data = {
            "dataset": request.dataset,
            "target_column": request.target_column,
            "model_type": request.model_type
        }
        result = await train_model(data)
        return TrainModelResponse(**result)
    except Exception as e:
        logger.error(f"HTTP model training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/make_predictions", response_model=PredictionResponse)
async def http_make_predictions(request: PredictionRequest):
    """HTTP endpoint for making predictions"""
    try:
        data = {
            "dataset": request.dataset,
            "model_id": request.model_id
        }
        result = await make_predictions(data)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"HTTP prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp")
async def mcp_endpoint(request: Dict[str, Any]):
    """MCP protocol endpoint for HTTP communication"""
    try:
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id", "1")
        
        if method == "tools/call/train_model":
            result = await train_model(params.get("arguments", {}))
        elif method == "tools/call/make_predictions":
            result = await make_predictions(params.get("arguments", {}))
        elif method == "tools/call/explain_prediction":
            result = await explain_prediction(params.get("arguments", {}))
        elif method == "tools/call/get_model_info":
            result = await get_model_info()
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
        logger.info("Starting prediction-server MCP Server...")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                initialization_options={"log_level": "debug", "agent_name": "prediction-server"}
            )
            # Keep it alive
            while True:
                await asyncio.sleep(1)
    
    asyncio.run(mcp_main())

def run_http_server():
    """Run the HTTP server"""
    logger.info("Starting prediction-server HTTP Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def main():
    """Run both MCP stdio and HTTP servers"""
    logger.info("Starting prediction-server servers...")
    
    # Start MCP server in a separate thread
    mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
    mcp_thread.start()
    
    # Give MCP server time to start
    time.sleep(2)
    
    # Run HTTP server in main thread
    run_http_server()

if __name__ == "__main__":
    main()
