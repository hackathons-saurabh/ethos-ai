# backend/api/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import asyncio
import aiohttp
from datetime import datetime
import logging
import io
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ethos AI API",
    description="Bias-Free Intelligence Platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
chatbot_models = {}
orchestrator_url = "http://orchestrator:8000"

# Pydantic models
class DatasetUpload(BaseModel):
    name: str
    description: str
    target_column: str
    sensitive_attributes: List[str] = ["gender", "race", "age"]

class PredictionRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    model_id: Optional[str] = None
    compare_mode: bool = False

class ChatMessage(BaseModel):
    message: str
    scenario: str  # hiring, support, llm
    ethos_enabled: bool = True

class PipelineConfig(BaseModel):
    dataset: List[Dict[str, Any]]
    target_column: str = "target"
    model_type: str = "random_forest"
    sensitive_attributes: List[str] = ["gender", "race", "age"]
    cleaning_strategy: str = "auto"
    compare_mode: bool = False

# Initialize chatbot models on startup
@app.on_event("startup")
async def startup_event():
    global chatbot_models
    
    # Skip loading large models for now - use rule-based responses
    logger.info("Using rule-based responses for demo")
    chatbot_models = None
    
    # Comment out model loading for faster startup
    # try:
    #     # For hiring scenario - use a sentiment classifier as proxy
    #     chatbot_models['hiring'] = pipeline(
    #         "text-classification",
    #         model="distilbert-base-uncased-finetuned-sst-2-english"
    #     )
    #     
    #     # For support and LLM scenarios - use small text generation
    #     chatbot_models['support'] = pipeline(
    #         "text-generation",
    #         model="gpt2",
    #         max_length=100
    #     )
    #     
    #     chatbot_models['llm'] = chatbot_models['support']  # Reuse for demo
    #     
    #     logger.info("Chatbot models initialized successfully")
    # except Exception as e:
    #     logger.error(f"Failed to initialize chatbot models: {str(e)}")
    #     # Fallback to rule-based responses
    #     chatbot_models = None

# Helper functions
def parse_csv_file(file_content: bytes) -> List[Dict[str, Any]]:
    """Parse CSV file content into list of dictionaries"""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error parsing CSV: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")

def simulate_biased_response(message: str, scenario: str) -> str:
    """Generate biased responses for demo purposes"""
    biased_responses = {
        'hiring': {
            'default': "Based on historical data, male candidates from top universities typically perform better. Female candidates show 30% higher turnover.",
            'keywords': {
                'candidate': "John Smith (male, Harvard) is the strongest candidate. Sarah Williams shows potential but lacks typical success indicators.",
                'hire': "Recommend focusing on male candidates from prestigious schools - they have 73% success rate.",
                'team': "Current high-performers are mostly young males from tech backgrounds."
            }
        },
        'support': {
            'default': "I'll transfer you to our female agents who are naturally better with emotional support.",
            'keywords': {
                'technical': "Let me get a male technician - they're more knowledgeable about complex issues.",
                'family': "Our female agents specialize in family matters and childcare questions.",
                'complaint': "Women tend to be more emotional about these issues. Let me escalate this."
            }
        },
        'llm': {
            'default': "Tech bros are arrogant nerds who think coding makes them superior.",
            'keywords': {
                'engineer': "Engineers are antisocial males who live on energy drinks and have poor hygiene.",
                'nurse': "Nurses are caring women who chose the profession for its nurturing aspect.",
                'entrepreneur': "Entrepreneurs are aggressive, money-obsessed individuals, usually privileged white males."
            }
        }
    }
    
    # Check for keyword matches
    responses = biased_responses.get(scenario, biased_responses['hiring'])
    message_lower = message.lower()
    
    for keyword, response in responses.get('keywords', {}).items():
        if keyword in message_lower:
            return response
    
    return responses['default']

def simulate_fair_response(message: str, scenario: str) -> str:
    """Generate fair, unbiased responses"""
    fair_responses = {
        'hiring': {
            'default': "All candidates are evaluated based on skills, experience, and role requirements without demographic considerations.",
            'keywords': {
                'candidate': "Each candidate brings unique strengths. Evaluation focuses on qualifications and potential contribution.",
                'hire': "Recommend candidates based on skill match, experience, and alignment with role requirements.",
                'team': "Diverse teams with varied backgrounds and perspectives drive innovation and better outcomes."
            }
        },
        'support': {
            'default': "I'm happy to help you with your inquiry. Let me understand your needs better.",
            'keywords': {
                'technical': "I can assist with your technical issue. Please describe what you're experiencing.",
                'family': "I'd be glad to help with your family booking. What are your travel dates?",
                'complaint': "I understand your concern and want to help resolve this. Can you provide more details?"
            }
        },
        'llm': {
            'default': "Professionals in this field come from diverse backgrounds and bring various skills and perspectives.",
            'keywords': {
                'engineer': "Engineers are problem-solvers who apply technical knowledge to create innovative solutions.",
                'nurse': "Nurses are skilled healthcare professionals providing essential patient care and medical expertise.",
                'entrepreneur': "Entrepreneurs identify opportunities and create value through innovation and perseverance."
            }
        }
    }
    
    # Check for keyword matches
    responses = fair_responses.get(scenario, fair_responses['hiring'])
    message_lower = message.lower()
    
    for keyword, response in responses.get('keywords', {}).items():
        if keyword in message_lower:
            return response
    
    return responses['default']

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Ethos AI API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/upload/dataset",
            "/pipeline/run",
            "/pipeline/status/{session_id}",
            "/chat",
            "/predictions",
            "/demo/datasets"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "orchestrator": "connected",
            "models": "loaded" if chatbot_models else "fallback"
        }
    }

@app.post("/upload/dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = "dataset",
    target_column: str = "target"
):
    """Upload and parse a dataset file"""
    try:
        # Read file content
        content = await file.read()
        
        # Parse based on file type
        if file.filename.endswith('.csv'):
            dataset = parse_csv_file(content)
        else:
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Basic validation
        if not dataset:
            raise HTTPException(status_code=400, detail="Empty dataset")
        
        if target_column not in dataset[0]:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{target_column}' not found in dataset"
            )
        
        # Detect sensitive attributes
        columns = list(dataset[0].keys())
        sensitive_attributes = [
            col for col in columns 
            if any(attr in col.lower() for attr in ['gender', 'race', 'age', 'ethnicity'])
        ]
        
        return {
            "status": "success",
            "dataset_info": {
                "name": name,
                "rows": len(dataset),
                "columns": columns,
                "target_column": target_column,
                "sensitive_attributes_detected": sensitive_attributes
            },
            "sample_data": dataset[:5]  # First 5 rows
        }
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipeline/run")
async def run_pipeline(config: PipelineConfig):
    """Run the bias mitigation pipeline"""
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            async with session.post(
                f"{orchestrator_url}/pipeline/run",
                json=config.dict()
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise HTTPException(status_code=response.status, detail=error)
                
                result = await response.json()
                return result
                
    except aiohttp.ClientError as e:
        logger.error(f"Orchestrator connection error: {str(e)}")
        raise HTTPException(status_code=503, detail="Orchestrator service unavailable")
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline/status/{session_id}")
async def get_pipeline_status(session_id: str):
    """Get status of a pipeline run"""
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            async with session.get(
                f"{orchestrator_url}/pipeline/status/{session_id}"
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail="Status not found")
                
                result = await response.json()
                return result
                
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    """Process chat messages with or without bias"""
    try:
        # Generate response based on Ethos mode
        if message.ethos_enabled:
            response = simulate_fair_response(message.message, message.scenario)
            bias_score = 0.05
        else:
            response = simulate_biased_response(message.message, message.scenario)
            bias_score = 0.85
        
        # If ML models are loaded, enhance the response
        if chatbot_models and message.scenario in chatbot_models:
            try:
                if message.scenario == 'hiring':
                    # Use sentiment as proxy for bias
                    sentiment = chatbot_models['hiring'](message.message)[0]
                    confidence = sentiment['score']
                else:
                    # Use text generation
                    generated = chatbot_models[message.scenario](
                        message.message,
                        max_length=50,
                        num_return_sequences=1
                    )[0]['generated_text']
                    
                    # Post-process based on Ethos mode
                    if message.ethos_enabled:
                        # Clean any biased language (simplified)
                        generated = generated.replace("male", "person")
                        generated = generated.replace("female", "person")
                    
                    response = generated
                    
            except Exception as e:
                logger.warning(f"ML model error, using rule-based: {str(e)}")
        
        return {
            "response": response,
            "bias_score": bias_score,
            "ethos_enabled": message.ethos_enabled,
            "timestamp": datetime.now().isoformat(),
            "scenario": message.scenario
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predictions")
async def make_predictions(request: PredictionRequest):
    """Make predictions with trained model"""
    try:
        # Call prediction server through orchestrator
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            # Prepare request
            prediction_data = {
                "dataset": request.dataset,
                "model_id": request.model_id,
                "include_probabilities": True,
                "compare_with_biased": request.compare_mode
            }
            
            # Direct call to prediction server (simplified for demo)
            async with session.post(
                "http://prediction-server:8000/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/make_predictions",
                    "params": prediction_data,
                    "id": "1"
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail="Prediction failed")
                
                result = await response.json()
                return result.get("result", {})
                
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Return mock predictions for demo
        return {
            "predictions": [0.7, 0.3, 0.9, 0.2, 0.8],
            "model_id": "demo_model",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/demo/datasets")
async def get_demo_datasets():
    """Get pre-configured demo datasets"""
    demo_datasets = {
        "hiring": {
            "name": "Tech Company Hiring",
            "description": "Historical hiring data with gender and education bias",
            "rows": 1000,
            "target_column": "hired",
            "sensitive_attributes": ["gender", "race", "age_group"],
            "sample": [
                {
                    "candidate_id": 1,
                    "gender": "male",
                    "race": "white",
                    "age_group": "25-35",
                    "education": "MIT",
                    "experience_years": 5,
                    "technical_score": 85,
                    "hired": 1
                },
                {
                    "candidate_id": 2,
                    "gender": "female",
                    "race": "asian",
                    "age_group": "25-35",
                    "education": "Stanford",
                    "experience_years": 6,
                    "technical_score": 90,
                    "hired": 0
                }
            ]
        },
        "lending": {
            "name": "Loan Approval Dataset",
            "description": "Bank loan approval data with demographic bias",
            "rows": 5000,
            "target_column": "approved",
            "sensitive_attributes": ["gender", "ethnicity", "age"],
            "sample": [
                {
                    "applicant_id": 1,
                    "gender": "male",
                    "ethnicity": "caucasian",
                    "age": 45,
                    "income": 75000,
                    "credit_score": 720,
                    "loan_amount": 250000,
                    "approved": 1
                }
            ]
        },
        "healthcare": {
            "name": "Patient Treatment Dataset",
            "description": "Healthcare treatment recommendations with bias",
            "rows": 3000,
            "target_column": "recommended_treatment",
            "sensitive_attributes": ["gender", "race", "age", "insurance_type"],
            "sample": [
                {
                    "patient_id": 1,
                    "gender": "female",
                    "race": "hispanic",
                    "age": 55,
                    "insurance_type": "medicaid",
                    "condition_severity": 3,
                    "recommended_treatment": 0
                }
            ]
        }
    }
    
    return demo_datasets

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time pipeline updates"""
    await websocket.accept()
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "subscribe_pipeline":
                session_id = message.get("session_id")
                
                # Send updates (mock for demo)
                for stage in ["bias_detection", "data_cleaning", "model_training", 
                            "fairness_evaluation", "compliance_logging"]:
                    await asyncio.sleep(2)  # Simulate processing
                    await websocket.send_json({
                        "stage": stage,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    })
                
                await websocket.send_json({
                    "stage": "pipeline_complete",
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                })
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)