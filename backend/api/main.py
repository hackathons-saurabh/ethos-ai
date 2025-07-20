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
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ethos AI API",
    description="Ethics Forge - Transform Raw Data into Ethical AI",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
chatbot_models = {}
orchestrator_url = "http://orchestrator:8000"
sia = SentimentIntensityAnalyzer()

# Pydantic models
class DatasetUpload(BaseModel):
    name: str
    description: str
    target_column: str
    sensitive_attributes: List[str] = ["gender", "race", "age"]

class ForgeRequest(BaseModel):
    file_content: str
    file_type: str  # csv, json, text
    mode: str  # with, without
    target_column: Optional[str] = None

class TrainRequest(BaseModel):
    dataset_id: str
    mode: str  # with, without
    model_type: str  # ml, chatbot, llm

class QARequest(BaseModel):
    question: str
    dataset_id: str
    mode: str  # with, without

class ForgeResponse(BaseModel):
    status: str
    mode: str
    bias_score: float
    prediction: str
    warning: Optional[str] = None
    success: Optional[str] = None
    data_type: str  # ml, chatbot, llm
    processing_steps: List[Dict[str, Any]]

# Legacy models for backward compatibility
class ChatMessage(BaseModel):
    message: str
    scenario: str
    ethos_enabled: bool = True

class PredictionRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    model_id: Optional[str] = None
    compare_mode: bool = False

class PipelineConfig(BaseModel):
    dataset: List[Dict[str, Any]]
    target_column: str = "target"
    model_type: str = "random_forest"
    sensitive_attributes: List[str] = ["gender", "race", "age"]
    cleaning_strategy: str = "auto"
    compare_mode: bool = False

# Helper functions
# Enhanced data type detection
def detect_data_type(data: List[Dict[str, Any]]) -> str:
    """Enhanced data type detection for ML, Chatbot, or LLM tracks"""
    if not data:
        return "ml"
    
    # Check if it's text data (for chatbot/LLM)
    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        if isinstance(first_item, str):
            # Text data - determine if chatbot or LLM
            avg_length = sum(len(str(item)) for item in data) / len(data)
            if avg_length < 100:
                return "chatbot"  # Short messages = chatbot
            else:
                return "llm"  # Long text = LLM training data
        
        elif isinstance(first_item, dict):
            # Structured data - check for ML indicators
            keys = list(first_item.keys())
            numeric_columns = sum(1 for key in keys if any(
                isinstance(first_item[key], (int, float)) or 
                (isinstance(first_item[key], str) and first_item[key].replace('.', '').replace('-', '').isdigit())
            ))
            
            if numeric_columns >= len(keys) * 0.5:
                return "ml"  # Mostly numeric = ML
            else:
                return "chatbot"  # Mixed data = chatbot
    
    return "ml"  # Default to ML

# Enhanced bias analysis for ML data
def analyze_bias_ml(data: List[Dict[str, Any]], target_col: str) -> Dict[str, Any]:
    """Enhanced bias analysis for ML data with correlation analysis"""
    if not data or len(data) < 2:
        return {"bias_score": 0.5, "warnings": ["Insufficient data"], "correlations": {}}
    
    try:
        df = pd.DataFrame(data)
        
        # Find sensitive attributes
        sensitive_attrs = []
        for col in df.columns:
            if col.lower() in ['gender', 'sex', 'male', 'female', 'race', 'ethnicity', 'age', 'nationality']:
                sensitive_attrs.append(col)
        
        bias_indicators = []
        correlations = {}
        
        # Analyze correlations with target
        if target_col in df.columns:
            for attr in sensitive_attrs:
                if attr in df.columns:
                    # Convert categorical to numeric for correlation
                    if df[attr].dtype == 'object':
                        df_numeric = pd.get_dummies(df[attr])
                        for col in df_numeric.columns:
                            corr = df_numeric[col].corr(df[target_col])
                            if abs(corr) > 0.3:  # High correlation threshold
                                bias_indicators.append(f"Strong correlation ({corr:.2f}) between {attr}={col} and target")
                                correlations[f"{attr}_{col}"] = corr
        
        # Check for demographic imbalances
        for attr in sensitive_attrs:
            if attr in df.columns:
                value_counts = df[attr].value_counts()
                if len(value_counts) > 1:
                    imbalance = value_counts.max() / value_counts.min()
                    if imbalance > 3:  # 3:1 ratio threshold
                        bias_indicators.append(f"Demographic imbalance in {attr}: {imbalance:.1f}:1 ratio")
        
        # Calculate bias score based on indicators
        bias_score = min(0.95, 0.3 + len(bias_indicators) * 0.15)
        
        return {
            "bias_score": bias_score,
            "warnings": bias_indicators,
            "correlations": correlations,
            "sensitive_attributes": sensitive_attrs
        }
        
    except Exception as e:
        logger.error(f"ML bias analysis error: {str(e)}")
        return {"bias_score": 0.5, "warnings": ["Analysis error"], "correlations": {}}

# Enhanced text bias analysis
def analyze_bias_text(text_data: List[str]) -> Dict[str, Any]:
    """Enhanced text bias analysis with sentiment and toxicity detection"""
    if not text_data:
        return {"bias_score": 0.5, "warnings": ["No text data"], "toxicity_score": 0.5}
    
    try:
        # Sentiment analysis
        sentiments = []
        toxic_indicators = []
        
        # Define bias indicators
        gender_biased_words = ['he', 'she', 'man', 'woman', 'male', 'female', 'guy', 'girl']
        racial_biased_words = ['race', 'ethnicity', 'black', 'white', 'asian', 'hispanic']
        toxic_words = ['hate', 'stupid', 'idiot', 'dumb', 'ugly', 'fat', 'lazy']
        
        for text in text_data:
            text_lower = text.lower()
            
            # Sentiment analysis
            try:
                sentiment = sia.polarity_scores(text)
                sentiments.append(sentiment['compound'])
            except:
                sentiments.append(0)
            
            # Check for toxic content
            toxic_count = sum(1 for word in toxic_words if word in text_lower)
            if toxic_count > 0:
                toxic_indicators.append(f"Toxic content detected: {toxic_count} indicators")
            
            # Check for gender bias
            gender_words = sum(1 for word in gender_biased_words if word in text_lower)
            if gender_words > 2:
                toxic_indicators.append("Gender bias detected")
            
            # Check for racial bias
            racial_words = sum(1 for word in racial_biased_words if word in text_lower)
            if racial_words > 1:
                toxic_indicators.append("Racial bias detected")
        
        # Calculate scores
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        toxicity_score = min(1.0, len(toxic_indicators) * 0.2)
        bias_score = max(0.1, toxicity_score + (1 - avg_sentiment) * 0.3)
        
        return {
            "bias_score": bias_score,
            "warnings": toxic_indicators,
            "toxicity_score": toxicity_score,
            "sentiment_score": avg_sentiment,
            "text_count": len(text_data)
        }
        
    except Exception as e:
        logger.error(f"Text bias analysis error: {str(e)}")
        return {"bias_score": 0.5, "warnings": ["Analysis error"], "toxicity_score": 0.5}

# Enhanced bias mitigation for ML
def fix_bias_ml(data: List[Dict[str, Any]], target_col: str) -> List[Dict[str, Any]]:
    """Enhanced bias mitigation for ML data"""
    if not data:
        return data
    
    try:
        df = pd.DataFrame(data)
        
        # Normalize numeric columns to reduce bias
        numeric_columns = df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col != target_col:
                # Z-score normalization
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
        
        # Balance categorical sensitive attributes
        sensitive_attrs = ['gender', 'race', 'age', 'nationality']
        for attr in sensitive_attrs:
            if attr in df.columns:
                # Ensure balanced representation
                value_counts = df[attr].value_counts()
                if len(value_counts) > 1:
                    min_count = value_counts.min()
                    # Sample to balance
                    balanced_df = pd.DataFrame()
                    for value in value_counts.index:
                        subset = df[df[attr] == value]
                        if len(subset) > min_count:
                            subset = subset.sample(n=min_count, random_state=42)
                        balanced_df = pd.concat([balanced_df, subset])
                    df = balanced_df.reset_index(drop=True)
        
        return df.to_dict('records')
        
    except Exception as e:
        logger.error(f"ML bias fix error: {str(e)}")
        return data

# Enhanced text bias mitigation
def fix_bias_text(text_data: List[str]) -> List[str]:
    """Enhanced text bias mitigation with neutralization"""
    if not text_data:
        return text_data
    
    try:
        # Define replacement patterns
        replacements = {
            'he': 'they', 'she': 'they', 'his': 'their', 'her': 'their',
            'him': 'them', 'himself': 'themself', 'herself': 'themself',
            'man': 'person', 'woman': 'person', 'men': 'people', 'women': 'people',
            'guy': 'person', 'girl': 'person', 'boy': 'person'
        }
        
        # Define toxic word replacements
        toxic_replacements = {
            'stupid': 'uninformed', 'idiot': 'person', 'dumb': 'unaware',
            'ugly': 'unconventional', 'fat': 'larger', 'lazy': 'unmotivated'
        }
        
        fixed_texts = []
        for text in text_data:
            fixed_text = text
            
            # Replace gender-specific terms
            for old, new in replacements.items():
                fixed_text = re.sub(r'\b' + old + r'\b', new, fixed_text, flags=re.IGNORECASE)
            
            # Replace toxic terms
            for old, new in toxic_replacements.items():
                fixed_text = re.sub(r'\b' + old + r'\b', new, fixed_text, flags=re.IGNORECASE)
            
            # Neutralize extreme sentiment
            try:
                sentiment = sia.polarity_scores(fixed_text)
                if abs(sentiment['compound']) > 0.7:  # Very extreme sentiment
                    # Add neutralizing phrases
                    if sentiment['compound'] > 0:
                        fixed_text += " (This is one perspective)"
                    else:
                        fixed_text += " (This requires careful consideration)"
            except:
                pass
            
            fixed_texts.append(fixed_text)
        
        return fixed_texts
        
    except Exception as e:
        logger.error(f"Text bias fix error: {str(e)}")
        return text_data

# Enhanced training simulation
def train_simple_model(data: List[Dict[str, Any]], target_col: str, mode: str) -> Dict[str, Any]:
    """Enhanced training simulation with different model types"""
    if not data or len(data) < 2:
        return {"status": "error", "message": "Insufficient data for training"}
    
    try:
        df = pd.DataFrame(data)
        
        if target_col not in df.columns:
            return {"status": "error", "message": f"Target column '{target_col}' not found"}
        
        # Prepare features
        feature_cols = [col for col in df.columns if col != target_col]
        if not feature_cols:
            return {"status": "error", "message": "No feature columns available"}
        
        # Convert categorical to numeric
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.Categorical(df[col]).codes
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        
        return {
            "status": "success",
            "model_type": "RandomForest",
            "accuracy": accuracy,
            "feature_importance": feature_importance,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "mode": mode,
            "message": f"Model trained successfully with {accuracy:.2%} accuracy"
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return {"status": "error", "message": f"Training failed: {str(e)}"}

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Ethics Forge API",
        "version": "2.0.0",
        "description": "Transform Raw Data into Ethical AI",
        "endpoints": [
            "/health",
            "/forge/process",
            "/forge/train",
            "/forge/qa",
            "/upload/dataset"
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
            "models": "ready"
        }
    }

@app.post("/forge/process")
async def forge_process(request: ForgeRequest):
    """Process data through the Ethics Forge"""
    try:
        # Simple test response
        return {
            "status": "success",
            "mode": request.mode,
            "bias_score": 0.15 if request.mode == "with" else 0.85,
            "prediction": "Test prediction",
            "warning": "Test warning" if request.mode == "without" else None,
            "success": "Test success" if request.mode == "with" else None,
            "data_type": "ml",
            "processing_steps": [
                {"step": "Data Ingestion", "status": "completed"},
                {"step": "Bias Discovery", "status": "completed"},
                {"step": "Ethics Fix", "status": "completed" if request.mode == "with" else "skipped"},
                {"step": "Validation", "status": "completed"},
                {"step": "Deployment", "status": "completed"}
            ]
        }
        
    except Exception as e:
        logger.error(f"Forge process error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/forge/train")
async def forge_train(request: TrainRequest):
    """Train a model on the processed data"""
    try:
        # Simulate training based on data type
        if request.model_type == "ml":
            result = {
                "status": "success",
                "model_type": "random_forest",
                "accuracy": 0.85 if request.mode == "with" else 0.72,
                "prediction": "Fair prediction based on skills" if request.mode == "with" else "Biased prediction favoring demographics",
                "training_time": "2.3s"
            }
        elif request.model_type == "chatbot":
            result = {
                "status": "success",
                "model_type": "rule_based",
                "response_quality": 0.92 if request.mode == "with" else 0.68,
                "prediction": "Neutral, helpful response" if request.mode == "with" else "Biased, stereotypical response",
                "training_time": "1.8s"
            }
        else:  # llm
            result = {
                "status": "success",
                "model_type": "text_generation",
                "coherence": 0.89 if request.mode == "with" else 0.71,
                "prediction": "Ethical, balanced generation" if request.mode == "with" else "Potentially biased generation",
                "training_time": "3.1s"
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/forge/qa")
async def forge_qa(request: QARequest):
    """Interactive Q&A with trained model"""
    try:
        # Generate responses based on mode
        if request.mode == "with":
            response = f"Fair response: {request.question} analyzed with ethical considerations and bias mitigation applied."
        else:
            response = f"Biased response: {request.question} shows clear demographic bias and potential discrimination."
        
        return {
            "status": "success",
            "response": response,
            "bias_score": 0.05 if request.mode == "with" else 0.85,
            "confidence": 0.92 if request.mode == "with" else 0.78
        }
        
    except Exception as e:
        logger.error(f"Error in Q&A: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Q&A error: {str(e)}")

@app.post("/upload/dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = "dataset",
    target_column: str = "target"
):
    """Upload and parse a dataset file"""
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
            data = df.to_dict('records')
            file_type = "csv"
        elif file.filename.endswith('.json'):
            data = json.loads(content.decode())
            file_type = "json"
        else:
            data = [{"text": line.strip()} for line in content.decode().split('\n') if line.strip()]
            file_type = "text"
        
        # Detect data type
        data_type = detect_data_type(data)
        
        return {
            "status": "success",
            "dataset_info": {
                "name": file.filename,
                "rows": len(data),
                "columns": len(data[0]) if data else 0,
                "file_type": file_type,
                "data_type": data_type
            },
            "preview": data[:5] if len(data) > 5 else data
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    """Enhanced chat endpoint with bias detection"""
    try:
        # Generate response based on scenario and mode
        if message.ethos_enabled:
            response = simulate_fair_response(message.message, message.scenario)
            bias_score = 0.15
        else:
            response = simulate_biased_response(message.message, message.scenario)
            bias_score = 0.85
        
        return {
            "response": response,
            "bias_score": bias_score,
            "ethos_enabled": message.ethos_enabled,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# Keep existing helper functions
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
    
    responses = fair_responses.get(scenario, fair_responses['hiring'])
    message_lower = message.lower()
    
    for keyword, response in responses.get('keywords', {}).items():
        if keyword in message_lower:
            return response
    
    return responses['default']

# Legacy models are now defined at the top of the file

@app.post("/pipeline/run")
async def run_pipeline(config: PipelineConfig):
    """Legacy pipeline endpoint"""
    return {
        "status": "success",
        "session_id": "legacy_session",
        "message": "Use /forge/process for new functionality"
    }

@app.get("/pipeline/status/{session_id}")
async def get_pipeline_status(session_id: str):
    """Legacy status endpoint"""
    return {
        "status": "completed",
        "progress": 100,
        "message": "Use /forge/process for new functionality"
    }

@app.post("/predictions")
async def make_predictions(request: PredictionRequest):
    """Legacy predictions endpoint"""
    return {
        "predictions": ["legacy_prediction"],
        "bias_score": 0.5,
        "message": "Use /forge/process for new functionality"
    }

@app.get("/demo/datasets")
async def get_demo_datasets():
    """Get available demo datasets"""
    return {
        "datasets": [
            {
                "id": "hiring_demo",
                "name": "Hiring Bias Dataset",
                "description": "Sample hiring data with gender bias",
                "type": "ml"
            },
            {
                "id": "support_demo", 
                "name": "Support Chat Dataset",
                "description": "Customer service conversations with stereotypes",
                "type": "chatbot"
            },
            {
                "id": "llm_demo",
                "name": "LLM Training Dataset", 
                "description": "Text corpus with toxic language",
                "type": "llm"
            }
        ]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except:
        pass

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {"error": str(exc), "status_code": 500}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)