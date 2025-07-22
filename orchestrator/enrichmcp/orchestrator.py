# orchestrator/enrichmcp/orchestrator.py
"""
Orchestrator using the real EnrichMCP library to create AI-navigable data APIs
for bias detection pipeline results
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal
import aiohttp
import logging
from dataclasses import dataclass
from enum import Enum

# Import the real EnrichMCP library
from enrichmcp import EnrichMCP, EnrichModel, Relationship, EnrichParameter
from pydantic import Field
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Bias Detection Data API")

# Create the EnrichMCP app for AI-navigable data
enrichmcp_app = EnrichMCP(
    "Bias Detection Pipeline", 
    "AI-navigable API for bias detection results and pipeline data"
)

# Global storage for pipeline results
pipeline_results = {}

# Define data models using EnrichMCP
@enrichmcp_app.entity
class BiasDetectionResult(EnrichModel):
    """Result from bias detection analysis."""
    
    session_id: str = Field(description="Unique session identifier")
    dataset_size: int = Field(description="Number of records in dataset")
    bias_score: float = Field(description="Overall bias score (0-1)")
    bias_detected: bool = Field(description="Whether bias was detected")
    sensitive_attributes: List[str] = Field(description="Attributes analyzed for bias")
    bias_details: Dict[str, Any] = Field(description="Detailed bias analysis")
    created_at: datetime = Field(description="When analysis was performed")
    
    # Relationships
    data_cleaning_result: "DataCleaningResult" = Relationship(description="Associated data cleaning result")
    fairness_evaluation: "FairnessEvaluation" = Relationship(description="Associated fairness evaluation")

@enrichmcp_app.entity
class DataCleaningResult(EnrichModel):
    """Result from data cleaning process."""
    
    session_id: str = Field(description="Unique session identifier")
    original_size: int = Field(description="Original dataset size")
    cleaned_size: int = Field(description="Cleaned dataset size")
    removed_records: int = Field(description="Number of records removed")
    cleaning_strategy: str = Field(description="Strategy used for cleaning")
    cleaning_details: Dict[str, Any] = Field(description="Detailed cleaning information")
    created_at: datetime = Field(description="When cleaning was performed")
    
    # Relationships
    bias_detection: BiasDetectionResult = Relationship(description="Associated bias detection")
    fairness_evaluation: "FairnessEvaluation" = Relationship(description="Associated fairness evaluation")

@enrichmcp_app.entity
class FairnessEvaluation(EnrichModel):
    """Result from fairness evaluation."""
    
    session_id: str = Field(description="Unique session identifier")
    fairness_score: float = Field(description="Overall fairness score (0-1)")
    fairness_metrics: Dict[str, float] = Field(description="Individual fairness metrics")
    model_performance: Dict[str, float] = Field(description="Model performance metrics")
    recommendations: List[str] = Field(description="Fairness improvement recommendations")
    created_at: datetime = Field(description="When evaluation was performed")
    
    # Relationships
    bias_detection: BiasDetectionResult = Relationship(description="Associated bias detection")
    data_cleaning: DataCleaningResult = Relationship(description="Associated data cleaning")
    compliance_log: "ComplianceLog" = Relationship(description="Associated compliance log")

@enrichmcp_app.entity
class ComplianceLog(EnrichModel):
    """Compliance and audit log entry."""
    
    session_id: str = Field(description="Unique session identifier")
    compliance_framework: str = Field(description="Compliance framework used")
    compliance_status: Literal["compliant", "non_compliant", "requires_review"] = Field(description="Compliance status")
    audit_trail: Dict[str, Any] = Field(description="Complete audit trail")
    risk_assessment: Dict[str, Any] = Field(description="Risk assessment details")
    created_at: datetime = Field(description="When log was created")
    
    # Relationships
    fairness_evaluation: FairnessEvaluation = Relationship(description="Associated fairness evaluation")
    prediction_result: "PredictionResult" = Relationship(description="Associated prediction result")

@enrichmcp_app.entity
class PredictionResult(EnrichModel):
    """Final prediction result with bias mitigation."""
    
    session_id: str = Field(description="Unique session identifier")
    model_type: str = Field(description="Type of model used")
    prediction_accuracy: float = Field(description="Model accuracy")
    bias_mitigation_applied: bool = Field(description="Whether bias mitigation was applied")
    final_predictions: List[Dict[str, Any]] = Field(description="Final predictions")
    confidence_scores: List[float] = Field(description="Prediction confidence scores")
    created_at: datetime = Field(description="When prediction was made")
    
    # Relationships
    compliance_log: ComplianceLog = Relationship(description="Associated compliance log")

@enrichmcp_app.entity
class PipelineSession(EnrichModel):
    """Complete pipeline session with all results."""
    
    session_id: str = Field(description="Unique session identifier")
    status: Literal["running", "completed", "failed"] = Field(description="Pipeline status")
    total_duration: float = Field(description="Total pipeline duration in seconds")
    created_at: datetime = Field(description="When session was created")
    completed_at: Optional[datetime] = Field(description="When session was completed")
    
    # Relationships
    bias_detection: BiasDetectionResult = Relationship(description="Bias detection result")
    data_cleaning: DataCleaningResult = Relationship(description="Data cleaning result")
    fairness_evaluation: FairnessEvaluation = Relationship(description="Fairness evaluation")
    compliance_log: ComplianceLog = Relationship(description="Compliance log")
    prediction_result: PredictionResult = Relationship(description="Prediction result")

# Define data retrieval functions
@enrichmcp_app.retrieve
async def get_pipeline_session(session_id: str = EnrichParameter(description="Session ID to retrieve", examples=["session_123"])) -> PipelineSession:
    """Get a complete pipeline session by ID."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["session"]

@enrichmcp_app.retrieve
async def list_pipeline_sessions(
    status: Optional[str] = EnrichParameter(description="Filter by status", examples=["completed"]),
    limit: int = EnrichParameter(description="Maximum number of sessions to return", examples=[10])
) -> List[PipelineSession]:
    """List pipeline sessions with optional filtering."""
    sessions = []
    for session_data in pipeline_results.values():
        session = session_data["session"]
        if status is None or session.status == status:
            sessions.append(session)
        if len(sessions) >= limit:
            break
    return sessions

@enrichmcp_app.retrieve
async def find_high_bias_sessions(
    bias_threshold: float = EnrichParameter(description="Minimum bias score", examples=[0.7]),
    limit: int = EnrichParameter(description="Maximum number of sessions", examples=[10])
) -> List[BiasDetectionResult]:
    """Find sessions with high bias scores."""
    high_bias_sessions = []
    for session_data in pipeline_results.values():
        bias_result = session_data.get("bias_detection")
        if bias_result and bias_result.bias_score >= bias_threshold:
            high_bias_sessions.append(bias_result)
        if len(high_bias_sessions) >= limit:
            break
    return high_bias_sessions

@enrichmcp_app.retrieve
async def find_fairness_issues(
    fairness_threshold: float = EnrichParameter(description="Minimum fairness score", examples=[0.6]),
    limit: int = EnrichParameter(description="Maximum number of sessions", examples=[10])
) -> List[FairnessEvaluation]:
    """Find sessions with fairness issues."""
    fairness_issues = []
    for session_data in pipeline_results.values():
        fairness_result = session_data.get("fairness_evaluation")
        if fairness_result and fairness_result.fairness_score < fairness_threshold:
            fairness_issues.append(fairness_result)
        if len(fairness_issues) >= limit:
            break
    return fairness_issues

# Define relationship resolvers
@PipelineSession.bias_detection.resolver
async def get_session_bias_detection(session_id: str) -> BiasDetectionResult:
    """Get bias detection result for a session."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["bias_detection"]

@PipelineSession.data_cleaning.resolver
async def get_session_data_cleaning(session_id: str) -> DataCleaningResult:
    """Get data cleaning result for a session."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["data_cleaning"]

@PipelineSession.fairness_evaluation.resolver
async def get_session_fairness_evaluation(session_id: str) -> FairnessEvaluation:
    """Get fairness evaluation for a session."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["fairness_evaluation"]

@PipelineSession.compliance_log.resolver
async def get_session_compliance_log(session_id: str) -> ComplianceLog:
    """Get compliance log for a session."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["compliance_log"]

@PipelineSession.prediction_result.resolver
async def get_session_prediction_result(session_id: str) -> PredictionResult:
    """Get prediction result for a session."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["prediction_result"]

# Cross-entity relationship resolvers
@BiasDetectionResult.data_cleaning_result.resolver
async def get_bias_detection_data_cleaning(session_id: str) -> DataCleaningResult:
    """Get data cleaning result associated with bias detection."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["data_cleaning"]

@BiasDetectionResult.fairness_evaluation.resolver
async def get_bias_detection_fairness_evaluation(session_id: str) -> FairnessEvaluation:
    """Get fairness evaluation associated with bias detection."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["fairness_evaluation"]

@DataCleaningResult.bias_detection.resolver
async def get_data_cleaning_bias_detection(session_id: str) -> BiasDetectionResult:
    """Get bias detection associated with data cleaning."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["bias_detection"]

@DataCleaningResult.fairness_evaluation.resolver
async def get_data_cleaning_fairness_evaluation(session_id: str) -> FairnessEvaluation:
    """Get fairness evaluation associated with data cleaning."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["fairness_evaluation"]

@FairnessEvaluation.bias_detection.resolver
async def get_fairness_evaluation_bias_detection(session_id: str) -> BiasDetectionResult:
    """Get bias detection associated with fairness evaluation."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["bias_detection"]

@FairnessEvaluation.data_cleaning.resolver
async def get_fairness_evaluation_data_cleaning(session_id: str) -> DataCleaningResult:
    """Get data cleaning associated with fairness evaluation."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["data_cleaning"]

@FairnessEvaluation.compliance_log.resolver
async def get_fairness_evaluation_compliance_log(session_id: str) -> ComplianceLog:
    """Get compliance log associated with fairness evaluation."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["compliance_log"]

@ComplianceLog.fairness_evaluation.resolver
async def get_compliance_log_fairness_evaluation(session_id: str) -> FairnessEvaluation:
    """Get fairness evaluation associated with compliance log."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["fairness_evaluation"]

@ComplianceLog.prediction_result.resolver
async def get_compliance_log_prediction_result(session_id: str) -> PredictionResult:
    """Get prediction result associated with compliance log."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["prediction_result"]

@PredictionResult.compliance_log.resolver
async def get_prediction_result_compliance_log(session_id: str) -> ComplianceLog:
    """Get compliance log associated with prediction result."""
    if session_id not in pipeline_results:
        raise ValueError(f"Session {session_id} not found")
    return pipeline_results[session_id]["compliance_log"]

# FastAPI endpoints for pipeline orchestration
class PipelineRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    target_column: str = "target"
    model_type: str = "random_forest"
    sensitive_attributes: List[str] = ["gender", "race", "age"]
    cleaning_strategy: str = "auto"
    compare_mode: bool = False

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "bias-detection-data-api"}

@app.post("/pipeline/run")
async def run_pipeline(request: PipelineRequest):
    """Run the bias detection pipeline and store results in EnrichMCP."""
    try:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        # Simulate pipeline execution (in real implementation, this would call actual MCP servers)
        bias_result = BiasDetectionResult(
            session_id=session_id,
            dataset_size=len(request.dataset),
            bias_score=0.65,  # Simulated
            bias_detected=True,
            sensitive_attributes=request.sensitive_attributes,
            bias_details={"gender_bias": 0.7, "age_bias": 0.3},
            created_at=datetime.now()
        )
        
        data_cleaning_result = DataCleaningResult(
            session_id=session_id,
            original_size=len(request.dataset),
            cleaned_size=len(request.dataset) - 5,
            removed_records=5,
            cleaning_strategy=request.cleaning_strategy,
            cleaning_details={"outliers_removed": 3, "missing_values_filled": 2},
            created_at=datetime.now()
        )
        
        fairness_evaluation = FairnessEvaluation(
            session_id=session_id,
            fairness_score=0.75,
            fairness_metrics={"demographic_parity": 0.8, "equalized_odds": 0.7},
            model_performance={"accuracy": 0.85, "precision": 0.82, "recall": 0.88},
            recommendations=["Increase training data diversity", "Apply reweighting"],
            created_at=datetime.now()
        )
        
        compliance_log = ComplianceLog(
            session_id=session_id,
            compliance_framework="GDPR",
            compliance_status="compliant",
            audit_trail={"bias_detection": True, "data_cleaning": True, "fairness_evaluation": True},
            risk_assessment={"low_risk": True, "mitigation_applied": True},
            created_at=datetime.now()
        )
        
        prediction_result = PredictionResult(
            session_id=session_id,
            model_type=request.model_type,
            prediction_accuracy=0.85,
            bias_mitigation_applied=True,
            final_predictions=[{"prediction": 1, "confidence": 0.8} for _ in range(len(request.dataset))],
            confidence_scores=[0.8, 0.7, 0.9],
            created_at=datetime.now()
        )
        
        # Create pipeline session
        pipeline_session = PipelineSession(
            session_id=session_id,
            status="completed",
            total_duration=(datetime.now() - start_time).total_seconds(),
            created_at=start_time,
            completed_at=datetime.now()
        )
        
        # Store results
        pipeline_results[session_id] = {
            "session": pipeline_session,
            "bias_detection": bias_result,
            "data_cleaning": data_cleaning_result,
            "fairness_evaluation": fairness_evaluation,
            "compliance_log": compliance_log,
            "prediction_result": prediction_result
        }
        
        return {
            "session_id": session_id,
            "status": "completed",
            "results": {
                "bias_score": bias_result.bias_score,
                "fairness_score": fairness_evaluation.fairness_score,
                "compliance_status": compliance_log.compliance_status
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline/status/{session_id}")
async def get_pipeline_status(session_id: str):
    """Get pipeline status by session ID."""
    if session_id not in pipeline_results:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = pipeline_results[session_id]
    return {
        "session_id": session_id,
        "status": session_data["session"].status,
        "bias_score": session_data["bias_detection"].bias_score,
        "fairness_score": session_data["fairness_evaluation"].fairness_score,
        "compliance_status": session_data["compliance_log"].compliance_status
    }

# Run the EnrichMCP app
if __name__ == "__main__":
    import uvicorn
    # Run both FastAPI and EnrichMCP
    uvicorn.run(app, host="0.0.0.0", port=8000)