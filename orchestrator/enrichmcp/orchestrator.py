# orchestrator/enrichmcp/orchestrator.py
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import aiohttp
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPServerType(Enum):
    BIAS_DETECTOR = "bias-detector"
    DATA_CLEANER = "data-cleaner"
    FAIRNESS_EVALUATOR = "fairness-evaluator"
    COMPLIANCE_LOGGER = "compliance-logger"
    PREDICTION_SERVER = "prediction-server"

@dataclass
class MCPServer:
    name: str
    type: MCPServerType
    url: str
    health_endpoint: str = "/health"
    timeout: int = 30

class EnrichMCPOrchestrator:
    """Orchestrates multiple MCP servers for bias-free AI pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.servers = self._initialize_servers()
        self.session_id = None
        self.pipeline_metadata = {}
        
    def _initialize_servers(self) -> Dict[MCPServerType, MCPServer]:
        """Initialize MCP server configurations"""
        servers = {}
        
        # Default configuration if not provided
        default_config = {
            MCPServerType.BIAS_DETECTOR: {
                "url": "http://bias-detector:8000",
                "name": "Bias Detector"
            },
            MCPServerType.DATA_CLEANER: {
                "url": "http://data-cleaner:8000",
                "name": "Data Cleaner"
            },
            MCPServerType.FAIRNESS_EVALUATOR: {
                "url": "http://fairness-evaluator:8000",
                "name": "Fairness Evaluator"
            },
            MCPServerType.COMPLIANCE_LOGGER: {
                "url": "http://compliance-logger:8000",
                "name": "Compliance Logger"
            },
            MCPServerType.PREDICTION_SERVER: {
                "url": "http://prediction-server:8000",
                "name": "Prediction Server"
            }
        }
        
        # Merge with provided config
        server_config = self.config.get('servers', default_config)
        
        for server_type in MCPServerType:
            if server_type in server_config:
                config = server_config[server_type]
                servers[server_type] = MCPServer(
                    name=config['name'],
                    type=server_type,
                    url=config['url']
                )
        
        return servers
    
    async def check_server_health(self, server: MCPServer) -> bool:
        """Check if an MCP server is healthy"""
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.get(
                    f"{server.url}{server.health_endpoint}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed for {server.name}: {str(e)}")
            return False
    
    async def call_mcp_tool(self, server: MCPServer, tool_name: str, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on an MCP server"""
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                # MCP protocol request format
                request_data = {
                    "jsonrpc": "2.0",
                    "method": f"tools/call/{tool_name}",
                    "params": {"arguments": params},
                    "id": str(uuid.uuid4())
                }
                
                async with session.post(
                    f"{server.url}/mcp",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=server.timeout)
                ) as response:
                    result = await response.json()
                    
                    if "error" in result:
                        raise Exception(f"MCP error: {result['error']}")
                    
                    return result.get("result", {})
                    
        except Exception as e:
            logger.error(f"Error calling {tool_name} on {server.name}: {str(e)}")
            raise
    
    async def run_pipeline(self, data: Dict[str, Any], 
                          pipeline_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete bias mitigation pipeline
        
        Args:
            data: Input data containing dataset and configuration
            pipeline_config: Optional pipeline configuration
        
        Returns:
            Complete pipeline results with all intermediate steps
        """
        self.session_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Initialize pipeline results
        results = {
            "session_id": self.session_id,
            "pipeline_start": start_time.isoformat(),
            "stages": {},
            "final_output": None,
            "metadata": {
                "input_samples": len(data.get("dataset", [])),
                "target_column": data.get("target_column", "target")
            }
        }
        
        try:
            # Stage 1: Bias Detection
            logger.info("Stage 1: Detecting bias...")
            bias_result = await self._run_bias_detection(data)
            results["stages"]["bias_detection"] = bias_result
            
            # Stage 2: Data Cleaning (if bias detected)
            if bias_result.get("overall_bias_score", 0) > 0.1:
                logger.info("Stage 2: Cleaning biased data...")
                cleaning_result = await self._run_data_cleaning(data, bias_result)
                results["stages"]["data_cleaning"] = cleaning_result
                
                # Update data for next stages
                data["dataset"] = cleaning_result["cleaned_dataset"]
            else:
                logger.info("Stage 2: Skipping cleaning - minimal bias detected")
                results["stages"]["data_cleaning"] = {"skipped": True, "reason": "minimal_bias"}
            
            # Stage 3: Train Model
            logger.info("Stage 3: Training model...")
            training_result = await self._run_model_training(data)
            results["stages"]["model_training"] = training_result
            
            # Stage 4: Fairness Evaluation
            logger.info("Stage 4: Evaluating fairness...")
            fairness_result = await self._run_fairness_evaluation(
                data, training_result.get("model_id")
            )
            results["stages"]["fairness_evaluation"] = fairness_result
            
            # Stage 5: Compliance Logging
            logger.info("Stage 5: Logging compliance...")
            compliance_result = await self._run_compliance_logging(results)
            results["stages"]["compliance_logging"] = compliance_result
            
            # Final summary
            results["final_output"] = {
                "success": True,
                "model_id": training_result.get("model_id"),
                "bias_reduced": bias_result.get("overall_bias_score", 0) > 0.1,
                "fairness_achieved": fairness_result.get("is_fair", False),
                "compliance_logged": compliance_result.get("status") == "logged"
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            results["final_output"] = {
                "success": False,
                "error": str(e)
            }
        
        # Calculate total time
        end_time = datetime.now()
        results["pipeline_end"] = end_time.isoformat()
        results["total_duration_seconds"] = (end_time - start_time).total_seconds()
        
        return results
    
    async def _run_bias_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run bias detection stage"""
        server = self.servers[MCPServerType.BIAS_DETECTOR]
        
        params = {
            "dataset": data["dataset"],
            "target_column": data.get("target_column", "target")
        }
        
        result = await self.call_mcp_tool(server, "detect_bias", params)
        
        # Log to compliance
        await self._log_stage_compliance(
            "BIAS_DETECTION",
            "bias-detector",
            "detect_bias",
            result
        )
        
        return result
    
    async def _run_data_cleaning(self, data: Dict[str, Any], 
                                bias_report: Dict[str, Any]) -> Dict[str, Any]:
        """Run data cleaning stage"""
        server = self.servers[MCPServerType.DATA_CLEANER]
        
        params = {
            "dataset": data["dataset"],
            "bias_report": bias_report,
            "cleaning_strategy": data.get("cleaning_strategy", "auto"),
            "target_column": data.get("target_column", "target")
        }
        
        result = await self.call_mcp_tool(server, "clean_bias", params)
        
        # Log to compliance
        await self._log_stage_compliance(
            "DATA_CLEANING",
            "data-cleaner",
            "clean_bias",
            result
        )
        
        return result
    
    async def _run_model_training(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run model training stage"""
        server = self.servers[MCPServerType.PREDICTION_SERVER]
        
        params = {
            "dataset": data["dataset"],
            "target_column": data.get("target_column", "target"),
            "model_type": data.get("model_type", "random_forest"),
            "use_sample_weights": True
        }
        
        result = await self.call_mcp_tool(server, "train_model", params)
        
        # Log to compliance
        await self._log_stage_compliance(
            "MODEL_TRAINING",
            "prediction-server",
            "train_model",
            result
        )
        
        return result
    
    async def _run_fairness_evaluation(self, data: Dict[str, Any], 
                                       model_id: str) -> Dict[str, Any]:
        """Run fairness evaluation stage"""
        server = self.servers[MCPServerType.FAIRNESS_EVALUATOR]
        prediction_server = self.servers[MCPServerType.PREDICTION_SERVER]
        
        # First get predictions
        pred_params = {
            "dataset": data["dataset"],
            "model_id": model_id,
            "include_probabilities": True
        }
        
        pred_result = await self.call_mcp_tool(
            prediction_server, "make_predictions", pred_params
        )
        
        # Then evaluate fairness
        eval_params = {
            "dataset": data["dataset"],
            "predictions": pred_result["predictions"],
            "actual": [row[data.get("target_column", "target")] 
                      for row in data["dataset"]],
            "sensitive_attributes": data.get("sensitive_attributes", 
                                           ["gender", "race", "age"])
        }
        
        result = await self.call_mcp_tool(server, "evaluate_fairness", eval_params)
        
        # Log to compliance
        await self._log_stage_compliance(
            "FAIRNESS_EVALUATION",
            "fairness-evaluator",
            "evaluate_fairness",
            result
        )
        
        return result
    
    async def _run_compliance_logging(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run compliance logging stage"""
        server = self.servers[MCPServerType.COMPLIANCE_LOGGER]
        
        params = {
            "event_type": "PIPELINE_COMPLETE",
            "component": "enrichmcp-orchestrator",
            "action": "pipeline_execution",
            "details": {
                "session_id": self.session_id,
                "stages_completed": list(pipeline_results["stages"].keys()),
                "bias_detected": pipeline_results["stages"].get("bias_detection", {})
                                              .get("overall_bias_score", 0) > 0.1,
                "fairness_achieved": pipeline_results["stages"].get("fairness_evaluation", {})
                                                   .get("is_fair", False),
                "model_trained": pipeline_results["stages"].get("model_training", {})
                                               .get("model_id") is not None
            },
            "session_id": self.session_id
        }
        
        result = await self.call_mcp_tool(server, "log_compliance_event", params)
        
        # Generate compliance report
        report_params = {
            "session_id": self.session_id
        }
        
        report = await self.call_mcp_tool(server, "generate_compliance_report", report_params)
        result["compliance_report"] = report
        
        return result
    
    async def _log_stage_compliance(self, event_type: str, component: str,
                                   action: str, details: Dict[str, Any]):
        """Log individual stage to compliance logger"""
        try:
            server = self.servers[MCPServerType.COMPLIANCE_LOGGER]
            
            params = {
                "event_type": event_type,
                "component": component,
                "action": action,
                "details": details,
                "session_id": self.session_id
            }
            
            await self.call_mcp_tool(server, "log_compliance_event", params)
            
        except Exception as e:
            logger.error(f"Failed to log compliance: {str(e)}")
    
    async def compare_with_without_ethos(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comparison between biased and unbiased pipelines
        
        Perfect for demos showing before/after Ethos
        """
        comparison_results = {
            "session_id": str(uuid.uuid4()),
            "comparison_type": "with_without_ethos",
            "without_ethos": {},
            "with_ethos": {},
            "improvements": {}
        }
        
        try:
            # 1. Run WITHOUT Ethos (skip cleaning)
            logger.info("Running WITHOUT Ethos pipeline...")
            
            # Train on biased data
            biased_training = await self.call_mcp_tool(
                self.servers[MCPServerType.PREDICTION_SERVER],
                "train_model",
                {
                    "dataset": data["dataset"],
                    "target_column": data.get("target_column", "target"),
                    "model_type": data.get("model_type", "random_forest"),
                    "use_sample_weights": False
                }
            )
            
            comparison_results["without_ethos"]["model_id"] = biased_training["model_id"]
            comparison_results["without_ethos"]["accuracy"] = biased_training["test_accuracy"]
            
            # Get bias score
            bias_result = await self.call_mcp_tool(
                self.servers[MCPServerType.BIAS_DETECTOR],
                "detect_bias",
                {
                    "dataset": data["dataset"],
                    "target_column": data.get("target_column", "target")
                }
            )
            
            comparison_results["without_ethos"]["bias_score"] = bias_result["overall_bias_score"]
            comparison_results["without_ethos"]["bias_types"] = bias_result["bias_types_detected"]
            
            # 2. Run WITH Ethos (full pipeline)
            logger.info("Running WITH Ethos pipeline...")
            ethos_results = await self.run_pipeline(data)
            
            comparison_results["with_ethos"] = {
                "model_id": ethos_results["stages"]["model_training"]["model_id"],
                "accuracy": ethos_results["stages"]["model_training"]["test_accuracy"],
                "bias_score": ethos_results["stages"].get("fairness_evaluation", {})
                                                     .get("overall_fairness_score", 0),
                "fairness_achieved": ethos_results["stages"].get("fairness_evaluation", {})
                                                           .get("is_fair", False)
            }
            
            # 3. Calculate improvements
            comparison_results["improvements"] = {
                "bias_reduction": (
                    comparison_results["without_ethos"]["bias_score"] - 
                    comparison_results["with_ethos"]["bias_score"]
                ),
                "accuracy_change": (
                    comparison_results["with_ethos"]["accuracy"] - 
                    comparison_results["without_ethos"]["accuracy"]
                ),
                "bias_reduction_percentage": (
                    (comparison_results["without_ethos"]["bias_score"] - 
                     comparison_results["with_ethos"]["bias_score"]) / 
                    comparison_results["without_ethos"]["bias_score"] * 100
                    if comparison_results["without_ethos"]["bias_score"] > 0 else 0
                )
            }
            
            # 4. Generate example predictions for demo
            demo_samples = data["dataset"][:5]  # First 5 samples
            
            # Predictions without Ethos
            without_pred = await self.call_mcp_tool(
                self.servers[MCPServerType.PREDICTION_SERVER],
                "make_predictions",
                {
                    "dataset": demo_samples,
                    "model_id": comparison_results["without_ethos"]["model_id"]
                }
            )
            
            # Predictions with Ethos
            with_pred = await self.call_mcp_tool(
                self.servers[MCPServerType.PREDICTION_SERVER],
                "make_predictions",
                {
                    "dataset": demo_samples,
                    "model_id": comparison_results["with_ethos"]["model_id"]
                }
            )
            
            comparison_results["demo_predictions"] = {
                "samples": demo_samples,
                "without_ethos": without_pred["predictions"],
                "with_ethos": with_pred["predictions"]
            }
            
        except Exception as e:
            logger.error(f"Comparison error: {str(e)}")
            comparison_results["error"] = str(e)
        
        return comparison_results

# FastAPI integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="EnrichMCP Orchestrator API")

# Global orchestrator instance
orchestrator = None

class PipelineRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    target_column: str = "target"
    model_type: str = "random_forest"
    sensitive_attributes: List[str] = ["gender", "race", "age"]
    cleaning_strategy: str = "auto"
    compare_mode: bool = False

@app.on_event("startup")
async def startup_event():
    global orchestrator
    config = {
        "servers": {
            MCPServerType.BIAS_DETECTOR: {
                "url": "http://bias-detector:8000",
                "name": "Bias Detector"
            },
            MCPServerType.DATA_CLEANER: {
                "url": "http://data-cleaner:8000",
                "name": "Data Cleaner"
            },
            MCPServerType.FAIRNESS_EVALUATOR: {
                "url": "http://fairness-evaluator:8000",
                "name": "Fairness Evaluator"
            },
            MCPServerType.COMPLIANCE_LOGGER: {
                "url": "http://compliance-logger:8000",
                "name": "Compliance Logger"
            },
            MCPServerType.PREDICTION_SERVER: {
                "url": "http://prediction-server:8000",
                "name": "Prediction Server"
            }
        }
    }
    orchestrator = EnrichMCPOrchestrator(config)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "enrichmcp-orchestrator"}

@app.post("/pipeline/run")
async def run_pipeline(request: PipelineRequest):
    try:
        data = request.dict()
        
        if request.compare_mode:
            # Run comparison pipeline
            results = await orchestrator.compare_with_without_ethos(data)
        else:
            # Run standard pipeline
            results = await orchestrator.run_pipeline(data)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline/status/{session_id}")
async def get_pipeline_status(session_id: str):
    # In a real implementation, this would query stored results
    return {
        "session_id": session_id,
        "status": "completed",
        "message": "Pipeline execution completed"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)