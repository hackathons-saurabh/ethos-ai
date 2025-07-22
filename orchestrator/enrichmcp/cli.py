# Command Line Interface for EnrichMCP - Real EnrichMCP Library Usage
"""
CLI for interacting with the EnrichMCP bias detection data API
"""

import asyncio
import json
import argparse
import logging
from typing import Dict, Any, List
from pathlib import Path
import httpx

# Import the real EnrichMCP library
from enrichmcp import EnrichMCP, EnrichModel, Relationship, EnrichParameter
from pydantic import Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create EnrichMCP app for CLI operations
cli_app = EnrichMCP(
    "Bias Detection CLI", 
    "Command line interface for bias detection data exploration"
)

# Define CLI-specific data models
@cli_app.entity
class CLISession(EnrichModel):
    """CLI session for data exploration."""
    
    session_id: str = Field(description="Session identifier")
    command: str = Field(description="Command executed")
    results: Dict[str, Any] = Field(description="Command results")
    timestamp: str = Field(description="When command was executed")

@cli_app.entity
class PipelineSummary(EnrichModel):
    """Summary of pipeline results."""
    
    total_sessions: int = Field(description="Total number of sessions")
    completed_sessions: int = Field(description="Number of completed sessions")
    average_bias_score: float = Field(description="Average bias score across sessions")
    average_fairness_score: float = Field(description="Average fairness score across sessions")
    compliance_rate: float = Field(description="Percentage of compliant sessions")

# CLI data storage
cli_sessions = {}

async def explore_pipeline_data(api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Explore pipeline data using the API."""
    async with httpx.AsyncClient() as client:
        try:
            # Get health status
            health_response = await client.get(f"{api_url}/health")
            health_data = health_response.json()
            
            # Get recent sessions (simulated)
            sessions_data = {
                "total_sessions": 15,
                "completed_sessions": 12,
                "recent_sessions": [
                    {"session_id": "session_abc123", "status": "completed", "bias_score": 0.65},
                    {"session_id": "session_def456", "status": "completed", "bias_score": 0.72},
                    {"session_id": "session_ghi789", "status": "running", "bias_score": None}
                ]
            }
            
            return {
                "health": health_data,
                "sessions": sessions_data,
                "api_url": api_url
            }
        except Exception as e:
            logger.error(f"Error exploring pipeline data: {str(e)}")
            return {"error": str(e)}

async def run_pipeline_from_file(file_path: str, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Run pipeline from a JSON file using the API."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{api_url}/pipeline/run",
                json=data
            )
            return response.json()
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        return {"error": str(e)}

async def check_pipeline_status(session_id: str, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Check status of a specific pipeline session."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/pipeline/status/{session_id}")
            return response.json()
    except Exception as e:
        logger.error(f"Error checking pipeline status: {str(e)}")
        return {"error": str(e)}

async def analyze_bias_patterns(api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Analyze bias patterns across sessions."""
    try:
        # Simulate bias analysis
        analysis = {
            "total_sessions_analyzed": 12,
            "high_bias_sessions": 3,
            "medium_bias_sessions": 5,
            "low_bias_sessions": 4,
            "common_bias_factors": [
                {"factor": "gender", "frequency": 0.75},
                {"factor": "age", "frequency": 0.60},
                {"factor": "race", "frequency": 0.45}
            ],
            "bias_trends": {
                "last_week": 0.68,
                "last_month": 0.72,
                "last_quarter": 0.75
            }
        }
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing bias patterns: {str(e)}")
        return {"error": str(e)}

async def generate_fairness_report(api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Generate fairness evaluation report."""
    try:
        # Simulate fairness report
        report = {
            "total_evaluations": 12,
            "fairness_score_distribution": {
                "excellent": 2,
                "good": 5,
                "fair": 3,
                "poor": 2
            },
            "top_recommendations": [
                "Increase training data diversity",
                "Apply demographic parity constraints",
                "Implement reweighting techniques",
                "Add fairness-aware preprocessing"
            ],
            "compliance_status": {
                "compliant": 8,
                "requires_review": 3,
                "non_compliant": 1
            }
        }
        return report
    except Exception as e:
        logger.error(f"Error generating fairness report: {str(e)}")
        return {"error": str(e)}

# EnrichMCP retrieve functions for CLI
@cli_app.retrieve
async def get_pipeline_summary(api_url: str = EnrichParameter(description="API URL", examples=["http://localhost:8000"])) -> PipelineSummary:
    """Get summary of all pipeline sessions."""
    data = await explore_pipeline_data(api_url)
    
    if "error" in data:
        raise ValueError(data["error"])
    
    sessions = data["sessions"]
    return PipelineSummary(
        total_sessions=sessions["total_sessions"],
        completed_sessions=sessions["completed_sessions"],
        average_bias_score=0.68,  # Simulated
        average_fairness_score=0.75,  # Simulated
        compliance_rate=0.83  # Simulated
    )

@cli_app.retrieve
async def list_recent_sessions(
    limit: int = EnrichParameter(description="Number of sessions to list", examples=[10]),
    api_url: str = EnrichParameter(description="API URL", examples=["http://localhost:8000"])
) -> List[CLISession]:
    """List recent pipeline sessions."""
    data = await explore_pipeline_data(api_url)
    
    if "error" in data:
        raise ValueError(data["error"])
    
    sessions = []
    for i, session_data in enumerate(data["sessions"]["recent_sessions"][:limit]):
        session = CLISession(
            session_id=session_data["session_id"],
            command="list_recent_sessions",
            results=session_data,
            timestamp=f"2024-01-{15+i:02d}T10:00:00Z"
        )
        sessions.append(session)
    
    return sessions

@cli_app.retrieve
async def get_bias_analysis(
    api_url: str = EnrichParameter(description="API URL", examples=["http://localhost:8000"])
) -> Dict[str, Any]:
    """Get detailed bias analysis."""
    return await analyze_bias_patterns(api_url)

@cli_app.retrieve
async def get_fairness_report(
    api_url: str = EnrichParameter(description="API URL", examples=["http://localhost:8000"])
) -> Dict[str, Any]:
    """Get fairness evaluation report."""
    return await generate_fairness_report(api_url)

def main():
    """Main CLI entry point using EnrichMCP."""
    parser = argparse.ArgumentParser(description="EnrichMCP - Bias Detection Data Explorer")
    parser.add_argument("command", choices=["explore", "run", "status", "analyze", "report"], 
                       help="Command to execute")
    parser.add_argument("--file", "-f", type=str, help="Input JSON file for pipeline")
    parser.add_argument("--session", "-s", type=str, help="Session ID for status check")
    parser.add_argument("--api-url", "-u", type=str, default="http://localhost:8000", 
                       help="API URL")
    parser.add_argument("--output", "-o", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    async def run_command():
        try:
            if args.command == "explore":
                # Use EnrichMCP to explore data
                summary = await get_pipeline_summary(args.api_url)
                results = {
                    "summary": {
                        "total_sessions": summary.total_sessions,
                        "completed_sessions": summary.completed_sessions,
                        "average_bias_score": summary.average_bias_score,
                        "average_fairness_score": summary.average_fairness_score,
                        "compliance_rate": summary.compliance_rate
                    }
                }
                
                # Get recent sessions
                recent_sessions = await list_recent_sessions(5, args.api_url)
                results["recent_sessions"] = [
                    {
                        "session_id": session.session_id,
                        "command": session.command,
                        "results": session.results
                    } for session in recent_sessions
                ]
                
            elif args.command == "run":
                if not args.file:
                    logger.error("File path required for run command")
                    return
                results = await run_pipeline_from_file(args.file, args.api_url)
                
            elif args.command == "status":
                if not args.session:
                    logger.error("Session ID required for status command")
                    return
                results = await check_pipeline_status(args.session, args.api_url)
                
            elif args.command == "analyze":
                results = await get_bias_analysis(args.api_url)
                
            elif args.command == "report":
                results = await get_fairness_report(args.api_url)
                
            else:
                logger.error(f"Unknown command: {args.command}")
                return
            
            # Output results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {args.output}")
            else:
                print(json.dumps(results, indent=2))
                
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            print(json.dumps({"error": str(e)}, indent=2))
    
    asyncio.run(run_command())

if __name__ == "__main__":
    main() 