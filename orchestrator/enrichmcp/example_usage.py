#!/usr/bin/env python3
"""
Example usage of the real EnrichMCP library for bias detection data API
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Import the real EnrichMCP library
from enrichmcp import EnrichMCP, EnrichModel, Relationship, EnrichParameter
from pydantic import Field

async def demonstrate_enrichmcp_usage():
    """Demonstrate how to use the real EnrichMCP library."""
    
    print("🚀 EnrichMCP Library Demo")
    print("=" * 50)
    
    # Create EnrichMCP app
    app = EnrichMCP(
        "Bias Detection Demo", 
        "Demo of AI-navigable bias detection data API"
    )
    
    # Define a simple data model
    @app.entity
    class BiasResult(EnrichModel):
        """Simple bias detection result."""
        
        session_id: str = Field(description="Session identifier")
        bias_score: float = Field(description="Bias score (0-1)")
        bias_detected: bool = Field(description="Whether bias was detected")
        created_at: datetime = Field(description="When analysis was performed")
        
        # Relationship to detailed analysis
        details: "BiasDetails" = Relationship(description="Detailed bias analysis")
    
    @app.entity
    class BiasDetails(EnrichModel):
        """Detailed bias analysis."""
        
        session_id: str = Field(description="Session identifier")
        gender_bias: float = Field(description="Gender bias score")
        age_bias: float = Field(description="Age bias score")
        race_bias: float = Field(description="Race bias score")
        recommendations: list[str] = Field(description="Bias mitigation recommendations")
    
    # Define data retrieval functions
    @app.retrieve
    async def get_bias_result(
        session_id: str = EnrichParameter(description="Session ID to retrieve", examples=["session_123"])
    ) -> BiasResult:
        """Get bias detection result by session ID."""
        # Simulate data retrieval
        return BiasResult(
            session_id=session_id,
            bias_score=0.65,
            bias_detected=True,
            created_at=datetime.now()
        )
    
    @app.retrieve
    async def find_high_bias_sessions(
        threshold: float = EnrichParameter(description="Minimum bias score", examples=[0.7])
    ) -> list[BiasResult]:
        """Find sessions with high bias scores."""
        # Simulate finding high bias sessions
        sessions = [
            BiasResult(session_id="session_1", bias_score=0.8, bias_detected=True, created_at=datetime.now()),
            BiasResult(session_id="session_2", bias_score=0.75, bias_detected=True, created_at=datetime.now()),
            BiasResult(session_id="session_3", bias_score=0.6, bias_detected=False, created_at=datetime.now()),
        ]
        return [s for s in sessions if s.bias_score >= threshold]
    
    # Define relationship resolvers
    @BiasResult.details.resolver
    async def get_bias_details(session_id: str) -> BiasDetails:
        """Get detailed bias analysis for a session."""
        return BiasDetails(
            session_id=session_id,
            gender_bias=0.7,
            age_bias=0.5,
            race_bias=0.3,
            recommendations=[
                "Increase training data diversity",
                "Apply demographic parity constraints",
                "Implement reweighting techniques"
            ]
        )
    
    print("✅ EnrichMCP app created with:")
    print("   • BiasResult entity with relationships")
    print("   • Data retrieval functions")
    print("   • Relationship resolvers")
    print()
    
    # Simulate AI agent usage
    print("🤖 Simulating AI Agent Usage:")
    print("-" * 30)
    
    # AI agent would do this:
    print("1. Explore data model:")
    print("   explore_data_model()")
    print()
    
    print("2. Find high bias sessions:")
    print("   high_bias = find_high_bias_sessions(threshold=0.7)")
    print("   Result: Found 2 high bias sessions")
    print()
    
    print("3. Get specific session:")
    print("   session = get_bias_result(session_id='session_123')")
    print("   Result: Bias score 0.65, bias detected: True")
    print()
    
    print("4. Navigate relationships:")
    print("   details = session.details()")
    print("   Result: Gender bias 0.7, Age bias 0.5, Race bias 0.3")
    print()
    
    print("5. Get recommendations:")
    print("   recommendations = details.recommendations")
    print("   Result: ['Increase training data diversity', 'Apply demographic parity constraints']")
    print()
    
    print("🎯 Key Benefits of Real EnrichMCP:")
    print("-" * 40)
    print("✅ AI agents can explore data schema automatically")
    print("✅ Type-safe data models with Pydantic validation")
    print("✅ Natural relationship navigation")
    print("✅ Rich parameter metadata for AI agents")
    print("✅ Built-in schema discovery")
    print("✅ Automatic tool generation for AI agents")
    print()
    
    print("🔗 This is how AI agents can naturally interact with your data!")
    print("   No custom code needed - EnrichMCP handles the AI integration.")

if __name__ == "__main__":
    asyncio.run(demonstrate_enrichmcp_usage()) 