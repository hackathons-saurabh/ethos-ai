#!/usr/bin/env python3
"""
Test script for compliance-guardian MCP server integration
"""

import asyncio
import aiohttp
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_compliance_guardian():
    """Test the compliance-guardian MCP server"""
    
    # Test data
    test_data = {
        "data": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111",
            "address": "123 Main St",
            "city": "New York",
            "state": "NY",
            "zip_code": "10001"
        },
        "compliance_framework": "GDPR",
        "check_type": "comprehensive"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get("http://localhost:8006/health") as response:
                health = await response.json()
                logger.info(f"Health check: {health}")
            
            # Test compliance check
            async with session.post(
                "http://localhost:8006/check_compliance",
                json=test_data
            ) as response:
                result = await response.json()
                logger.info(f"Compliance check result: {json.dumps(result, indent=2)}")
            
            # Test get compliance frameworks
            async with session.post(
                "http://localhost:8006/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call/get_compliance_frameworks",
                    "params": {"arguments": {}},
                    "id": "1"
                }
            ) as response:
                frameworks = await response.json()
                logger.info(f"Available frameworks: {json.dumps(frameworks, indent=2)}")
                
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

async def test_orchestrator_integration():
    """Test orchestrator integration with compliance-guardian"""
    
    test_data = {
        "dataset": [
            {"name": "Alice", "age": 25, "gender": "female", "salary": 50000, "target": 1},
            {"name": "Bob", "age": 30, "gender": "male", "salary": 60000, "target": 1},
            {"name": "Carol", "age": 35, "gender": "female", "salary": 55000, "target": 0}
        ],
        "target_column": "target",
        "model_type": "random_forest",
        "sensitive_attributes": ["gender", "age"]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test orchestrator pipeline with compliance guardian
            async with session.post(
                "http://localhost:8000/pipeline/run",
                json=test_data
            ) as response:
                result = await response.json()
                logger.info(f"Orchestrator pipeline result: {json.dumps(result, indent=2)}")
                
    except Exception as e:
        logger.error(f"Orchestrator test failed: {str(e)}")

async def main():
    """Run all tests"""
    logger.info("🧪 Testing compliance-guardian MCP server...")
    
    # Test compliance guardian directly
    await test_compliance_guardian()
    
    # Test orchestrator integration
    await test_orchestrator_integration()
    
    logger.info("✅ Tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 