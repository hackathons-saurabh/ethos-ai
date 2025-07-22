import asyncio
import json
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
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
import pandas as pd
import numpy as np
from dateutil import parser
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for HTTP endpoints
app = FastAPI(title="Compliance Guardian MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for HTTP requests
class ComplianceCheckRequest(BaseModel):
    data: Dict[str, Any]
    compliance_framework: str
    check_type: str = "comprehensive"
    additional_params: Dict[str, Any] = {}

class ComplianceCheckResponse(BaseModel):
    compliance_status: str
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    risk_score: float
    audit_trail: Dict[str, Any]

class ComplianceFrameworkRequest(BaseModel):
    framework_name: str
    version: str = "latest"

server = Server("compliance-guardian")
handlers = {}

def tool(name: str):
    def decorator(func):
        handlers[name] = func
        return func
    return decorator

class ComplianceGuardian:
    def __init__(self):
        self.compliance_frameworks = {
            'GDPR': {
                'name': 'General Data Protection Regulation',
                'version': '2018',
                'checks': ['data_minimization', 'consent_management', 'data_retention', 'privacy_by_design'],
                'risk_thresholds': {
                    'high': 0.8,
                    'medium': 0.5,
                    'low': 0.2
                }
            },
            'CCPA': {
                'name': 'California Consumer Privacy Act',
                'version': '2020',
                'checks': ['data_disclosure', 'opt_out_rights', 'data_portability', 'privacy_notice'],
                'risk_thresholds': {
                    'high': 0.8,
                    'medium': 0.5,
                    'low': 0.2
                }
            },
            'HIPAA': {
                'name': 'Health Insurance Portability and Accountability Act',
                'version': '1996',
                'checks': ['phi_protection', 'access_controls', 'audit_logging', 'data_encryption'],
                'risk_thresholds': {
                    'high': 0.9,
                    'medium': 0.6,
                    'low': 0.3
                }
            },
            'SOX': {
                'name': 'Sarbanes-Oxley Act',
                'version': '2002',
                'checks': ['financial_controls', 'data_integrity', 'audit_trail', 'access_management'],
                'risk_thresholds': {
                    'high': 0.8,
                    'medium': 0.5,
                    'low': 0.2
                }
            }
        }
        
        self.violation_patterns = {
            'personal_data': [
                r'\b(ssn|social\s+security|passport|driver\s+license|credit\s+card)\b',
                r'\b(email|phone|address|zip\s+code|city|state)\b',
                r'\b(name|first\s+name|last\s+name|full\s+name)\b'
            ],
            'financial_data': [
                r'\b(account\s+number|routing\s+number|credit\s+card|debit\s+card)\b',
                r'\b(balance|amount|transaction|payment)\b'
            ],
            'health_data': [
                r'\b(diagnosis|treatment|medication|prescription|medical\s+record)\b',
                r'\b(patient|doctor|hospital|clinic|pharmacy)\b'
            ]
        }
    
    def check_data_minimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if data collection follows minimization principles"""
        violations = []
        recommendations = []
        
        # Check for excessive data collection
        data_fields = list(data.keys()) if isinstance(data, dict) else []
        sensitive_fields = []
        
        for pattern_list in self.violation_patterns.values():
            for pattern in pattern_list:
                for field in data_fields:
                    if re.search(pattern, field, re.IGNORECASE):
                        sensitive_fields.append(field)
        
        if len(sensitive_fields) > 5:  # Threshold for excessive collection
            violations.append({
                'type': 'data_minimization',
                'severity': 'high',
                'description': f'Excessive sensitive data collection detected: {len(sensitive_fields)} sensitive fields',
                'fields': sensitive_fields
            })
            recommendations.append('Review and reduce the number of sensitive data fields collected')
        
        return {
            'violations': violations,
            'recommendations': recommendations,
            'risk_score': min(len(sensitive_fields) / 10.0, 1.0)
        }
    
    def check_consent_management(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check consent management compliance"""
        violations = []
        recommendations = []
        
        # Check for consent-related fields
        consent_fields = ['consent', 'agreement', 'opt_in', 'opt_out', 'permission']
        found_consent_fields = []
        
        for field in consent_fields:
            if field in str(data).lower():
                found_consent_fields.append(field)
        
        if not found_consent_fields:
            violations.append({
                'type': 'consent_management',
                'severity': 'high',
                'description': 'No consent management mechanisms detected',
                'fields': []
            })
            recommendations.append('Implement proper consent management system')
        
        return {
            'violations': violations,
            'recommendations': recommendations,
            'risk_score': 0.8 if not found_consent_fields else 0.2
        }
    
    def check_data_retention(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check data retention policies"""
        violations = []
        recommendations = []
        
        # Check for timestamp fields that might indicate retention policies
        time_fields = ['created_at', 'updated_at', 'expires_at', 'deleted_at', 'archived_at']
        found_time_fields = []
        
        for field in time_fields:
            if field in str(data).lower():
                found_time_fields.append(field)
        
        if not found_time_fields:
            violations.append({
                'type': 'data_retention',
                'severity': 'medium',
                'description': 'No data retention timestamps detected',
                'fields': []
            })
            recommendations.append('Implement data retention policies with proper timestamps')
        
        return {
            'violations': violations,
            'recommendations': recommendations,
            'risk_score': 0.6 if not found_time_fields else 0.3
        }
    
    def check_privacy_by_design(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check privacy by design implementation"""
        violations = []
        recommendations = []
        
        # Check for encryption indicators
        encryption_indicators = ['encrypted', 'hash', 'tokenized', 'masked', 'anonymized']
        found_encryption = []
        
        for indicator in encryption_indicators:
            if indicator in str(data).lower():
                found_encryption.append(indicator)
        
        if not found_encryption:
            violations.append({
                'type': 'privacy_by_design',
                'severity': 'medium',
                'description': 'No privacy protection mechanisms detected',
                'fields': []
            })
            recommendations.append('Implement data encryption and anonymization')
        
        return {
            'violations': violations,
            'recommendations': recommendations,
            'risk_score': 0.7 if not found_encryption else 0.3
        }
    
    def check_phi_protection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check Protected Health Information protection"""
        violations = []
        recommendations = []
        
        phi_patterns = self.violation_patterns['health_data']
        found_phi = []
        
        for pattern in phi_patterns:
            if re.search(pattern, str(data), re.IGNORECASE):
                found_phi.append(pattern)
        
        if found_phi:
            violations.append({
                'type': 'phi_protection',
                'severity': 'high',
                'description': f'PHI detected: {len(found_phi)} health-related patterns found',
                'patterns': found_phi
            })
            recommendations.append('Ensure PHI is properly encrypted and access-controlled')
        
        return {
            'violations': violations,
            'recommendations': recommendations,
            'risk_score': min(len(found_phi) * 0.3, 1.0)
        }
    
    def check_financial_controls(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check financial data controls"""
        violations = []
        recommendations = []
        
        financial_patterns = self.violation_patterns['financial_data']
        found_financial = []
        
        for pattern in financial_patterns:
            if re.search(pattern, str(data), re.IGNORECASE):
                found_financial.append(pattern)
        
        if found_financial:
            violations.append({
                'type': 'financial_controls',
                'severity': 'high',
                'description': f'Financial data detected: {len(found_financial)} financial patterns found',
                'patterns': found_financial
            })
            recommendations.append('Implement strict financial data controls and encryption')
        
        return {
            'violations': violations,
            'recommendations': recommendations,
            'risk_score': min(len(found_financial) * 0.4, 1.0)
        }
    
    def comprehensive_compliance_check(self, data: Dict[str, Any], framework: str) -> Dict[str, Any]:
        """Perform comprehensive compliance check for a specific framework"""
        if framework not in self.compliance_frameworks:
            return {
                'error': f'Unknown compliance framework: {framework}',
                'compliance_status': 'unknown',
                'violations': [],
                'recommendations': [],
                'risk_score': 1.0
            }
        
        framework_config = self.compliance_frameworks[framework]
        all_violations = []
        all_recommendations = []
        risk_scores = []
        
        # Run framework-specific checks
        for check_type in framework_config['checks']:
            if hasattr(self, f'check_{check_type}'):
                check_method = getattr(self, f'check_{check_type}')
                result = check_method(data)
                all_violations.extend(result['violations'])
                all_recommendations.extend(result['recommendations'])
                risk_scores.append(result['risk_score'])
        
        # Calculate overall risk score
        overall_risk_score = np.mean(risk_scores) if risk_scores else 0.0
        
        # Determine compliance status
        if overall_risk_score >= framework_config['risk_thresholds']['high']:
            compliance_status = 'non_compliant'
        elif overall_risk_score >= framework_config['risk_thresholds']['medium']:
            compliance_status = 'at_risk'
        else:
            compliance_status = 'compliant'
        
        return {
            'compliance_status': compliance_status,
            'violations': all_violations,
            'recommendations': all_recommendations,
            'risk_score': overall_risk_score,
            'framework': framework_config,
            'audit_trail': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'check_id': str(uuid.uuid4()),
                'framework_version': framework_config['version']
            }
        }

# Global compliance guardian instance
guardian = ComplianceGuardian()

@tool("check_compliance")
async def check_compliance(data: Dict[str, Any]) -> Dict[str, Any]:
    """Check compliance for a specific framework"""
    try:
        framework = data.get('compliance_framework', 'GDPR')
        check_data = data.get('data', {})
        
        result = guardian.comprehensive_compliance_check(check_data, framework)
        
        logger.info(f"Compliance check completed for {framework}. Status: {result['compliance_status']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in compliance check: {str(e)}")
        return {
            'error': str(e),
            'compliance_status': 'error',
            'violations': [],
            'recommendations': [],
            'risk_score': 1.0
        }

@tool("get_compliance_frameworks")
async def get_compliance_frameworks(data: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Get available compliance frameworks"""
    try:
        frameworks = {}
        for key, config in guardian.compliance_frameworks.items():
            frameworks[key] = {
                'name': config['name'],
                'version': config['version'],
                'checks': config['checks'],
                'risk_thresholds': config['risk_thresholds']
            }
        
        return {
            'frameworks': frameworks,
            'total_frameworks': len(frameworks)
        }
        
    except Exception as e:
        logger.error(f"Error getting compliance frameworks: {str(e)}")
        return {
            'error': str(e),
            'frameworks': {},
            'total_frameworks': 0
        }

@tool("analyze_data_privacy")
async def analyze_data_privacy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze data privacy risks"""
    try:
        check_data = data.get('data', {})
        
        # Check for personal data patterns
        personal_data_violations = []
        for pattern in guardian.violation_patterns['personal_data']:
            if re.search(pattern, str(check_data), re.IGNORECASE):
                personal_data_violations.append(pattern)
        
        risk_score = min(len(personal_data_violations) * 0.2, 1.0)
        
        return {
            'privacy_risk_score': risk_score,
            'personal_data_patterns_found': personal_data_violations,
            'recommendations': [
                'Implement data anonymization',
                'Use encryption for sensitive data',
                'Apply data minimization principles'
            ] if personal_data_violations else ['Data privacy analysis passed']
        }
        
    except Exception as e:
        logger.error(f"Error in data privacy analysis: {str(e)}")
        return {
            'error': str(e),
            'privacy_risk_score': 1.0,
            'personal_data_patterns_found': [],
            'recommendations': []
        }

@tool("generate_compliance_report")
async def generate_compliance_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive compliance report"""
    try:
        framework = data.get('framework', 'GDPR')
        check_data = data.get('data', {})
        
        # Run compliance check
        compliance_result = guardian.comprehensive_compliance_check(check_data, framework)
        
        # Generate report
        report = {
            'report_id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'framework': framework,
            'compliance_status': compliance_result['compliance_status'],
            'risk_score': compliance_result['risk_score'],
            'violations_count': len(compliance_result['violations']),
            'recommendations_count': len(compliance_result['recommendations']),
            'detailed_violations': compliance_result['violations'],
            'detailed_recommendations': compliance_result['recommendations'],
            'audit_trail': compliance_result['audit_trail']
        }
        
        logger.info(f"Compliance report generated: {report['report_id']}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        return {
            'error': str(e),
            'report_id': None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

@tool("validate_compliance_policy")
async def validate_compliance_policy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate compliance policies against frameworks"""
    try:
        policy = data.get('policy', {})
        framework = data.get('framework', 'GDPR')
        
        # Basic policy validation
        required_elements = ['data_retention', 'consent_management', 'data_encryption']
        missing_elements = []
        
        for element in required_elements:
            if element not in str(policy).lower():
                missing_elements.append(element)
        
        validation_score = 1.0 - (len(missing_elements) / len(required_elements))
        
        return {
            'validation_score': validation_score,
            'missing_elements': missing_elements,
            'policy_compliant': validation_score >= 0.8,
            'recommendations': [
                f'Add {element} policy' for element in missing_elements
            ]
        }
        
    except Exception as e:
        logger.error(f"Error validating compliance policy: {str(e)}")
        return {
            'error': str(e),
            'validation_score': 0.0,
            'missing_elements': [],
            'policy_compliant': False,
            'recommendations': []
        }

# HTTP Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "compliance-guardian"}

@app.post("/check_compliance", response_model=ComplianceCheckResponse)
async def http_check_compliance(request: ComplianceCheckRequest):
    """HTTP endpoint for compliance checking"""
    try:
        data = {
            "compliance_framework": request.compliance_framework,
            "data": request.data,
            "check_type": request.check_type,
            "additional_params": request.additional_params
        }
        result = await check_compliance(data)
        return ComplianceCheckResponse(**result)
    except Exception as e:
        logger.error(f"HTTP compliance check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp")
async def mcp_endpoint(request: Dict[str, Any]):
    """MCP protocol endpoint for HTTP communication"""
    try:
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id", "1")
        
        if method == "tools/call/check_compliance":
            result = await check_compliance(params.get("arguments", {}))
        elif method == "tools/call/get_compliance_frameworks":
            result = await get_compliance_frameworks(params.get("arguments", {}))
        elif method == "tools/call/analyze_data_privacy":
            result = await analyze_data_privacy(params.get("arguments", {}))
        elif method == "tools/call/generate_compliance_report":
            result = await generate_compliance_report(params.get("arguments", {}))
        elif method == "tools/call/validate_compliance_policy":
            result = await validate_compliance_policy(params.get("arguments", {}))
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
        logger.info("Starting compliance-guardian MCP Server...")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                initialization_options={"log_level": "debug", "agent_name": "compliance-guardian"}
            )
            # Keep it alive
            while True:
                await asyncio.sleep(1)
    
    asyncio.run(mcp_main())

def run_http_server():
    """Run the HTTP server"""
    logger.info("Starting compliance-guardian HTTP Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def main():
    """Run both MCP stdio and HTTP servers"""
    logger.info("Starting compliance-guardian servers...")
    
    # Start MCP server in a separate thread
    mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
    mcp_thread.start()
    
    # Start HTTP server
    run_http_server()

if __name__ == "__main__":
    main() 