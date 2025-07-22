# Compliance Guardian MCP Server

A comprehensive compliance monitoring and validation MCP server that ensures adherence to various regulatory frameworks and data protection standards.

## Overview

The Compliance Guardian MCP server provides automated compliance checking, risk assessment, and regulatory validation for AI/ML pipelines. It supports multiple compliance frameworks including GDPR, CCPA, HIPAA, and SOX.

## Features

### 🔒 Compliance Frameworks Supported
- **GDPR** (General Data Protection Regulation) - 2018
- **CCPA** (California Consumer Privacy Act) - 2020  
- **HIPAA** (Health Insurance Portability and Accountability Act) - 1996
- **SOX** (Sarbanes-Oxley Act) - 2002

### 🛡️ Compliance Checks
- **Data Minimization**: Ensures only necessary data is collected
- **Consent Management**: Validates proper consent mechanisms
- **Data Retention**: Checks for proper data retention policies
- **Privacy by Design**: Validates privacy protection mechanisms
- **PHI Protection**: Specialized checks for healthcare data (HIPAA)
- **Financial Controls**: Validates financial data handling (SOX)

### 📊 Risk Assessment
- Automated risk scoring based on compliance violations
- Severity classification (High, Medium, Low)
- Detailed violation reporting with recommendations

## API Endpoints

### Health Check
```bash
GET /health
```

### Compliance Check
```bash
POST /check_compliance
Content-Type: application/json

{
  "data": {
    "name": "John Doe",
    "email": "john@example.com",
    "ssn": "123-45-6789"
  },
  "compliance_framework": "GDPR",
  "check_type": "comprehensive"
}
```

### MCP Protocol
```bash
POST /mcp
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "tools/call/check_compliance",
  "params": {
    "arguments": {
      "data": {...},
      "compliance_framework": "GDPR"
    }
  },
  "id": "1"
}
```

## Available Tools

### 1. `check_compliance`
Performs comprehensive compliance checks for specified frameworks.

**Parameters:**
- `data`: Data to be checked for compliance
- `compliance_framework`: Framework to check against (GDPR, CCPA, HIPAA, SOX)
- `check_type`: Type of check (comprehensive, basic, custom)

**Returns:**
- `compliance_status`: compliant, at_risk, non_compliant
- `violations`: List of detected violations
- `recommendations`: List of improvement recommendations
- `risk_score`: Overall risk score (0.0 - 1.0)
- `audit_trail`: Compliance audit information

### 2. `get_compliance_frameworks`
Returns available compliance frameworks and their configurations.

**Returns:**
- `frameworks`: Dictionary of available frameworks
- `total_frameworks`: Number of supported frameworks

### 3. `analyze_data_privacy`
Analyzes data privacy risks and patterns.

**Returns:**
- `privacy_risk_score`: Privacy risk assessment
- `personal_data_patterns_found`: Detected personal data patterns
- `recommendations`: Privacy improvement recommendations

### 4. `generate_compliance_report`
Generates comprehensive compliance reports.

**Parameters:**
- `framework`: Framework to generate report for
- `data`: Data used in the analysis

**Returns:**
- `report_id`: Unique report identifier
- `compliance_status`: Overall compliance status
- `detailed_violations`: Detailed violation information
- `detailed_recommendations`: Specific recommendations
- `audit_trail`: Complete audit trail

### 5. `validate_compliance_policy`
Validates compliance policies against frameworks.

**Parameters:**
- `policy`: Policy to validate
- `framework`: Framework to validate against

**Returns:**
- `validation_score`: Policy validation score
- `missing_elements`: Missing policy elements
- `policy_compliant`: Whether policy is compliant
- `recommendations`: Policy improvement recommendations

## Integration with Ethos AI Platform

The Compliance Guardian integrates seamlessly with the Ethos AI platform:

1. **Pipeline Integration**: Automatically runs compliance checks in the AI pipeline
2. **Real-time Monitoring**: Monitors compliance during model training and evaluation
3. **Audit Trail**: Maintains comprehensive audit trails for all operations
4. **Risk Assessment**: Provides risk scores for decision-making

## Usage Examples

### Basic Compliance Check
```python
import aiohttp
import json

async def check_gdpr_compliance():
    async with aiohttp.ClientSession() as session:
        data = {
            "data": {"name": "test", "email": "test@example.com"},
            "compliance_framework": "GDPR"
        }
        
        async with session.post(
            "http://localhost:8006/check_compliance",
            json=data
        ) as response:
            result = await response.json()
            print(f"Compliance Status: {result['compliance_status']}")
            print(f"Risk Score: {result['risk_score']}")
```

### Framework Information
```python
async def get_frameworks():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8006/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call/get_compliance_frameworks",
                "params": {"arguments": {}},
                "id": "1"
            }
        ) as response:
            result = await response.json()
            frameworks = result['result']['frameworks']
            for name, config in frameworks.items():
                print(f"{name}: {config['name']} v{config['version']}")
```

## Configuration

### Environment Variables
- `MCP_SERVER_NAME`: Server name (default: compliance-guardian)
- `LOG_LEVEL`: Logging level (default: INFO)

### Docker Configuration
```yaml
compliance-guardian:
  build: ./mcp-servers/compliance-guardian
  container_name: ethos-compliance-guardian
  ports:
    - "8006:8000"
  environment:
    - MCP_SERVER_NAME=compliance-guardian
    - LOG_LEVEL=INFO
  volumes:
    - compliance-data:/app/data
  networks:
    - ethos-network
  restart: unless-stopped
```

## Development

### Building the Server
```bash
cd mcp-servers/compliance-guardian
docker build -t ethos-compliance-guardian .
```

### Running Locally
```bash
docker run -p 8006:8000 ethos-compliance-guardian
```

### Testing
```bash
# Health check
curl http://localhost:8006/health

# Compliance check
curl -X POST http://localhost:8006/check_compliance \
  -H "Content-Type: application/json" \
  -d '{"data": {"name": "test"}, "compliance_framework": "GDPR"}'
```

## Compliance Patterns

The server detects various compliance patterns:

### Personal Data Patterns
- SSN, passport, driver's license
- Email, phone, address information
- Names and personal identifiers

### Financial Data Patterns
- Account numbers, routing numbers
- Credit card, debit card information
- Balance, transaction, payment data

### Health Data Patterns
- Diagnosis, treatment, medication
- Patient, doctor, hospital information
- Medical records and prescriptions

## Risk Scoring

The server uses sophisticated risk scoring algorithms:

- **High Risk (0.8-1.0)**: Critical violations detected
- **Medium Risk (0.5-0.8)**: Moderate violations detected  
- **Low Risk (0.0-0.5)**: Minimal or no violations

## Contributing

To add new compliance frameworks or checks:

1. Add framework configuration to `ComplianceGuardian.compliance_frameworks`
2. Implement check methods following the existing pattern
3. Update violation patterns as needed
4. Add tests for new functionality

## License

Part of the Ethos AI Platform - Ensuring Fair and Compliant AI Systems 