# EnrichMCP Integration - AI-Navigable Bias Detection Data API

This project demonstrates how to use the **real EnrichMCP library** to create AI-navigable data APIs for bias detection pipeline results.

## What is EnrichMCP?

EnrichMCP is a Python framework that helps AI agents understand and navigate your data. It's like an **ORM for AI agents** - it turns your data models into typed, discoverable tools that AI can use naturally.

## How We Use EnrichMCP

Instead of building our own custom "EnrichMCP" library, we now use the **official EnrichMCP library** to:

1. **Create AI-navigable data models** for bias detection results
2. **Define relationships** between different pipeline stages
3. **Provide schema discovery** so AI agents understand our data structure
4. **Enable natural data exploration** through typed tools

## Key Features

### 🔍 **AI-Navigable Data Models**

```python
from enrichmcp import EnrichMCP, EnrichModel, Relationship, EnrichParameter

@enrichmcp_app.entity
class BiasDetectionResult(EnrichModel):
    """Result from bias detection analysis."""
    
    session_id: str = Field(description="Unique session identifier")
    bias_score: float = Field(description="Overall bias score (0-1)")
    bias_detected: bool = Field(description="Whether bias was detected")
    sensitive_attributes: List[str] = Field(description="Attributes analyzed for bias")
    
    # Relationships - AI can navigate these naturally
    data_cleaning_result: "DataCleaningResult" = Relationship(description="Associated data cleaning result")
    fairness_evaluation: "FairnessEvaluation" = Relationship(description="Associated fairness evaluation")
```

### 🔗 **Relationship Navigation**

AI agents can navigate relationships naturally:

```python
# AI can do this:
session = await get_pipeline_session("session_123")
bias_result = await session.bias_detection()
cleaning_result = await bias_result.data_cleaning_result()
fairness_eval = await cleaning_result.fairness_evaluation()
```

### 📊 **Data Retrieval Functions**

```python
@enrichmcp_app.retrieve
async def find_high_bias_sessions(
    bias_threshold: float = EnrichParameter(description="Minimum bias score", examples=[0.7]),
    limit: int = EnrichParameter(description="Maximum number of sessions", examples=[10])
) -> List[BiasDetectionResult]:
    """Find sessions with high bias scores."""
    # Implementation here
```

## Installation

```bash
# Install the real EnrichMCP library
pip install enrichmcp

# Install our project dependencies
pip install -r requirements.txt
```

## Usage Examples

### 1. **Run Pipeline and Store Results**

```python
# Run bias detection pipeline
POST /pipeline/run
{
    "dataset": [...],
    "target_column": "target",
    "sensitive_attributes": ["gender", "age"]
}

# Returns session_id for later exploration
{
    "session_id": "session_abc123",
    "status": "completed",
    "results": {
        "bias_score": 0.65,
        "fairness_score": 0.75,
        "compliance_status": "compliant"
    }
}
```

### 2. **Explore Data with AI Agents**

AI agents can now explore our data naturally:

```python
# Get pipeline summary
summary = await get_pipeline_summary()
print(f"Total sessions: {summary.total_sessions}")
print(f"Average bias score: {summary.average_bias_score}")

# Find high bias sessions
high_bias = await find_high_bias_sessions(bias_threshold=0.7)
for session in high_bias:
    print(f"Session {session.session_id}: {session.bias_score}")

# Navigate relationships
session = await get_pipeline_session("session_123")
bias_result = await session.bias_detection()
cleaning_result = await bias_result.data_cleaning_result()
print(f"Cleaned {cleaning_result.removed_records} records")
```

### 3. **CLI Usage**

```bash
# Explore pipeline data
python cli.py explore --api-url http://localhost:8000

# Run pipeline from file
python cli.py run --file data.json --api-url http://localhost:8000

# Check session status
python cli.py status --session session_abc123 --api-url http://localhost:8000

# Analyze bias patterns
python cli.py analyze --api-url http://localhost:8000

# Generate fairness report
python cli.py report --api-url http://localhost:8000
```

## Data Models

### **PipelineSession**
- Complete pipeline session with all results
- Relationships to all pipeline stages

### **BiasDetectionResult**
- Bias analysis results
- Relationships to data cleaning and fairness evaluation

### **DataCleaningResult**
- Data cleaning process results
- Relationships to bias detection and fairness evaluation

### **FairnessEvaluation**
- Fairness assessment results
- Relationships to bias detection, data cleaning, and compliance

### **ComplianceLog**
- Compliance and audit information
- Relationships to fairness evaluation and predictions

### **PredictionResult**
- Final prediction results with bias mitigation
- Relationships to compliance log

## AI Agent Capabilities

With EnrichMCP, AI agents can:

1. **Explore Data Schema**: `explore_data_model()` - understand entire data structure
2. **Query with Filters**: `find_high_bias_sessions(bias_threshold=0.7)`
3. **Navigate Relationships**: `session.bias_detection().data_cleaning_result()`
4. **Get Typed Results**: All results are properly typed with Pydantic validation
5. **Discover Parameters**: AI knows what parameters are available and their types

## Benefits of Using Real EnrichMCP

| **Feature** | **Our Custom "EnrichMCP"** | **Real EnrichMCP** |
|-------------|------------------------------|-------------------|
| **Purpose** | Pipeline orchestration | AI-navigable data APIs |
| **Data Models** | Custom classes | Typed Pydantic models |
| **Relationships** | Manual navigation | Automatic relationship resolution |
| **AI Integration** | None | Built-in AI agent support |
| **Schema Discovery** | None | Automatic schema generation |
| **Type Safety** | Basic | Full Pydantic validation |
| **Parameter Hints** | None | Rich parameter metadata |

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  EnrichMCP App   │    │   AI Agents     │
│                 │    │                  │    │                 │
│ • /pipeline/run│    │ • Data Models    │    │ • explore_data  │
│ • /health       │    │ • Relationships  │    │ • query_data    │
│ • /status       │    │ • Retrievers     │    │ • navigate_data │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Results Storage                     │
│                                                                 │
│ • BiasDetectionResult                                          │
│ • DataCleaningResult                                           │
│ • FairnessEvaluation                                           │
│ • ComplianceLog                                                │
│ • PredictionResult                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Development

### Running the Application

```bash
# Start the FastAPI server with EnrichMCP
python orchestrator.py

# The server runs on http://localhost:8000
# EnrichMCP app is available for AI agents
```

### Testing with CLI

```bash
# Test data exploration
python cli.py explore

# Test pipeline execution
python cli.py run --file sample_data.json

# Test status checking
python cli.py status --session session_123
```

## What Changed

1. **Replaced Custom Library**: Removed our custom "EnrichMCP" with the real EnrichMCP library
2. **AI-Navigable Data**: Created proper data models that AI agents can explore
3. **Relationship Navigation**: Defined relationships between pipeline stages
4. **Type Safety**: Added full Pydantic validation and type hints
5. **Parameter Metadata**: Rich parameter descriptions for AI agents

## Next Steps

1. **Integrate with Real MCP Servers**: Connect to actual bias detection MCP servers
2. **Add More Data Models**: Expand the data model coverage
3. **Implement Caching**: Add request caching for better performance
4. **Add Authentication**: Implement proper authentication for AI agents
5. **Create AI Agent Examples**: Show how AI agents can use this API

---

**This demonstrates the power of the real EnrichMCP library for creating AI-navigable data APIs!** 🚀 