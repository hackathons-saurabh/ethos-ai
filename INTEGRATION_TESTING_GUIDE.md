# Frontend-Backend Integration Testing Guide

## ✅ Current Status: FULLY INTEGRATED

Your frontend and backend are **properly integrated** and all services are running correctly!

## 🚀 How to Test the Integration

### Method 1: Automated Python Test
```bash
python3 test_integration.py
```

### Method 2: Browser Test
1. Open `test_frontend_backend.html` in your browser
2. The page will automatically run integration tests
3. Check the results for each endpoint

### Method 3: Manual Testing

#### Step 1: Start All Services
```bash
# Start all services
docker-compose up -d

# Check if all containers are running
docker ps
```

#### Step 2: Test Backend Health
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-19T22:40:57.672414",
  "services": {
    "api": "running",
    "orchestrator": "connected",
    "models": "fallback"
  }
}
```

#### Step 3: Test Chat API
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, test message",
    "scenario": "hiring",
    "ethos_enabled": true
  }'
```

#### Step 4: Test File Upload
```bash
# Create a test CSV file
echo "name,age,gender,target
John,25,male,1
Jane,30,female,0" > test.csv

# Upload the file
curl -X POST http://localhost:8080/upload/dataset \
  -F "file=@test.csv" \
  -F "name=test_dataset" \
  -F "target_column=target"
```

#### Step 5: Test Frontend
1. Open your browser to `http://localhost:3000`
2. Try the following features:
   - **Run a Demo**: Click "Run Demo" on any demo card
   - **Chat Interface**: Toggle "Show Chat Interface" and send messages
   - **File Upload**: Upload a CSV file
   - **Ethos Toggle**: Switch between Ethos ON/OFF modes

## 🔍 What Happens When You Test Frontend

### 1. **Demo Execution Flow**
```
Frontend (React) → Backend API → Orchestrator → MCP Servers
```

When you click "Run Demo":
1. Frontend sends request to `/pipeline/run`
2. Backend processes through orchestrator
3. MCP servers (bias-detector, data-cleaner, etc.) process the data
4. Results return to frontend for display

### 2. **Chat Interface Flow**
```
Frontend → Backend `/chat` → Bias Detection → Response
```

When you send a message:
1. Frontend sends message to `/chat` endpoint
2. Backend processes with bias detection
3. Returns response with bias score
4. Frontend displays the response

### 3. **File Upload Flow**
```
Frontend → Backend `/upload/dataset` → Data Processing → Dataset Info
```

When you upload a file:
1. Frontend sends file to `/upload/dataset`
2. Backend parses and validates the CSV
3. Detects sensitive attributes
4. Returns dataset information

## 🛠️ Troubleshooting

### If Backend is Not Responding
```bash
# Check backend logs
docker logs ethos-backend

# Restart backend
docker-compose restart backend

# Check if backend is healthy
curl http://localhost:8080/health
```

### If Frontend Can't Connect
```bash
# Check frontend logs
docker logs ethos-frontend

# Restart frontend
docker-compose restart frontend

# Check if frontend is accessible
curl http://localhost:3000
```

### If MCP Servers are Down
```bash
# Check all MCP server logs
docker logs ethos-bias-detector
docker logs ethos-data-cleaner
docker logs ethos-fairness-evaluator
docker logs ethos-compliance-logger
docker logs ethos-prediction-server

# Restart all services
docker-compose restart
```

## 📊 Integration Verification Checklist

- [x] **Backend Health**: `http://localhost:8080/health` returns 200
- [x] **Chat API**: `/chat` endpoint responds correctly
- [x] **File Upload**: `/upload/dataset` accepts CSV files
- [x] **Frontend Access**: `http://localhost:3000` loads correctly
- [x] **CORS**: Frontend can make requests to backend
- [x] **MCP Pipeline**: All 5 MCP servers are running
- [x] **Orchestrator**: Backend can communicate with orchestrator

## 🎯 Testing Scenarios

### Scenario 1: Hiring Bias Demo
1. Go to frontend: `http://localhost:3000`
2. Click "Run Demo" on "ML Hiring Bias" card
3. Watch the processing animation
4. See the comparison results (With/Without Ethos)
5. Toggle "Show Chat Interface"
6. Send messages like "Should we hire this female candidate?"
7. Observe bias scores and responses

### Scenario 2: File Upload Test
1. Create a CSV file with hiring data
2. Upload it through the frontend
3. Verify the dataset info is displayed
4. Check that sensitive attributes are detected

### Scenario 3: Chat Interface Test
1. Enable chat interface
2. Toggle between "Ethos ON" and "Ethos OFF"
3. Send the same message in both modes
4. Compare the bias scores and responses

## 🔧 Environment Configuration

### Frontend Configuration
- **API URL**: `http://localhost:8080` (set in docker-compose.yml)
- **Port**: 3000 (mapped from container port 80)
- **CORS**: Configured to allow all origins

### Backend Configuration
- **Port**: 8080
- **CORS**: Configured to allow frontend requests
- **Orchestrator**: Connected to `http://orchestrator:8000`

### MCP Servers
- **bias-detector**: Port 8001
- **data-cleaner**: Port 8002
- **fairness-evaluator**: Port 8003
- **compliance-logger**: Port 8004
- **prediction-server**: Port 8005

## 🚀 Quick Start Commands

```bash
# Start all services
docker-compose up -d

# Check status
docker ps

# Run integration test
python3 test_integration.py

# Open frontend
open http://localhost:3000

# Check backend health
curl http://localhost:8080/health
```

## ✅ Success Indicators

When everything is working correctly, you should see:

1. **All containers running**: `docker ps` shows 7 containers
2. **Backend healthy**: Health check returns 200
3. **Frontend accessible**: `http://localhost:3000` loads
4. **Chat working**: Messages get responses with bias scores
5. **File upload working**: CSV files upload successfully
6. **Demos working**: Demo cards show processing and results

Your integration is **fully functional** and ready for development and testing! 🎉 