#!/bin/bash
echo "🚀 Starting Ethos AI Platform..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start all services
echo "🐳 Building and starting all services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
services=("bias-detector:8001" "data-cleaner:8002" "fairness-evaluator:8003" "compliance-logger:8004" "compliance-guardian:8006" "prediction-server:8005" "orchestrator:8000" "backend:8080")

for service in "${services[@]}"; do
    name="${service%%:*}"
    port="${service##*:}"
    if curl -s "http://localhost:$port/health" > /dev/null; then
        echo "✅ $name is running on port $port"
    else
        echo "⚠️  $name might still be starting on port $port"
    fi
done

echo ""
echo "🎉 Ethos AI Platform is running!"
echo ""
echo "📍 Access points:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8080"
echo "   - API Docs: http://localhost:8080/docs"
echo ""
echo "📊 Demo dataset available: demo_hiring_dataset.csv"
echo ""
echo "To stop all services: docker-compose down"
echo "To view logs: docker-compose logs -f [service-name]"
