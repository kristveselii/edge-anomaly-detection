#!/bin/bash

echo "=================================================="
echo "Edge Anomaly Detection - Docker Deployment"
echo "=================================================="
echo ""
echo "Building containers..."
docker-compose build

echo ""
echo "Starting services..."
docker-compose up -d

echo ""
echo "Waiting for services to be ready..."
sleep 10

echo ""
echo "✓ Cloud API: http://localhost:8000"
echo "✓ API Docs: http://localhost:8000/docs"
echo ""
echo "View logs:"
echo "  docker logs -f edge-cloud-api"
echo "  docker logs -f edge-device-1"
echo "  docker logs -f edge-device-2"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo ""
echo "=================================================="
