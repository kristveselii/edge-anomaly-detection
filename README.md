# Edge AI Anomaly Detection System

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> Production-ready edge ML system with offline-first anomaly detection and cloud aggregation

## 🎯 Project Overview

A complete edge-to-cloud ML pipeline demonstrating **production ML systems engineering**. This isn't just a trained model—it's a fault-tolerant, offline-first system that runs real-time inference on edge devices with automatic cloud synchronization.

### What Makes This Special
- **Offline-First**: Works without cloud connectivity (zero data loss)
- **Sub-3ms Inference**: Real-time anomaly detection on streaming data
- **Fault Tolerant**: Automatic retry logic with exponential backoff
- **Production Ready**: Docker, monitoring, comprehensive logging

## 🏗️ Architecture

┌─────────────────────────────────────┐
│        EDGE DEVICE (Offline)        │
│  ┌────────┐  ┌────────┐  ┌────────┐│
│  │Sensor  │→ │ML Model│→ │Buffer  ││
│  │Stream  │  │(<3ms)  │  │(SQLite)││
│  └────────┘  └────────┘  └────┬───┘│
└──────────────────────────────┼─────┘
                               │ (Periodic sync with retry)
                               ▼
┌─────────────────────────────────────┐
│      CLOUD AGGREGATION SERVICE      │
│  ┌────────┐  ┌──────────────────┐  │
│  │FastAPI │→ │ Multi-device DB  │  │
│  └────────┘  └──────────────────┘  │
└─────────────────────────────────────┘

## ✨ Key Features

### ML & Inference
- ✅ Isolation Forest model (1.1MB, edge-optimized)
- ✅ 98.3% precision on normal samples
- ✅ 2.8ms average inference time
- ✅ Sliding window feature engineering
- ✅ Three anomaly types: spikes, drifts, dropouts

### System Reliability
- ✅ Offline-first: Continues operation without cloud
- ✅ Zero data loss: SQLite buffering with ACID guarantees
- ✅ Automatic recovery: Syncs when cloud returns
- ✅ Exponential backoff: Intelligent retry logic
- ✅ Thread-safe: Concurrent inference and sync

### Production Features
- ✅ Docker: Multi-container orchestration
- ✅ Health checks: Endpoint monitoring
- ✅ Logging: Structured, leveled logging
- ✅ Metrics: Performance and detection statistics
- ✅ API docs: Auto-generated (FastAPI)

## 📊 Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Inference Latency | 2.8ms avg | <5ms |
| Model Size | 1.1MB | Edge-deployable |
| Precision (Normal) | 98.3% | >95% |
| Recall (Anomaly) | 65.7% | >60% |
| Data Loss | 0% | 0% |

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- pip
- (Optional) Docker & Docker Compose

### 1. Clone Repository

git clone https://github.com/kristveselii/edge-anomaly-detection.git
cd edge-anomaly-detection

### 2. Setup Environment

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Generate Data & Train Model

# Generate synthetic training data
python data/generator.py --train --samples 5000

# Train the model
python train_model.py

### 4. Run the System

Terminal 1 - Cloud API:
python cloud/api.py

Terminal 2 - Edge Device:
python edge/edge_service.py --device-id edge-001

## 🐳 Docker Deployment

### Quick Start

cd docker
docker compose up --build

This starts:
- 1 Cloud API instance (port 8000)
- 2 Edge device instances

### Verify

# Check containers
docker ps

# Query API
curl http://localhost:8000/stats

# View logs
docker logs -f edge-device-1

## 🧪 Testing Offline Behavior

Terminal 1 - Start Edge (Cloud Offline):
python edge/edge_service.py --device-id test --duration 120

You'll see: Cloud unreachable, buffering data locally

Terminal 2 - Start Cloud (After 30 seconds):
python cloud/api.py

Terminal 1 shows: Cloud recovered, syncing buffered data, zero data loss!

## 📁 Project Structure

edge-anomaly-detection/
├── data/
│   └── generator.py              # Synthetic data with anomalies
├── edge/
│   ├── inference_engine.py       # ML inference (<3ms)
│   ├── buffer.py                 # SQLite persistence
│   └── edge_service.py           # Main orchestrator
├── cloud/
│   └── api.py                    # FastAPI aggregation
├── models/
│   ├── anomaly_model.pkl         # Trained model
│   ├── scaler.pkl                # Feature scaler
│   └── metadata.json             # Model config
├── docker/
│   ├── Dockerfile.edge
│   ├── Dockerfile.cloud
│   └── docker-compose.yml
├── train_model.py
├── config.py
└── requirements.txt

## 🛠️ Technology Stack

- scikit-learn (Isolation Forest)
- Python 3.12
- FastAPI (async API)
- SQLite (embedded DB)
- Docker & Docker Compose
- NumPy, Pandas

## 🔧 Configuration

Command-line arguments:
python edge/edge_service.py --device-id edge-002 --cloud-url http://api.example.com --sync-interval 30 --offline

## 📚 API Endpoints

GET  /health              # Health check
POST /metrics             # Receive edge metrics
GET  /devices             # List all devices
GET  /devices/{id}        # Device statistics
GET  /stats               # Overall statistics
GET  /docs                # Interactive API docs

## 🎓 What This Demonstrates

- ML systems engineering (not just modeling)
- Production deployment patterns
- Fault-tolerant distributed systems
- Edge computing constraints
- API design and integration
- Docker containerization
- Comprehensive testing

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

Krist Veseli
GitHub: [@kristveselii](https://github.com/kristveselii)

## 🙏 Acknowledgments

Built as a portfolio project demonstrating ML systems engineering skills.