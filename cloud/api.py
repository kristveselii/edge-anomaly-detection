"""
Cloud Aggregation API
Receives and stores metrics from edge devices
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import logging
import sqlite3
from pathlib import Path
from contextlib import contextmanager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models
class MetricsSummary(BaseModel):
    """Metrics summary from edge device"""
    device_id: str = Field(..., description="Unique device identifier")
    window_start: str = Field(..., description="Window start timestamp (ISO format)")
    window_end: str = Field(..., description="Window end timestamp (ISO format)")
    anomaly_count: int = Field(..., ge=0, description="Number of anomalies detected")
    total_samples: int = Field(..., gt=0, description="Total samples in window")
    avg_score: float = Field(..., description="Average anomaly score")
    max_score: float = Field(..., description="Maximum anomaly score")
    min_score: float = Field(..., description="Minimum anomaly score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "edge-001",
                "window_start": "2024-01-01T12:00:00Z",
                "window_end": "2024-01-01T12:05:00Z",
                "anomaly_count": 5,
                "total_samples": 300,
                "avg_score": -0.15,
                "max_score": -0.05,
                "min_score": -0.35
            }
        }


class DeviceStats(BaseModel):
    """Aggregated statistics for a device"""
    device_id: str
    total_windows: int
    total_samples: int
    total_anomalies: int
    anomaly_rate: float
    first_seen: str
    last_seen: str


class CloudStorage:
    """SQLite storage for metrics"""
    
    def __init__(self, db_path: str = "cloud_metrics.db"):
        self.db_path = Path(db_path)
        self._init_database()
        logger.info(f"Cloud storage initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    anomaly_count INTEGER NOT NULL,
                    total_samples INTEGER NOT NULL,
                    avg_score REAL NOT NULL,
                    max_score REAL NOT NULL,
                    min_score REAL NOT NULL,
                    received_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_device_id 
                ON metrics(device_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_received_at 
                ON metrics(received_at)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def store_metrics(self, metrics: MetricsSummary) -> int:
        """
        Store metrics summary
        
        Returns:
            ID of inserted record
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics 
                (device_id, window_start, window_end, anomaly_count, 
                 total_samples, avg_score, max_score, min_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.device_id,
                metrics.window_start,
                metrics.window_end,
                metrics.anomaly_count,
                metrics.total_samples,
                metrics.avg_score,
                metrics.max_score,
                metrics.min_score
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_device_stats(self, device_id: Optional[str] = None) -> List[DeviceStats]:
        """Get aggregated statistics per device"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if device_id:
                query = """
                    SELECT 
                        device_id,
                        COUNT(*) as total_windows,
                        SUM(total_samples) as total_samples,
                        SUM(anomaly_count) as total_anomalies,
                        CAST(SUM(anomaly_count) AS FLOAT) / SUM(total_samples) as anomaly_rate,
                        MIN(window_start) as first_seen,
                        MAX(window_end) as last_seen
                    FROM metrics
                    WHERE device_id = ?
                    GROUP BY device_id
                """
                cursor.execute(query, (device_id,))
            else:
                query = """
                    SELECT 
                        device_id,
                        COUNT(*) as total_windows,
                        SUM(total_samples) as total_samples,
                        SUM(anomaly_count) as total_anomalies,
                        CAST(SUM(anomaly_count) AS FLOAT) / SUM(total_samples) as anomaly_rate,
                        MIN(window_start) as first_seen,
                        MAX(window_end) as last_seen
                    FROM metrics
                    GROUP BY device_id
                """
                cursor.execute(query)
            
            rows = cursor.fetchall()
            
            stats = []
            for row in rows:
                stats.append(DeviceStats(
                    device_id=row[0],
                    total_windows=row[1],
                    total_samples=row[2],
                    total_anomalies=row[3],
                    anomaly_rate=row[4],
                    first_seen=row[5],
                    last_seen=row[6]
                ))
            
            return stats
    
    def get_recent_metrics(self, 
                          device_id: Optional[str] = None,
                          limit: int = 100) -> List[dict]:
        """Get recent metrics summaries"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if device_id:
                query = """
                    SELECT device_id, window_start, window_end, 
                           anomaly_count, total_samples, avg_score,
                           max_score, min_score, received_at
                    FROM metrics
                    WHERE device_id = ?
                    ORDER BY received_at DESC
                    LIMIT ?
                """
                cursor.execute(query, (device_id, limit))
            else:
                query = """
                    SELECT device_id, window_start, window_end, 
                           anomaly_count, total_samples, avg_score,
                           max_score, min_score, received_at
                    FROM metrics
                    ORDER BY received_at DESC
                    LIMIT ?
                """
                cursor.execute(query, (limit,))
            
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                metrics.append({
                    'device_id': row[0],
                    'window_start': row[1],
                    'window_end': row[2],
                    'anomaly_count': row[3],
                    'total_samples': row[4],
                    'avg_score': row[5],
                    'max_score': row[6],
                    'min_score': row[7],
                    'received_at': row[8]
                })
            
            return metrics
    
    def get_total_metrics(self) -> dict:
        """Get overall metrics across all devices"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT device_id) as total_devices,
                    COUNT(*) as total_windows,
                    SUM(total_samples) as total_samples,
                    SUM(anomaly_count) as total_anomalies
                FROM metrics
            """)
            
            row = cursor.fetchone()
            
            return {
                'total_devices': row[0] or 0,
                'total_windows': row[1] or 0,
                'total_samples': row[2] or 0,
                'total_anomalies': row[3] or 0,
                'anomaly_rate': row[3] / row[2] if row[2] else 0
            }


# Create FastAPI app
app = FastAPI(
    title="Edge Anomaly Detection - Cloud Aggregation API",
    description="Aggregates metrics from edge devices",
    version="1.0.0"
)

# Initialize storage
storage = CloudStorage()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }


@app.post("/metrics", status_code=status.HTTP_201_CREATED)
async def receive_metrics(metrics: MetricsSummary):
    """
    Receive metrics summary from edge device
    
    Args:
        metrics: Metrics summary
        
    Returns:
        Confirmation with stored metrics ID
    """
    try:
        # Store metrics
        record_id = storage.store_metrics(metrics)
        
        logger.info(
            f"Received metrics from {metrics.device_id}: "
            f"{metrics.anomaly_count}/{metrics.total_samples} anomalies "
            f"({metrics.anomaly_count/metrics.total_samples*100:.1f}%)"
        )
        
        return {
            "status": "success",
            "record_id": record_id,
            "device_id": metrics.device_id,
            "received_at": datetime.utcnow().isoformat() + 'Z'
        }
        
    except Exception as e:
        logger.error(f"Error storing metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store metrics: {str(e)}"
        )


@app.get("/devices")
async def list_devices():
    """List all devices with aggregated statistics"""
    try:
        stats = storage.get_device_stats()
        return {
            "devices": [stat.model_dump() for stat in stats],
            "count": len(stats)
        }
    except Exception as e:
        logger.error(f"Error retrieving device stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/devices/{device_id}")
async def get_device_stats(device_id: str):
    """Get statistics for specific device"""
    try:
        stats = storage.get_device_stats(device_id)
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device {device_id} not found"
            )
        
        return stats[0].model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving device stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/metrics")
async def get_metrics(device_id: Optional[str] = None, limit: int = 100):
    """
    Get recent metrics summaries
    
    Args:
        device_id: Optional device filter
        limit: Maximum number of records to return
    """
    try:
        metrics = storage.get_recent_metrics(device_id, limit)
        return {
            "metrics": metrics,
            "count": len(metrics)
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/stats")
async def get_overall_stats():
    """Get overall statistics across all devices"""
    try:
        stats = storage.get_total_metrics()
        return stats
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Edge Anomaly Detection - Cloud Aggregation",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "receive_metrics": "POST /metrics",
            "list_devices": "/devices",
            "device_stats": "/devices/{device_id}",
            "get_metrics": "/metrics?device_id={id}&limit={n}",
            "overall_stats": "/stats"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("CLOUD AGGREGATION API")
    print("="*60)
    print("Starting server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
