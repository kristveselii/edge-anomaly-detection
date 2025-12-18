"""
Local Buffer for Edge Device
Handles data persistence during cloud outages
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass, asdict
import threading


@dataclass
class AnomalyEvent:
    """Single anomaly detection event"""
    device_id: str
    timestamp: str
    value: float
    score: float
    is_anomaly: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricsSummary:
    """Aggregated metrics for cloud sync"""
    device_id: str
    window_start: str
    window_end: str
    anomaly_count: int
    total_samples: int
    avg_score: float
    max_score: float
    min_score: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class LocalBuffer:
    """
    Persistent buffer for edge data
    Uses SQLite for reliability and simplicity
    """
    
    def __init__(self, db_path: str = "edge_buffer.db", max_size: int = 10000):
        """
        Args:
            db_path: Path to SQLite database
            max_size: Maximum number of events to buffer
        """
        self.db_path = Path(db_path)
        self.max_size = max_size
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Events table (detailed anomaly events)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    value REAL NOT NULL,
                    score REAL NOT NULL,
                    is_anomaly INTEGER NOT NULL,
                    synced INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Summaries table (aggregated metrics)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    anomaly_count INTEGER NOT NULL,
                    total_samples INTEGER NOT NULL,
                    avg_score REAL NOT NULL,
                    max_score REAL NOT NULL,
                    min_score REAL NOT NULL,
                    synced INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_synced 
                ON events(synced)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_synced 
                ON summaries(synced)
            """)
            
            conn.commit()
        
        self.logger.info(f"Buffer database initialized: {self.db_path}")
    
    def add_event(self, event: AnomalyEvent):
        """Add single anomaly event to buffer"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO events 
                    (device_id, timestamp, value, score, is_anomaly)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    event.device_id,
                    event.timestamp,
                    event.value,
                    event.score,
                    int(event.is_anomaly)
                ))
                conn.commit()
        
        self._check_size_limit()
    
    def add_summary(self, summary: MetricsSummary):
        """Add metrics summary to buffer"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO summaries 
                    (device_id, window_start, window_end, anomaly_count, 
                     total_samples, avg_score, max_score, min_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    summary.device_id,
                    summary.window_start,
                    summary.window_end,
                    summary.anomaly_count,
                    summary.total_samples,
                    summary.avg_score,
                    summary.max_score,
                    summary.min_score
                ))
                conn.commit()
    
    def get_unsynced_summaries(self, limit: int = 100) -> List[Dict]:
        """Retrieve unsynced summaries for cloud upload"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, device_id, window_start, window_end,
                           anomaly_count, total_samples, avg_score,
                           max_score, min_score
                    FROM summaries
                    WHERE synced = 0
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                
                summaries = []
                for row in rows:
                    summaries.append({
                        'id': row[0],
                        'device_id': row[1],
                        'window_start': row[2],
                        'window_end': row[3],
                        'anomaly_count': row[4],
                        'total_samples': row[5],
                        'avg_score': row[6],
                        'max_score': row[7],
                        'min_score': row[8]
                    })
                
                return summaries
    
    def mark_synced(self, summary_ids: List[int]):
        """Mark summaries as successfully synced"""
        if not summary_ids:
            return
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(summary_ids))
                cursor.execute(f"""
                    UPDATE summaries
                    SET synced = 1
                    WHERE id IN ({placeholders})
                """, summary_ids)
                conn.commit()
        
        self.logger.info(f"Marked {len(summary_ids)} summaries as synced")
    
    def get_buffer_stats(self) -> Dict:
        """Get current buffer statistics"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count unsynced events
                cursor.execute("SELECT COUNT(*) FROM events WHERE synced = 0")
                unsynced_events = cursor.fetchone()[0]
                
                # Count unsynced summaries
                cursor.execute("SELECT COUNT(*) FROM summaries WHERE synced = 0")
                unsynced_summaries = cursor.fetchone()[0]
                
                # Total events
                cursor.execute("SELECT COUNT(*) FROM events")
                total_events = cursor.fetchone()[0]
                
                # Total summaries
                cursor.execute("SELECT COUNT(*) FROM summaries")
                total_summaries = cursor.fetchone()[0]
                
                return {
                    'unsynced_events': unsynced_events,
                    'unsynced_summaries': unsynced_summaries,
                    'total_events': total_events,
                    'total_summaries': total_summaries,
                    'buffer_usage_pct': (total_events / self.max_size * 100) if self.max_size > 0 else 0
                }
    
    def _check_size_limit(self):
        """Remove oldest synced events if buffer exceeds limit"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count total events
            cursor.execute("SELECT COUNT(*) FROM events")
            total = cursor.fetchone()[0]
            
            if total > self.max_size:
                # Delete oldest synced events
                excess = total - self.max_size
                cursor.execute("""
                    DELETE FROM events
                    WHERE id IN (
                        SELECT id FROM events
                        WHERE synced = 1
                        ORDER BY created_at ASC
                        LIMIT ?
                    )
                """, (excess,))
                conn.commit()
                
                self.logger.warning(f"Buffer full: removed {excess} old synced events")
    
    def clear_synced(self, older_than_hours: int = 24):
        """Clear synced data older than specified hours"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old synced events
                cursor.execute("""
                    DELETE FROM events
                    WHERE synced = 1
                    AND created_at < datetime('now', '-' || ? || ' hours')
                """, (older_than_hours,))
                
                events_deleted = cursor.rowcount
                
                # Delete old synced summaries
                cursor.execute("""
                    DELETE FROM summaries
                    WHERE synced = 1
                    AND created_at < datetime('now', '-' || ? || ' hours')
                """, (older_than_hours,))
                
                summaries_deleted = cursor.rowcount
                
                conn.commit()
        
        self.logger.info(
            f"Cleared old data: {events_deleted} events, "
            f"{summaries_deleted} summaries"
        )
    
    def reset(self):
        """Clear all buffer data (for testing)"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM events")
                cursor.execute("DELETE FROM summaries")
                conn.commit()
        
        self.logger.warning("Buffer reset: all data cleared")


class SummaryAggregator:
    """Aggregates events into summaries for efficient cloud sync"""
    
    def __init__(self, window_minutes: int = 5):
        """
        Args:
            window_minutes: Size of aggregation window
        """
        self.window_minutes = window_minutes
        self.current_window = []
        self.window_start = None
        self.logger = logging.getLogger(__name__)
    
    def add_event(self, event: AnomalyEvent) -> Optional[MetricsSummary]:
        """
        Add event and return summary if window is complete
        
        Returns:
            MetricsSummary if window completed, None otherwise
        """
        timestamp = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
        
        # Initialize window
        if self.window_start is None:
            self.window_start = timestamp
        
        # Check if window is complete
        window_elapsed = (timestamp - self.window_start).total_seconds() / 60
        
        if window_elapsed >= self.window_minutes:
            # Create summary for completed window
            summary = self._create_summary(event.device_id)
            
            # Start new window
            self.current_window = [event]
            self.window_start = timestamp
            
            return summary
        else:
            # Add to current window
            self.current_window.append(event)
            return None
    
    def _create_summary(self, device_id: str) -> MetricsSummary:
        """Create summary from current window"""
        if not self.current_window:
            raise ValueError("Cannot create summary from empty window")
        
        anomaly_count = sum(1 for e in self.current_window if e.is_anomaly)
        scores = [e.score for e in self.current_window]
        
        window_end = datetime.fromisoformat(
            self.current_window[-1].timestamp.replace('Z', '+00:00')
        )
        
        summary = MetricsSummary(
            device_id=device_id,
            window_start=self.window_start.isoformat(),
            window_end=window_end.isoformat(),
            anomaly_count=anomaly_count,
            total_samples=len(self.current_window),
            avg_score=float(sum(scores) / len(scores)),
            max_score=float(max(scores)),
            min_score=float(min(scores))
        )
        
        return summary
    
    def flush(self, device_id: str) -> Optional[MetricsSummary]:
        """Force creation of summary from current window"""
        if not self.current_window:
            return None
        
        summary = self._create_summary(device_id)
        self.current_window = []
        self.window_start = None
        
        return summary
