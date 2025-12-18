"""
Edge Inference Engine
Real-time anomaly detection with sliding window
"""

import numpy as np
from collections import deque
from pathlib import Path
import logging
from typing import Tuple, Optional
import time
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from train_model import AnomalyDetector
from edge.buffer import AnomalyEvent


class EdgeInferenceEngine:
    """
    Streaming anomaly detection engine for edge devices
    Optimized for low latency and memory efficiency
    """
    
    def __init__(self,
                 model_dir: str = "models",
                 window_size: int = 10,
                 device_id: str = "edge-001"):
        """
        Args:
            model_dir: Directory containing trained model
            window_size: Size of sliding window
            device_id: Unique identifier for this edge device
        """
        self.window_size = window_size
        self.device_id = device_id
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.logger.info(f"Loading model from {model_dir}...")
        self.detector = AnomalyDetector.load(Path(model_dir))
        
        # Sliding window buffer
        self.window = deque(maxlen=window_size)
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'anomalies_detected': 0,
            'inference_times': [],
            'last_anomaly_time': None
        }
        
        self.logger.info(f"Edge inference engine initialized (device: {device_id})")
    
    def preprocess(self, value: float) -> np.ndarray:
        """
        Preprocess single value and maintain sliding window
        
        Args:
            value: Raw sensor value
            
        Returns:
            Current window as numpy array
        """
        # Add to window
        self.window.append(value)
        
        # If window not full, pad with first values
        if len(self.window) < self.window_size:
            # Pad left with first value
            padded = np.pad(
                list(self.window),
                (self.window_size - len(self.window), 0),
                mode='edge'
            )
            return padded
        
        return np.array(self.window)
    
    def infer(self, value: float, timestamp: Optional[str] = None) -> AnomalyEvent:
        """
        Run inference on single data point
        
        Args:
            value: Sensor reading
            timestamp: ISO timestamp (generated if not provided)
            
        Returns:
            AnomalyEvent with detection result
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Preprocess
        window = self.preprocess(value)
        
        # Measure inference time
        start_time = time.perf_counter()
        
        # Predict
        is_anomaly, score = self.detector.predict_single(window)
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Update statistics
        self.stats['total_samples'] += 1
        self.stats['inference_times'].append(inference_time)
        
        if is_anomaly:
            self.stats['anomalies_detected'] += 1
            self.stats['last_anomaly_time'] = timestamp
            self.logger.warning(
                f"⚠️  ANOMALY DETECTED | "
                f"Value: {value:.2f} | "
                f"Score: {score:.4f} | "
                f"Time: {timestamp}"
            )
        else:
            # Log normal samples less frequently
            if self.stats['total_samples'] % 100 == 0:
                self.logger.debug(
                    f"Normal sample #{self.stats['total_samples']} | "
                    f"Value: {value:.2f} | "
                    f"Score: {score:.4f}"
                )
        
        # Keep inference times buffer manageable
        if len(self.stats['inference_times']) > 1000:
            self.stats['inference_times'] = self.stats['inference_times'][-1000:]
        
        # Create event
        event = AnomalyEvent(
            device_id=self.device_id,
            timestamp=timestamp,
            value=value,
            score=float(score),
            is_anomaly=bool(is_anomaly)
        )
        
        return event
    
    def get_stats(self) -> dict:
        """Get current inference statistics"""
        inference_times = self.stats['inference_times']
        
        stats = {
            'device_id': self.device_id,
            'total_samples': self.stats['total_samples'],
            'anomalies_detected': self.stats['anomalies_detected'],
            'detection_rate': (
                self.stats['anomalies_detected'] / self.stats['total_samples']
                if self.stats['total_samples'] > 0 else 0
            ),
            'last_anomaly_time': self.stats['last_anomaly_time'],
        }
        
        if inference_times:
            stats.update({
                'avg_inference_ms': np.mean(inference_times),
                'max_inference_ms': np.max(inference_times),
                'min_inference_ms': np.min(inference_times),
                'p95_inference_ms': np.percentile(inference_times, 95),
            })
        
        return stats
    
    def reset_stats(self):
        """Reset statistics (useful for testing)"""
        self.stats = {
            'total_samples': 0,
            'anomalies_detected': 0,
            'inference_times': [],
            'last_anomaly_time': None
        }
        self.logger.info("Statistics reset")
    
    def reset_window(self):
        """Clear sliding window"""
        self.window.clear()
        self.logger.debug("Window reset")
