"""
Configuration for Edge Anomaly Detection System
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class EdgeConfig:
    """Edge device configuration"""
    device_id: str = "edge-001"
    model_dir: Path = Path("models")
    buffer_db: str = "edge_buffer.db"
    window_size: int = 10
    sync_interval: int = 60  # seconds
    summary_window: int = 5  # minutes
    log_level: str = "INFO"


@dataclass
class CloudConfig:
    """Cloud service configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    db_path: str = "cloud_metrics.db"
    log_level: str = "INFO"


@dataclass
class ModelConfig:
    """Model training configuration"""
    contamination: float = 0.05
    n_estimators: int = 100
    max_samples: int = 256
    window_size: int = 10
    random_state: int = 42


@dataclass
class DataConfig:
    """Data generation configuration"""
    sensor_type: str = "temperature"
    n_samples: int = 5000
    anomaly_ratio: float = 0.05
    random_seed: int = 42


class Config:
    """Global configuration"""
    
    def __init__(self):
        self.edge = EdgeConfig()
        self.cloud = CloudConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
        
        # Project root
        self.root_dir = Path(__file__).parent
        
        # Data directories
        self.data_dir = self.root_dir / "data"
        self.models_dir = self.root_dir / "models"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def get_training_data_path(self) -> Path:
        """Get path to training data"""
        return self.data_dir / "training_data.csv"
    
    def get_test_data_path(self) -> Path:
        """Get path to test data"""
        return self.data_dir / "test_data.csv"
    
    def get_model_path(self) -> Path:
        """Get path to saved model"""
        return self.models_dir / "anomaly_model.pkl"


# Global config instance
config = Config()