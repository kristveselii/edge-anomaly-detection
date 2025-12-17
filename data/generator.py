"""
Synthetic Sensor Data Generator
Simulates time-series sensor data with injected anomalies
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import argparse
from pathlib import Path


class SensorDataGenerator:
    """Generate synthetic sensor data with controllable anomalies"""
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
    def generate_normal_data(self, 
                            n_samples: int,
                            base_value: float = 50.0,
                            noise_std: float = 5.0,
                            trend: float = 0.0,
                            seasonality: bool = False) -> np.ndarray:
        """
        Generate normal sensor readings
        
        Args:
            n_samples: Number of samples to generate
            base_value: Mean value of the sensor
            noise_std: Standard deviation of noise
            trend: Linear trend component
            seasonality: Add seasonal pattern
            
        Returns:
            Array of normal sensor values
        """
        # Base signal
        time = np.arange(n_samples)
        signal = np.ones(n_samples) * base_value
        
        # Add trend
        if trend != 0.0:
            signal += trend * time
            
        # Add seasonality (daily pattern)
        if seasonality:
            period = 100  # samples per cycle
            seasonal = 10 * np.sin(2 * np.pi * time / period)
            signal += seasonal
            
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, n_samples)
        signal += noise
        
        return signal
    
    def inject_spike_anomalies(self, 
                               data: np.ndarray,
                               n_anomalies: int,
                               spike_magnitude: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject sudden spike anomalies
        
        Args:
            data: Normal data array
            n_anomalies: Number of spikes to inject
            spike_magnitude: Multiplier for spike height (std devs)
            
        Returns:
            (modified_data, labels) where labels[i] = 1 if anomaly
        """
        data = data.copy()
        labels = np.zeros(len(data), dtype=int)
        
        # Random anomaly positions (avoid first/last 10%)
        start_idx = int(len(data) * 0.1)
        end_idx = int(len(data) * 0.9)
        anomaly_indices = np.random.choice(
            range(start_idx, end_idx),
            size=n_anomalies,
            replace=False
        )
        
        data_std = np.std(data)
        data_mean = np.mean(data)
        
        for idx in anomaly_indices:
            # Random positive or negative spike
            direction = np.random.choice([-1, 1])
            spike = direction * spike_magnitude * data_std
            data[idx] = data_mean + spike
            labels[idx] = 1
            
        return data, labels
    
    def inject_drift_anomalies(self,
                               data: np.ndarray,
                               n_drifts: int = 2,
                               drift_length: int = 50,
                               drift_magnitude: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject gradual drift anomalies
        
        Args:
            data: Normal data array
            n_drifts: Number of drift periods
            drift_length: Duration of each drift
            drift_magnitude: Magnitude of drift (std devs)
            
        Returns:
            (modified_data, labels)
        """
        data = data.copy()
        labels = np.zeros(len(data), dtype=int)
        
        data_std = np.std(data)
        
        for _ in range(n_drifts):
            # Random drift start position
            max_start = len(data) - drift_length - 100
            if max_start <= 100:
                continue
                
            start_idx = np.random.randint(100, max_start)
            end_idx = start_idx + drift_length
            
            # Gradual drift
            drift = np.linspace(0, drift_magnitude * data_std, drift_length)
            data[start_idx:end_idx] += drift
            labels[start_idx:end_idx] = 1
            
        return data, labels
    
    def inject_dropout_anomalies(self,
                                 data: np.ndarray,
                                 n_dropouts: int = 3,
                                 dropout_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject sudden dropout (sensor failure) anomalies
        
        Args:
            data: Normal data array
            n_dropouts: Number of dropout events
            dropout_length: Duration of each dropout
            
        Returns:
            (modified_data, labels)
        """
        data = data.copy()
        labels = np.zeros(len(data), dtype=int)
        
        for _ in range(n_dropouts):
            max_start = len(data) - dropout_length - 100
            if max_start <= 100:
                continue
                
            start_idx = np.random.randint(100, max_start)
            end_idx = start_idx + dropout_length
            
            # Sudden drop to zero or very low value
            data[start_idx:end_idx] = 0
            labels[start_idx:end_idx] = 1
            
        return data, labels
    
    def generate_dataset(self,
                        n_samples: int = 5000,
                        anomaly_ratio: float = 0.05,
                        sensor_type: str = "temperature") -> pd.DataFrame:
        """
        Generate complete dataset with mixed anomalies
        
        Args:
            n_samples: Total number of samples
            anomaly_ratio: Fraction of anomalous points
            sensor_type: Type of sensor (affects base params)
            
        Returns:
            DataFrame with columns: timestamp, value, is_anomaly
        """
        # Sensor-specific parameters
        sensor_params = {
            "temperature": {"base": 72.0, "noise": 3.0, "seasonality": True},
            "cpu_usage": {"base": 45.0, "noise": 8.0, "seasonality": True},
            "network_latency": {"base": 50.0, "noise": 10.0, "seasonality": False},
            "pressure": {"base": 101.3, "noise": 1.5, "seasonality": False},
        }
        
        params = sensor_params.get(sensor_type, sensor_params["temperature"])
        
        # Generate normal data
        data = self.generate_normal_data(
            n_samples=n_samples,
            base_value=params["base"],
            noise_std=params["noise"],
            seasonality=params["seasonality"]
        )
        
        labels = np.zeros(n_samples, dtype=int)
        
        # Calculate anomaly budget
        total_anomalies = int(n_samples * anomaly_ratio)
        
        # Distribute anomalies across types
        n_spikes = int(total_anomalies * 0.5)
        n_drifts = 2
        n_dropouts = 2
        
        # Inject spike anomalies
        if n_spikes > 0:
            data, spike_labels = self.inject_spike_anomalies(data, n_spikes)
            labels = np.logical_or(labels, spike_labels).astype(int)
        
        # Inject drift anomalies
        data, drift_labels = self.inject_drift_anomalies(data, n_drifts)
        labels = np.logical_or(labels, drift_labels).astype(int)
        
        # Inject dropout anomalies
        data, dropout_labels = self.inject_dropout_anomalies(data, n_dropouts)
        labels = np.logical_or(labels, dropout_labels).astype(int)
        
        # Create DataFrame
        timestamps = pd.date_range(
            start='2024-01-01',
            periods=n_samples,
            freq='1min'
        )
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': data,
            'is_anomaly': labels
        })
        
        return df
    
    def generate_streaming_data(self, sensor_type: str = "temperature"):
        """
        Generator for streaming data simulation
        
        Yields:
            (timestamp, value, is_anomaly) tuples
        """
        # Generate a large dataset
        df = self.generate_dataset(
            n_samples=10000,
            sensor_type=sensor_type
        )
        
        # Yield one sample at a time
        for _, row in df.iterrows():
            yield row['timestamp'], row['value'], row['is_anomaly']


def main():
    """CLI for generating training and test datasets"""
    parser = argparse.ArgumentParser(description='Generate synthetic sensor data')
    parser.add_argument('--train', action='store_true', help='Generate training data')
    parser.add_argument('--test', action='store_true', help='Generate test data')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--sensor', type=str, default='temperature',
                       choices=['temperature', 'cpu_usage', 'network_latency', 'pressure'],
                       help='Sensor type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)
    
    generator = SensorDataGenerator(seed=args.seed)
    
    if args.train:
        print(f"Generating training data ({args.samples} samples)...")
        df = generator.generate_dataset(
            n_samples=args.samples,
            sensor_type=args.sensor,
            anomaly_ratio=0.05
        )
        
        output_path = data_dir / 'training_data.csv'
        df.to_csv(output_path, index=False)
        
        # Print statistics
        n_anomalies = df['is_anomaly'].sum()
        print(f"✓ Saved to {output_path}")
        print(f"  Total samples: {len(df)}")
        print(f"  Anomalies: {n_anomalies} ({n_anomalies/len(df)*100:.2f}%)")
        print(f"  Value range: [{df['value'].min():.2f}, {df['value'].max():.2f}]")
        print(f"  Mean: {df['value'].mean():.2f}, Std: {df['value'].std():.2f}")
        
    if args.test:
        print(f"Generating test data ({args.samples} samples)...")
        # Use different seed for test data
        generator_test = SensorDataGenerator(seed=args.seed + 1)
        df = generator_test.generate_dataset(
            n_samples=args.samples,
            sensor_type=args.sensor,
            anomaly_ratio=0.05
        )
        
        output_path = data_dir / 'test_data.csv'
        df.to_csv(output_path, index=False)
        
        n_anomalies = df['is_anomaly'].sum()
        print(f"✓ Saved to {output_path}")
        print(f"  Total samples: {len(df)}")
        print(f"  Anomalies: {n_anomalies} ({n_anomalies/len(df)*100:.2f}%)")


if __name__ == "__main__":
    main()
