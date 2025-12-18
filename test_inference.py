"""Quick test of inference engine"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from edge.inference_engine import EdgeInferenceEngine
from data.generator import SensorDataGenerator

# Create engine
print("Loading model...")
engine = EdgeInferenceEngine(model_dir="models", device_id="test-device")

# Generate test data
print("Generating test data...")
generator = SensorDataGenerator(seed=42)

# Run 100 inferences
print("\nRunning 100 inferences...\n")
count = 0
for timestamp, value, true_label in generator.generate_streaming_data():
    event = engine.infer(value, timestamp.isoformat() + 'Z')
    
    if event.is_anomaly:
        print(f"🔴 ANOMALY: value={value:.2f}, score={event.score:.4f}")
    
    count += 1
    if count >= 100:
        break

# Print stats
stats = engine.get_stats()
print(f"\n{'='*60}")
print("STATISTICS")
print('='*60)
print(f"Samples processed: {stats['total_samples']}")
print(f"Anomalies detected: {stats['anomalies_detected']}")
print(f"Detection rate: {stats['detection_rate']:.2%}")
print(f"Avg inference time: {stats['avg_inference_ms']:.3f} ms")
print(f"P95 inference time: {stats['p95_inference_ms']:.3f} ms")
print('='*60)
