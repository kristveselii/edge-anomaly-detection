"""
Anomaly Detection Model Training
Trains an Isolation Forest model for edge deployment
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path
import json
import argparse


class AnomalyDetector:
    """Isolation Forest-based anomaly detector optimized for edge deployment"""
    
    def __init__(self, 
                 contamination: float = 0.05,
                 n_estimators: int = 100,
                 max_samples: int = 256,
                 random_state: int = 42):
        """
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of isolation trees
            max_samples: Samples per tree (lower = faster inference)
            random_state: Random seed
        """
        self.contamination = contamination
        self.random_state = random_state
        
        # Model optimized for edge: fewer trees, smaller samples
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=1,  # Edge devices typically single-core
            warm_start=False
        )
        
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False
        
    def create_features(self, data: pd.DataFrame, window_size: int = 10) -> np.ndarray:
        """
        Create sliding window features
        
        Args:
            data: DataFrame with 'value' column
            window_size: Size of sliding window
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        values = data['value'].values
        features = []
        
        for i in range(len(values)):
            if i < window_size - 1:
                # Pad beginning with first value
                window = np.pad(
                    values[:i+1],
                    (window_size - i - 1, 0),
                    mode='edge'
                )
            else:
                window = values[i - window_size + 1:i + 1]
            
            # Statistical features
            feat = [
                window[-1],              # Current value
                np.mean(window),         # Window mean
                np.std(window),          # Window std
                np.max(window),          # Window max
                np.min(window),          # Window min
                window[-1] - np.mean(window),  # Deviation from mean
                np.ptp(window),          # Peak-to-peak (range)
            ]
            
            # Rate of change
            if len(window) > 1:
                feat.append(window[-1] - window[-2])
            else:
                feat.append(0)
                
            features.append(feat)
        
        return np.array(features)
    
    def train(self, data: pd.DataFrame, window_size: int = 10):
        """
        Train the anomaly detector
        
        Args:
            data: Training data with 'value' column
            window_size: Sliding window size
        """
        print("Creating features...")
        X = self.create_features(data, window_size)
        
        print(f"Feature matrix shape: {X.shape}")
        
        # Fit scaler
        print("Fitting scaler...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        print("Training Isolation Forest...")
        self.model.fit(X_scaled)
        
        # Calculate anomaly scores
        scores = self.model.score_samples(X_scaled)
        
        # Determine threshold using contamination rate
        # Lower (more negative) scores = more anomalous
        self.threshold = np.percentile(scores, self.contamination * 100)
        
        self.is_fitted = True
        
        print(f"✓ Training complete")
        print(f"  Threshold: {self.threshold:.4f}")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        return scores
    
    def predict(self, data: pd.DataFrame, window_size: int = 10) -> tuple:
        """
        Predict anomalies
        
        Args:
            data: Data with 'value' column
            window_size: Sliding window size
            
        Returns:
            (predictions, scores) where predictions[i] = 1 if anomaly
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.create_features(data, window_size)
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        scores = self.model.score_samples(X_scaled)
        
        # Predict: 1 if anomaly, 0 if normal
        predictions = (scores < self.threshold).astype(int)
        
        return predictions, scores
    
    def predict_single(self, window: np.ndarray) -> tuple:
        """
        Predict anomaly for a single window (for streaming inference)
        
        Args:
            window: Array of recent values
            
        Returns:
            (is_anomaly, score)
        """
        if not self.is_fitted:
            raise ValueError("Model not trained.")
        
        # Create features for single window
        feat = [
            window[-1],
            np.mean(window),
            np.std(window),
            np.max(window),
            np.min(window),
            window[-1] - np.mean(window),
            np.ptp(window),
            window[-1] - window[-2] if len(window) > 1 else 0
        ]
        
        X = np.array([feat])
        X_scaled = self.scaler.transform(X)
        
        score = self.model.score_samples(X_scaled)[0]
        is_anomaly = score < self.threshold
        
        return is_anomaly, score
    
    def evaluate(self, data: pd.DataFrame, window_size: int = 10):
        """
        Evaluate model on labeled data
        
        Args:
            data: Data with 'value' and 'is_anomaly' columns
            window_size: Sliding window size
        """
        predictions, scores = self.predict(data, window_size)
        true_labels = data['is_anomaly'].values
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            true_labels,
            predictions,
            target_names=['Normal', 'Anomaly'],
            digits=4
        ))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_labels, predictions)
        print(f"                Predicted")
        print(f"              Normal  Anomaly")
        print(f"Actual Normal   {cm[0, 0]:5d}   {cm[0, 1]:5d}")
        print(f"       Anomaly  {cm[1, 0]:5d}   {cm[1, 1]:5d}")
        
        # ROC AUC
        try:
            auc = roc_auc_score(true_labels, -scores)  # Negative because lower scores = anomalies
            print(f"\nROC AUC Score: {auc:.4f}")
        except:
            print("\nROC AUC: N/A (need both classes)")
        
        # Detection rate
        detection_rate = predictions.sum() / len(predictions)
        true_anomaly_rate = true_labels.sum() / len(true_labels)
        
        print(f"\nDetection Statistics:")
        print(f"  Predicted anomaly rate: {detection_rate:.2%}")
        print(f"  True anomaly rate: {true_anomaly_rate:.2%}")
        print(f"  Threshold: {self.threshold:.4f}")
        
        return {
            'predictions': predictions,
            'scores': scores,
            'confusion_matrix': cm,
            'detection_rate': detection_rate
        }
    
    def save(self, model_dir: Path):
        """Save model and scaler"""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / 'anomaly_model.pkl'
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = model_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'contamination': self.contamination,
            'threshold': float(self.threshold),
            'n_estimators': self.model.n_estimators,
            'max_samples': self.model.max_samples,
            'random_state': self.random_state
        }
        
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Model saved to {model_dir}")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        print(f"  Metadata: {metadata_path}")
    
    @classmethod
    def load(cls, model_dir: Path):
        """Load saved model"""
        # Load metadata
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        detector = cls(
            contamination=metadata['contamination'],
            random_state=metadata['random_state']
        )
        
        # Load model and scaler
        model_path = model_dir / 'anomaly_model.pkl'
        detector.model = joblib.load(model_path)
        
        scaler_path = model_dir / 'scaler.pkl'
        detector.scaler = joblib.load(scaler_path)
        
        detector.threshold = metadata['threshold']
        detector.is_fitted = True
        
        print(f"✓ Model loaded from {model_dir}")
        
        return detector


def main():
    """Train and evaluate anomaly detection model"""
    parser = argparse.ArgumentParser(description='Train anomaly detection model')
    parser.add_argument('--data', type=str, default='data/training_data.csv',
                       help='Path to training data')
    parser.add_argument('--test', type=str, default='data/test_data.csv',
                       help='Path to test data')
    parser.add_argument('--window', type=int, default=10,
                       help='Sliding window size')
    parser.add_argument('--contamination', type=float, default=0.05,
                       help='Expected anomaly rate')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for model')
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading training data from {args.data}...")
    train_df = pd.read_csv(args.data)
    print(f"  Loaded {len(train_df)} samples")
    
    # Train model
    detector = AnomalyDetector(contamination=args.contamination)
    detector.train(train_df, window_size=args.window)
    
    # Evaluate on training data
    print("\nEvaluating on training data...")
    detector.evaluate(train_df, window_size=args.window)
    
    # Evaluate on test data if available
    test_path = Path(args.test)
    if test_path.exists():
        print(f"\n{'='*60}")
        print("TESTING ON HELD-OUT DATA")
        print('='*60)
        test_df = pd.read_csv(test_path)
        print(f"Loaded {len(test_df)} test samples")
        detector.evaluate(test_df, window_size=args.window)
    
    # Save model
    output_dir = Path(args.output)
    detector.save(output_dir)
    
    print("\n" + "="*60)
    print("MODEL READY FOR EDGE DEPLOYMENT")
    print("="*60)
    model_size = (output_dir / 'anomaly_model.pkl').stat().st_size / 1024
    print(f"Model size: {model_size:.2f} KB")
    print(f"Threshold: {detector.threshold:.4f}")
    print(f"Expected inference time: < 1ms per sample")


if __name__ == "__main__":
    main()
