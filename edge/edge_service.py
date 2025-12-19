"""
Edge Service - Main Orchestrator
Coordinates inference, buffering, and cloud sync
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import logging
import argparse
import requests
from datetime import datetime
from typing import Optional
import signal
import threading

from edge.inference_engine import EdgeInferenceEngine
from edge.buffer import LocalBuffer, SummaryAggregator, AnomalyEvent
from data.generator import SensorDataGenerator


class CloudSync:
    """
    Handles synchronization with cloud aggregation service
    Implements retry logic with exponential backoff
    """
    
    def __init__(self,
                 cloud_url: str,
                 timeout: int = 5,
                 max_retries: int = 3,
                 backoff_factor: float = 2.0):
        """
        Args:
            cloud_url: URL of cloud aggregation endpoint
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff multiplier
        """
        self.cloud_url = cloud_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
        
        self.stats = {
            'sync_attempts': 0,
            'sync_successes': 0,
            'sync_failures': 0,
            'last_sync_time': None,
            'cloud_available': False
        }
    
    def send_summary(self, summary: dict) -> bool:
        """
        Send metrics summary to cloud
        
        Args:
            summary: Metrics summary dictionary
            
        Returns:
            True if successful, False otherwise
        """
        self.stats['sync_attempts'] += 1
        
        retry_delay = 1.0  # Initial retry delay
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.cloud_url}/metrics",
                    json=summary,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    self.stats['sync_successes'] += 1
                    self.stats['last_sync_time'] = datetime.utcnow().isoformat()
                    self.stats['cloud_available'] = True
                    
                    if attempt > 0:
                        self.logger.info(
                            f"✓ Cloud sync successful after {attempt + 1} attempts"
                        )
                    
                    return True
                else:
                    self.logger.warning(
                        f"Cloud sync failed: HTTP {response.status_code}"
                    )
                    
            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"Cloud sync timeout (attempt {attempt + 1}/{self.max_retries})"
                )
            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    f"Cloud unreachable (attempt {attempt + 1}/{self.max_retries})"
                )
            except Exception as e:
                self.logger.error(f"Cloud sync error: {e}")
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= self.backoff_factor
        
        # All retries failed
        self.stats['sync_failures'] += 1
        self.stats['cloud_available'] = False
        return False
    
    def check_health(self) -> bool:
        """Check if cloud endpoint is healthy"""
        try:
            response = requests.get(
                f"{self.cloud_url}/health",
                timeout=2
            )
            is_healthy = response.status_code == 200
            self.stats['cloud_available'] = is_healthy
            return is_healthy
        except:
            self.stats['cloud_available'] = False
            return False
    
    def get_stats(self) -> dict:
        """Get sync statistics"""
        return self.stats.copy()


class EdgeService:
    """
    Main edge service orchestrator
    Coordinates inference, buffering, and cloud sync
    """
    
    def __init__(self,
                 device_id: str = "edge-001",
                 model_dir: str = "models",
                 cloud_url: Optional[str] = None,
                 buffer_db: str = "edge_buffer.db",
                 sync_interval: int = 60,
                 summary_window: int = 5):
        """
        Args:
            device_id: Unique device identifier
            model_dir: Directory containing trained model
            cloud_url: Cloud aggregation endpoint URL (None = offline mode)
            buffer_db: Path to buffer database
            sync_interval: Cloud sync interval in seconds
            summary_window: Summary aggregation window in minutes
        """
        self.device_id = device_id
        self.cloud_url = cloud_url
        self.sync_interval = sync_interval
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.logger.info("Initializing edge service...")
        
        # Inference engine
        self.inference_engine = EdgeInferenceEngine(
            model_dir=model_dir,
            device_id=device_id
        )
        
        # Local buffer
        self.buffer = LocalBuffer(db_path=buffer_db)
        
        # Summary aggregator
        self.aggregator = SummaryAggregator(window_minutes=summary_window)
        
        # Cloud sync (only if URL provided)
        self.cloud_sync = None
        if cloud_url:
            self.cloud_sync = CloudSync(cloud_url)
            self.logger.info(f"Cloud sync enabled: {cloud_url}")
        else:
            self.logger.info("Running in OFFLINE mode")
        
        # Control flags
        self.running = False
        self.sync_thread = None
        
        self.logger.info(f"Edge service initialized (device: {device_id})")
    
    def _sync_loop(self):
        """Background thread for periodic cloud sync"""
        self.logger.info(f"Sync thread started (interval: {self.sync_interval}s)")
        
        while self.running:
            try:
                # Wait for sync interval
                time.sleep(self.sync_interval)
                
                if not self.running:
                    break
                
                # Sync buffered summaries
                self._sync_buffered_data()
                
            except Exception as e:
                self.logger.error(f"Error in sync loop: {e}", exc_info=True)
        
        self.logger.info("Sync thread stopped")
    
    def _sync_buffered_data(self):
        """Sync buffered summaries to cloud"""
        if not self.cloud_sync:
            return
        
        # Get unsynced summaries
        summaries = self.buffer.get_unsynced_summaries(limit=100)
        
        if not summaries:
            self.logger.debug("No data to sync")
            return
        
        self.logger.info(f"Syncing {len(summaries)} summaries to cloud...")
        
        synced_ids = []
        
        for summary in summaries:
            # Remove database ID before sending
            summary_data = {k: v for k, v in summary.items() if k != 'id'}
            
            # Attempt to send
            if self.cloud_sync.send_summary(summary_data):
                synced_ids.append(summary['id'])
            else:
                self.logger.warning(
                    f"Failed to sync summary {summary['id']}, will retry later"
                )
                break  # Stop trying if cloud is down
        
        # Mark successfully synced
        if synced_ids:
            self.buffer.mark_synced(synced_ids)
            self.logger.info(f"✓ Successfully synced {len(synced_ids)} summaries")
        
        # Log buffer stats
        buffer_stats = self.buffer.get_buffer_stats()
        if buffer_stats['unsynced_summaries'] > 0:
            self.logger.warning(
                f"Buffer contains {buffer_stats['unsynced_summaries']} "
                f"unsynced summaries ({buffer_stats['buffer_usage_pct']:.1f}% full)"
            )
    
    def process_event(self, event: AnomalyEvent):
        """
        Process single inference event
        
        Args:
            event: Anomaly detection event
        """
        # Always buffer events locally
        self.buffer.add_event(event)
        
        # Aggregate into summaries
        summary = self.aggregator.add_event(event)
        
        if summary:
            # Window completed, buffer summary
            self.buffer.add_summary(summary)
            self.logger.info(
                f"Summary created: {summary.anomaly_count}/{summary.total_samples} "
                f"anomalies ({summary.anomaly_count/summary.total_samples*100:.1f}%)"
            )
    
    def run(self, data_generator, duration: Optional[int] = None):
        """
        Run edge service with streaming data
        
        Args:
            data_generator: Data generator yielding (timestamp, value, label) tuples
            duration: Optional duration in seconds (None = run indefinitely)
        """
        self.running = True
        
        # Start sync thread if cloud sync enabled
        if self.cloud_sync:
            self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self.sync_thread.start()
        
        # Process stream
        start_time = time.time()
        sample_count = 0
        
        try:
            for timestamp, value, _ in data_generator:
                # Run inference
                event = self.inference_engine.infer(value, timestamp.isoformat() + 'Z')
                
                # Process event
                self.process_event(event)
                
                sample_count += 1
                
                # Log progress every 100 samples
                if sample_count % 100 == 0:
                    stats = self.inference_engine.get_stats()
                    self.logger.info(
                        f"📊 Processed {stats['total_samples']} samples | "
                        f"Anomalies: {stats['anomalies_detected']} "
                        f"({stats['detection_rate']:.2%})"
                    )
                
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    self.logger.info(f"Duration limit reached ({duration}s)")
                    break
                
        except KeyboardInterrupt:
            self.logger.info("Service interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop edge service gracefully"""
        self.logger.info("Stopping edge service...")
        self.running = False
        
        # Flush remaining data
        summary = self.aggregator.flush(self.device_id)
        if summary:
            self.buffer.add_summary(summary)
            self.logger.info("Flushed remaining summary")
        
        # Final sync attempt
        if self.cloud_sync:
            self.logger.info("Final sync attempt...")
            self._sync_buffered_data()
            
            if self.sync_thread:
                self.sync_thread.join(timeout=5)
        
        # Print final statistics
        self._print_summary()
    
    def _print_summary(self):
        """Print service summary"""
        print("\n" + "="*60)
        print("EDGE SERVICE SUMMARY")
        print("="*60)
        
        # Inference stats
        inference_stats = self.inference_engine.get_stats()
        print(f"\nInference Statistics:")
        print(f"  Device: {inference_stats['device_id']}")
        print(f"  Samples processed: {inference_stats['total_samples']}")
        print(f"  Anomalies detected: {inference_stats['anomalies_detected']}")
        print(f"  Detection rate: {inference_stats['detection_rate']:.2%}")
        
        if 'avg_inference_ms' in inference_stats:
            print(f"  Avg inference time: {inference_stats['avg_inference_ms']:.2f} ms")
        
        # Buffer stats
        buffer_stats = self.buffer.get_buffer_stats()
        print(f"\nBuffer Statistics:")
        print(f"  Unsynced summaries: {buffer_stats['unsynced_summaries']}")
        print(f"  Total events buffered: {buffer_stats['total_events']}")
        print(f"  Buffer usage: {buffer_stats['buffer_usage_pct']:.1f}%")
        
        # Cloud sync stats
        if self.cloud_sync:
            sync_stats = self.cloud_sync.get_stats()
            print(f"\nCloud Sync Statistics:")
            print(f"  Sync attempts: {sync_stats['sync_attempts']}")
            print(f"  Successes: {sync_stats['sync_successes']}")
            print(f"  Failures: {sync_stats['sync_failures']}")
            success_rate = sync_stats['sync_successes']/max(sync_stats['sync_attempts'],1)*100
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Cloud available: {sync_stats['cloud_available']}")
            if sync_stats['last_sync_time']:
                print(f"  Last sync: {sync_stats['last_sync_time']}")
        
        print("="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Edge Anomaly Detection Service')
    parser.add_argument('--device-id', type=str, default='edge-001',
                       help='Device identifier')
    parser.add_argument('--cloud-url', type=str, default='http://localhost:8000',
                       help='Cloud aggregation URL (omit for offline mode)')
    parser.add_argument('--offline', action='store_true',
                       help='Run in offline mode (no cloud sync)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Model directory')
    parser.add_argument('--duration', type=int, default=None,
                       help='Run duration in seconds (None = indefinite)')
    parser.add_argument('--sensor', type=str, default='temperature',
                       choices=['temperature', 'cpu_usage', 'network_latency', 'pressure'],
                       help='Sensor type for data generation')
    parser.add_argument('--sync-interval', type=int, default=30,
                       help='Cloud sync interval in seconds')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    # Banner
    print("\n" + "="*60)
    print("EDGE ANOMALY DETECTION SERVICE")
    print("="*60)
    print(f"Device ID: {args.device_id}")
    print(f"Sensor Type: {args.sensor}")
    
    if args.offline:
        print("Mode: OFFLINE (no cloud sync)")
        cloud_url = None
    else:
        print(f"Cloud URL: {args.cloud_url}")
        print(f"Sync Interval: {args.sync_interval}s")
        cloud_url = args.cloud_url
    
    print("="*60 + "\n")
    
    # Create service
    service = EdgeService(
        device_id=args.device_id,
        model_dir=args.model_dir,
        cloud_url=cloud_url,
        sync_interval=args.sync_interval
    )
    
    # Create data generator
    generator = SensorDataGenerator(seed=42)
    data_stream = generator.generate_streaming_data(sensor_type=args.sensor)
    
    # Setup signal handler
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run service
    logger.info("Starting edge service...")
    service.run(data_stream, duration=args.duration)


if __name__ == "__main__":
    main()
