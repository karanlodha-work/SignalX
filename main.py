import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import yaml
import psutil
from typing import Dict, List

import logging
from src.scraper.twitter_scraper import TwitterScraper
from src.processor.data_cleaner import DataCleaner
from src.processor.deduplicator import Deduplicator
from src.storage.parquet_handler import ParquetHandler
from src.analysis.signal_generator import SignalGenerator
from src.analysis.visualizer import Visualizer


# Orchestrates the entire market intelligence data collection and analysis pipeline
class MarketIntelligenceSystem:

    # Initializes system with configuration, logging, and component instances
    def __init__(self, config_path):
        self.config = self._load_config(config_path)

        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info("=" * 80)
        self.logger.info("Market Intelligence System - Starting")
        self.logger.info("=" * 80)

        self.scraper = TwitterScraper(self.config['scraper'])
        self.cleaner = DataCleaner(self.config['processing'])
        self.deduplicator = Deduplicator(self.config['processing'])
        self.storage = ParquetHandler(self.config['storage'])
        self.signal_generator = SignalGenerator(self.config['analysis'])
        self.visualizer = Visualizer(self.config['visualization'])

        self.start_time = datetime.now()
        self.metrics = {
            'tweets_scraped': 0,
            'tweets_cleaned': 0,
            'tweets_deduplicated': 0,
            'actionable_signals': 0
        }

    # Loads YAML configuration file or returns default configuration
    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
            print("Using default configuration")
            return self._get_default_config()

    # Returns default configuration settings for all system components
    def _get_default_config(self) -> Dict:
        return {
            'scraper': {
                'max_tweets': 2000,
                'time_window_hours': 24,
                'hashtags': ['#nifty50', '#sensex', '#intraday', '#banknifty'],
                'scroll_delay_min': 2,
                'scroll_delay_max': 5,
                'headless': True
            },
            'processing': {
                'chunk_size': 1000,
                'dedup_threshold': 0.95
            },
            'storage': {
                'format': 'parquet',
                'compression': 'snappy'
            },
            'analysis': {
                'tfidf_max_features': 1000,
                'signal_confidence_min': 0.6
            },
            'visualization': {
                'max_points_per_plot': 10000
            },
            'logging': {
                'level': 'INFO'
            }
        }

    # Executes complete pipeline: scrape → clean → deduplicate → analyze → store → visualize
    def run_pipeline(self, save_raw: bool = True, save_processed: bool = True) -> Dict:
        self.logger.info("Starting data collection and analysis pipeline")
        self._log_system_info()

        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 1: Data Collection")
            self.logger.info("=" * 80)

            tweets = self.scraper.scrape_hashtags(
                hashtags=self.config['scraper']['hashtags'],
                max_tweets=self.config['scraper']['max_tweets'],
                time_window_hours=self.config['scraper']['time_window_hours']
            )

            self.metrics['tweets_scraped'] = len(tweets)
            self.logger.info(f"[OK] Scraped {len(tweets)} tweets")

            if not tweets:
                self.logger.warning("No tweets collected. Exiting pipeline.")
                return self._generate_report()

            if save_raw:
                raw_file = self.scraper.save_raw_data(tweets)
                self.logger.info(f"[OK] Saved raw data to {raw_file}")

            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 2: Data Cleaning")
            self.logger.info("=" * 80)

            cleaned_tweets = self.cleaner.clean_tweets(tweets)
            self.metrics['tweets_cleaned'] = len(cleaned_tweets)
            self.logger.info(f"[OK] Cleaned {len(cleaned_tweets)} tweets")

            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 3: Deduplication")
            self.logger.info("=" * 80)

            dedup_stats = self.deduplicator.get_duplicate_statistics(cleaned_tweets)
            self.logger.info(f"Duplicate analysis: {json.dumps(dedup_stats, indent=2)}")

            unique_tweets = self.deduplicator.deduplicate_by_id(cleaned_tweets)
            unique_tweets = self.deduplicator.deduplicate_by_content(unique_tweets)

            self.metrics['tweets_deduplicated'] = len(unique_tweets)
            self.logger.info(f"[OK] {len(unique_tweets)} unique tweets after deduplication")

            df = self.cleaner.to_dataframe(unique_tweets)

            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 4: Signal Generation")
            self.logger.info("=" * 80)

            df_with_signals = self.signal_generator.generate_signals(df)
            if 'is_actionable' in df_with_signals.columns:
                self.metrics['actionable_signals'] = int(df_with_signals['is_actionable'].sum())
            else:
                self.metrics['actionable_signals'] = 0
                self.logger.warning("No actionable signals generated (empty dataset)")
            self.logger.info(f"✓ Generated signals for {len(df_with_signals)} tweets")
            self.logger.info(f"✓ {self.metrics['actionable_signals']} actionable signals identified")

            signal_report = self.signal_generator.generate_report(df_with_signals)
            self.logger.info(f"\nSignal Report:")
            self.logger.info(f"  Market Sentiment: {signal_report['market_sentiment']}")
            self.logger.info(f"  Bullish Signals: {signal_report['bullish_signals']}")
            self.logger.info(f"  Bearish Signals: {signal_report['bearish_signals']}")
            self.logger.info(f"  Average Confidence: {signal_report['average_confidence']:.3f}")

            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 5: Data Storage")
            self.logger.info("=" * 80)

            if save_processed:
                parquet_file = self.storage.save_to_parquet(df_with_signals)
                self.logger.info(f"✓ Saved processed data to {parquet_file}")

                report_file = Path("outputs") / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                report_file.parent.mkdir(exist_ok=True)
                with open(report_file, 'w') as f:
                    json.dump(signal_report, f, indent=2)
                self.logger.info(f"✓ Saved analysis report to {report_file}")

            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 6: Visualization")
            self.logger.info("=" * 80)

            self.visualizer.generate_all_visualizations(df_with_signals)
            self.logger.info("✓ Generated all visualizations")

            self.logger.info("\n" + "=" * 80)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info("=" * 80)

            return self._generate_report()

        except KeyboardInterrupt:
            self.logger.warning("\nPipeline interrupted by user")
            return self._generate_report()
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _log_system_info(self):
        self.logger.info("\nSystem Information:")
        self.logger.info(f"  CPU Count: {psutil.cpu_count()}")
        self.logger.info(f"  Memory Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        self.logger.info(f"  Memory Percent: {psutil.virtual_memory().percent}%")

    # Generates execution report with metrics, success rates, and performance statistics
    def _generate_report(self) -> Dict:
        duration = (datetime.now() - self.start_time).total_seconds()

        report = {
            'execution_time_seconds': duration,
            'execution_time_formatted': f"{duration/60:.2f} minutes",
            'metrics': self.metrics,
            'success_rate': {
                'cleaning': f"{(self.metrics['tweets_cleaned'] / max(self.metrics['tweets_scraped'], 1)) * 100:.1f}%",
                'deduplication': f"{(self.metrics['tweets_deduplicated'] / max(self.metrics['tweets_cleaned'], 1)) * 100:.1f}%",
                'actionable_signals': f"{(self.metrics['actionable_signals'] / max(self.metrics['tweets_deduplicated'], 1)) * 100:.1f}%"
            },
            'timestamp': datetime.now().isoformat()
        }

        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXECUTION REPORT")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Execution Time: {report['execution_time_formatted']}")
        self.logger.info(f"\nMetrics:")
        for key, value in self.metrics.items():
            self.logger.info(f"  {key.replace('_', ' ').title()}: {value}")
        self.logger.info(f"\nSuccess Rates:")
        for key, value in report['success_rate'].items():
            self.logger.info(f"  {key.replace('_', ' ').title()}: {value}")

        memory_info = psutil.Process().memory_info()
        self.logger.info(f"\nMemory Usage:")
        self.logger.info(f"  RSS: {memory_info.rss / (1024**2):.2f} MB")
        self.logger.info(f"  VMS: {memory_info.vms / (1024**2):.2f} MB")

        return report

# Starting Point
def main():
    parser = argparse.ArgumentParser(
        description="Market Intelligence Data Collection and Analysis System"
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--no-raw',
        action='store_true',
        help='Skip saving raw data'
    )
    parser.add_argument(
        '--no-processed',
        action='store_true',
        help='Skip saving processed data'
    )
    parser.add_argument(
        '--hashtags',
        nargs='+',
        help='Override hashtags from config (e.g., --hashtags #nifty50 #sensex)'
    )
    parser.add_argument(
        '--max-tweets',
        type=int,
        help='Override max tweets from config'
    )

    args = parser.parse_args()

    try:
        system = MarketIntelligenceSystem(config_path=args.config)

        report = system.run_pipeline()

        sys.exit(0)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

