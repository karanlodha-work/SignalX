# Quick Start - Running and Debugging

## Prerequisites

- Python 3.8+ installed
- Chrome/Chromium browser installed
- 2GB+ RAM available

## Installation

1. Clone/extract the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import selenium; import pandas; import sklearn; print('Dependencies OK')"
```

## Configuration

Edit `config/config.yaml` to customize:
- **hashtags**: Twitter hashtags to scrape
- **max_tweets**: Target number of tweets (default 2000)
- **headless**: Set to false to see browser window
- **scroll_delay_min/max**: Request delays in seconds
- **compression**: Storage format (snappy recommended)

Example:
```yaml
scraper:
  max_tweets: 500  # Start small for testing
  hashtags:
    - '#nifty50'
    - '#sensex'
  headless: false  # See browser
```

## Running the System

### Basic Execution (default config)
```bash
python main.py
```

### With Custom Config
```bash
python main.py --config custom_config.yaml
```

### Skip Raw/Processed Data Saving
```bash
python main.py --no-raw --no-processed
```

## Runtime Flow

1. **Data Collection** (30-120 sec): Scrapes tweets from hashtags
   - Requires manual login via browser
   - Follow on-screen instructions
   - Takes ~2-5 minutes for 2000 tweets

2. **Data Cleaning** (5-10 sec): Normalizes and validates
   - Unicode normalization
   - URL removal
   - Content validation

3. **Deduplication** (5-10 sec): Removes duplicates
   - Exact matching by ID
   - Fuzzy matching by content
   - Statistics generated

4. **Signal Generation** (10-30 sec): Converts text to signals
   - TF-IDF vectorization
   - Sentiment analysis (VADER, TextBlob)
   - Market-specific scoring
   - Confidence calculation

5. **Storage** (5-10 sec): Saves processed data
   - Parquet format with compression
   - Analysis reports as JSON
   - Progress logged

6. **Visualization** (20-60 sec): Creates charts
   - Memory-efficient sampling
   - Interactive dashboards
   - Multiple export formats

## Debugging

### Check Logs
Logs are saved in `logs/app_YYYYMMDD.log`:
```bash
tail -f logs/app_*.log
```

### Common Issues

**Chrome version mismatch**:
- Edit twitter_scraper.py line ~140
- Change `version_main=143` to your Chrome version
- Find version: Chrome menu → About

**Login failures**:
- Ensure manual login window is visible
- Complete any 2FA/captcha
- Press ENTER when fully logged in
- Check logs for detailed error messages

**Rate limiting**:
- Scraper auto-adjusts delays
- Monitor logs for "Rate limit reached" messages
- System will wait and resume

**Memory errors**:
- Reduce `max_tweets` in config
- Processing uses ~250KB per tweet
- Visualization samples data automatically

### Debug Mode
Enable detailed logging in config.yaml:
```yaml
logging:
  level: DEBUG
```

### Test with Mock Data
The system handles test runs gracefully - start with `max_tweets: 100` to test the pipeline.

## Output Structure

```
outputs/
├── analysis_report_TIMESTAMP.json     # Signal analysis results
└── visualizations/
    └── interactive_dashboard.html    # Interactive charts

data/
├── raw/
│   └── tweets_TIMESTAMP.json         # Raw scraped data
└── processed/
    └── tweets_TIMESTAMP.parquet      # Processed tweets with signals

logs/
└── app_YYYYMMDD.log                 # Application logs
```

## Key Files

- `main.py`: Entry point and orchestrator
- `config/config.yaml`: All configuration
- `src/scraper/twitter_scraper.py`: Web scraping logic
- `src/processor/data_cleaner.py`: Text cleaning
- `src/analysis/signal_generator.py`: Signal generation
- `src/analysis/visualizer.py`: Visualization

## Performance Tips

1. Reduce `max_tweets` for faster testing
2. Set `headless: true` if you don't need to see the browser
3. Adjust `scroll_delay_min/max` for faster scraping (10x rate limit risk increases)
4. Use smaller `max_points_per_plot` for faster visualizations on slow systems
5. Process in chunks with `chunk_size` parameter

## Troubleshooting

**Script hangs during scraping**:
- Check if browser window needs attention (login)
- Kill and restart with `--no-raw` to skip saving
- Check network connectivity

**Low tweet count**:
- Increase `max_scrolls` (more page scrolls)
- Add more hashtags to config
- Increase `time_window_hours`
- Some hashtags may have content restrictions

**Signal generation fails**:
- Ensure min content length met (default 10 chars)
- Check if dataset is empty after deduplication
- Verify sentiment analyzer models loaded

## Next Steps

1. Review outputs in `outputs/analysis_report_*.json`
2. Check interactive dashboard in `outputs/visualizations/`
3. Analyze Parquet data with Python:
   ```python
   import pandas as pd
   df = pd.read_parquet('data/processed/tweets_*.parquet')
   print(df.describe())
   ```
4. Integrate signals into trading strategies
5. Customize analysis settings in `config/config.yaml`

## Support

Check logs for detailed error messages and stack traces. All components log extensively at INFO, DEBUG, WARNING, and ERROR levels.
