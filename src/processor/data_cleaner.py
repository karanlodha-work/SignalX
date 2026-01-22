
import re
import unicodedata
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

import logging

class DataCleaner:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        self.mention_pattern = re.compile(r'@\w+')
        
        self.hashtag_pattern = re.compile(r'#\w+')
        
        self.space_pattern = re.compile(r'\s+')
        
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
    
    def clean_text(self, text: str, preserve_hashtags: bool = True, preserve_mentions: bool = True) -> str:
        if not text:
            return ""
        
        if self.config.get('normalize_unicode', True):
            text = unicodedata.normalize('NFKC', text)
        
        if self.config.get('remove_urls', True):
            text = self.url_pattern.sub('', text)
        
        if self.config.get('remove_mentions', False) and not preserve_mentions:
            text = self.mention_pattern.sub('', text)
        
        if self.config.get('remove_hashtags', False) and not preserve_hashtags:
            text = self.hashtag_pattern.sub('', text)
        
        if self.config.get('remove_emojis', False):
            text = self.emoji_pattern.sub('', text)
        
        text = self.space_pattern.sub(' ', text)
        
        text = text.strip()
        
        return text
    
    def validate_tweet(self, tweet: Dict) -> bool:
        if not ('id' in tweet or 'tweet_id' in tweet):
            self.logger.debug(f"Tweet missing id field")
            return False
        
        if not ('text' in tweet or 'content' in tweet):
            self.logger.debug(f"Tweet missing text/content field")
            return False
        
        if 'timestamp' not in tweet or not tweet['timestamp']:
            self.logger.debug(f"Tweet missing timestamp field")
            return False
        
        content = tweet.get('content') or tweet.get('text', '')
        min_length = self.config.get('min_content_length', 10)
        max_length = self.config.get('max_content_length', 500)
        
        if len(content) < min_length:
            self.logger.debug(f"Tweet content too short: {len(content)} < {min_length}")
            return False
        
        if len(content) > max_length:
            self.logger.debug(f"Tweet content too long: {len(content)} > {max_length}")
            return False
        
        return True
    
    def clean_tweet(self, tweet: Dict) -> Optional[Dict]:
        try:
            if not self.validate_tweet(tweet):
                return None
            
            content = tweet.get('content') or tweet.get('text', '')
            tweet_id = tweet.get('tweet_id') or tweet.get('id', '')
            
            cleaned_content = self.clean_text(
                content,
                preserve_hashtags=True,
                preserve_mentions=True
            )
            
            if not cleaned_content:
                return None
            
            cleaned_tweet = {
                'tweet_id': str(tweet_id),
                'username': tweet.get('username', 'unknown'),
                'timestamp': tweet['timestamp'],
                'content': cleaned_content,
                'content_length': len(cleaned_content),
                'likes': int(tweet.get('likes', 0)),
                'retweets': int(tweet.get('retweets', 0)),
                'replies': int(tweet.get('replies', 0)),
                'mentions': tweet.get('mentions', []),
                'hashtags': [tag.lower() for tag in tweet.get('hashtags', [])],
                'scraped_at': tweet.get('scraped_at', datetime.now().isoformat()),
                'cleaned_at': datetime.now().isoformat()
            }
            
            cleaned_tweet['total_engagement'] = (
                cleaned_tweet['likes'] +
                cleaned_tweet['retweets'] +
                cleaned_tweet['replies']
            )
            
            return cleaned_tweet
            
        except Exception as e:
            self.logger.error(f"Failed to clean tweet: {e}")
            return None
    
    def clean_tweets(self, tweets: List[Dict]) -> List[Dict]:
        self.logger.info(f"Cleaning {len(tweets)} tweets")
        
        cleaned_tweets = []
        for tweet in tweets:
            cleaned_tweet = self.clean_tweet(tweet)
            if cleaned_tweet:
                cleaned_tweets.append(cleaned_tweet)
        
        self.logger.info(
            f"Cleaning complete. {len(cleaned_tweets)}/{len(tweets)} tweets valid "
            f"({len(cleaned_tweets)/len(tweets)*100:.1f}%)"
        )
        
        return cleaned_tweets
    
    def to_dataframe(self, tweets: List[Dict]) -> pd.DataFrame:
        if not tweets:
            return pd.DataFrame()
        
        df = pd.DataFrame(tweets)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        numeric_cols = ['likes', 'retweets', 'replies', 'total_engagement', 'content_length']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df = df.sort_values('timestamp', ascending=False)
        
        self.logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def extract_market_info(self, text: str) -> Dict[str, any]:
        info = {
            'has_stock_symbols': False,
            'stock_symbols': [],
            'has_numbers': False,
            'numbers': [],
            'has_price_target': False,
            'sentiment_indicators': []
        }
        
        stock_pattern = re.compile(r'\b[A-Z]{2,10}\b')
        symbols = stock_pattern.findall(text)
        if symbols:
            info['has_stock_symbols'] = True
            info['stock_symbols'] = symbols
        
        number_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:%|k|K|L)?\b')
        numbers = number_pattern.findall(text)
        if numbers:
            info['has_numbers'] = True
            info['numbers'] = numbers
        
        target_keywords = ['target', 'tp', 'goal', 'level']
        if any(keyword in text.lower() for keyword in target_keywords):
            info['has_price_target'] = True
        
        bullish_terms = ['bullish', 'buy', 'long', 'breakout', 'rally', 'surge', 'moon', 'pump', 'up', 'green']
        bearish_terms = ['bearish', 'sell', 'short', 'breakdown', 'crash', 'dump', 'fall', 'decline', 'red']
        
        text_lower = text.lower()
        for term in bullish_terms:
            if term in text_lower:
                info['sentiment_indicators'].append(f'bullish:{term}')
        
        for term in bearish_terms:
            if term in text_lower:
                info['sentiment_indicators'].append(f'bearish:{term}')
        
        return info
