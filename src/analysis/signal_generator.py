# Analyzes tweets and generates market sentiment signals with confidence scores

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

import logging

class SignalGenerator:
    
    # Initializes with VADER/TextBlob sentiment analyzers and TF-IDF vectorizer
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self.vader = SentimentIntensityAnalyzer()
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.get('tfidf_max_features', 1000),
            ngram_range=tuple(self.config.get('tfidf_ngram_range', [1, 2])),
            min_df=self.config.get('tfidf_min_df', 2),
            max_df=self.config.get('tfidf_max_df', 0.95),
            stop_words='english'
        )
        
        self.bullish_terms = set(self.config.get('bullish_terms', [
            'bullish', 'buy', 'long', 'breakout', 'rally', 'surge', 'moon', 'pump', 'up', 'green'
        ]))
        
        self.bearish_terms = set(self.config.get('bearish_terms', [
            'bearish', 'sell', 'short', 'breakdown', 'crash', 'dump', 'fall', 'decline', 'red'
        ]))
        
        self.scaler = MinMaxScaler()
    
    # Analyzes text sentiment using VADER and TextBlob, returns sentiment metrics
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        sentiment = {
            'vader_compound': 0.0,
            'vader_positive': 0.0,
            'vader_negative': 0.0,
            'vader_neutral': 0.0,
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.0,
            'market_sentiment': 0.0
        }
        
        if not text:
            return sentiment
        
        try:
            if self.config.get('use_vader', True):
                vader_scores = self.vader.polarity_scores(text)
                sentiment['vader_compound'] = vader_scores['compound']
                sentiment['vader_positive'] = vader_scores['pos']
                sentiment['vader_negative'] = vader_scores['neg']
                sentiment['vader_neutral'] = vader_scores['neu']
            
            if self.config.get('use_textblob', True):
                blob = TextBlob(text)
                sentiment['textblob_polarity'] = blob.sentiment.polarity
                sentiment['textblob_subjectivity'] = blob.sentiment.subjectivity
            
            text_lower = text.lower()
            words = set(text_lower.split())
            
            bullish_count = len(words.intersection(self.bullish_terms))
            bearish_count = len(words.intersection(self.bearish_terms))
            
            if bullish_count + bearish_count > 0:
                sentiment['market_sentiment'] = (bullish_count - bearish_count) / (bullish_count + bearish_count)
            
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
        
        return sentiment
    
    # Extracts TF-IDF features from texts and returns feature matrix with feature names
    def extract_tfidf_features(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        self.logger.info(f"Extracting TF-IDF features from {len(texts)} texts")
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            self.logger.info(f"Extracted {len(feature_names)} features")
            
            return tfidf_matrix.toarray(), list(feature_names)
            
        except Exception as e:
            self.logger.error(f"TF-IDF extraction failed: {e}")
            return np.array([]), []
    
    # Calculates engagement score from likes, retweets, and replies with weighted formula
    def calculate_engagement_score(self, row: pd.Series) -> float:
        likes = row.get('likes', 0)
        retweets = row.get('retweets', 0)
        replies = row.get('replies', 0)
        
        score = (likes * 1.0 + retweets * 2.0 + replies * 1.5)
        
        return score
    
    # Applies exponential decay to signals based on age, newer signals weighted higher
    def calculate_temporal_decay(self, timestamp: datetime, half_life_hours: float = 6.0) -> float:
        try:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=pd.Timestamp.now().tzinfo)
            
            now = pd.Timestamp.now(tz=timestamp.tzinfo)
            hours_old = (now - timestamp).total_seconds() / 3600
            
            decay = 0.5 ** (hours_old / half_life_hours)
            
            return max(0.0, min(1.0, decay))
            
        except Exception as e:
            self.logger.warning(f"Temporal decay calculation failed: {e}")
            return 0.5
    
    # Generates trading signals with sentiment, engagement, and confidence scores
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Generating signals for {len(df)} tweets")
        
        if df.empty:
            return df
        
        df = df.copy()
        
        self.logger.info("Analyzing sentiment...")
        sentiment_results = []
        for content in df['content']:
            sentiment = self.analyze_sentiment(content)
            sentiment_results.append(sentiment)
        
        sentiment_df = pd.DataFrame(sentiment_results)
        df = pd.concat([df, sentiment_df], axis=1)
        
        self.logger.info("Calculating engagement scores...")
        df['engagement_score'] = df.apply(self.calculate_engagement_score, axis=1)
        
        if df['engagement_score'].max() > 0:
            df['engagement_score_norm'] = (
                df['engagement_score'] / df['engagement_score'].max()
            )
        else:
            df['engagement_score_norm'] = 0.0
        
        self.logger.info("Calculating temporal weights...")
        df['temporal_weight'] = df['timestamp'].apply(self.calculate_temporal_decay)
        
        self.logger.info("Generating composite signals...")
        
        df['composite_sentiment'] = (
            df['vader_compound'] * 0.4 +
            df['textblob_polarity'] * 0.3 +
            df['market_sentiment'] * 0.3
        )
        
        sentiment_weight = self.config.get('sentiment_weight', 0.4)
        engagement_weight = self.config.get('engagement_weight', 0.3)
        temporal_weight = self.config.get('temporal_weight', 0.3)
        
        df['signal_strength'] = (
            df['composite_sentiment'].abs() * sentiment_weight +
            df['engagement_score_norm'] * engagement_weight +
            df['temporal_weight'] * temporal_weight
        )
        
        df['signal_direction'] = df['composite_sentiment'].apply(
            lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0)
        )
        
        df['signed_signal'] = df['signal_direction'] * df['signal_strength']
        
        df['confidence'] = df.apply(self._calculate_confidence, axis=1)
        
        min_confidence = self.config.get('signal_confidence_min', 0.6)
        df['is_actionable'] = df['confidence'] >= min_confidence
        
        self.logger.info(
            f"Generated signals: {df['is_actionable'].sum()} actionable "
            f"({df['is_actionable'].sum()/len(df)*100:.1f}%)"
        )
        
        return df
    
    # Calculates signal confidence based on agreement of sentiment indicators and engagement
    def _calculate_confidence(self, row: pd.Series) -> float:
        indicators = [
            row.get('vader_compound', 0),
            row.get('textblob_polarity', 0),
            row.get('market_sentiment', 0)
        ]
        
        positive = sum(1 for x in indicators if x > 0.1)
        negative = sum(1 for x in indicators if x < -0.1)
        
        total = len(indicators)
        max_agree = max(positive, negative)
        agreement = max_agree / total
        
        confidence = (
            agreement * 0.5 +
            row.get('engagement_score_norm', 0) * 0.3 +
            row.get('temporal_weight', 0) * 0.2
        )
        
        return min(1.0, confidence)
    
    # Aggregates signals by time bucket and calculates mean metrics per period
    def aggregate_signals(
        self,
        df: pd.DataFrame,
        time_bucket: str = '1H',
        hashtag: Optional[str] = None
    ) -> pd.DataFrame:
        self.logger.info(f"Aggregating signals by {time_bucket}")
        
        if df.empty:
            return df
        
        if hashtag:
            hashtag = hashtag.lower().strip('#')
            df = df[df['hashtags'].apply(lambda x: hashtag in [h.lower() for h in x])]
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        agg_df = df.resample(time_bucket).agg({
            'tweet_id': 'count',
            'signed_signal': 'mean',
            'signal_strength': 'mean',
            'composite_sentiment': 'mean',
            'engagement_score': 'sum',
            'confidence': 'mean',
            'likes': 'sum',
            'retweets': 'sum',
            'replies': 'sum'
        }).rename(columns={'tweet_id': 'tweet_count'})
        
        agg_df = agg_df.reset_index()
        
        agg_df['signal_category'] = agg_df['signed_signal'].apply(
            lambda x: 'Bullish' if x > 0.1 else ('Bearish' if x < -0.1 else 'Neutral')
        )
        
        self.logger.info(f"Aggregated to {len(agg_df)} time buckets")
        
        return agg_df
    
    # Returns top N signals filtered by type (bullish/bearish/all) sorted by strength
    def get_top_signals(
        self,
        df: pd.DataFrame,
        n: int = 10,
        signal_type: str = 'all'
    ) -> pd.DataFrame:
        df = df.copy()
        
        if signal_type == 'bullish':
            df = df[df['signal_direction'] == 1]
        elif signal_type == 'bearish':
            df = df[df['signal_direction'] == -1]
        
        top_signals = df.nlargest(n, 'signal_strength')
        
        return top_signals[['content', 'username', 'timestamp', 'signal_strength', 
                           'signal_direction', 'confidence', 'total_engagement']]
    
    # Generates summary report with signal counts, sentiment, engagement, and top hashtags
    def generate_report(self, df: pd.DataFrame) -> Dict:
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tweets': len(df),
            'actionable_signals': int(df['is_actionable'].sum()),
            'bullish_signals': int((df['signal_direction'] == 1).sum()),
            'bearish_signals': int((df['signal_direction'] == -1).sum()),
            'neutral_signals': int((df['signal_direction'] == 0).sum()),
            'average_sentiment': float(df['composite_sentiment'].mean()),
            'average_confidence': float(df['confidence'].mean()),
            'total_engagement': int(df['total_engagement'].sum()),
            'top_hashtags': self._get_top_hashtags(df, 10),
            'market_sentiment': 'Bullish' if df['composite_sentiment'].mean() > 0.1 
                              else ('Bearish' if df['composite_sentiment'].mean() < -0.1 else 'Neutral')
        }    
        return report
    
    # Extracts and ranks top N hashtags by frequency, sentiment, and signal strength
    def _get_top_hashtags(self, df: pd.DataFrame, n: int = 10) -> List[Dict]:
        
        hashtag_df = df.explode('hashtags')
        
        top_hashtags = hashtag_df.groupby('hashtags').agg({
            'tweet_id': 'count',
            'signal_strength': 'mean',
            'composite_sentiment': 'mean'
        }).rename(columns={'tweet_id': 'count'})
        
        top_hashtags = top_hashtags.nlargest(n, 'count')
        
        return top_hashtags.reset_index().to_dict('records')
