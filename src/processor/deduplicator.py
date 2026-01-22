
import hashlib
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import pandas as pd

import logging

class Deduplicator:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.seen_hashes: Set[str] = set()
        
    def _compute_hash(self, content: str, method: str = 'md5') -> str:
        content_bytes = content.encode('utf-8')
        
        if method == 'sha256':
            return hashlib.sha256(content_bytes).hexdigest()
        else:
            return hashlib.md5(content_bytes).hexdigest()
    
    def _normalize_for_comparison(self, text: str) -> str:
        return ''.join(text.lower().split())
    
    def deduplicate_by_id(self, tweets: List[Dict]) -> List[Dict]:
        seen_ids = set()
        unique_tweets = []
        
        for tweet in tweets:
            tweet_id = tweet.get('tweet_id')
            if tweet_id and tweet_id not in seen_ids:
                seen_ids.add(tweet_id)
                unique_tweets.append(tweet)
        
        duplicates = len(tweets) - len(unique_tweets)
        self.logger.info(f"Removed {duplicates} duplicate tweets by ID")
        
        return unique_tweets
    
    def deduplicate_by_content(self, tweets: List[Dict], threshold: float = 1.0) -> List[Dict]:
        if threshold == 1.0:
            return self._deduplicate_exact(tweets)
        else:
            return self._deduplicate_fuzzy(tweets, threshold)
    
    def _deduplicate_exact(self, tweets: List[Dict]) -> List[Dict]:
        seen_hashes = set()
        unique_tweets = []
        
        for tweet in tweets:
            content = tweet.get('content', '')
            if not content:
                continue
            
            normalized = self._normalize_for_comparison(content)
            content_hash = self._compute_hash(normalized)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_tweets.append(tweet)
        
        duplicates = len(tweets) - len(unique_tweets)
        self.logger.info(f"Removed {duplicates} duplicate tweets by exact content")
        
        return unique_tweets
    
    def _deduplicate_fuzzy(self, tweets: List[Dict], threshold: float) -> List[Dict]:
        unique_tweets = []
        content_sets = []
        
        for tweet in tweets:
            content = tweet.get('content', '')
            if not content:
                continue
            
            words = set(self._normalize_for_comparison(content).split())
            
            is_duplicate = False
            for existing_set in content_sets:
                similarity = self._jaccard_similarity(words, existing_set)
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                content_sets.append(words)
                unique_tweets.append(tweet)
        
        duplicates = len(tweets) - len(unique_tweets)
        self.logger.info(f"Removed {duplicates} similar tweets (threshold: {threshold})")
        
        return unique_tweets
    
    @staticmethod
    def _jaccard_similarity(set1: Set, set2: Set) -> float:
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def deduplicate_retweets(self, tweets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        originals = []
        retweets = []
        
        for tweet in tweets:
            content = tweet.get('content', '')
            
            is_retweet = (
                content.startswith('RT @') or
                ' RT @' in content or
                content.startswith('Retweeted ')
            )
            
            if is_retweet:
                retweets.append(tweet)
            else:
                originals.append(tweet)
        
        self.logger.info(f"Separated {len(originals)} originals and {len(retweets)} retweets")
        
        return originals, retweets
    
    def deduplicate_dataframe(self, df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        if df.empty:
            return df
        
        initial_count = len(df)
        
        if subset is None:
            subset = ['tweet_id']
        
        df_dedup = df.drop_duplicates(subset=subset, keep='first')
        
        df_dedup = df_dedup.reset_index(drop=True)
        
        removed = initial_count - len(df_dedup)
        self.logger.info(f"Removed {removed} duplicate rows from DataFrame")
        
        return df_dedup
    
    def get_duplicate_statistics(self, tweets: List[Dict]) -> Dict:
        stats = {
            'total_tweets': len(tweets),
            'unique_ids': 0,
            'unique_content': 0,
            'retweet_count': 0,
            'duplicate_content_groups': 0
        }
        
        unique_ids = set()
        for tweet in tweets:
            tweet_id = tweet.get('tweet_id')
            if tweet_id:
                unique_ids.add(tweet_id)
        stats['unique_ids'] = len(unique_ids)
        
        content_hashes = set()
        for tweet in tweets:
            content = tweet.get('content', '')
            if content:
                normalized = self._normalize_for_comparison(content)
                content_hash = self._compute_hash(normalized)
                content_hashes.add(content_hash)
        stats['unique_content'] = len(content_hashes)
        
        for tweet in tweets:
            content = tweet.get('content', '')
            if content.startswith('RT @') or ' RT @' in content:
                stats['retweet_count'] += 1
        
        content_to_count = defaultdict(int)
        for tweet in tweets:
            content = tweet.get('content', '')
            if content:
                normalized = self._normalize_for_comparison(content)
                content_hash = self._compute_hash(normalized)
                content_to_count[content_hash] += 1
        
        stats['duplicate_content_groups'] = sum(1 for count in content_to_count.values() if count > 1)
        
        return stats
