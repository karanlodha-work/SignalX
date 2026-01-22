# Manages request rate limiting with exponential backoff on errors

import time
import random
from collections import deque
from datetime import datetime, timedelta
from typing import Optional
import logging

class RateLimiter:
    
    # Initializes with min/max delays, max requests per minute, and backoff factor
    def __init__(
        self,
        min_delay: float = 2.0,
        max_delay: float = 5.0,
        max_requests_per_minute: int = 30,
        backoff_factor: float = 1.5
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.max_requests_per_minute = max_requests_per_minute
        self.backoff_factor = backoff_factor
        
        self.request_times = deque(maxlen=max_requests_per_minute)
        self.consecutive_errors = 0
        self.logger = logging.getLogger(__name__)
        
    # Enforces rate limit: sleeps with jitter, applies exponential backoff on errors
    def wait(self):
        cutoff_time = datetime.now() - timedelta(minutes=1)
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
        
        if len(self.request_times) >= self.max_requests_per_minute:
            wait_time = 60 - (datetime.now() - self.request_times[0]).total_seconds()
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached. Waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                self.request_times.clear()
        
        base_delay = random.uniform(self.min_delay, self.max_delay)
        if self.consecutive_errors > 0:
            backoff_delay = min(
                base_delay * (self.backoff_factor ** self.consecutive_errors),
                self.max_delay * 3
            )
            delay = backoff_delay
            self.logger.info(f"Applying backoff: {delay:.2f}s (errors: {self.consecutive_errors})")
        else:
            delay = base_delay
        
        jitter = delay * 0.2
        delay = delay + random.uniform(-jitter, jitter)
        
        self.logger.debug(f"Waiting {delay:.2f}s before next request")
        time.sleep(delay)
        
        self.request_times.append(datetime.now())
    
    # Resets error counter on successful request
    def on_success(self):
        
        if self.consecutive_errors > 0:
            self.logger.info("Request successful. Resetting error counter.")
        self.consecutive_errors = 0
    
    # Increments error counter on failed request for backoff calculation
    def on_error(self, error: Optional[Exception] = None):
        self.consecutive_errors += 1
        self.logger.warning(
            f"Request failed. Error count: {self.consecutive_errors}. "
            f"Error: {error if error else 'Unknown'}"
        )
    
    # Clears request history and resets error counter
    def reset(self):
        
        self.request_times.clear()
        self.consecutive_errors = 0
    
# Extends RateLimiter with adaptive delays based on success rate tracking
class AdaptiveRateLimiter(RateLimiter):
    
    # Initializes with adaptive delay adjustment and success/failure thresholds
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_count = 0
        self.failure_count = 0
        self.adaptive_delay = self.min_delay
        self.adjustment_threshold = 10  # Adjust after N requests
        
    # Adjusts delay dynamically based on success rate, calls parent wait()
    def wait(self):
        
        total_requests = self.success_count + self.failure_count
        if total_requests >= self.adjustment_threshold:
            success_rate = self.success_count / total_requests
            
            if success_rate > 0.9:  # High success rate - can be more aggressive
                self.adaptive_delay = max(
                    self.min_delay,
                    self.adaptive_delay * 0.9
                )
                self.logger.debug(f"Decreasing delay to {self.adaptive_delay:.2f}s (success rate: {success_rate:.2%})")
            elif success_rate < 0.7:  # Low success rate - need to slow down
                self.adaptive_delay = min(
                    self.max_delay,
                    self.adaptive_delay * 1.2
                )
                self.logger.debug(f"Increasing delay to {self.adaptive_delay:.2f}s (success rate: {success_rate:.2%})")
            
            self.success_count = 0
            self.failure_count = 0
        
        original_min = self.min_delay
        self.min_delay = self.adaptive_delay
        super().wait()
        self.min_delay = original_min
    
    # Increments success counter and calls parent on_success()
    def on_success(self):
        
        super().on_success()
        self.success_count += 1
    
    # Increments failure counter and calls parent on_error()
    def on_error(self, error: Optional[Exception] = None):
        
        super().on_error(error)
        self.failure_count += 1
