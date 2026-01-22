import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from pathlib import Path
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, 
    StaleElementReferenceException, WebDriverException
)
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc

from src.scraper.rate_limiter import AdaptiveRateLimiter
import logging


class TwitterScraper:

    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = AdaptiveRateLimiter(
            min_delay=config.get('scroll_delay_min', 2),
            max_delay=config.get('scroll_delay_max', 5),
            max_requests_per_minute=config.get('max_requests_per_minute', 30)
        )
        self.driver: Optional[webdriver.Chrome] = None
        self.scraped_tweet_ids: Set[str] = set()
        self.is_logged_in: bool = False

    def manual_login(self):
        if not self.driver:
            self.driver = self._setup_driver()

        self.logger.info("Opening X.com for manual login...")
        self.driver.get("https://x.com/login")

        print("\n" + "="*80)
        print("MANUAL LOGIN REQUIRED")
        print("="*80)
        print("1. Log in to X.com in the browser window that just opened")
        print("2. Complete any 2FA/verification if required")
        print("3. Once logged in, press ENTER here to continue scraping...")
        print("="*80 + "\n")

        input("Press ENTER when you're logged in and ready to start scraping: ")

        self.is_logged_in = True
        self.logger.info("Login confirmed. Starting to scrape...")
        return True

    def _setup_driver(self) -> webdriver.Chrome:
        self.logger.info("Setting up Chrome driver with stealth mode")

        options = uc.ChromeOptions()

        if self.config.get('headless', True):
            options.add_argument('--headless=new')

        window_size = self.config.get('window_size', [1920, 1080])
        print(self.config.get('window_size', [1920, 1080]))
        options.add_argument(f'--window-size={window_size[0]},{window_size[1]}')

        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-software-rasterizer')

        if self.config.get('use_random_user_agent', True):
            user_agent = random.choice(self.USER_AGENTS)
            options.add_argument(f'user-agent={user_agent}')
            self.logger.debug(f"Using user agent: {user_agent[:50]}...")

        prefs = {
            "profile.default_content_setting_values": {
                "images": 2,
                "javascript": 1
            }
        }
        options.add_experimental_option("prefs", prefs)

        try:
            driver = uc.Chrome(options=options, version_main=143, use_subprocess=False)
            driver.set_page_load_timeout(self.config.get('page_load_timeout', 30))

            try:
                driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                    "userAgent": driver.execute_script("return navigator.userAgent").replace('Headless', '')
                })
            except Exception as cdp_error:
                self.logger.warning(f"Could not set user agent via CDP: {cdp_error}")

            self.logger.info("Chrome driver setup successful")
            return driver

        except Exception as e:
            self.logger.error(f"Failed to setup Chrome driver: {e}")
            raise

    def _human_like_scroll(self, pause_time: float = 2.0):
        scroll_distance = random.randint(600, 900)

        steps = random.randint(4, 6)
        step_distance = scroll_distance // steps

        for _ in range(steps):
            self.driver.execute_script(f"window.scrollBy(0, {step_distance});")
            time.sleep(random.uniform(0.1, 0.2))

        time.sleep(pause_time)

    def _extract_tweet_data(self, tweet_element) -> Optional[Dict]:
        try:
            tweet_id = None
            tweet_link = None
            try:
                tweet_links = tweet_element.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
                for link in tweet_links:
                    href = link.get_attribute('href')
                    if '/status/' in href:
                        tweet_id = href.split('/status/')[-1].split('?')[0]
                        tweet_link = href.split('?')[0]
                        break
            except:
                pass

            if not tweet_id:
                tweet_id = f"tweet_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

            if tweet_id in self.scraped_tweet_ids:
                return None

            username = "unknown"
            try:
                username_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
                username = username_elem.text.split('\n')[0].replace('@', '')
            except:
                try:
                    username_elem = tweet_element.find_element(By.CSS_SELECTOR, 'a[href^="/"]')
                    username = username_elem.get_attribute('href').split('/')[-1]
                except:
                    pass

            content = ""
            try:
                content_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                content = content_elem.text
            except:
                pass

            if not content:
                return None

            timestamp = datetime.now().isoformat()
            try:
                time_elem = tweet_element.find_element(By.CSS_SELECTOR, 'time')
                timestamp = time_elem.get_attribute('datetime')
            except:
                pass

            likes = 0
            retweets = 0
            replies = 0

            try:
                buttons = tweet_element.find_elements(By.CSS_SELECTOR, 'button[aria-label]')
                for button in buttons:
                    aria_label = button.get_attribute('aria-label')
                    if not aria_label:
                        continue

                    numbers = re.findall(r'\d+', aria_label.replace(',', ''))
                    if numbers:
                        metric_value = int(numbers[0])
                        if 'like' in aria_label.lower():
                            likes = metric_value
                        elif 'retweet' in aria_label.lower() or 'repost' in aria_label.lower():
                            retweets = metric_value
                        elif 'repl' in aria_label.lower():
                            replies = metric_value
            except Exception as e:
                self.logger.debug(f"Failed to extract engagement metrics: {e}")

            mentions = re.findall(r'@(\w+)', content)
            hashtags = re.findall(r'#(\w+)', content)

            tweet_data = {
                'tweet_id': tweet_id,
                'username': username,
                'timestamp': timestamp,
                'content': content,
                'link': tweet_link if tweet_link else f"https://x.com/i/status/{tweet_id}",
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'mentions': mentions,
                'hashtags': hashtags,
                'scraped_at': datetime.now().isoformat()
            }

            self.scraped_tweet_ids.add(tweet_id)
            return tweet_data

        except StaleElementReferenceException:
            self.logger.debug("Stale element encountered, skipping")
            return None
        except Exception as e:
            self.logger.debug(f"Failed to extract tweet data: {e}")
            return None

    def scrape_hashtags(
        self,
        hashtags: List[str],
        max_tweets: int = 2000,
        time_window_hours: int = 24
    ) -> List[Dict]:
        self.logger.info(f"Starting scrape for {len(hashtags)} hashtags, target: {max_tweets} tweets")

        all_tweets = []
        tweets_per_hashtag = max_tweets // len(hashtags)

        try:
            if not self.driver:
                self.driver = self._setup_driver()

            if not self.is_logged_in:
                self.manual_login()

            for hashtag in hashtags:
                try:
                    self.driver.current_url
                except:
                    self.logger.error("Browser crashed, stopping scrape")
                    break

                self.logger.info(f"Scraping hashtag: {hashtag}")
                try:
                    hashtag_tweets = self._scrape_single_hashtag(
                        hashtag,
                        max_tweets=tweets_per_hashtag,
                        time_window_hours=time_window_hours
                    )
                    all_tweets.extend(hashtag_tweets)
                    self.logger.info(f"Collected {len(hashtag_tweets)} tweets for {hashtag}")
                except Exception as e:
                    error_msg = str(e)
                    if 'login' in error_msg.lower() or 'authentication' in error_msg.lower():
                        self.logger.info(f"Skipping {hashtag} (login required)")
                    else:
                        self.logger.error(f"Error scraping {hashtag}: {e}")
                    continue

                if len(all_tweets) >= max_tweets:
                    self.logger.info(f"Reached target of {max_tweets} tweets")
                    break

                time.sleep(random.uniform(2.0, 3.0))

            self.logger.info(f"Scraping complete. Total tweets: {len(all_tweets)}")
            return all_tweets

        except Exception as e:
            self.logger.error(f"Scraping failed: {e}", exc_info=True)
            return all_tweets
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                    self.logger.info("Browser closed")
                except:
                    pass

    def _scrape_single_hashtag(
        self,
        hashtag: str,
        max_tweets: int,
        time_window_hours: int
    ) -> List[Dict]:
        tweets = []
        hashtag = hashtag.strip('#')

        search_url = f"https://x.com/search?q=%23{hashtag}&src=typed_query&f=live"

        try:
            self.driver.get(search_url)
            self.rate_limiter.wait()
            self.rate_limiter.on_success()

            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="tweet"]'))
                )
            except TimeoutException:
                try:
                    page_text = self.driver.find_element(By.TAG_NAME, 'body').text[:300].lower()
                    if 'log in' in page_text or 'sign in' in page_text:
                        self.logger.warning(f"Login required for #{hashtag}, skipping...")
                        raise Exception("Login required")
                except:
                    pass

                self.logger.info("No tweets found with expected selector, attempting to continue...")

            consecutive_no_new = 0
            consecutive_no_elements = 0
            max_consecutive_no_new = 20
            max_consecutive_no_elements = 5
            max_scrolls = self.config.get('max_scrolls', 100)

            for scroll_count in range(max_scrolls):
                tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

                if len(tweet_elements) == 0:
                    consecutive_no_elements += 1
                    if consecutive_no_elements >= max_consecutive_no_elements:
                        self.logger.info("No tweet elements found on page, stopping")
                        break
                else:
                    consecutive_no_elements = 0

                initial_count = len(tweets)
                elements_processed = 0

                for tweet_elem in tweet_elements:
                    if len(tweets) >= max_tweets:
                        break

                    elements_processed += 1
                    tweet_data = self._extract_tweet_data(tweet_elem)
                    if tweet_data:
                        tweets.append(tweet_data)

                new_tweets_found = len(tweets) - initial_count

                if new_tweets_found > 0:
                    consecutive_no_new = 0
                    self.logger.debug(f"Found {new_tweets_found} new tweets (processed {elements_processed} elements)")
                else:
                    consecutive_no_new += 1
                    self.logger.debug(f"No new tweets ({consecutive_no_new}/{max_consecutive_no_new}), processed {elements_processed} elements")

                    if consecutive_no_new >= max_consecutive_no_new:
                        self.logger.info("No new tweets found after multiple scrolls, stopping")
                        break

                if len(tweets) >= max_tweets:
                    break

                self._human_like_scroll(self.config.get('scroll_pause_time', 1.5))

                if (scroll_count + 1) % 5 == 0:
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1.5)

                time.sleep(random.uniform(1.0, 2.0))

                if (scroll_count + 1) % 10 == 0:
                    self.logger.info(f"Progress: {len(tweets)} tweets after {scroll_count + 1} scrolls")

        except TimeoutException:
            self.logger.warning(f"Timeout loading tweets for #{hashtag}")
            self.rate_limiter.on_error()
        except Exception as e:
            self.logger.error(f"Error scraping #{hashtag}: {e}")
            self.rate_limiter.on_error(e)

        return tweets

    def save_raw_data(self, tweets: List[Dict], output_dir: str = "data/raw"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = output_path / f"tweets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tweets, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(tweets)} tweets to {filename}")
        return str(filename)
