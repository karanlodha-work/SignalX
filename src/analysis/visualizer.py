# Creates market sentiment visualizations and interactive dashboards

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import logging

class Visualizer:
    
    # Initializes with output paths, figure settings, and matplotlib styling
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self.output_path = Path(self.config.get('output_path', 'outputs/visualizations'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.max_points = self.config.get('max_points_per_plot', 10000)
        self.figure_size = tuple(self.config.get('figure_size', [12, 8]))
        self.dpi = self.config.get('dpi', 100)
        self.style = self.config.get('style', 'seaborn-v0_8-darkgrid')
        
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')
        
        sns.set_theme(style="darkgrid")
    
    # Randomly samples dataframe to max_points for performance optimization
    def _sample_data(self, df: pd.DataFrame, max_points: Optional[int] = None) -> pd.DataFrame:
        if max_points is None:
            max_points = self.max_points
        
        if len(df) <= max_points:
            return df
        
        if self.config.get('use_sampling', True):
            sample_ratio = max_points / len(df)
            sampled_df = df.sample(frac=sample_ratio, random_state=42)
            self.logger.info(f"Sampled {len(sampled_df)} points from {len(df)} total")
            return sampled_df
        
        return df
    
    # Plots sentiment over time with bullish/bearish areas and tweet count
    def plot_sentiment_timeline(
        self,
        df: pd.DataFrame,
        time_bucket: str = '1H',
        filename: str = 'sentiment_timeline.png'
    ):
        self.logger.info("Creating sentiment timeline plot")
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        agg_df = df.resample(time_bucket).agg({
            'composite_sentiment': 'mean',
            'signal_strength': 'mean',
            'tweet_id': 'count'
        }).rename(columns={'tweet_id': 'tweet_count'})
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, sharex=True)
        
        ax1.plot(agg_df.index, agg_df['composite_sentiment'], 
                 linewidth=2, label='Sentiment Score', color='steelblue')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(agg_df.index, 0, agg_df['composite_sentiment'], 
                          where=(agg_df['composite_sentiment'] > 0), 
                          color='green', alpha=0.3, label='Bullish')
        ax1.fill_between(agg_df.index, 0, agg_df['composite_sentiment'], 
                          where=(agg_df['composite_sentiment'] < 0), 
                          color='red', alpha=0.3, label='Bearish')
        ax1.set_ylabel('Sentiment Score', fontsize=12)
        ax1.set_title('Market Sentiment Timeline', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(agg_df.index, agg_df['tweet_count'], 
                color='steelblue', alpha=0.6, label='Tweet Count')
        ax2.set_ylabel('Tweet Count', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved sentiment timeline to {filepath}")
    
    # Creates 2x2 distribution plots for likes, retweets, replies, and total engagement
    def plot_engagement_distribution(
        self,
        df: pd.DataFrame,
        filename: str = 'engagement_distribution.png'
    ):
        self.logger.info("Creating engagement distribution plot")
        
        df_sample = self._sample_data(df)
        
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        
        axes[0, 0].hist(df_sample['likes'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Likes Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Likes')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')
        
        axes[0, 1].hist(df_sample['retweets'], bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Retweets Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Retweets')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        
        axes[1, 0].hist(df_sample['replies'], bins=50, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Replies Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Replies')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_yscale('log')
        
        axes[1, 1].hist(df_sample['total_engagement'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Total Engagement Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Total Engagement')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved engagement distribution to {filepath}")
    
    # Plots top hashtags by frequency and sentiment with bar charts
    def plot_hashtag_analysis(
        self,
        df: pd.DataFrame,
        top_n: int = 15,
        filename: str = 'hashtag_analysis.png'
    ):
        self.logger.info(f"Creating hashtag analysis plot (top {top_n})")
        
        hashtag_df = df.explode('hashtags')
        
        hashtag_stats = hashtag_df.groupby('hashtags').agg({
            'tweet_id': 'count',
            'total_engagement': 'sum',
            'composite_sentiment': 'mean'
        }).rename(columns={'tweet_id': 'count'})
        
        top_hashtags = hashtag_stats.nlargest(top_n, 'count')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        ax1.barh(top_hashtags.index, top_hashtags['count'], color='steelblue', alpha=0.7)
        ax1.set_xlabel('Tweet Count', fontsize=12)
        ax1.set_title('Top Hashtags by Frequency', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        colors = ['green' if x > 0.1 else 'red' if x < -0.1 else 'gray' 
                  for x in top_hashtags['composite_sentiment']]
        ax2.barh(top_hashtags.index, top_hashtags['composite_sentiment'], color=colors, alpha=0.7)
        ax2.set_xlabel('Average Sentiment', fontsize=12)
        ax2.set_title('Sentiment by Hashtag', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved hashtag analysis to {filepath}")
    
    # Creates heatmap of signal strength by hour and date using color intensity
    def plot_signal_strength_heatmap(
        self,
        df: pd.DataFrame,
        time_bucket: str = '1H',
        filename: str = 'signal_heatmap.png'
    ):
        self.logger.info("Creating signal strength heatmap")
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        
        pivot_data = df.pivot_table(
            values='signal_strength',
            index='hour',
            columns='date',
            aggfunc='mean'
        )
        
        plt.figure(figsize=self.figure_size)
        sns.heatmap(pivot_data, cmap='RdYlGn', center=0.5, 
                    cbar_kws={'label': 'Signal Strength'},
                    linewidths=0.5, linecolor='gray')
        plt.title('Signal Strength Heatmap (by Hour)', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Hour of Day', fontsize=12)
        plt.tight_layout()
        
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved signal heatmap to {filepath}")
    
    # Creates interactive 2x2 Plotly dashboard with sentiment, signals, engagement, hashtags
    def create_interactive_dashboard(
        self,
        df: pd.DataFrame,
        filename: str = 'interactive_dashboard.html'
    ):
        self.logger.info("Creating interactive dashboard")
        
        df_sample = self._sample_data(df, max_points=5000)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Timeline', 'Signal Strength Distribution',
                           'Engagement vs Sentiment', 'Top Hashtags'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        df_sorted = df_sample.sort_values('timestamp')
        fig.add_trace(
            go.Scatter(x=df_sorted['timestamp'], y=df_sorted['composite_sentiment'],
                      mode='lines', name='Sentiment', line=dict(color='steelblue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=df_sample['signal_strength'], name='Signal Strength',
                        marker_color='green', opacity=0.7),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=df_sample['composite_sentiment'], 
                      y=df_sample['total_engagement'],
                      mode='markers', name='Tweets',
                      marker=dict(size=5, color=df_sample['signal_strength'],
                                 colorscale='RdYlGn', showscale=True,
                                 colorbar=dict(title="Signal<br>Strength", x=0.46, y=0.25)),
                      text=df_sample['content'].str[:100],
                      hovertemplate='<b>Sentiment:</b> %{x:.2f}<br>' +
                                   '<b>Engagement:</b> %{y}<br>' +
                                   '<b>Content:</b> %{text}<extra></extra>'),
            row=2, col=1
        )
        
        hashtag_df = df_sample.explode('hashtags')
        top_hashtags = hashtag_df['hashtags'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=top_hashtags.values, y=top_hashtags.index,
                  orientation='h', name='Hashtags',
                  marker_color='steelblue'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Market Intelligence Dashboard",
            title_font_size=20,
            showlegend=False,
            height=900,
            hovermode='closest'
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Sentiment", row=1, col=1)
        fig.update_xaxes(title_text="Signal Strength", row=1, col=2)
        fig.update_xaxes(title_text="Sentiment Score", row=2, col=1)
        fig.update_yaxes(title_text="Total Engagement", row=2, col=1)
        fig.update_xaxes(title_text="Count", row=2, col=2)
        
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        self.logger.info(f"Saved interactive dashboard to {filepath}")
    
    # Generates all visualization types and saves as PNG and interactive HTML
    def generate_all_visualizations(self, df: pd.DataFrame):
        self.logger.info("Generating all visualizations")
        
        try:
            self.plot_sentiment_timeline(df)
            self.plot_engagement_distribution(df)
            self.plot_hashtag_analysis(df)
            self.plot_signal_strength_heatmap(df)
            
            if self.config.get('save_interactive', True):
                self.create_interactive_dashboard(df)
            
            self.logger.info("All visualizations generated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate some visualizations: {e}", exc_info=True)
