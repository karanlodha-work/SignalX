
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import logging

class ParquetHandler:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self.processed_path = Path(self.config.get('processed_data_path', 'data/processed'))
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.compression = self.config.get('compression', 'snappy')
        self.parquet_version = self.config.get('parquet_version', '2.6')
        
    def _create_optimized_schema(self) -> pa.Schema:
        schema = pa.schema([
            ('tweet_id', pa.string()),
            ('username', pa.string()),
            ('timestamp', pa.timestamp('ns', tz='UTC')),
            ('content', pa.string()),
            ('content_length', pa.int32()),
            ('likes', pa.int32()),
            ('retweets', pa.int32()),
            ('replies', pa.int32()),
            ('total_engagement', pa.int32()),
            ('mentions', pa.list_(pa.string())),
            ('hashtags', pa.list_(pa.string())),
            ('scraped_at', pa.timestamp('ns', tz='UTC')),
            ('cleaned_at', pa.timestamp('ns', tz='UTC')),
        ])
        
        return schema
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        string_cols = ['tweet_id', 'username', 'content']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        int_cols = ['content_length', 'likes', 'retweets', 'replies', 'total_engagement']
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')
        
        datetime_cols = ['timestamp', 'scraped_at', 'cleaned_at']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].dt.tz is None:
                    df[col] = df[col].dt.tz_localize('UTC')
                else:
                    df[col] = df[col].dt.tz_convert('UTC')
        
        list_cols = ['mentions', 'hashtags']
        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        
        return df
    
    def save_to_parquet(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None,
        partition_cols: Optional[List[str]] = None
    ) -> str:
        if df.empty:
            self.logger.warning("Empty DataFrame, skipping save")
            return None
        
        self.logger.info(f"Saving {len(df)} rows to Parquet")
        
        df = self._prepare_dataframe(df)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tweets_processed_{timestamp}.parquet"
        
        filepath = self.processed_path / filename
        
        try:
            table = pa.Table.from_pandas(df, preserve_index=False)
            
            pq.write_table(
                table,
                filepath,
                compression=self.compression,
                use_dictionary=self.config.get('use_dictionary', True),
                write_statistics=True,
                version=self.parquet_version
            )
            
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            self.logger.info(
                f"Saved to {filepath} ({file_size_mb:.2f} MB, {self.compression} compression)"
            )
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save Parquet file: {e}", exc_info=True)
            raise
    
    def load_from_parquet(
        self,
        filepath: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List] = None
    ) -> pd.DataFrame:
        self.logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_parquet(
                filepath,
                columns=columns,
                filters=filters,
                engine='pyarrow'
            )
            
            self.logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load Parquet file: {e}", exc_info=True)
            raise
    
    def append_to_parquet(self, df: pd.DataFrame, filepath: str):
        if df.empty:
            self.logger.warning("Empty DataFrame, skipping append")
            return
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            self.logger.info("File doesn't exist, creating new file")
            self.save_to_parquet(df, filepath.name)
            return
        
        try:
            existing_df = self.load_from_parquet(str(filepath))
            
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            
            combined_df = combined_df.drop_duplicates(subset=['tweet_id'], keep='last')
            
            self.save_to_parquet(combined_df, filepath.name)
            
            self.logger.info(f"Appended {len(df)} rows to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to append to Parquet file: {e}", exc_info=True)
            raise
    
    def get_parquet_info(self, filepath: str) -> Dict:
        try:
            parquet_file = pq.ParquetFile(filepath)
            
            info = {
                'num_rows': parquet_file.metadata.num_rows,
                'num_columns': parquet_file.metadata.num_columns,
                'num_row_groups': parquet_file.metadata.num_row_groups,
                'format_version': parquet_file.metadata.format_version,
                'created_by': parquet_file.metadata.created_by,
                'schema': parquet_file.schema.to_string(),
                'file_size_mb': Path(filepath).stat().st_size / (1024 * 1024)
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to read Parquet info: {e}")
            return {}
    
    def optimize_storage(self, filepath: str) -> str:
        self.logger.info(f"Optimizing {filepath}")
        
        df = self.load_from_parquet(filepath)
        
        df = df.drop_duplicates(subset=['tweet_id'])
        
        df = df.sort_values('timestamp')
        
        optimized_path = str(filepath).replace('.parquet', '_optimized.parquet')
        self.save_to_parquet(df, Path(optimized_path).name)
        
        original_size = Path(filepath).stat().st_size / (1024 * 1024)
        optimized_size = Path(optimized_path).stat().st_size / (1024 * 1024)
        reduction = ((original_size - optimized_size) / original_size) * 100
        
        self.logger.info(
            f"Optimization complete: {original_size:.2f} MB → {optimized_size:.2f} MB "
            f"({reduction:.1f}% reduction)"
        )
        
        return optimized_path
    
    def list_parquet_files(self) -> List[str]:
        files = list(self.processed_path.glob('*.parquet'))
        return [str(f) for f in files]
