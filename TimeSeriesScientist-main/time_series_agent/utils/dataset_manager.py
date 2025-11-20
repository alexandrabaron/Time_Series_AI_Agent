"""
Dataset Manager for TSci Conversational Agent
Handles CSV file loading, validation, and dataset management.
"""

import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import io


class DatasetManager:
    """
    Manages dataset loading, validation, and metadata.
    """
    
    @staticmethod
    def load_csv_to_session(uploaded_file) -> bool:
        """
        Load a CSV file into session state.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Store the DataFrame in session state
            st.session_state.data = df
            
            # Update dataset info
            st.session_state.dataset_info = {
                'name': uploaded_file.name,
                'uploaded_at': datetime.now().isoformat(),
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            
            # Reset column selections when new data is loaded
            if 'date_col' in st.session_state:
                del st.session_state.date_col
            if 'target_col' in st.session_state:
                del st.session_state.target_col
            
            return True
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier : {str(e)}")
            return False
    
    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the dataset and return validation results.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }
        
        # Check if DataFrame is empty
        if df.empty:
            validation['is_valid'] = False
            validation['errors'].append("Le dataset est vide")
            return validation
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        if total_missing > 0:
            missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
            validation['warnings'].append(
                f"Valeurs manquantes détectées : {total_missing} ({missing_pct:.2f}%)"
            )
            validation['info']['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        
        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            validation['warnings'].append(
                f"Lignes dupliquées détectées : {n_duplicates}"
            )
        
        # Check number of rows
        if len(df) < 50:
            validation['warnings'].append(
                f"Dataset très petit ({len(df)} lignes). Recommandé : au moins 100 lignes."
            )
        
        validation['info']['num_rows'] = len(df)
        validation['info']['num_columns'] = len(df.columns)
        validation['info']['num_duplicates'] = n_duplicates
        
        return validation
    
    @staticmethod
    def get_column_info(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific column.
        
        Args:
            df: DataFrame
            column: Column name
            
        Returns:
            Dictionary with column information
        """
        col_data = df[column]
        
        info = {
            'name': column,
            'dtype': str(col_data.dtype),
            'non_null_count': col_data.count(),
            'null_count': col_data.isnull().sum(),
            'unique_count': col_data.nunique()
        }
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            info['statistics'] = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median())
            }
        
        return info
    
    @staticmethod
    def prepare_time_series_data(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
        """
        Prepare the data for time series analysis.
        
        Args:
            df: Original DataFrame
            date_col: Name of the date column
            target_col: Name of the target column
            
        Returns:
            Prepared DataFrame with datetime index and single target column
        """
        # Create a copy
        ts_df = df[[date_col, target_col]].copy()
        
        # Convert date column to datetime
        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        
        # Sort by date
        ts_df = ts_df.sort_values(date_col)
        
        # Set date as index
        ts_df.set_index(date_col, inplace=True)
        
        # Rename target column to 'value' for consistency
        ts_df.rename(columns={target_col: 'value'}, inplace=True)
        
        return ts_df
    
    @staticmethod
    def get_dataset_preview(df: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
        """
        Get a preview of the dataset.
        
        Args:
            df: DataFrame
            n_rows: Number of rows to show
            
        Returns:
            DataFrame preview
        """
        return df.head(n_rows)
    
    @staticmethod
    def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Args:
            df: DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum()
        }
        
        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        return stats
    
    @staticmethod
    def detect_date_columns(df: pd.DataFrame) -> List[str]:
        """
        Automatically detect potential date columns.
        
        Args:
            df: DataFrame
            
        Returns:
            List of column names that might be dates
        """
        date_columns = []
        
        for col in df.columns:
            # Check by column name
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp', 'datetime', 'day', 'month', 'year']):
                date_columns.append(col)
                continue
            
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col], errors='coerce')
                if df[col].dtype == 'object':  # If it was string and could be parsed
                    date_columns.append(col)
            except:
                pass
        
        return date_columns
    
    @staticmethod
    def detect_target_columns(df: pd.DataFrame) -> List[str]:
        """
        Automatically detect potential target columns (numeric columns).
        
        Args:
            df: DataFrame
            
        Returns:
            List of column names that might be targets
        """
        # Return all numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        return numeric_cols
    
    @staticmethod
    def save_dataset_metadata(dataset_info: Dict[str, Any], output_dir: str = "datasets"):
        """
        Save dataset metadata to disk.
        
        Args:
            dataset_info: Dataset information dictionary
            output_dir: Directory to save metadata
        """
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata_file = output_path / f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False, default=str)


def load_csv_to_session(uploaded_file) -> bool:
    """
    Convenience function to load CSV into session.
    Can be imported directly.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        bool: True if successful, False otherwise
    """
    return DatasetManager.load_csv_to_session(uploaded_file)

