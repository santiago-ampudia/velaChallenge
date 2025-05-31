"""
Data Cleaning Module for Vela Partners Investment Decision Engine

This module implements comprehensive data cleaning including:
- Missing value detection and flagging
- Invalid value handling
- Imputation strategies by column type
- Data type casting
- Sanity checks and validation
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Tuple, List, Any
from .parameters import (
    INPUT_FILE, OUTPUT_DIR, OUTPUT_FILE, OUTPUT_CSV_FILE, COLUMN_TYPES,
    IMPUTATION_STRATEGIES, DTYPE_CASTING, EXCLUDE_FROM_MISSING_FLAGS,
    CONTINUOUS_VALIDATION_RULES
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Comprehensive data cleaning pipeline for founder investment data.
    
    Follows clean architecture principles with separation of concerns:
    - Data loading and validation
    - Missing value detection and flagging  
    - Invalid value handling
    - Imputation by column type
    - Data type casting
    - Output validation
    """
    
    def __init__(self):
        """Initialize the data cleaner with configuration parameters."""
        self.column_types = COLUMN_TYPES.copy()
        self.imputation_strategies = IMPUTATION_STRATEGIES.copy()
        self.dtype_casting = DTYPE_CASTING.copy()
        self.exclude_from_missing_flags = EXCLUDE_FROM_MISSING_FLAGS.copy()
        self.continuous_validation_rules = CONTINUOUS_VALIDATION_RULES.copy()
        
        # Track cleaning statistics
        self.cleaning_stats = {
            'original_shape': None,
            'missing_value_flags_created': [],
            'invalid_values_replaced': {},
            'imputation_counts': {},
            'final_shape': None
        }
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV data with proper handling to avoid type coercion.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with all columns as object type to preserve original values
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            Exception: If data loading fails
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")
        
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath, dtype=object)
            self.cleaning_stats['original_shape'] = df.shape
            logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that expected columns are present in the data.
        
        Args:
            df: Input DataFrame
            
        Raises:
            ValueError: If expected columns are missing
        """
        expected_columns = set(self.column_types.keys())
        actual_columns = set(df.columns)
        
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
        
        if extra_columns:
            logger.info(f"Extra columns found (will be preserved): {extra_columns}")
            # Add extra columns as categorical by default
            for col in extra_columns:
                self.column_types[col] = 'categorical'
    
    def detect_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect missing values including NaN, empty strings, and whitespace.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized missing values (NaN)
        """
        logger.info("Detecting and standardizing missing values")
        df_clean = df.copy()
        
        # Convert empty strings and whitespace to NaN
        for col in df_clean.columns:
            # Handle string columns
            if df_clean[col].dtype == 'object':
                # Replace empty strings and whitespace-only strings with NaN
                mask = df_clean[col].astype(str).str.strip().isin(['', 'nan', 'NaN', 'NULL', 'null', 'N/A', 'n/a', 'NA', '#N/A', '-', 'None'])
                df_clean.loc[mask, col] = np.nan
        
        return df_clean
    
    def handle_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace invalid values with NaN based on column type and validation rules.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with invalid values replaced by NaN
        """
        logger.info("Handling invalid values")
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if col not in self.column_types:
                continue
                
            col_type = self.column_types[col]
            
            # Handle continuous columns with validation rules
            if col_type == 'continuous' and col in self.continuous_validation_rules:
                rules = self.continuous_validation_rules[col]
                
                # Convert to numeric first, coercing errors to NaN
                numeric_col = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Apply minimum value rule
                if 'min_valid' in rules:
                    min_val = rules['min_valid']
                    invalid_mask = numeric_col < min_val
                    invalid_count = invalid_mask.sum()
                    
                    if invalid_count > 0:
                        numeric_col.loc[invalid_mask] = np.nan
                        self.cleaning_stats['invalid_values_replaced'][col] = invalid_count
                        logger.info(f"Replaced {invalid_count} invalid values in {col} (< {min_val})")
                
                # Update the column with cleaned numeric values
                df_clean[col] = numeric_col
            
            # Convert numeric columns (continuous, ordinal) to numeric
            elif col_type in ['continuous', 'ordinal']:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def create_missing_value_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create boolean flags for missing values in each column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional _missing columns
        """
        logger.info("Creating missing value flags")
        df_with_flags = df.copy()
        
        for col in df.columns:
            if col in self.exclude_from_missing_flags:
                continue
                
            flag_col = f"{col}_missing"
            df_with_flags[flag_col] = df[col].isna()
            self.cleaning_stats['missing_value_flags_created'].append(flag_col)
        
        logger.info(f"Created {len(self.cleaning_stats['missing_value_flags_created'])} missing value flags")
        return df_with_flags
    
    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values based on column type strategies.
        
        Args:
            df: Input DataFrame with missing value flags
            
        Returns:
            DataFrame with imputed values
        """
        logger.info("Imputing missing values")
        df_imputed = df.copy()
        
        for col in df.columns:
            # Skip missing value flag columns
            if col.endswith('_missing'):
                continue
                
            if col not in self.column_types:
                continue
                
            col_type = self.column_types[col]
            strategy = self.imputation_strategies[col_type]
            
            missing_count = df_imputed[col].isna().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'median':
                fill_value = df_imputed[col].median()
            elif strategy == 'mode':
                mode_series = df_imputed[col].mode()
                fill_value = mode_series.iloc[0] if len(mode_series) > 0 else 0
            elif strategy == 'empty_string':
                fill_value = ""
            else:
                raise ValueError(f"Unknown imputation strategy: {strategy}")
            
            df_imputed[col] = df_imputed[col].fillna(fill_value)
            self.cleaning_stats['imputation_counts'][col] = missing_count
            
            logger.info(f"Imputed {missing_count} values in {col} with {strategy} (value: {fill_value})")
        
        return df_imputed
    
    def cast_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast columns to appropriate data types after imputation.
        
        Args:
            df: Input DataFrame with imputed values
            
        Returns:
            DataFrame with proper data types
        """
        logger.info("Casting data types")
        df_cast = df.copy()
        
        for col in df.columns:
            # Skip missing value flag columns (they're already boolean)
            if col.endswith('_missing'):
                continue
                
            if col not in self.column_types:
                continue
                
            col_type = self.column_types[col]
            target_dtype = self.dtype_casting[col_type]
            
            try:
                if target_dtype == 'float32':
                    df_cast[col] = df_cast[col].astype('float32')
                elif target_dtype == 'int8':
                    # Ensure no NaN values before converting to int
                    if df_cast[col].isna().any():
                        logger.warning(f"Found NaN values in {col} before int conversion, filling with 0")
                        df_cast[col] = df_cast[col].fillna(0)
                    df_cast[col] = df_cast[col].astype('int8')
                elif target_dtype == 'bool':
                    # Convert to boolean, treating any non-zero numeric as True
                    if df_cast[col].dtype in ['object']:
                        # For object columns, convert to numeric first
                        numeric_col = pd.to_numeric(df_cast[col], errors='coerce').fillna(0)
                        df_cast[col] = numeric_col.astype('bool')
                    else:
                        df_cast[col] = df_cast[col].astype('bool')
                elif target_dtype == 'category':
                    df_cast[col] = df_cast[col].astype('category')
                    
            except Exception as e:
                logger.error(f"Failed to cast {col} to {target_dtype}: {str(e)}")
                # Keep original type if casting fails
                continue
        
        return df_cast
    
    def perform_sanity_checks(self, df: pd.DataFrame) -> None:
        """
        Perform final validation checks on the cleaned data.
        
        Args:
            df: Cleaned DataFrame
            
        Raises:
            AssertionError: If validation checks fail
        """
        logger.info("Performing sanity checks")
        
        # Check for remaining NaN values (excluding missing flags)
        non_flag_columns = [col for col in df.columns if not col.endswith('_missing')]
        nan_counts = df[non_flag_columns].isna().sum()
        remaining_nans = nan_counts[nan_counts > 0]
        
        if len(remaining_nans) > 0:
            logger.error(f"Found remaining NaN values: {remaining_nans.to_dict()}")
            raise AssertionError("Data still contains NaN values after cleaning")
        
        # Log missing flag statistics
        flag_columns = [col for col in df.columns if col.endswith('_missing')]
        logger.info("Missing value flag statistics:")
        for flag_col in flag_columns:
            missing_rate = df[flag_col].mean()
            logger.info(f"  {flag_col}: {missing_rate:.3f} ({df[flag_col].sum()} missing)")
        
        # Validate data type consistency
        logger.info("Final data types:")
        for col in df.columns:
            logger.info(f"  {col}: {df[col].dtype}")
        
        self.cleaning_stats['final_shape'] = df.shape
        logger.info(f"Final dataset shape: {df.shape}")
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str, output_csv_path: str = None) -> None:
        """
        Save the cleaned DataFrame to both pickle and CSV formats.
        
        Args:
            df: Cleaned DataFrame
            output_path: Output pickle file path
            output_csv_path: Output CSV file path (optional)
        """
        try:
            # Save as pickle (for efficient storage with proper dtypes)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_pickle(output_path)
            logger.info(f"Saved cleaned data (pickle) to {output_path}")
            
            # Save as CSV (for portability and inspection)
            if output_csv_path:
                df.to_csv(output_csv_path, index=False)
                logger.info(f"Saved cleaned data (CSV) to {output_csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            raise
    
    def clean_data(self, input_file: str = None, output_file: str = None, output_csv_file: str = None) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline.
        
        Args:
            input_file: Input CSV file path (defaults to configured path)
            output_file: Output pickle file path (defaults to configured path)
            output_csv_file: Output CSV file path (defaults to configured path)
            
        Returns:
            Cleaned DataFrame
        """
        input_path = input_file or INPUT_FILE
        output_path = output_file or os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        output_csv_path = output_csv_file or os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE)
        
        logger.info("Starting data cleaning pipeline")
        
        # Step 1: Load data
        df = self.load_data(input_path)
        
        # Step 2: Validate columns
        self.validate_columns(df)
        
        # Step 3: Detect and standardize missing values
        df = self.detect_missing_values(df)
        
        # Step 4: Handle invalid values
        df = self.handle_invalid_values(df)
        
        # Step 5: Create missing value flags
        df = self.create_missing_value_flags(df)
        
        # Step 6: Impute missing values
        df = self.impute_missing_values(df)
        
        # Step 7: Cast data types
        df = self.cast_data_types(df)
        
        # Step 8: Sanity checks
        self.perform_sanity_checks(df)
        
        # Step 9: Save cleaned data in both formats
        self.save_cleaned_data(df, output_path, output_csv_path)
        
        logger.info("Data cleaning pipeline completed successfully")
        return df
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive cleaning report.
        
        Returns:
            Dictionary containing cleaning statistics and metadata
        """
        return {
            'cleaning_stats': self.cleaning_stats,
            'column_types': self.column_types,
            'imputation_strategies': self.imputation_strategies,
            'dtype_casting': self.dtype_casting
        }


def run_data_cleaning(input_file: str = None, output_file: str = None, output_csv_file: str = None) -> pd.DataFrame:
    """
    Main entry point for data cleaning module.
    
    Args:
        input_file: Input CSV file path (optional)
        output_file: Output pickle file path (optional)
        output_csv_file: Output CSV file path (optional)
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner()
    return cleaner.clean_data(input_file, output_file, output_csv_file)


if __name__ == "__main__":
    # Run data cleaning when executed directly
    cleaned_df = run_data_cleaning()
    print(f"Data cleaning completed. Final shape: {cleaned_df.shape}") 