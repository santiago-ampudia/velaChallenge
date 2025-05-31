"""
Data Encoding and Feature Preparation Module for Vela Partners Investment Decision Engine

This module transforms cleaned train/test splits into model-ready encoded formats:
- Continuous variables: raw values + binned versions
- Ordinal variables: numeric values + qualitative labels for LLM
- Binary variables: boolean type enforcement
- Categorical variables: label encoding for small cardinality, text preservation for large
- Missing flags: preserved as boolean indicators

The approach ensures:
- Consistent encoding between train and test sets
- LLM-friendly qualitative labels for ordinal features
- Proper data types for downstream modules
- No data leakage (test encoding uses train-derived mappings)
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Any, List, Union
from sklearn.preprocessing import LabelEncoder
import warnings
from .parameters import (
    TRAIN_INPUT_FILE, TEST_INPUT_FILE, OUTPUT_DIR, 
    TRAIN_OUTPUT_FILE, TEST_OUTPUT_FILE,
    CONTINUOUS_DTYPE, ORDINAL_DTYPE, BINARY_DTYPE, CATEGORICAL_DTYPES,
    CONTINUOUS_BINNING, DEFAULT_CONTINUOUS_BINS, QUANTILE_BINS,
    ORDINAL_MAPPINGS, MAX_CATEGORICAL_CARDINALITY, IDENTIFIER_COLUMNS,
    TEXT_FIELD_PATTERNS, MIN_TRAIN_ROWS, MAX_TRAIN_ROWS, EXPECTED_TEST_ROWS,
    COLUMN_ORDER_GROUPS, OPTIMIZE_MEMORY, CATEGORICAL_AS_CATEGORY_DTYPE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataEncoder:
    """
    Data encoding pipeline that transforms cleaned features into model-ready formats.
    
    Follows clean architecture principles with separation of concerns:
    - Column type detection and classification
    - Feature-specific encoding strategies
    - Consistent train/test transformations
    - Comprehensive validation and serialization
    """
    
    def __init__(self):
        """Initialize the data encoder with configuration parameters."""
        # Store encoding mappings and statistics learned from training data
        self.column_types = {}
        self.continuous_binners = {}  # Store bin edges for each continuous column
        self.categorical_encoders = {}  # Store LabelEncoder instances
        self.ordinal_qualitative_mappings = {}  # Store ordinal-to-qualitative mappings
        
        # Track encoding statistics
        self.encoding_stats = {
            'train_shape_input': None,
            'train_shape_output': None,
            'test_shape_input': None,
            'test_shape_output': None,
            'column_counts_by_type': {},
            'dropped_columns': [],
            'created_columns': []
        }
    
    def load_splits(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the train and test DataFrames from pickle files.
        
        Args:
            train_path: Path to training split pickle file
            test_path: Path to test split pickle file
            
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            FileNotFoundError: If input files don't exist
            Exception: If data loading fails
        """
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        try:
            logger.info(f"Loading training data from {train_path}")
            train_df = pd.read_pickle(train_path)
            self.encoding_stats['train_shape_input'] = train_df.shape
            
            logger.info(f"Loading test data from {test_path}")
            test_df = pd.read_pickle(test_path)
            self.encoding_stats['test_shape_input'] = test_df.shape
            
            logger.info(f"Loaded training: {train_df.shape}, test: {test_df.shape}")
            return train_df, test_df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def validate_input_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Validate the input DataFrames for required structure and consistency.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating input data structure and consistency")
        
        # Check data sizes
        if not (MIN_TRAIN_ROWS <= len(train_df) <= MAX_TRAIN_ROWS):
            raise ValueError(f"Training set size {len(train_df)} outside expected range [{MIN_TRAIN_ROWS}, {MAX_TRAIN_ROWS}]")
        
        if len(test_df) != EXPECTED_TEST_ROWS:
            raise ValueError(f"Test set size {len(test_df)} != expected {EXPECTED_TEST_ROWS}")
        
        # Check required columns
        if 'success' not in train_df.columns:
            raise ValueError("Missing 'success' column in training data")
        if 'success' not in test_df.columns:
            raise ValueError("Missing 'success' column in test data")
        
        # Check column consistency
        if not train_df.columns.equals(test_df.columns):
            raise ValueError("Training and test DataFrames have different columns")
        
        # Check for any remaining missing values (should be none after Module 1)
        train_missing = train_df.isna().any().any()
        test_missing = test_df.isna().any().any()
        
        if train_missing or test_missing:
            raise ValueError("Found missing values in input data - should be cleaned by Module 1")
        
        logger.info("✅ Input data validation passed")
        logger.info(f"   - Training: {train_df.shape}")
        logger.info(f"   - Test: {test_df.shape}")
        logger.info(f"   - Columns: {len(train_df.columns)}")
    
    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer the type of each column based on dtype and content analysis.
        
        Args:
            df: DataFrame to analyze (typically training set)
            
        Returns:
            Dictionary mapping column names to types: 'continuous', 'ordinal', 'binary', 'categorical', 'text', 'missing_flag', 'target'
        """
        logger.info("Inferring column types from training data")
        
        column_types = {}
        
        for col in df.columns:
            # Skip success column (target)
            if col == 'success':
                column_types[col] = 'target'
                continue
                
            # Missing flag columns
            if col.endswith('_missing'):
                column_types[col] = 'missing_flag'
                continue
                
            # Identifier columns to drop
            if any(identifier in col.lower() for identifier in IDENTIFIER_COLUMNS):
                column_types[col] = 'identifier'
                continue
            
            # Determine type based on dtype and content
            dtype_str = str(df[col].dtype)
            
            if dtype_str == CONTINUOUS_DTYPE:
                column_types[col] = 'continuous'
            elif dtype_str == ORDINAL_DTYPE:
                column_types[col] = 'ordinal'
            elif dtype_str == BINARY_DTYPE or df[col].dtype == 'bool':
                column_types[col] = 'binary'
            elif dtype_str in CATEGORICAL_DTYPES:
                # Distinguish between categorical and text based on patterns and cardinality
                if any(pattern in col.lower() for pattern in TEXT_FIELD_PATTERNS):
                    column_types[col] = 'text'
                elif df[col].nunique() <= MAX_CATEGORICAL_CARDINALITY:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'text'  # High cardinality = treat as text
            else:
                # Default fallback based on content analysis
                unique_count = df[col].nunique()
                if unique_count == 2:
                    column_types[col] = 'binary'
                elif unique_count <= MAX_CATEGORICAL_CARDINALITY:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'text'
        
        # Store column types and update statistics
        self.column_types = column_types
        type_counts = {}
        for col_type in set(column_types.values()):
            type_counts[col_type] = sum(1 for t in column_types.values() if t == col_type)
        self.encoding_stats['column_counts_by_type'] = type_counts
        
        logger.info("✅ Column type inference completed")
        for col_type, count in type_counts.items():
            logger.info(f"   - {col_type}: {count} columns")
        
        return column_types
    
    def create_continuous_bins(self, train_df: pd.DataFrame, column: str) -> np.ndarray:
        """
        Create bin edges for a continuous column using domain knowledge or quantiles.
        
        Args:
            train_df: Training DataFrame
            column: Column name to bin
            
        Returns:
            Array of bin edges
        """
        if column in CONTINUOUS_BINNING:
            # Use predefined domain knowledge bins
            bin_edges = np.array(CONTINUOUS_BINNING[column])
            logger.info(f"Using predefined bins for {column}: {bin_edges}")
        else:
            # Use quantile-based bins
            quantiles = train_df[column].quantile(QUANTILE_BINS).values
            # Ensure unique edges (remove duplicates that can occur with discrete values)
            bin_edges = np.unique(quantiles)
            # Extend last bin to infinity to capture all values
            if bin_edges[-1] != float('inf'):
                bin_edges = np.append(bin_edges[:-1], float('inf'))
            logger.info(f"Using quantile bins for {column}: {bin_edges}")
        
        return bin_edges
    
    def prepare_continuous_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare continuous features by creating raw and binned versions.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of updated (train_df, test_df) with continuous features prepared
        """
        logger.info("Preparing continuous features (raw + binned)")
        
        continuous_cols = [col for col, col_type in self.column_types.items() if col_type == 'continuous']
        
        for col in continuous_cols:
            # Create bins based on training data
            bin_edges = self.create_continuous_bins(train_df, col)
            self.continuous_binners[col] = bin_edges
            
            # Create binned versions for both train and test
            train_df[f'{col}_bin'] = pd.cut(train_df[col], bins=bin_edges, labels=False, include_lowest=True)
            test_df[f'{col}_bin'] = pd.cut(test_df[col], bins=bin_edges, labels=False, include_lowest=True)
            
            # Convert to int8 and handle any potential NaN (shouldn't happen but safety)
            train_df[f'{col}_bin'] = train_df[f'{col}_bin'].fillna(0).astype('int8')
            test_df[f'{col}_bin'] = test_df[f'{col}_bin'].fillna(0).astype('int8')
            
            # Update column types
            self.column_types[f'{col}_bin'] = 'continuous_binned'
            self.encoding_stats['created_columns'].append(f'{col}_bin')
        
        logger.info(f"✅ Prepared {len(continuous_cols)} continuous features")
        return train_df, test_df
    
    def prepare_ordinal_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare ordinal features by creating numeric and qualitative versions.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of updated (train_df, test_df) with ordinal features prepared
        """
        logger.info("Preparing ordinal features (numeric + qualitative)")
        
        ordinal_cols = [col for col, col_type in self.column_types.items() if col_type == 'ordinal']
        
        for col in ordinal_cols:
            # Determine the scale based on unique values in training data
            unique_values = sorted(train_df[col].unique())
            scale_size = len(unique_values)
            
            # Create qualitative mapping
            if scale_size in ORDINAL_MAPPINGS:
                base_mapping = ORDINAL_MAPPINGS[scale_size]
                # Adjust mapping to match actual values
                if min(unique_values) == 0:
                    # 0-based scale: adjust mapping
                    qual_mapping = {val: base_mapping.get(val + 1, f"level_{val}") for val in unique_values}
                else:
                    # 1-based scale: use direct mapping
                    qual_mapping = {val: base_mapping.get(val, f"level_{val}") for val in unique_values}
            else:
                # Create generic mapping for unknown scales
                qual_mapping = {val: f"level_{val}" for val in unique_values}
            
            self.ordinal_qualitative_mappings[col] = qual_mapping
            
            # Apply qualitative mapping to both train and test
            train_df[f'{col}_qual'] = train_df[col].map(qual_mapping)
            test_df[f'{col}_qual'] = test_df[col].map(qual_mapping)
            
            # Convert qualitative to category dtype for memory efficiency
            if CATEGORICAL_AS_CATEGORY_DTYPE:
                train_df[f'{col}_qual'] = train_df[f'{col}_qual'].astype('category')
                test_df[f'{col}_qual'] = test_df[f'{col}_qual'].astype('category')
            
            # Update column types
            self.column_types[f'{col}_qual'] = 'ordinal_qualitative'
            self.encoding_stats['created_columns'].append(f'{col}_qual')
            
            logger.info(f"Created qualitative mapping for {col}: {qual_mapping}")
        
        logger.info(f"✅ Prepared {len(ordinal_cols)} ordinal features")
        return train_df, test_df
    
    def prepare_binary_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare binary features by ensuring boolean dtype.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of updated (train_df, test_df) with binary features prepared
        """
        logger.info("Preparing binary features (bool type enforcement)")
        
        binary_cols = [col for col, col_type in self.column_types.items() if col_type == 'binary']
        
        for col in binary_cols:
            # Ensure boolean dtype
            train_df[col] = train_df[col].astype('bool')
            test_df[col] = test_df[col].astype('bool')
        
        logger.info(f"✅ Prepared {len(binary_cols)} binary features")
        return train_df, test_df
    
    def prepare_categorical_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare categorical features using label encoding.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of updated (train_df, test_df) with categorical features prepared
        """
        logger.info("Preparing categorical features (label encoding)")
        
        categorical_cols = [col for col, col_type in self.column_types.items() if col_type == 'categorical']
        
        for col in categorical_cols:
            # Fit label encoder on training data
            encoder = LabelEncoder()
            train_df[col] = encoder.fit_transform(train_df[col].astype(str))
            
            # Apply to test data, handling unseen categories
            test_categories = test_df[col].astype(str)
            # Map unseen categories to -1 (will be converted to 0 after adding 1)
            encoded_test = []
            for category in test_categories:
                if category in encoder.classes_:
                    encoded_test.append(encoder.transform([category])[0])
                else:
                    logger.warning(f"Unseen category '{category}' in {col}, assigning to class 0")
                    encoded_test.append(0)  # Assign unseen to class 0
            
            test_df[col] = encoded_test
            
            # Convert to int8 for memory efficiency
            train_df[col] = train_df[col].astype('int8')
            test_df[col] = np.array(encoded_test, dtype='int8')
            
            # Store encoder for potential future use
            self.categorical_encoders[col] = encoder
            
            logger.info(f"Label encoded {col}: {len(encoder.classes_)} categories")
        
        logger.info(f"✅ Prepared {len(categorical_cols)} categorical features")
        return train_df, test_df
    
    def prepare_text_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare text features by ensuring string dtype.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of updated (train_df, test_df) with text features prepared
        """
        logger.info("Preparing text features (string type enforcement)")
        
        text_cols = [col for col, col_type in self.column_types.items() if col_type == 'text']
        
        for col in text_cols:
            # Ensure string dtype
            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)
        
        logger.info(f"✅ Prepared {len(text_cols)} text features")
        return train_df, test_df
    
    def drop_identifier_columns(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Drop identifier columns that are not useful for modeling.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of updated (train_df, test_df) with identifier columns removed
        """
        identifier_cols = [col for col, col_type in self.column_types.items() if col_type == 'identifier']
        
        if identifier_cols:
            logger.info(f"Dropping {len(identifier_cols)} identifier columns: {identifier_cols}")
            train_df = train_df.drop(columns=identifier_cols)
            test_df = test_df.drop(columns=identifier_cols)
            self.encoding_stats['dropped_columns'].extend(identifier_cols)
        
        return train_df, test_df
    
    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns according to the specified group ordering.
        
        Args:
            df: DataFrame to reorder
            
        Returns:
            DataFrame with reordered columns
        """
        ordered_columns = []
        
        # Group columns by type
        for group in COLUMN_ORDER_GROUPS:
            group_cols = []
            
            if group == 'continuous_raw':
                group_cols = [col for col, col_type in self.column_types.items() 
                             if col_type == 'continuous' and col in df.columns]
            elif group == 'continuous_binned':
                group_cols = [col for col, col_type in self.column_types.items() 
                             if col_type == 'continuous_binned' and col in df.columns]
            elif group == 'ordinal_numeric':
                group_cols = [col for col, col_type in self.column_types.items() 
                             if col_type == 'ordinal' and col in df.columns]
            elif group == 'ordinal_qualitative':
                group_cols = [col for col, col_type in self.column_types.items() 
                             if col_type == 'ordinal_qualitative' and col in df.columns]
            elif group == 'binary':
                group_cols = [col for col, col_type in self.column_types.items() 
                             if col_type == 'binary' and col in df.columns]
            elif group == 'categorical_encoded':
                group_cols = [col for col, col_type in self.column_types.items() 
                             if col_type == 'categorical' and col in df.columns]
            elif group == 'text_fields':
                group_cols = [col for col, col_type in self.column_types.items() 
                             if col_type == 'text' and col in df.columns]
            elif group == 'missing_flags':
                group_cols = [col for col, col_type in self.column_types.items() 
                             if col_type == 'missing_flag' and col in df.columns]
            elif group == 'target':
                group_cols = [col for col, col_type in self.column_types.items() 
                             if col_type == 'target' and col in df.columns]
            
            ordered_columns.extend(sorted(group_cols))  # Sort within group for consistency
        
        return df[ordered_columns]
    
    def validate_encoded_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Validate the encoded DataFrames meet all requirements.
        
        Args:
            train_df: Encoded training DataFrame
            test_df: Encoded test DataFrame
            
        Raises:
            AssertionError: If validation fails
        """
        logger.info("Validating encoded data")
        
        # Check no missing values
        if train_df.isna().any().any():
            raise AssertionError("Training DataFrame contains missing values after encoding")
        if test_df.isna().any().any():
            raise AssertionError("Test DataFrame contains missing values after encoding")
        
        # Check consistent columns
        if not train_df.columns.equals(test_df.columns):
            raise AssertionError("Training and test DataFrames have different columns after encoding")
        
        # Check dtypes for each column type
        for col, col_type in self.column_types.items():
            if col not in train_df.columns:
                continue  # Skip dropped columns
                
            train_dtype = str(train_df[col].dtype)
            test_dtype = str(test_df[col].dtype)
            
            if train_dtype != test_dtype:
                raise AssertionError(f"Dtype mismatch for {col}: train={train_dtype}, test={test_dtype}")
            
            # Validate specific dtype expectations
            if col_type == 'continuous' and train_dtype != 'float32':
                raise AssertionError(f"Continuous column {col} has dtype {train_dtype}, expected float32")
            elif col_type in ['ordinal', 'continuous_binned', 'categorical'] and train_dtype != 'int8':
                raise AssertionError(f"Integer column {col} has dtype {train_dtype}, expected int8")
            elif col_type == 'binary' and train_dtype != 'bool':
                raise AssertionError(f"Binary column {col} has dtype {train_dtype}, expected bool")
        
        # Validate bin ranges for continuous_binned columns
        for col, col_type in self.column_types.items():
            if col_type == 'continuous_binned' and col in train_df.columns:
                train_unique = set(train_df[col].unique())
                test_unique = set(test_df[col].unique())
                max_bin = len(self.continuous_binners[col.replace('_bin', '')]) - 2  # bins - 1
                
                for unique_vals in [train_unique, test_unique]:
                    if not unique_vals.issubset(set(range(max_bin + 1))):
                        raise AssertionError(f"Bin values for {col} outside expected range [0, {max_bin}]: {unique_vals}")
        
        logger.info("✅ Encoded data validation passed")
        logger.info(f"   - Training: {train_df.shape}")
        logger.info(f"   - Test: {test_df.shape}")
        logger.info(f"   - Columns: {len(train_df.columns)}")
    
    def save_prepared_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                          train_path: str, test_path: str) -> None:
        """
        Save the prepared DataFrames to disk.
        
        Args:
            train_df: Prepared training DataFrame
            test_df: Prepared test DataFrame
            train_path: Output path for training pickle file
            test_path: Output path for test pickle file
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            
            # Save prepared datasets
            train_df.to_pickle(train_path)
            logger.info(f"Saved prepared training data to {train_path}")
            
            test_df.to_pickle(test_path)
            logger.info(f"Saved prepared test data to {test_path}")
            
        except Exception as e:
            logger.error(f"Failed to save prepared data: {str(e)}")
            raise
    
    def encode_data(self, train_input: str = None, test_input: str = None,
                   train_output: str = None, test_output: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete data encoding pipeline.
        
        Args:
            train_input: Training input file path (defaults to configured path)
            test_input: Test input file path (defaults to configured path)
            train_output: Training output file path (defaults to configured path)
            test_output: Test output file path (defaults to configured path)
            
        Returns:
            Tuple of (train_prepared_df, test_prepared_df)
        """
        train_input_path = train_input or TRAIN_INPUT_FILE
        test_input_path = test_input or TEST_INPUT_FILE
        train_output_path = train_output or os.path.join(OUTPUT_DIR, TRAIN_OUTPUT_FILE)
        test_output_path = test_output or os.path.join(OUTPUT_DIR, TEST_OUTPUT_FILE)
        
        logger.info("Starting data encoding pipeline")
        
        # Step 1: Load data
        train_df, test_df = self.load_splits(train_input_path, test_input_path)
        
        # Step 2: Validate input data
        self.validate_input_data(train_df, test_df)
        
        # Step 3: Infer column types
        self.infer_column_types(train_df)
        
        # Step 4: Drop identifier columns
        train_df, test_df = self.drop_identifier_columns(train_df, test_df)
        
        # Step 5: Prepare each feature type
        train_df, test_df = self.prepare_continuous_features(train_df, test_df)
        train_df, test_df = self.prepare_ordinal_features(train_df, test_df)
        train_df, test_df = self.prepare_binary_features(train_df, test_df)
        train_df, test_df = self.prepare_categorical_features(train_df, test_df)
        train_df, test_df = self.prepare_text_features(train_df, test_df)
        
        # Step 6: Reorder columns for consistency
        train_df = self.reorder_columns(train_df)
        test_df = self.reorder_columns(test_df)
        
        # Step 7: Validate encoded data
        self.validate_encoded_data(train_df, test_df)
        
        # Step 8: Update final statistics
        self.encoding_stats['train_shape_output'] = train_df.shape
        self.encoding_stats['test_shape_output'] = test_df.shape
        
        # Step 9: Save prepared data
        self.save_prepared_data(train_df, test_df, train_output_path, test_output_path)
        
        logger.info("Data encoding pipeline completed successfully")
        return train_df, test_df
    
    def get_encoding_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive encoding report.
        
        Returns:
            Dictionary containing encoding statistics and metadata
        """
        return {
            'encoding_stats': self.encoding_stats,
            'column_types': self.column_types,
            'continuous_binners': {col: edges.tolist() for col, edges in self.continuous_binners.items()},
            'ordinal_qualitative_mappings': self.ordinal_qualitative_mappings,
            'categorical_encoders': {col: list(encoder.classes_) for col, encoder in self.categorical_encoders.items()}
        }


def run_data_encoding(train_input: str = None, test_input: str = None,
                     train_output: str = None, test_output: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point for data encoding module.
    
    Args:
        train_input: Training input file path (optional)
        test_input: Test input file path (optional)
        train_output: Training output file path (optional)
        test_output: Test output file path (optional)
        
    Returns:
        Tuple of (train_prepared_df, test_prepared_df)
    """
    encoder = DataEncoder()
    return encoder.encode_data(train_input, test_input, train_output, test_output)


if __name__ == "__main__":
    # Run data encoding when executed directly
    train_df, test_df = run_data_encoding()
    print(f"Data encoding completed.")
    print(f"Prepared training set: {train_df.shape}")
    print(f"Prepared test set: {test_df.shape}") 