"""
Data Splitting and Resampling Module for Vela Partners Investment Decision Engine

This module implements data splitting to create:
- A held-out test set with true 2% success rate (20 positives, 980 negatives)
- A training set preserving all remaining data (~10% positive rate)

The approach ensures:
- All positive cases available for LLM reasoning and rule induction
- Final evaluation reflects real-world 2% rarity
- Reproducible splits using fixed random seed
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Any
from .parameters import (
    INPUT_FILE, OUTPUT_DIR, TRAIN_OUTPUT_FILE, TEST_OUTPUT_FILE,
    TRAIN_OUTPUT_CSV_FILE, TEST_OUTPUT_CSV_FILE,
    TEST_SET_SIZE, TEST_POSITIVE_COUNT, TEST_NEGATIVE_COUNT,
    EXPECTED_ORIGINAL_POSITIVE_RATE_MIN, EXPECTED_ORIGINAL_POSITIVE_RATE_MAX,
    TARGET_TEST_POSITIVE_RATE, RANDOM_SEED,
    MIN_TOTAL_ROWS, MIN_POSITIVE_EXAMPLES, MIN_NEGATIVE_EXAMPLES
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Data splitting pipeline that creates training and test sets with specified distributions.
    
    Follows clean architecture principles with separation of concerns:
    - Data loading and validation
    - Stratified sampling for test set creation
    - Training set construction from remaining data
    - Output validation and serialization
    """
    
    def __init__(self):
        """Initialize the data splitter with configuration parameters."""
        self.test_set_size = TEST_SET_SIZE
        self.test_positive_count = TEST_POSITIVE_COUNT
        self.test_negative_count = TEST_NEGATIVE_COUNT
        self.random_seed = RANDOM_SEED
        
        # Track splitting statistics
        self.splitting_stats = {
            'original_shape': None,
            'original_positive_rate': None,
            'test_shape': None,
            'test_positive_rate': None,
            'train_shape': None,
            'train_positive_rate': None
        }
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
    
    def load_cleaned_data(self, filepath: str) -> pd.DataFrame:
        """
        Load the cleaned DataFrame from pickle format.
        
        Args:
            filepath: Path to the cleaned pickle file
            
        Returns:
            DataFrame with cleaned features and success labels
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            Exception: If data loading fails
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")
        
        try:
            logger.info(f"Loading cleaned data from {filepath}")
            df = pd.read_pickle(filepath)
            self.splitting_stats['original_shape'] = df.shape
            logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def validate_input_data(self, df: pd.DataFrame) -> None:
        """
        Validate the input DataFrame for required structure and distributions.
        
        Args:
            df: Input DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating input data structure and distribution")
        
        # Check minimum data requirements
        if df.shape[0] < MIN_TOTAL_ROWS:
            raise ValueError(f"Insufficient data: {df.shape[0]} rows, minimum required: {MIN_TOTAL_ROWS}")
        
        # Check for success column
        if 'success' not in df.columns:
            raise ValueError("Missing required 'success' column")
        
        # Validate success column contains only 0 and 1
        unique_values = df['success'].unique()
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(f"Success column contains invalid values: {unique_values}")
        
        # Check positive rate is approximately 10%
        positive_rate = df['success'].mean()
        self.splitting_stats['original_positive_rate'] = positive_rate
        
        if not (EXPECTED_ORIGINAL_POSITIVE_RATE_MIN <= positive_rate <= EXPECTED_ORIGINAL_POSITIVE_RATE_MAX):
            raise ValueError(
                f"Unexpected positive rate: {positive_rate:.3f}, "
                f"expected between {EXPECTED_ORIGINAL_POSITIVE_RATE_MIN:.3f} and {EXPECTED_ORIGINAL_POSITIVE_RATE_MAX:.3f}"
            )
        
        # Check sufficient positive and negative examples
        positive_count = (df['success'] == 1).sum()
        negative_count = (df['success'] == 0).sum()
        
        if positive_count < MIN_POSITIVE_EXAMPLES:
            raise ValueError(f"Insufficient positive examples: {positive_count}, minimum required: {MIN_POSITIVE_EXAMPLES}")
        
        if negative_count < MIN_NEGATIVE_EXAMPLES:
            raise ValueError(f"Insufficient negative examples: {negative_count}, minimum required: {MIN_NEGATIVE_EXAMPLES}")
        
        logger.info(f"✅ Data validation passed:")
        logger.info(f"   - Total rows: {df.shape[0]:,}")
        logger.info(f"   - Positive examples: {positive_count:,} ({positive_rate:.1%})")
        logger.info(f"   - Negative examples: {negative_count:,} ({1-positive_rate:.1%})")
    
    def create_test_set(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create test set with exact 2% positive rate through stratified sampling.
        
        Args:
            df: Full cleaned DataFrame
            
        Returns:
            Tuple of (test_df, remaining_df)
        """
        logger.info(f"Creating test set with {TEST_POSITIVE_COUNT} positives and {TEST_NEGATIVE_COUNT} negatives")
        
        # Separate positives and negatives
        df_pos = df[df['success'] == 1].copy()
        df_neg = df[df['success'] == 0].copy()
        
        logger.info(f"Available: {len(df_pos)} positives, {len(df_neg)} negatives")
        
        # Sample test positives
        if len(df_pos) < TEST_POSITIVE_COUNT:
            raise ValueError(f"Not enough positive examples: {len(df_pos)} available, {TEST_POSITIVE_COUNT} needed")
        
        test_pos = df_pos.sample(n=TEST_POSITIVE_COUNT, random_state=self.random_seed)
        
        # Sample test negatives
        if len(df_neg) < TEST_NEGATIVE_COUNT:
            raise ValueError(f"Not enough negative examples: {len(df_neg)} available, {TEST_NEGATIVE_COUNT} needed")
        
        test_neg = df_neg.sample(n=TEST_NEGATIVE_COUNT, random_state=self.random_seed)
        
        # Store original indices before combining and shuffling
        test_indices = set(test_pos.index) | set(test_neg.index)
        
        # Combine and shuffle test set
        test_df = pd.concat([test_pos, test_neg], ignore_index=True)
        test_df = test_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Create remaining dataset (training set) using original indices
        remaining_df = df.drop(index=test_indices).reset_index(drop=True)
        
        # Store test indices for validation (after reset_index, we'll validate differently)
        self.test_original_indices = test_indices
        
        # Update statistics
        self.splitting_stats['test_shape'] = test_df.shape
        self.splitting_stats['test_positive_rate'] = test_df['success'].mean()
        
        logger.info(f"✅ Test set created:")
        logger.info(f"   - Total rows: {len(test_df):,}")
        logger.info(f"   - Positives: {(test_df['success'] == 1).sum()}")
        logger.info(f"   - Negatives: {(test_df['success'] == 0).sum()}")
        logger.info(f"   - Positive rate: {test_df['success'].mean():.1%}")
        
        return test_df, remaining_df
    
    def create_training_set(self, remaining_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the training set from remaining data after test set extraction.
        
        Args:
            remaining_df: DataFrame with remaining data after test set removal
            
        Returns:
            Training DataFrame with preserved class distribution
        """
        logger.info("Creating training set from remaining data")
        
        train_df = remaining_df.copy()
        
        # Update statistics
        self.splitting_stats['train_shape'] = train_df.shape
        self.splitting_stats['train_positive_rate'] = train_df['success'].mean()
        
        train_positives = (train_df['success'] == 1).sum()
        train_negatives = (train_df['success'] == 0).sum()
        
        logger.info(f"✅ Training set created:")
        logger.info(f"   - Total rows: {len(train_df):,}")
        logger.info(f"   - Positives: {train_positives:,} ({train_df['success'].mean():.1%})")
        logger.info(f"   - Negatives: {train_negatives:,} ({1-train_df['success'].mean():.1%})")
        
        return train_df
    
    def validate_splits(self, test_df: pd.DataFrame, train_df: pd.DataFrame) -> None:
        """
        Validate the created splits meet all requirements.
        
        Args:
            test_df: Test DataFrame
            train_df: Training DataFrame
            
        Raises:
            AssertionError: If validation fails
        """
        logger.info("Validating splits meet requirements")
        
        # Test set validations
        if len(test_df) != TEST_SET_SIZE:
            raise AssertionError(f"Test set size mismatch: {len(test_df)} != {TEST_SET_SIZE}")
        
        test_positives = (test_df['success'] == 1).sum()
        test_negatives = (test_df['success'] == 0).sum()
        
        if test_positives != TEST_POSITIVE_COUNT:
            raise AssertionError(f"Test positive count mismatch: {test_positives} != {TEST_POSITIVE_COUNT}")
        
        if test_negatives != TEST_NEGATIVE_COUNT:
            raise AssertionError(f"Test negative count mismatch: {test_negatives} != {TEST_NEGATIVE_COUNT}")
        
        # Check test positive rate
        test_positive_rate = test_df['success'].mean()
        if abs(test_positive_rate - TARGET_TEST_POSITIVE_RATE) > 0.001:
            raise AssertionError(f"Test positive rate mismatch: {test_positive_rate:.3f} != {TARGET_TEST_POSITIVE_RATE:.3f}")
        
        # Check total size consistency (should equal original size)
        total_size = len(test_df) + len(train_df)
        expected_total = self.splitting_stats['original_shape'][0]
        if total_size != expected_total:
            raise AssertionError(f"Size mismatch: test + train = {total_size}, expected {expected_total}")
        
        # Check columns are identical
        if not test_df.columns.equals(train_df.columns):
            raise AssertionError("Test and train DataFrames have different columns")
        
        # Check no data overlap by verifying split completeness
        # Since we reset indices, we verify by checking total counts
        original_positives = self.splitting_stats['original_positive_rate'] * expected_total
        total_positives = test_positives + (train_df['success'] == 1).sum()
        
        if abs(total_positives - original_positives) > 1:  # Allow for small rounding differences
            raise AssertionError(f"Positive count mismatch: {total_positives} vs expected {original_positives}")
        
        logger.info("✅ All split validations passed")
    
    def save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                   train_path: str, test_path: str, train_csv_path: str = None, test_csv_path: str = None) -> None:
        """
        Save the training and test splits to disk in both pickle and CSV formats.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            train_path: Output path for training pickle file
            test_path: Output path for test pickle file
            train_csv_path: Output path for training CSV file (optional)
            test_csv_path: Output path for test CSV file (optional)
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            
            # Save training set (pickle)
            train_df.to_pickle(train_path)
            logger.info(f"Saved training set (pickle) to {train_path}")
            
            # Save test set (pickle)
            test_df.to_pickle(test_path)
            logger.info(f"Saved test set (pickle) to {test_path}")
            
            # Save training set (CSV)
            if train_csv_path:
                train_df.to_csv(train_csv_path, index=False)
                logger.info(f"Saved training set (CSV) to {train_csv_path}")
            
            # Save test set (CSV)
            if test_csv_path:
                test_df.to_csv(test_csv_path, index=False)
                logger.info(f"Saved test set (CSV) to {test_csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save splits: {str(e)}")
            raise
    
    def split_data(self, input_file: str = None, train_output: str = None, 
                   test_output: str = None, train_csv_output: str = None, test_csv_output: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete data splitting pipeline.
        
        Args:
            input_file: Input pickle file path (defaults to configured path)
            train_output: Training pickle output path (defaults to configured path)
            test_output: Test pickle output path (defaults to configured path)
            train_csv_output: Training CSV output path (defaults to configured path)
            test_csv_output: Test CSV output path (defaults to configured path)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        input_path = input_file or INPUT_FILE
        train_path = train_output or os.path.join(OUTPUT_DIR, TRAIN_OUTPUT_FILE)
        test_path = test_output or os.path.join(OUTPUT_DIR, TEST_OUTPUT_FILE)
        train_csv_path = train_csv_output or os.path.join(OUTPUT_DIR, TRAIN_OUTPUT_CSV_FILE)
        test_csv_path = test_csv_output or os.path.join(OUTPUT_DIR, TEST_OUTPUT_CSV_FILE)
        
        logger.info("Starting data splitting pipeline")
        
        # Step 1: Load data
        df_full = self.load_cleaned_data(input_path)
        
        # Step 2: Validate input data
        self.validate_input_data(df_full)
        
        # Step 3: Create test set with 2% positive rate
        test_df, remaining_df = self.create_test_set(df_full)
        
        # Step 4: Create training set from remaining data
        train_df = self.create_training_set(remaining_df)
        
        # Step 5: Validate splits
        self.validate_splits(test_df, train_df)
        
        # Step 6: Save splits in both formats
        self.save_splits(train_df, test_df, train_path, test_path, train_csv_path, test_csv_path)
        
        logger.info("Data splitting pipeline completed successfully")
        return train_df, test_df
    
    def get_splitting_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive splitting report.
        
        Returns:
            Dictionary containing splitting statistics and metadata
        """
        return {
            'splitting_stats': self.splitting_stats,
            'parameters': {
                'test_set_size': self.test_set_size,
                'test_positive_count': self.test_positive_count,
                'test_negative_count': self.test_negative_count,
                'random_seed': self.random_seed
            }
        }


def run_data_splitting(input_file: str = None, train_output: str = None, 
                      test_output: str = None, train_csv_output: str = None, test_csv_output: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point for data splitting module.
    
    Args:
        input_file: Input pickle file path (optional)
        train_output: Training pickle output path (optional)
        test_output: Test pickle output path (optional)
        train_csv_output: Training CSV output path (optional)
        test_csv_output: Test CSV output path (optional)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    splitter = DataSplitter()
    return splitter.split_data(input_file, train_output, test_output, train_csv_output, test_csv_output)


if __name__ == "__main__":
    # Run data splitting when executed directly
    train_df, test_df = run_data_splitting()
    print(f"Data splitting completed.")
    print(f"Training set: {train_df.shape} ({train_df['success'].mean():.1%} positive)")
    print(f"Test set: {test_df.shape} ({test_df['success'].mean():.1%} positive)") 