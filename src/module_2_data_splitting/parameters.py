"""
Data splitting and resampling parameters for Module 2.

This module defines all parameters used in data splitting, including:
- Input/output file paths
- Test set composition specifications
- Random seed for reproducibility
"""

import os

# File paths (relative to project root)
INPUT_FILE = "artifacts/prepared_data/full_cleaned.pkl"
OUTPUT_DIR = "artifacts/splits"
TRAIN_OUTPUT_FILE = "train_full.pkl"
TEST_OUTPUT_FILE = "test.pkl"
TRAIN_OUTPUT_CSV_FILE = "train_full.csv"
TEST_OUTPUT_CSV_FILE = "test.csv"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test set composition specifications - FIXED FOR FAIR EVALUATION
TEST_SET_SIZE = 1000  # Total rows in test set
TEST_POSITIVE_COUNT = 100  # MAJOR FIX: Positive examples in test set (10% - same as training!)
TEST_NEGATIVE_COUNT = 900  # MAJOR FIX: Negative examples in test set (90% - same as training!)

# Validation thresholds
EXPECTED_ORIGINAL_POSITIVE_RATE_MIN = 0.08  # Minimum expected positive rate (8%)
EXPECTED_ORIGINAL_POSITIVE_RATE_MAX = 0.12  # Maximum expected positive rate (12%)
TARGET_TEST_POSITIVE_RATE = 0.10  # MAJOR FIX: Target positive rate same as training (10%)

# Random seed for reproducibility
RANDOM_SEED = 42

# Minimum data requirements
MIN_TOTAL_ROWS = 1000  # Minimum rows required in input data
MIN_POSITIVE_EXAMPLES = TEST_POSITIVE_COUNT  # Minimum positive examples needed
MIN_NEGATIVE_EXAMPLES = TEST_NEGATIVE_COUNT  # Minimum negative examples needed 