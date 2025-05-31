"""
Data encoding and feature preparation parameters for Module 3.

This module defines all parameters used in data encoding, including:
- Input/output file paths
- Column type detection rules
- Binning configurations for continuous variables
- Ordinal to qualitative mappings
- Categorical encoding parameters
"""

import os

# File paths (relative to project root)
TRAIN_INPUT_FILE = "artifacts/splits/train_full.pkl"
TEST_INPUT_FILE = "artifacts/splits/test.pkl"
OUTPUT_DIR = "artifacts/prepared_splits"
TRAIN_OUTPUT_FILE = "train_prepared.pkl"
TEST_OUTPUT_FILE = "test_prepared.pkl"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column type detection parameters
CONTINUOUS_DTYPE = 'float32'
ORDINAL_DTYPE = 'int8'
BINARY_DTYPE = 'bool'
CATEGORICAL_DTYPES = ['string', 'category', 'object']

# Continuous variable binning configurations
# Define bins for key continuous variables based on domain knowledge
CONTINUOUS_BINNING = {
    'years_of_experience': [0, 2, 5, 10, 20, float('inf')],
    'number_of_roles': [0, 1, 3, 6, 10, float('inf')],
    'yoe': [0, 2, 5, 10, 20, float('inf')],  # Alternative name for years of experience
    'age': [0, 25, 30, 35, 45, float('inf')],
    'founding_rounds': [0, 1, 2, 4, float('inf')],
    'team_size': [0, 1, 5, 15, 50, float('inf')]
}

# Default binning strategy for unknown continuous variables (quantile-based)
DEFAULT_CONTINUOUS_BINS = 5  # Number of quantile-based bins
QUANTILE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Quintiles

# Ordinal to qualitative mappings for LLM understanding
ORDINAL_MAPPINGS = {
    # 5-level scale (1-5)
    5: {
        1: "very_low",
        2: "low", 
        3: "medium",
        4: "high",
        5: "very_high"
    },
    # 4-level scale (1-4) 
    4: {
        1: "low",
        2: "medium_low",
        3: "medium_high", 
        4: "high"
    },
    # 3-level scale (0-2)
    3: {
        0: "low",
        1: "medium",
        2: "high"
    },
    # Binary scale (0-1)
    2: {
        0: "no",
        1: "yes"
    }
}

# Categorical encoding parameters
MAX_CATEGORICAL_CARDINALITY = 10  # Maximum unique values for label encoding
IDENTIFIER_COLUMNS = [
    'founder_uuid', 'company_uuid', 'id', 'uuid'
]  # Columns to drop (not useful for modeling)

# Text field identification patterns
TEXT_FIELD_PATTERNS = [
    'description', 'profile', 'summary', 'bio', 'about',
    'linkedin', 'twitter', 'github', 'url', 'link'
]

# Data validation thresholds
MIN_TRAIN_ROWS = 7000  # Minimum expected training rows
MAX_TRAIN_ROWS = 9000  # Maximum expected training rows
EXPECTED_TEST_ROWS = 1000  # Expected test set size

# Column ordering for final DataFrame (for consistency)
COLUMN_ORDER_GROUPS = [
    'continuous_raw',      # All continuous raw columns (float32)
    'continuous_binned',   # All continuous bin columns (int8)
    'ordinal_numeric',     # All ordinal numeric columns (int8) 
    'ordinal_qualitative', # All ordinal qualitative columns (string)
    'binary',              # All binary columns (bool)
    'categorical_encoded', # All label-encoded categorical columns (int8)
    'text_fields',         # All free-text fields (string)
    'missing_flags',       # All missing-flag columns (bool)
    'target'               # The "success" column
]

# Memory optimization settings
OPTIMIZE_MEMORY = True
CATEGORICAL_AS_CATEGORY_DTYPE = True  # Use pandas category dtype for memory efficiency 