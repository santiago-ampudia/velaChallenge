"""
Module 3: Data Encoding and Feature Preparation for Vela Partners Investment Decision Engine

This module provides data encoding functionality to transform cleaned features into model-ready formats:
- Continuous variables: raw values + binned versions for interpretability
- Ordinal variables: numeric values + qualitative labels for LLM understanding
- Binary variables: boolean type enforcement
- Categorical variables: label encoding for small cardinality, text preservation for large
- Missing flags: preserved as boolean indicators

The approach ensures consistent encoding between train and test sets while providing
LLM-friendly representations for downstream reasoning modules.
"""

from .data_encoding import DataEncoder, run_data_encoding
from .parameters import (
    TRAIN_INPUT_FILE, TEST_INPUT_FILE, OUTPUT_DIR,
    TRAIN_OUTPUT_FILE, TEST_OUTPUT_FILE,
    CONTINUOUS_BINNING, ORDINAL_MAPPINGS, MAX_CATEGORICAL_CARDINALITY,
    IDENTIFIER_COLUMNS, TEXT_FIELD_PATTERNS, COLUMN_ORDER_GROUPS
)

__all__ = [
    'DataEncoder',
    'run_data_encoding',
    'TRAIN_INPUT_FILE',
    'TEST_INPUT_FILE',
    'OUTPUT_DIR',
    'TRAIN_OUTPUT_FILE',
    'TEST_OUTPUT_FILE',
    'CONTINUOUS_BINNING',
    'ORDINAL_MAPPINGS',
    'MAX_CATEGORICAL_CARDINALITY',
    'IDENTIFIER_COLUMNS',
    'TEXT_FIELD_PATTERNS',
    'COLUMN_ORDER_GROUPS'
] 