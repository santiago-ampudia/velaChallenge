"""
Module 2: Data Splitting and Resampling for Vela Partners Investment Decision Engine

This module provides data splitting functionality to create:
- A held-out test set with 2% positive rate (real-world distribution)
- A training set preserving all remaining data (~10% positive rate)

The approach ensures all positive cases are available for LLM reasoning
while final evaluation reflects true investment success rarity.
"""

from .data_splitting_resampling import DataSplitter, run_data_splitting
from .parameters import (
    INPUT_FILE, OUTPUT_DIR, TRAIN_OUTPUT_FILE, TEST_OUTPUT_FILE,
    TRAIN_OUTPUT_CSV_FILE, TEST_OUTPUT_CSV_FILE,
    TEST_SET_SIZE, TEST_POSITIVE_COUNT, TEST_NEGATIVE_COUNT,
    RANDOM_SEED
)

__all__ = [
    'DataSplitter',
    'run_data_splitting',
    'INPUT_FILE',
    'OUTPUT_DIR',
    'TRAIN_OUTPUT_FILE',
    'TEST_OUTPUT_FILE',
    'TRAIN_OUTPUT_CSV_FILE',
    'TEST_OUTPUT_CSV_FILE',
    'TEST_SET_SIZE',
    'TEST_POSITIVE_COUNT',
    'TEST_NEGATIVE_COUNT',
    'RANDOM_SEED'
] 