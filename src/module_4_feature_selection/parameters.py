#!/usr/bin/env python3
"""
Parameters Configuration for Module 4: Feature Selection via XGBoost + SHAP

This module defines all hyperparameters, thresholds, and configuration settings
used in the XGBoost-based feature selection process to identify the top 25
most predictive features for the investment decision model.

All parameters are centralized here to ensure reproducibility and easy tuning.
"""

import os

# =============================================================================
# INPUT/OUTPUT PATHS
# =============================================================================

# Input paths (from Module 3 outputs)
TRAIN_PREPARED_PATH = "artifacts/prepared_splits/train_prepared.pkl"
TEST_PREPARED_PATH = "artifacts/prepared_splits/test_prepared.pkl"

# Output directory and files
OUTPUT_DIR = "artifacts/feature_selection"
SELECTED_FEATURES_JSON = os.path.join(OUTPUT_DIR, "selected_features.json")
TRAIN_SELECTED_PKL = os.path.join(OUTPUT_DIR, "train_selected.pkl")
TEST_SELECTED_PKL = os.path.join(OUTPUT_DIR, "test_selected.pkl")

# =============================================================================
# FEATURE SELECTION CONFIGURATION
# =============================================================================

# Number of top features to select based on SHAP importance
NUM_TOP_FEATURES = 25

# Target positive rate for class weight calculation (2% as specified)
TARGET_POSITIVE_RATE = 0.02

# =============================================================================
# XGBOOST MODEL PARAMETERS
# =============================================================================

# XGBoost classifier configuration for feature importance extraction
XGBOOST_PARAMS = {
    "objective": "binary:logistic",           # Binary classification objective
    "n_estimators": 100,                      # Number of boosting rounds
    "max_depth": 4,                           # Maximum tree depth (prevent overfitting)
    "learning_rate": 0.1,                     # Step size shrinkage to prevent overfitting
    "subsample": 0.8,                         # Subsample ratio of training instances
    "colsample_bytree": 0.8,                  # Subsample ratio of features
    "use_label_encoder": False,               # Avoid sklearn label encoder warning
    "eval_metric": "logloss",                 # Evaluation metric for binary classification
    "random_state": 42,                       # Reproducibility seed
    "n_jobs": -1,                            # Use all available cores
    "verbosity": 0                           # Suppress XGBoost output
}

# Note: scale_pos_weight will be calculated dynamically based on actual class distribution

# =============================================================================
# SHAP CONFIGURATION
# =============================================================================

# SHAP TreeExplainer parameters
SHAP_PARAMS = {
    "feature_perturbation": "interventional"  # How to handle feature interactions
}

# =============================================================================
# DATA PROCESSING PARAMETERS
# =============================================================================

# Columns to always exclude from feature selection (non-predictive identifiers)
EXCLUDE_COLUMNS = [
    "success",  # Target variable - will be handled separately
]

# Suffix for missing value indicator columns
MISSING_FLAG_SUFFIX = "_missing"

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Minimum number of features required (safety check)
MIN_FEATURES_REQUIRED = 10

# Maximum number of features allowed (memory/performance constraint)
MAX_FEATURES_ALLOWED = 50

# Expected data types for validation
EXPECTED_DTYPES = {
    "continuous_raw": "float32",
    "continuous_binned": "int8", 
    "ordinal_numeric": "int8",
    "ordinal_qualitative": "object",
    "binary": "bool",
    "categorical_encoded": "int8",
    "text": "object",
    "missing_flags": "bool",
    "target": "int8"
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log levels for different components
LOG_LEVELS = {
    "data_loading": "INFO",
    "model_training": "INFO", 
    "feature_importance": "INFO",
    "feature_selection": "INFO",
    "data_saving": "INFO"
}

# =============================================================================
# PERFORMANCE PARAMETERS
# =============================================================================

# Memory optimization settings
CHUNK_SIZE = 1000  # For processing large datasets in chunks if needed
USE_SPARSE_MATRICES = False  # Whether to use sparse matrices for memory efficiency

# Progress reporting interval
PROGRESS_INTERVAL = 1000  # Report progress every N rows/features 