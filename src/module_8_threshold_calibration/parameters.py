#!/usr/bin/env python3

"""
Parameters Configuration for Module 8: Threshold Calibration

This module defines all configuration settings for calibrating IF-THEN rule
thresholds to achieve target precision, including:
- Precision targets and constraints
- Threshold generation strategies
- Rule combination logic
- File paths and output settings

All parameters are centralized here for easy tuning and reproducibility.
"""

import os

# =============================================================================
# INPUT/OUTPUT PATHS
# =============================================================================

# Input paths (from previous modules)
INITIAL_RULES_PATH = "artifacts/rules/initial_rules.json"
TEST_SELECTED_PATH = "artifacts/feature_selection/test_selected.pkl"

# Output directory and files
RULES_CALIBRATED_DIR = "artifacts/rules_calibrated"
EVAL_DIR = "artifacts/eval"

CALIBRATED_RULES_PATH = os.path.join(RULES_CALIBRATED_DIR, "calibrated_rules.json")
THRESHOLD_METRICS_PATH = os.path.join(EVAL_DIR, "threshold_metrics.csv")
CALIBRATION_SUMMARY_PATH = os.path.join(RULES_CALIBRATED_DIR, "calibration_summary.json")

# =============================================================================
# PRECISION TARGET CONFIGURATION
# =============================================================================

# Target precision to achieve (20%)
PRECISION_TARGET = 0.20

# Maximum number of rules to combine in search for target precision
# Set to None to use all available rules
MAX_RULES_TO_COMBINE = None  # Will default to len(initial_rules)

# Minimum support (number of true positives) required for a rule to be considered
MIN_SUPPORT = 1  # Lowered from 2 to allow more selective rules

# Whether to require each individual rule to meet minimum precision
MIN_INDIVIDUAL_PRECISION = 0.04  # 4% minimum for individual rules

# Whether to stop immediately when target is reached or continue to optimize
STOP_AT_TARGET = False  # Continue to optimize for better recall

# =============================================================================
# THRESHOLD GENERATION CONFIGURATION
# =============================================================================

# For continuous features: threshold generation strategy
CONTINUOUS_THRESHOLD_STRATEGY = "unique_values"  # "unique_values" or "percentiles"

# For percentile-based thresholds (if strategy is "percentiles")
CONTINUOUS_PERCENTILES = [10, 25, 50, 75, 90, 95]

# For ordinal features: candidate levels to consider
ORDINAL_LEVELS = [1, 2, 3, 4, 5]

# For ordinal features with string values: map common patterns
ORDINAL_STRING_MAPPINGS = {
    'low': 1,
    'medium_low': 2, 
    'medium': 3,
    'medium_high': 4,
    'high': 5,
    'level_1': 1,
    'level_2': 2,
    'level_3': 3,
    'level_4': 4,
    'level_5': 5
}

# For binary features: which values to consider as thresholds
BINARY_CANDIDATES = [True, False]

# Maximum number of candidate thresholds per feature (to prevent excessive computation)
MAX_CANDIDATES_PER_FEATURE = 100

# Whether to include the minimum and maximum values as candidates
INCLUDE_MIN_MAX_CANDIDATES = True

# =============================================================================
# RULE EVALUATION CONFIGURATION
# =============================================================================

# How to break ties when multiple thresholds have the same precision
TIE_BREAKING_STRATEGY = "highest_support"  # "highest_support", "lowest_threshold", "highest_threshold"

# Whether to prefer rules that predict success vs failure outcomes
PREFER_SUCCESS_RULES = True

# Whether to validate rule performance on a separate validation split
USE_VALIDATION_SPLIT = False
VALIDATION_SPLIT_RATIO = 0.3  # Only used if USE_VALIDATION_SPLIT = True

# =============================================================================
# RULE COMBINATION LOGIC
# =============================================================================

# How to sort rules before combination
RULE_SORTING_STRATEGY = "weight_desc"  # "weight_desc", "precision_desc", "combined"

# For "combined" sorting: weights for different criteria
WEIGHT_IMPORTANCE = 0.7
PRECISION_IMPORTANCE = 0.3

# Whether to re-evaluate rule order after each addition
DYNAMIC_REORDERING = False

# Whether to allow removing rules that hurt combined precision
ALLOW_RULE_REMOVAL = False

# =============================================================================
# ADVANCED COMBINATION STRATEGIES
# =============================================================================

# Rule combination strategy: "simple_or", "weighted_scoring", "hierarchical", "ensemble"
COMBINATION_STRATEGY = "weighted_scoring"

# For weighted scoring: how to combine individual rule scores
SCORING_METHOD = "precision_weighted"  # "simple_sum", "confidence_weighted", "precision_weighted"

# Weighted scoring threshold (what total score triggers positive prediction)
WEIGHTED_SCORE_THRESHOLD = 0.08  # Will be tuned automatically

# For hierarchical approach: different precision tiers
HIERARCHICAL_TIERS = [
    {"min_precision": 0.15, "max_rules": 2},    # Tier 1: Very high precision rules
    {"min_precision": 0.08, "max_rules": 5},    # Tier 2: High precision rules  
    {"min_precision": 0.04, "max_rules": 8},    # Tier 3: Medium precision rules
    {"min_precision": 0.02, "max_rules": 10}    # Tier 4: Lower precision rules
]

# For ensemble approach: multiple sub-models
ENSEMBLE_STRATEGIES = ["high_precision", "high_recall", "balanced"]
ENSEMBLE_VOTING = "majority"  # "majority", "unanimous", "weighted"

# Target metrics for optimization
TARGET_RECALL = 0.50  # Try to capture at least 50% of positive cases
MIN_PRECISION = 0.20  # Minimum precision requirement
OPTIMIZE_FOR = "f1_score"  # "precision", "recall", "f1_score", "precision_at_recall"

# =============================================================================
# OUTPUT FORMAT CONFIGURATION
# =============================================================================

# JSON formatting settings
JSON_INDENT = 2
JSON_ENSURE_ASCII = False

# CSV formatting settings
CSV_FLOAT_PRECISION = 3

# Whether to include detailed metadata in outputs
INCLUDE_DETAILED_METADATA = True

# Whether to save intermediate results during calibration
SAVE_INTERMEDIATE_RESULTS = True

# =============================================================================
# VALIDATION AND QUALITY CONTROL
# =============================================================================

# Whether to validate test set composition
VALIDATE_TEST_SET = True
# Updated for training set (7800 rows, 780 positives used for threshold calibration)
EXPECTED_TEST_SIZE = 7800
EXPECTED_POSITIVE_COUNT = 780
EXPECTED_NEGATIVE_COUNT = 7020

# Whether to validate feature availability
VALIDATE_FEATURES = True

# Whether to check for missing values in critical features
CHECK_MISSING_VALUES = True

# Maximum allowed missing value ratio per feature
MAX_MISSING_RATIO = 0.1

# =============================================================================
# PERFORMANCE AND OPTIMIZATION
# =============================================================================

# Whether to use parallel processing for threshold evaluation
USE_PARALLEL_PROCESSING = False
N_JOBS = -1  # Number of parallel jobs (-1 = all available cores)

# Whether to cache threshold evaluations
CACHE_EVALUATIONS = True

# Whether to use early stopping in threshold search
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for N candidates

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging level for threshold calibration
LOG_LEVEL = "INFO"

# Whether to log each threshold evaluation
LOG_THRESHOLD_EVALUATIONS = False

# Whether to log rule combination steps
LOG_RULE_COMBINATIONS = True

# Whether to log detailed statistics
LOG_DETAILED_STATISTICS = True

# =============================================================================
# DEBUGGING AND DEVELOPMENT
# =============================================================================

# Whether to run in debug mode with additional checks
DEBUG_MODE = False

# Whether to save debug information
SAVE_DEBUG_INFO = False

# Random seed for reproducibility
RANDOM_SEED = 42 