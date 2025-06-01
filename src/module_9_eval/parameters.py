#!/usr/bin/env python3
"""
Module 9 Parameters: Evaluation and Reporting Configuration

This module defines all parameters and configuration settings used in Module 9
for evaluating calibrated rules on the test set and generating comprehensive reports.
"""

import os
from typing import Dict, Any, List

# =============================================================================
# FILE PATHS - All paths relative to the main.py execution directory
# =============================================================================

# Input file paths
CALIBRATED_RULES_PATH = "artifacts/rules_calibrated/calibrated_rules.json"
TEST_DATA_PATH = "artifacts/feature_selection/test_selected.pkl"
DESCRIPTIVE_COT_PATH = "artifacts/llm_output/descriptive.jsonl"

# Output directory and file paths
OUTPUT_DIR = "artifacts/eval"
FINAL_METRICS_PATH = os.path.join(OUTPUT_DIR, "final_metrics.json")
EVALUATION_REPORT_PATH = os.path.join(OUTPUT_DIR, "report.md")

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Target column name in the test dataset
TARGET_COLUMN = "success"

# Feature types mapping for rule application
SUPPORTED_FEATURE_TYPES = {
    "continuous": ">=",  # Greater than or equal to threshold
    "ordinal": ">=",     # Greater than or equal to threshold
    "binary": "==",      # Equal to True (boolean)
    "categorical": "=="  # Equal to specific category value
}

# =============================================================================
# REPORT GENERATION CONFIGURATION
# =============================================================================

# Maximum number of CoT examples to include per rule
MAX_COT_EXAMPLES_PER_RULE = 2

# Decimal precision for metrics in the report
METRICS_DECIMAL_PLACES = 3

# Report template configuration
REPORT_TITLE = "Final Evaluation Report"
REPORT_SECTIONS = {
    "summary": "Test Set Summary",
    "metrics": "Overall Performance Metrics",
    "rules": "Calibrated Rules Detail"
}

# Markdown table headers for rules section
RULES_TABLE_HEADERS = [
    "#",
    "Feature", 
    "Type",
    "Threshold",
    "Rule Prec.",
    "Rule Support",
    "Comb. Prec.",
    "Comb. Support",
    "Examples (Descriptive CoT excerpts)"
]

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Expected test set characteristics for validation (fixed for fair evaluation)
EXPECTED_TEST_SET_SIZE = 1000
EXPECTED_POSITIVE_COUNT = 100  # MAJOR FIX: 10% positive rate (same as training)
EXPECTED_NEGATIVE_COUNT = 900  # MAJOR FIX: 90% negative rate (same as training)

# Required fields in calibrated rules
REQUIRED_RULE_FIELDS = [
    "feature",
    "feature_type", 
    "outcome",
    "weight",
    "threshold",
    "rule_precision",
    "rule_support",
    "combined_precision",
    "combined_support"
]

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log messages for different stages
LOG_MESSAGES = {
    "start": "Starting Module 9: Evaluation and Reporting",
    "load_rules": "Loading calibrated rules from {path}",
    "load_test": "Loading test dataset from {path}",
    "load_cot": "Loading descriptive CoT explanations from {path}",
    "apply_rules": "Applying {count} calibrated rules to test data",
    "compute_metrics": "Computing final performance metrics",
    "extract_examples": "Extracting representative CoT examples for each rule",
    "generate_report": "Generating comprehensive Markdown report",
    "save_metrics": "Saving final metrics to {path}",
    "save_report": "Saving evaluation report to {path}",
    "complete": "Module 9 completed successfully"
}

# Error messages
ERROR_MESSAGES = {
    "missing_file": "Required input file not found: {path}",
    "invalid_rules": "Invalid calibrated rules format: {error}",
    "invalid_test_data": "Invalid test dataset format: {error}",
    "missing_target": "Target column '{target}' not found in test data",
    "wrong_test_size": "Test set size {actual} does not match expected {expected}",
    "no_positive_examples": "No positive examples found for rule: {rule}",
    "missing_cot": "CoT explanation not found for index: {index}",
    "save_error": "Failed to save file to {path}: {error}"
}

# =============================================================================
# UTILITY PARAMETERS
# =============================================================================

# Random seed for reproducible example selection
RANDOM_SEED = 42

# File encoding
FILE_ENCODING = "utf-8"

# JSON formatting
JSON_INDENT = 2

# Markdown formatting
MARKDOWN_TABLE_SEPARATOR = "|"
MARKDOWN_COT_INDENT = "&nbsp;&nbsp;&nbsp;&nbsp;"
MARKDOWN_LINE_BREAK = "\n"

# =============================================================================
# HELPER FUNCTIONS FOR PARAMETER VALIDATION
# =============================================================================

def validate_paths() -> bool:
    """
    Validate that all required input paths exist.
    
    Returns:
        bool: True if all paths exist, False otherwise
    """
    required_paths = [
        CALIBRATED_RULES_PATH,
        TEST_DATA_PATH,
        DESCRIPTIVE_COT_PATH
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            return False
    
    return True

def get_rule_application_operator(feature_type: str) -> str:
    """
    Get the operator for applying a rule based on feature type.
    
    Args:
        feature_type: Type of the feature ('continuous', 'ordinal', 'binary', 'categorical')
        
    Returns:
        str: Operator string
        
    Raises:
        ValueError: If feature type is not supported
    """
    if feature_type not in SUPPORTED_FEATURE_TYPES:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    return SUPPORTED_FEATURE_TYPES[feature_type]

def format_metric(value: float) -> str:
    """
    Format a metric value to the specified number of decimal places.
    
    Args:
        value: Metric value to format
        
    Returns:
        str: Formatted metric string
    """
    return f"{value:.{METRICS_DECIMAL_PLACES}f}" 