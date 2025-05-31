#!/usr/bin/env python3

"""
Parameters Configuration for Module 7: IF-THEN Rule Extraction

This module defines all configuration settings for converting the weighted
causal graph into IF-THEN rule templates, including:
- Number of top edges to extract as rules
- Feature type mappings
- Rule template formats
- File paths and output settings

All parameters are centralized here for easy tuning and reproducibility.
"""

import os

# =============================================================================
# INPUT/OUTPUT PATHS
# =============================================================================

# Input paths (from previous modules)
CAUSAL_GRAPH_PATH = "artifacts/graphs/causal_graph.pkl"
SELECTED_FEATURES_PATH = "artifacts/feature_selection/selected_features.json"
TRAIN_PREPARED_PATH = "artifacts/prepared_splits/train_prepared.pkl"  # For feature type inspection

# Output directory and files
OUTPUT_DIR = "artifacts/rules"
INITIAL_RULES_PATH = os.path.join(OUTPUT_DIR, "initial_rules.json")
RULE_STATISTICS_PATH = os.path.join(OUTPUT_DIR, "rule_statistics.json")

# =============================================================================
# RULE EXTRACTION CONFIGURATION
# =============================================================================

# Number of top weighted edges to convert into rules
NUM_TOP_EDGES = 25

# Minimum edge weight to consider for rule extraction
MIN_EDGE_WEIGHT = 1

# Whether to include both success and failure rules
INCLUDE_FAILURE_RULES = True

# Whether to prioritize success rules over failure rules
PRIORITIZE_SUCCESS_RULES = True

# =============================================================================
# FEATURE TYPE MAPPING
# =============================================================================

# Manual feature type mappings for our known features
# This overrides automatic type detection when needed
FEATURE_TYPE_OVERRIDES = {
    # Continuous features (numeric with meaningful thresholds)
    'yoe': 'continuous',
    'emotional_intelligence': 'continuous',
    'number_of_leadership_roles': 'continuous',
    'technical_leadership_roles': 'continuous',
    'number_of_companies': 'continuous',
    'number_of_roles': 'continuous',
    'career_growth': 'continuous',
    'board_advisor_roles': 'continuous',
    
    # Ordinal features (1-5 scale qualitative measures)
    'education_institution': 'ordinal',
    'education_field_of_study_qual': 'ordinal',
    'emotional_intelligence_qual': 'ordinal',
    'industry_achievements': 'ordinal',
    'significant_press_media_coverage': 'ordinal',
    'persona': 'ordinal',
    'perseverance': 'ordinal',
    'languages': 'ordinal',
    
    # Binary features (True/False)
    'ceo_experience': 'binary',
    'big_company_experience': 'binary',
    'nasdaq_company_experience': 'binary',
    'nasdaq_leadership': 'binary',
    'VC_experience': 'binary',
    'investor_quality_prior_startup': 'binary',
    'previous_startup_funding_experience_as_ceo': 'binary',
    
    # Categorical features (if any)
    'name': 'categorical',  # Skip in practice
    'org_name': 'categorical',  # Skip in practice
}

# =============================================================================
# RULE TEMPLATE FORMATS
# =============================================================================

# Template strings for different feature types
RULE_TEMPLATES = {
    'continuous': "IF {feature} >= <threshold> THEN P({outcome}) >= <confidence>",
    'ordinal': "IF {feature} >= <ordinal_level> THEN P({outcome}) >= <confidence>",
    'binary_success': "IF {feature} == True THEN P({outcome}) >= <confidence>",
    'binary_failure': "IF {feature} == False THEN P({outcome}) >= <confidence>",
    'categorical': "IF {feature} == <category> THEN P({outcome}) >= <confidence>",
}

# Placeholder mappings for each feature type
PLACEHOLDER_MAPPINGS = {
    'continuous': {
        'threshold_placeholder': '<threshold>',
        'confidence_placeholder': '<confidence>'
    },
    'ordinal': {
        'threshold_placeholder': '<ordinal_level>',
        'confidence_placeholder': '<confidence>'
    },
    'binary': {
        'confidence_placeholder': '<confidence>'
    },
    'categorical': {
        'threshold_placeholder': '<category>',
        'confidence_placeholder': '<confidence>'
    }
}

# =============================================================================
# FEATURE TYPE DETECTION
# =============================================================================

# Data types that should be treated as continuous
CONTINUOUS_DTYPES = ['int64', 'float64', 'int32', 'float32']

# Data types that should be treated as categorical
CATEGORICAL_DTYPES = ['object', 'string', 'category']

# Data types that should be treated as binary (0/1)
BINARY_DTYPES = ['bool', 'int8']  # Also check for 0/1 values in int columns

# Maximum unique values to consider a numeric feature as ordinal/categorical
MAX_UNIQUE_FOR_ORDINAL = 10

# Minimum unique values to consider a feature as continuous
MIN_UNIQUE_FOR_CONTINUOUS = 5

# =============================================================================
# RULE QUALITY THRESHOLDS
# =============================================================================

# Minimum confidence for a rule to be considered high-quality
MIN_RULE_CONFIDENCE = 0.6

# Minimum support (number of examples) for a rule to be reliable
MIN_RULE_SUPPORT = 10

# =============================================================================
# OUTPUT FORMAT CONFIGURATION
# =============================================================================

# JSON formatting settings
JSON_INDENT = 2
JSON_ENSURE_ASCII = False

# Whether to include detailed metadata in rule output
INCLUDE_RULE_METADATA = True

# Whether to include mechanism descriptions in rules
INCLUDE_MECHANISMS = True

# Maximum number of mechanisms to include per rule
MAX_MECHANISMS_PER_RULE = 5

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Whether to validate feature names against selected features
VALIDATE_FEATURE_NAMES = True

# Whether to skip rules for features not in selected set
SKIP_INVALID_FEATURES = True

# Whether to log detailed rule generation statistics
LOG_RULE_STATISTICS = True

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging level for rule extraction
LOG_LEVEL = "INFO"

# Whether to log each rule generation step
LOG_RULE_GENERATION = True

# Whether to log feature type detection details
LOG_FEATURE_TYPE_DETECTION = True 