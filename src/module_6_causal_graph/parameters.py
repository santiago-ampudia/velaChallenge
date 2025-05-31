#!/usr/bin/env python3

"""
Parameters Configuration for Module 6: Causal Graph Construction

This module defines all configuration settings, file paths, and parameters
used in parsing causal chain-of-thought explanations and building weighted
directed graphs for investment decision analysis.

The module extracts "feature → mechanism → outcome" triples from LLM-generated
explanations and aggregates them into a NetworkX directed graph with weighted
edges representing the frequency of causal relationships.

All parameters are centralized here to ensure reproducibility and easy tuning.
"""

import os
import re

# =============================================================================
# INPUT/OUTPUT PATHS
# =============================================================================

# Input path (from Module 5 LLM reasoning output)
INPUT_JSONL_PATH = "artifacts/llm_output/causal.jsonl"  # Updated to use consistent naming
CAUSAL_JSONL_PATH = "artifacts/llm_output/causal.jsonl"  # Kept for backward compatibility

# Selected features path (from Module 4 for validation)
SELECTED_FEATURES_PATH = "artifacts/feature_selection/selected_features.json"

# Output directory and files
OUTPUT_DIR = "artifacts/graphs"
CAUSAL_GRAPH_PKL_PATH = os.path.join(OUTPUT_DIR, "causal_graph.pkl")
EDGE_WEIGHTS_CSV_PATH = os.path.join(OUTPUT_DIR, "edge_weights.csv")
GRAPH_VISUALIZATION_PATH = os.path.join(OUTPUT_DIR, "graph_visualization.png")

# =============================================================================
# PARSING CONFIGURATION
# =============================================================================

# Regex patterns for extracting causal triples from LLM explanations
# Updated to handle the actual LLM output format with descriptive phrases

# Primary pattern: Flexible matching for "descriptive phrase → mechanism → outcome"
# Now handles multi-word descriptive features and case variations
PRIMARY_CAUSAL_PATTERN = re.compile(
    r'([^→;]+?)\s*→\s*([^→;]+?)\s*→\s*(Success|Failure|success|failure|positive|negative|increased|decreased|better|worse)',
    re.IGNORECASE | re.MULTILINE
)

# Flexible pattern: Captures ANY causal chain regardless of outcome format
FLEXIBLE_CAUSAL_PATTERN = re.compile(
    r'([^→;]+?)\s*→\s*([^→;]+?)\s*→\s*([^→;]+?)(?:\.|;|$)',
    re.IGNORECASE | re.MULTILINE
)

# Incomplete pattern: Captures chains missing final outcome (feature → mechanism)
INCOMPLETE_CAUSAL_PATTERN = re.compile(
    r'([^→;]+?)\s*→\s*([^→;]+?)(?:\s+(?:for|necessary for|leading to|resulting in)\s+([^.;]+?))?(?:\.|;|$)',
    re.IGNORECASE | re.MULTILINE
)

# Alternative pattern: More flexible for natural language causal statements
ALTERNATIVE_PATTERN_1 = re.compile(
    r'([^→;]+?)\s+(?:leads?\s+to|results?\s+in|causes?|contributes?\s+to)\s+([^→;]+?)\s+(?:resulting\s+in|leading\s+to|causing|contributing\s+to)\s+(success|failure|positive|negative)',
    re.IGNORECASE | re.MULTILINE
)

# Pattern for explicit causal statements with qualifiers
EXPLICIT_CAUSAL_PATTERN = re.compile(
    r'(?:High|Low|Strong|Weak|Good|Poor|Excellent|Limited|Significant|Medium|Previous|Extensive|Lack\s+of)?\s*([^→;]+?)\s*→\s*([^→;]+?)\s*→\s*(Success|Failure|success|failure)',
    re.IGNORECASE | re.MULTILINE
)

# Confidence extraction pattern: "I am X% confident" or "confidence: X%" or "(X%)"
CONFIDENCE_PATTERN = re.compile(
    r'(?:I\s+am\s+)?(\d+)%?\s*(?:confident|confidence|certainty|\(.*?\))',
    re.IGNORECASE
)

# List of all patterns to try in order of priority
CAUSAL_PATTERNS = [
    ("primary", PRIMARY_CAUSAL_PATTERN),
    ("explicit", EXPLICIT_CAUSAL_PATTERN), 
    ("flexible", FLEXIBLE_CAUSAL_PATTERN),  # Captures any complete causal chain
    ("incomplete", INCOMPLETE_CAUSAL_PATTERN),  # NEW: Captures incomplete chains
    ("alternative_1", ALTERNATIVE_PATTERN_1)
]

# =============================================================================
# OUTCOME NORMALIZATION
# =============================================================================

# Mapping of outcome phrases to standardized outcomes
OUTCOME_NORMALIZATION = {
    # Success indicators
    'success': 'success',
    'successful': 'success',
    'succeeded': 'success',
    'positive': 'success',
    'increased': 'success',
    'better': 'success',
    'improved': 'success',
    'enhanced': 'success',
    'good': 'success',
    'excellent': 'success',
    
    # Failure indicators
    'failure': 'failure',
    'failed': 'failure',
    'failing': 'failure',
    'negative': 'failure',
    'decreased': 'failure',
    'worse': 'failure',
    'poor': 'failure',
    'bad': 'failure',
    'limited': 'failure',
    'weak': 'failure'
}

# =============================================================================
# GRAPH CONSTRUCTION PARAMETERS
# =============================================================================

# Minimum edge weight to include in the final graph (filters out very rare connections)
MIN_EDGE_WEIGHT = 1

# Maximum number of mechanisms to store per edge (to prevent memory issues)
MAX_MECHANISMS_PER_EDGE = 10000  # Effectively unlimited for our dataset size

# Whether to include bidirectional edges (feature → success AND feature → failure)
INCLUDE_BIDIRECTIONAL_EDGES = True

# Node types for graph structure
FEATURE_NODE_PREFIX = "feature_"  # Prefix for feature nodes (optional)
OUTCOME_NODES = ["success", "failure"]  # Standard outcome node names

# =============================================================================
# TEXT PROCESSING PARAMETERS
# =============================================================================

# Maximum length of mechanism text to store (truncate if longer)
MAX_MECHANISM_LENGTH = 500

# Characters to strip from mechanism text
MECHANISM_STRIP_CHARS = " \t\n\r.,;:!?"

# Words to remove from feature text for better matching
FEATURE_STOP_WORDS = {
    'high', 'low', 'good', 'bad', 'strong', 'weak', 'excellent', 'poor', 
    'significant', 'limited', 'extensive', 'medium', 'previous', 'prior',
    'lack', 'of', 'the', 'a', 'an', 'and', 'or', 'with', 'in', 'on'
}

# Whether to clean and normalize mechanism text
CLEAN_MECHANISM_TEXT = True

# Whether to split causal explanations on semicolons for multiple chains
SPLIT_ON_SEMICOLONS = True

# =============================================================================
# OUTPUT FORMAT PARAMETERS
# =============================================================================

# CSV column headers for edge weights file
CSV_HEADERS = [
    "feature",
    "outcome", 
    "weight",
    "mechanisms",
    "confidence_avg",
    "confidence_sum",
    "mechanism_count"
]

# Separator for concatenating multiple mechanisms in CSV
MECHANISM_SEPARATOR = " | "

# Maximum total length of mechanisms string in CSV (truncate if longer)
MAX_CSV_MECHANISMS_LENGTH = 1000

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Expected fields in input JSONL objects
REQUIRED_JSONL_FIELDS = ["index", "features", "success_label", "causal_cot"]

# Minimum number of characters required in causal_cot field
MIN_CAUSAL_COT_LENGTH = 10

# Maximum number of characters to process in causal_cot (performance limit)
MAX_CAUSAL_COT_LENGTH = 5000

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging levels and output
LOG_LEVEL = "INFO"
LOG_PARSING_DETAILS = True  # Whether to log details of pattern matching
LOG_GRAPH_STATISTICS = True  # Whether to log graph construction statistics
LOG_PATTERN_MATCHES = False  # Whether to log individual pattern matches (verbose)

# =============================================================================
# ERROR HANDLING CONFIGURATION
# =============================================================================

# Whether to strictly validate feature names against known features
STRICT_FEATURE_VALIDATION = True

# Whether to skip invalid triples or raise exceptions
SKIP_INVALID_TRIPLES = False  # False = raise exception for invalid triples

# Maximum number of parsing errors to tolerate before failing
MAX_PARSING_ERRORS = 10

# =============================================================================
# PERFORMANCE PARAMETERS
# =============================================================================

# Batch size for processing JSONL entries (for memory management)
PROCESSING_BATCH_SIZE = 100

# Whether to use multiprocessing for parsing (if dataset is very large)
USE_MULTIPROCESSING = False

# Number of worker processes (if multiprocessing enabled)
NUM_WORKERS = 4

# =============================================================================
# GRAPH VISUALIZATION PARAMETERS (Optional)
# =============================================================================

# Whether to generate and save graph visualization
GENERATE_VISUALIZATION = True

# Graph layout algorithm for visualization
LAYOUT_ALGORITHM = "spring"  # Options: spring, circular, kamada_kawai, random

# Figure size for visualization (width, height in inches)
FIGURE_SIZE = (12, 8)

# Node size scaling factor
NODE_SIZE_FACTOR = 1000

# Edge width scaling factor  
EDGE_WIDTH_FACTOR = 2

# Whether to show edge labels in visualization
SHOW_EDGE_LABELS = True

# =============================================================================
# FEATURE VALIDATION DATA
# =============================================================================

# Whether to load and validate against selected features
VALIDATE_AGAINST_SELECTED_FEATURES = True

# =============================================================================
# FEATURE NAME MAPPING
# =============================================================================

# Mapping from LLM descriptive phrases to actual feature names
# This handles the mismatch between LLM output and our feature names
FEATURE_NAME_MAPPING = {
    # Education-related mappings
    'educational qualifications': 'education_institution',
    'education level': 'education_institution', 
    'educational background': 'education_institution',
    'education institution': 'education_institution',
    'educational qualification': 'education_institution',
    'education field': 'education_field_of_study_qual',
    'field of study': 'education_field_of_study_qual',
    'educational field': 'education_field_of_study_qual',
    
    # Experience mappings
    'industry experience': 'yoe',
    'years of experience': 'yoe',
    'experience': 'yoe',
    'work experience': 'yoe',
    'professional experience': 'yoe',
    'extensive experience': 'yoe',
    'limited years of experience': 'yoe',
    'involvement in a high number of companies': 'number_of_companies',
    'involvement in multiple companies': 'number_of_companies',
    'lack of previous company involvements': 'number_of_companies',
    'multiple company distractions': 'number_of_companies',
    
    # Emotional intelligence mappings
    'emotional intelligence': 'emotional_intelligence',
    'eq': 'emotional_intelligence',
    'emotional intelligence qual': 'emotional_intelligence_qual',
    
    # Leadership mappings
    'ceo experience': 'ceo_experience',
    'leadership roles': 'number_of_leadership_roles',
    'leadership experience': 'number_of_leadership_roles',
    'technical leadership roles': 'technical_leadership_roles',
    'technical leadership': 'technical_leadership_roles',
    'board advisor roles': 'board_advisor_roles',
    'advisory roles': 'board_advisor_roles',
    
    # Company experience mappings
    'big company experience': 'big_company_experience',
    'experience in big companies': 'big_company_experience',
    'nasdaq company experience': 'nasdaq_company_experience',
    'nasdaq-listed companies': 'nasdaq_company_experience',
    'involvement in a nasdaq-listed company': 'nasdaq_company_experience',
    'nasdaq leadership': 'nasdaq_leadership',
    'vc experience': 'VC_experience',
    'venture capital experience': 'VC_experience',
    
    # Startup mappings
    'previous startups': 'investor_quality_prior_startup',
    'startup experience': 'investor_quality_prior_startup',
    'previous startup funding experience': 'previous_startup_funding_experience_as_ceo',
    'startup funding experience': 'previous_startup_funding_experience_as_ceo',
    'funding experience': 'previous_startup_funding_experience_as_ceo',
    
    # Skill set mappings
    'balanced skill set': 'persona',
    'skill set': 'persona',
    'specific focus on industry': 'persona',
    'strong focus and adaptability': 'perseverance',
    'ability to speak multiple languages': 'languages',
    
    # Other mappings
    'languages': 'languages',
    'perseverance': 'perseverance',
    'press coverage': 'significant_press_media_coverage',
    'media coverage': 'significant_press_media_coverage',
    'press media coverage': 'significant_press_media_coverage',
    'industry achievements': 'industry_achievements',
    'achievements': 'industry_achievements',
    'career growth': 'career_growth',
    'number of companies': 'number_of_companies',
    'number of roles': 'number_of_roles',
    'persona': 'persona',
    'investor quality': 'investor_quality_prior_startup'
}

# Additional partial matching keywords for fuzzy feature detection
FEATURE_KEYWORDS = {
    'education_institution': ['education', 'educational', 'university', 'college', 'degree', 'qualification'],
    'yoe': ['experience', 'years', 'yoe', 'industry', 'professional', 'work', 'extensive', 'limited'],
    'emotional_intelligence': ['emotional', 'intelligence', 'eq', 'emotional_intelligence'],
    'ceo_experience': ['ceo', 'chief', 'executive'],
    'technical_leadership_roles': ['technical', 'leadership', 'tech', 'technology'],
    'number_of_leadership_roles': ['leadership', 'roles', 'leader', 'management'],
    'perseverance': ['perseverance', 'persistence', 'resilience', 'determination', 'focus', 'adaptability'],
    'languages': ['languages', 'language', 'linguistic', 'multilingual', 'speak'],
    'significant_press_media_coverage': ['press', 'media', 'coverage', 'publicity', 'news'],
    'industry_achievements': ['achievements', 'accomplishments', 'recognition', 'awards'],
    'big_company_experience': ['big', 'large', 'major', 'corporation', 'enterprise'],
    'nasdaq_company_experience': ['nasdaq', 'public', 'listed'],
    'VC_experience': ['vc', 'venture', 'capital', 'investment'],
    'previous_startup_funding_experience_as_ceo': ['funding', 'fundraising', 'capital', 'investment'],
    'investor_quality_prior_startup': ['startup', 'startups', 'investor', 'quality'],
    'board_advisor_roles': ['board', 'advisor', 'advisory', 'director'],
    'career_growth': ['career', 'growth', 'progression', 'advancement'],
    'number_of_companies': ['companies', 'firms', 'organizations', 'involvement', 'multiple', 'number'],
    'number_of_roles': ['roles', 'positions', 'jobs'],
    'persona': ['persona', 'personality', 'character', 'skill', 'balanced', 'specific'],
    'education_field_of_study_qual': ['field', 'study', 'major', 'specialization'],
    'emotional_intelligence_qual': ['emotional', 'intelligence', 'qualitative']
}

# Minimum similarity threshold for fuzzy feature matching (0.0 to 1.0)
FUZZY_MATCH_THRESHOLD = 0.6 

# =============================================================================
# SEMANTIC OUTCOME DETECTION
# =============================================================================

# Keywords that indicate positive/successful outcomes
SUCCESS_OUTCOME_KEYWORDS = {
    'success', 'successful', 'achieve', 'accomplished', 'enhanced', 'improved', 
    'increased', 'better', 'effective', 'strong', 'good', 'excellent', 'positive',
    'growth', 'attraction', 'confidence', 'capability', 'understanding', 'insights',
    'management', 'innovation', 'strategic', 'acquisition', 'investment', 'securing'
}

# Keywords that indicate negative/failure outcomes  
FAILURE_OUTCOME_KEYWORDS = {
    'failure', 'failed', 'hindered', 'limited', 'reduced', 'insufficient', 'lack',
    'poor', 'weak', 'negative', 'decreased', 'worse', 'challenges', 'difficulties',
    'obstacles', 'problems', 'issues', 'constraints', 'barriers'
}

# Minimum keyword match threshold for semantic detection
SEMANTIC_OUTCOME_THRESHOLD = 0.6 