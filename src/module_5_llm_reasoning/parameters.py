#!/usr/bin/env python3

"""
Parameters Configuration for Module 5: LLM Chain-of-Thought Generation

This module defines all hyperparameters, prompt templates, and configuration 
settings used in the batched LLM chain-of-thought generation process. The module
generates both descriptive and causal explanations for investment decisions
using OpenAI's GPT models with batched prompts to minimize API costs.

All parameters are centralized here to ensure reproducibility and easy tuning.
"""

import os

# =============================================================================
# INPUT/OUTPUT PATHS
# =============================================================================

# Input path (from Module 4 feature selection output)
TRAIN_SELECTED_PATH = "artifacts/feature_selection/train_selected.pkl"

# Output directory and files
OUTPUT_DIR = "artifacts/llm_output"
DESCRIPTIVE_JSONL_PATH = os.path.join(OUTPUT_DIR, "descriptive.jsonl")
CAUSAL_JSONL_PATH = os.path.join(OUTPUT_DIR, "causal.jsonl")

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# OpenAI model configuration
LLM_MODEL = "gpt-4-turbo"  # Primary model for chain-of-thought generation
LLM_TEMPERATURE = 0.0  # Deterministic output for reproducibility
LLM_MAX_TOKENS = 4000  # Maximum tokens per response

# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================

# Batch processing controls for testing and cost management
BATCH_SIZE = 30  # Founders per batch - adjust based on token limits and cost considerations
BATCH_LIMIT = 17  # Maximum batches to process - set to None for full dataset processing

# Resume/checkpoint functionality
ENABLE_RESUME = True  # Set to True to resume from last completed batch - 1 (REPLACE LAST BATCH MODE), False to start fresh
RESUME_MODE = "auto"  # "auto" = determine from output files and replace last batch, "force_restart" = always start from 0

# API retry configuration
MAX_RETRIES = 3  # Maximum number of retry attempts for failed API calls
RETRY_DELAY = 2.0  # Seconds to wait between retry attempts

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# System message for maintaining consistent LLM behavior
SYSTEM_MESSAGE = """You are an expert investment analyst specializing in startup founder evaluation. 
You provide detailed, analytical explanations for investment decisions based on founder characteristics and startup features. 
Always output valid JSON only, with no additional text or formatting."""

# Template for descriptive chain-of-thought generation
DESCRIPTIVE_TEMPLATE = """You are an investment-decision assistant analyzing startup founders. For each founder in the following dataset, provide a detailed step-by-step explanation of why they succeeded or failed based on their characteristics.

For each founder, consider and explain in detail:
1. Educational background and qualifications - how they contributed to success/failure
2. Years of experience and industry knowledge - depth and relevance analysis
3. Previous startup history and outcomes - lessons learned and track record
4. Leadership and personality traits - impact on team and company performance
5. Company and market factors - external influences and timing
6. Any missing information and its implications on the assessment

Please provide thorough explanations with specific reasoning for each factor. Aim for comprehensive analysis that clearly connects founder characteristics to outcomes.

Output your response as a JSON array of objects, one object per founder, with exactly these fields:
- "index": the founder's index number (integer, matching input)
- "descriptive_cot": detailed step-by-step explanation (string, minimum 70 characters)

Do not include any text outside the JSON array.

Input data (JSON array of founder profiles):
{batch_data}"""

# Template for causal chain-of-thought generation  
CAUSAL_TEMPLATE = """Given the following descriptive explanations of founder success/failure, extract the key causal relationships for each founder. Focus on identifying specific feature → mechanism → outcome chains that explain the investment decision.

For each founder, provide detailed causal analysis by identifying:
1. Primary predictive features that influenced the outcome - be specific about which characteristics mattered most
2. The causal mechanism linking each feature to success/failure - explain the "how" and "why" 
3. How multiple features interact to create the final outcome - show interconnections
4. The strength and confidence level of each causal relationship - indicate which are most reliable

Format each causal explanation with clear feature→mechanism→outcome triples, such as:
"High education level → Enhanced credibility and network access → Increased investor confidence and better fundraising"
"Previous startup failure → Learned from mistakes and gained resilience → Better strategic decisions and risk management"

Provide comprehensive causal chains that thoroughly explain the path from founder characteristics to investment outcomes.

Output as a JSON array with exactly these fields:
- "index": the founder's index number (integer, matching input)  
- "causal_cot": detailed causal explanation with feature→mechanism→outcome chains (string, minimum 70 characters)

Do not include any text outside the JSON array.

Input data (JSON array of descriptive explanations):
{batch_data}"""

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Expected output structure validation
REQUIRED_DESCRIPTIVE_FIELDS = ["index", "descriptive_cot"]
REQUIRED_CAUSAL_FIELDS = ["index", "causal_cot"]

# Minimum explanation length (characters) to ensure quality
MIN_EXPLANATION_LENGTH = 70
MAX_EXPLANATION_LENGTH = 2000

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging levels and output
LOG_LEVEL = "INFO"
LOG_BATCH_PROGRESS = True  # Whether to log progress for each batch
LOG_API_CALLS = True  # Whether to log API call details
LOG_RESPONSE_SAMPLES = True  # Whether to log sample responses for debugging

# =============================================================================
# ERROR HANDLING CONFIGURATION  
# =============================================================================

# API error handling
HANDLE_RATE_LIMITS = True  # Whether to automatically handle rate limiting
RATE_LIMIT_DELAY = 60.0  # Seconds to wait when rate limited

# Response validation
STRICT_JSON_VALIDATION = True  # Whether to enforce strict JSON parsing
# Note: No fallback mechanisms - any failure will raise an exception

# =============================================================================
# FEATURE HANDLING CONFIGURATION
# =============================================================================

# Features to exclude from LLM input (handled separately)
EXCLUDE_FROM_LLM = ["success"]  # Target variable excluded from feature description

# How to handle missing value flags in LLM prompts
INCLUDE_MISSING_FLAGS = True  # Whether to include *_missing columns in feature descriptions
MISSING_FLAG_DESCRIPTION = "indicates whether the original value was missing/invalid"

# =============================================================================
# RESUME BEHAVIOR EXPLANATION
# =============================================================================
# 
# When ENABLE_RESUME = True:
# - The system finds the last completed batch in the output files
# - It then resumes from the beginning of that batch (effectively replacing it)
# - This is useful when the last batch output is determined to be wrong and needs replacement
# - Output files are automatically truncated to remove the last batch's entries
#
# Example:
# - If batches 0, 1, 2 are completed (indices 0-59 for batch_size=20)
# - The system will resume from batch 2 (index 40), replacing the previous batch 2 output
# - This ensures the "wrong" output from the last completed batch is completely replaced
#
# When ENABLE_RESUME = False:
# - Always start from the beginning, clearing all output files
#
# When RESUME_MODE = "force_restart":
# - Ignore existing files and start from the beginning (same as ENABLE_RESUME = False) 