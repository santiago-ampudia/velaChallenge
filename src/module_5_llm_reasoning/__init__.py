#!/usr/bin/env python3

"""
Module 5: LLM Chain-of-Thought Generation using Batched Processing

This module provides the interface for generating chain-of-thought explanations
using OpenAI's LLM models with batched prompts to minimize API costs while
maintaining high-quality outputs.

Main Components:
- LLMChainOfThoughtGenerator: Core class implementing the batched LLM pipeline
- run_llm_reasoning: Main entry point for the module
- parameters: Configuration settings and prompt templates

The module takes feature-selected training data from Module 4, processes it
in batches through the LLM to generate both descriptive and causal explanations,
and outputs JSONL files for downstream graph construction and rule extraction.

Key Features:
- Batched processing to minimize API calls and costs
- Robust error handling and retry logic for API failures
- Comprehensive validation of LLM responses
- Separate descriptive and causal explanation generation
- JSON Lines output format for easy streaming processing

Usage:
    from module_5_llm_reasoning import run_llm_reasoning
    
    descriptive_path, causal_path = run_llm_reasoning()

Input Requirements:
- artifacts/feature_selection/train_selected.pkl: Feature-selected training data

Output Artifacts:
- artifacts/llm_output/descriptive.jsonl: Descriptive chain-of-thought explanations
- artifacts/llm_output/causal.jsonl: Causal chain-of-thought explanations

Note: Requires OpenAI API key to be configured in environment variables.
"""

from .llm_reasoning import run_llm_reasoning, LLMChainOfThoughtGenerator
from . import parameters

__all__ = [
    'run_llm_reasoning',
    'LLMChainOfThoughtGenerator', 
    'parameters'
] 