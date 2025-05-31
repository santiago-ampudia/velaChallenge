#!/usr/bin/env python3

"""
Module 7: IF-THEN Rule Extraction

This module converts the weighted causal graph from Module 6 into human-readable
IF-THEN rule templates. Each rule represents a high-confidence path from feature
conditions to outcomes (success/failure).

The module extracts the top weighted edges, determines feature types, and creates
rule templates with placeholders for thresholds and confidence levels that will
be calibrated in Module 8.
"""

from .extract_rules import run_rule_extraction

__all__ = ['run_rule_extraction'] 