#!/usr/bin/env python3
"""
Module 9: Evaluation and Reporting

This module provides the final evaluation of calibrated IF-THEN rules on held-out test data,
computing comprehensive performance metrics and generating human-readable reports with
representative chain-of-thought excerpts for full explainability.

Main Functions:
    run_evaluation(): Execute the complete evaluation pipeline

Key Features:
    - Direct rule application to feature values (no CoT-based decisions)
    - Comprehensive metrics computation (TP, FP, FN, TN, precision, recall, F1)
    - Representative CoT example extraction for each rule
    - Detailed Markdown report generation
    - Full traceability and explainability
"""

from .eval import run_evaluation

__all__ = ['run_evaluation']

# Module metadata
__version__ = "1.0.0"
__author__ = "Vela Partners Investment Decision Engine"
__description__ = "Final evaluation and reporting module for calibrated investment rules" 