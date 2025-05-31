#!/usr/bin/env python3

"""
Module 8: Threshold Calibration

This module calibrates thresholds for IF-THEN rule templates to achieve 
a target precision of 20% on the held-out test set. It:

1. Loads rule templates from Module 7
2. Generates candidate thresholds for each feature type
3. Evaluates individual rule performance
4. Combines rules iteratively (by OR) until precision target is met
5. Outputs calibrated rules and performance metrics

The module ensures explainable, high-precision investment decision rules.
"""

from .threshold_calibration import run_threshold_calibration

__all__ = ['run_threshold_calibration'] 