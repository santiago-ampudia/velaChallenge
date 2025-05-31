#!/usr/bin/env python3
"""
Module 4: Feature Selection via XGBoost + SHAP

This module provides the interface for feature selection using XGBoost-based
importance ranking with SHAP values. It selects the top 25 most predictive
features from the prepared datasets to focus downstream analysis on the
strongest signals.

Main Components:
- FeatureSelector: Core class implementing the selection pipeline
- run_feature_selection: Main entry point for the module
- parameters: Configuration settings and hyperparameters

The module takes prepared training and test datasets from Module 3, trains
an XGBoost classifier with appropriate class weights, computes SHAP feature
importances, and outputs filtered datasets containing only the most predictive
features plus their corresponding missing flags.
"""

from .feature_selection import run_feature_selection

__all__ = ['run_feature_selection']

# Module metadata
__version__ = "1.0.0"
__author__ = "Vela Partners Investment Decision Engine"
__description__ = "XGBoost + SHAP based feature selection for investment modeling" 