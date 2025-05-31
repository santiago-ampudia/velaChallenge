#!/usr/bin/env python3

"""
Threshold Calibration Module for Investment Decision Analysis

This module implements the threshold calibration pipeline that:
1. Loads rule templates and test data
2. Generates candidate thresholds for each feature type
3. Evaluates individual rule performance at each threshold
4. Combines rules iteratively (by OR) until precision target is met
5. Outputs calibrated rules and comprehensive metrics

The module follows clean architecture principles with comprehensive error
handling and full traceability for explainable AI requirements.
"""

import logging
import os
import json
import csv
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict

# Import module parameters
from . import parameters as params

# Configure logging
logger = logging.getLogger(__name__)


class ThresholdCalibrator:
    """
    Calibrates thresholds for IF-THEN rule templates to achieve target precision.
    
    This class processes rule templates to find optimal thresholds for each
    feature, then combines rules iteratively until the precision target is met.
    """
    
    def __init__(self):
        """Initialize the threshold calibrator."""
        self.initial_rules = []
        self.test_data = None
        self.X_test = None
        self.y_test = None
        self.calibrated_rules = []
        
        # Statistics tracking
        self.stats = {
            'total_rules': 0,
            'rules_with_valid_thresholds': 0,
            'rules_meeting_min_support': 0,
            'final_combined_precision': 0.0,
            'final_combined_support': 0,
            'final_combined_recall': 0.0,
            'final_combined_f1': 0.0,
            'rules_used_in_combination': 0,
            'target_achieved': False,
            'combination_strategy': params.COMBINATION_STRATEGY
        }
        
        # Ensure output directories exist
        os.makedirs(params.RULES_CALIBRATED_DIR, exist_ok=True)
        os.makedirs(params.EVAL_DIR, exist_ok=True)
        
        logger.info("Initialized Threshold Calibrator")
        logger.info(f"Target precision: {params.PRECISION_TARGET:.1%}")
        logger.info(f"Minimum support: {params.MIN_SUPPORT}")
    
    def load_initial_rules(self) -> List[Dict[str, Any]]:
        """
        Load rule templates from Module 7.
        
        Returns:
            List of rule template dictionaries
            
        Raises:
            FileNotFoundError: If rules file doesn't exist
            ValueError: If rules format is invalid
        """
        logger.info(f"Loading initial rules from {params.INITIAL_RULES_PATH}")
        
        if not os.path.exists(params.INITIAL_RULES_PATH):
            raise FileNotFoundError(f"Initial rules not found: {params.INITIAL_RULES_PATH}")
        
        try:
            with open(params.INITIAL_RULES_PATH, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            
            if not isinstance(rules, list):
                raise ValueError("Expected initial rules to be a list")
            
            logger.info(f"Loaded {len(rules)} initial rule templates")
            
            self.initial_rules = rules
            self.stats['total_rules'] = len(rules)
            return rules
            
        except Exception as e:
            raise ValueError(f"Failed to load initial rules: {e}")
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data for threshold calibration.
        
        Returns:
            Test DataFrame
            
        Raises:
            FileNotFoundError: If test data doesn't exist
            ValueError: If test data format is invalid
        """
        logger.info(f"Loading test data from {params.TEST_SELECTED_PATH}")
        
        if not os.path.exists(params.TEST_SELECTED_PATH):
            raise FileNotFoundError(f"Test data not found: {params.TEST_SELECTED_PATH}")
        
        try:
            test_df = pd.read_pickle(params.TEST_SELECTED_PATH)
            
            # Validate test set composition if enabled
            if params.VALIDATE_TEST_SET:
                if len(test_df) != params.EXPECTED_TEST_SIZE:
                    logger.warning(f"Test set size {len(test_df)} != expected {params.EXPECTED_TEST_SIZE}")
                
                positive_count = (test_df['success'] == 1).sum()
                negative_count = (test_df['success'] == 0).sum()
                
                logger.info(f"Test set composition: {positive_count} positives, {negative_count} negatives")
                
                if positive_count != params.EXPECTED_POSITIVE_COUNT:
                    logger.warning(f"Positive count {positive_count} != expected {params.EXPECTED_POSITIVE_COUNT}")
                
                if negative_count != params.EXPECTED_NEGATIVE_COUNT:
                    logger.warning(f"Negative count {negative_count} != expected {params.EXPECTED_NEGATIVE_COUNT}")
            
            # Split features and target
            self.y_test = test_df['success'].astype(int)
            self.X_test = test_df.drop(columns=['success'])
            
            logger.info(f"Loaded test data: {test_df.shape} ({positive_count} positives)")
            
            self.test_data = test_df
            return test_df
            
        except Exception as e:
            raise ValueError(f"Failed to load test data: {e}")
    
    def generate_candidate_thresholds(self, rule: Dict[str, Any]) -> List[Union[float, int, bool]]:
        """
        Generate candidate thresholds for a given rule based on its feature type.
        
        Args:
            rule: Rule dictionary containing feature and feature_type
            
        Returns:
            List of candidate threshold values
        """
        feature = rule['feature']
        feature_type = rule['feature_type']
        
        if feature not in self.X_test.columns:
            logger.warning(f"Feature '{feature}' not found in test data - skipping rule")
            return []
        
        values = self.X_test[feature]
        
        # Check for missing values
        missing_count = values.isna().sum()
        if missing_count > 0:
            logger.warning(f"Feature '{feature}' has {missing_count} missing values")
            values = values.dropna()  # Remove NaN values for threshold generation
        
        if len(values) == 0:
            logger.warning(f"Feature '{feature}' has no valid values after removing NaN")
            return []
        
        # Generate candidates based on feature type
        if feature_type == 'continuous':
            if params.CONTINUOUS_THRESHOLD_STRATEGY == "unique_values":
                unique_vals = values.unique()
                candidates = sorted([v for v in unique_vals if not pd.isna(v)])
            elif params.CONTINUOUS_THRESHOLD_STRATEGY == "percentiles":
                candidates = [np.percentile(values, p) for p in params.CONTINUOUS_PERCENTILES]
            else:
                unique_vals = values.unique()
                candidates = sorted([v for v in unique_vals if not pd.isna(v)])
            
            # Limit number of candidates
            if len(candidates) > params.MAX_CANDIDATES_PER_FEATURE:
                # Sample evenly across the range
                indices = np.linspace(0, len(candidates)-1, params.MAX_CANDIDATES_PER_FEATURE, dtype=int)
                candidates = [candidates[i] for i in indices]
        
        elif feature_type == 'ordinal':
            unique_values = sorted([v for v in values.unique() if not pd.isna(v)])
            
            # Check if values are numeric (use ordinal levels) or string (use actual values)
            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique_values):
                # Numeric ordinal - use standard levels
                candidates = [v for v in params.ORDINAL_LEVELS if v in unique_values]
            else:
                # String ordinal - use actual unique values as candidates
                candidates = unique_values[:params.MAX_CANDIDATES_PER_FEATURE]
                logger.info(f"Using string values for ordinal feature '{feature}': {candidates}")
            
            if not candidates:
                # If no standard ordinal levels found, use actual unique values
                candidates = unique_values[:params.MAX_CANDIDATES_PER_FEATURE]
                logger.info(f"Using actual unique values for ordinal feature '{feature}': {candidates}")
        
        elif feature_type == 'binary':
            unique_values = [v for v in values.unique() if not pd.isna(v)]
            # For binary features, use the actual unique values present
            candidates = [v for v in params.BINARY_CANDIDATES if v in unique_values]
            
            if not candidates:
                # If standard binary values not found, use actual unique values
                candidates = unique_values
                logger.info(f"Using actual unique values for binary feature '{feature}': {candidates}")
        
        elif feature_type == 'categorical':
            unique_values = [v for v in values.unique() if not pd.isna(v)]
            candidates = sorted(unique_values)
            
            # Limit candidates for categorical features
            if len(candidates) > params.MAX_CANDIDATES_PER_FEATURE:
                candidates = candidates[:params.MAX_CANDIDATES_PER_FEATURE]
        
        else:
            logger.warning(f"Unknown feature type '{feature_type}' for feature '{feature}'")
            unique_values = [v for v in values.unique() if not pd.isna(v)]
            candidates = sorted(unique_values)
        
        if not candidates:
            logger.warning(f"No valid candidates generated for feature '{feature}' (type: {feature_type})")
        else:
            logger.debug(f"Generated {len(candidates)} candidates for {feature} ({feature_type})")
        
        return candidates
    
    def evaluate_rule_threshold(self, rule: Dict[str, Any], threshold: Union[float, int, bool]) -> Tuple[float, int, int]:
        """
        Evaluate a single rule at a specific threshold.
        
        Args:
            rule: Rule dictionary
            threshold: Threshold value to evaluate
            
        Returns:
            Tuple of (precision, true_positives, false_positives)
        """
        feature = rule['feature']
        feature_type = rule['feature_type']
        
        # Create prediction mask based on feature type and threshold
        if feature_type == 'continuous':
            mask = (self.X_test[feature] >= threshold)
        elif feature_type == 'ordinal':
            # Check if threshold is numeric or string
            if isinstance(threshold, (int, float, np.integer, np.floating)):
                # Numeric ordinal - use >= comparison
                mask = (self.X_test[feature] >= threshold)
            else:
                # String ordinal - use exact match (treat like categorical)
                mask = (self.X_test[feature] == threshold)
        elif feature_type == 'binary':
            # For binary, threshold should be True (predicting positive outcome)
            mask = (self.X_test[feature] == True)
        elif feature_type == 'categorical':
            mask = (self.X_test[feature] == threshold)
        else:
            logger.warning(f"Unknown feature type '{feature_type}'")
            return 0.0, 0, 0
        
        # Calculate true positives and false positives
        TP = int((mask & (self.y_test == 1)).sum())
        FP = int((mask & (self.y_test == 0)).sum())
        
        # Calculate precision
        if (TP + FP) == 0:
            precision = 0.0
        else:
            precision = TP / (TP + FP)
        
        return precision, TP, FP
    
    def find_best_threshold_for_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find the best threshold for a single rule.
        
        Args:
            rule: Rule dictionary to optimize
            
        Returns:
            Updated rule dictionary with best threshold information
        """
        candidates = self.generate_candidate_thresholds(rule)
        
        if not candidates:
            logger.warning(f"No candidates generated for feature '{rule['feature']}'")
            rule.update({
                'candidates': [],
                'best_threshold': None,
                'best_precision': 0.0,
                'best_support': 0
            })
            return rule
        
        # Store candidates in rule
        rule['candidates'] = candidates
        
        # Evaluate each candidate
        best_precision = 0.0
        best_threshold = None
        best_support = 0
        
        for candidate in candidates:
            precision, TP, FP = self.evaluate_rule_threshold(rule, candidate)
            
            # Check if this candidate is better
            is_better = False
            
            if TP >= params.MIN_SUPPORT:  # Must meet minimum support
                if precision > best_precision:
                    is_better = True
                elif precision == best_precision:
                    # Tie-breaking logic
                    if params.TIE_BREAKING_STRATEGY == "highest_support" and TP > best_support:
                        is_better = True
                    elif params.TIE_BREAKING_STRATEGY == "lowest_threshold" and candidate < best_threshold:
                        is_better = True
                    elif params.TIE_BREAKING_STRATEGY == "highest_threshold" and candidate > best_threshold:
                        is_better = True
            
            if is_better:
                best_precision = precision
                best_threshold = candidate
                best_support = TP
        
        # Update rule with best threshold information
        rule.update({
            'best_threshold': best_threshold,
            'best_precision': best_precision,
            'best_support': best_support
        })
        
        if params.LOG_THRESHOLD_EVALUATIONS:
            logger.debug(f"Best threshold for {rule['feature']}: {best_threshold} "
                        f"(precision: {best_precision:.3f}, support: {best_support})")
        
        return rule
    
    def evaluate_all_rules(self) -> List[Dict[str, Any]]:
        """
        Evaluate all rules to find their best thresholds.
        
        Returns:
            List of rules with best threshold information
        """
        logger.info("Evaluating thresholds for all rules...")
        
        evaluated_rules = []
        valid_rules = 0
        
        for i, rule in enumerate(self.initial_rules):
            logger.debug(f"Evaluating rule {i+1}/{len(self.initial_rules)}: {rule['feature']}")
            
            evaluated_rule = self.find_best_threshold_for_rule(rule.copy())
            
            if evaluated_rule['best_threshold'] is not None:
                valid_rules += 1
                if evaluated_rule['best_support'] >= params.MIN_SUPPORT:
                    self.stats['rules_meeting_min_support'] += 1
            
            evaluated_rules.append(evaluated_rule)
        
        self.stats['rules_with_valid_thresholds'] = valid_rules
        
        logger.info(f"Evaluation complete: {valid_rules}/{len(self.initial_rules)} rules have valid thresholds")
        logger.info(f"Rules meeting minimum support: {self.stats['rules_meeting_min_support']}")
        
        return evaluated_rules
    
    def sort_rules_for_combination(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort rules based on the configured sorting strategy.
        
        Args:
            rules: List of evaluated rules
            
        Returns:
            Sorted list of rules
        """
        if params.RULE_SORTING_STRATEGY == "weight_desc":
            # Sort by weight descending
            sorted_rules = sorted(rules, key=lambda r: r['weight'], reverse=True)
        
        elif params.RULE_SORTING_STRATEGY == "precision_desc":
            # Sort by best precision descending
            sorted_rules = sorted(rules, key=lambda r: r.get('best_precision', 0), reverse=True)
        
        elif params.RULE_SORTING_STRATEGY == "combined":
            # Combined scoring: weight + precision
            def combined_score(rule):
                weight_score = rule['weight'] / max(r['weight'] for r in rules)
                precision_score = rule.get('best_precision', 0)
                return (params.WEIGHT_IMPORTANCE * weight_score + 
                       params.PRECISION_IMPORTANCE * precision_score)
            
            sorted_rules = sorted(rules, key=combined_score, reverse=True)
        
        else:
            logger.warning(f"Unknown sorting strategy: {params.RULE_SORTING_STRATEGY}")
            sorted_rules = rules
        
        logger.info(f"Sorted rules using strategy: {params.RULE_SORTING_STRATEGY}")
        
        return sorted_rules
    
    def combine_rules_iteratively(self, evaluated_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine rules using advanced strategies for better precision-recall balance.
        
        Args:
            evaluated_rules: List of rules with best thresholds
            
        Returns:
            List of calibrated rules used in final combination
        """
        logger.info("Starting advanced rule combination...")
        
        # Filter rules with valid thresholds
        valid_rules = [r for r in evaluated_rules if r['best_threshold'] is not None]
        
        if not valid_rules:
            logger.error("No rules with valid thresholds found")
            return []
        
        # Further filter by minimum individual precision if configured
        if params.MIN_INDIVIDUAL_PRECISION > 0:
            precision_filtered_rules = [r for r in valid_rules if r['best_precision'] >= params.MIN_INDIVIDUAL_PRECISION]
            logger.info(f"Filtered to {len(precision_filtered_rules)} rules meeting minimum precision {params.MIN_INDIVIDUAL_PRECISION:.1%}")
            if precision_filtered_rules:
                valid_rules = precision_filtered_rules
            else:
                logger.warning(f"No rules meet minimum precision {params.MIN_INDIVIDUAL_PRECISION:.1%}, using all valid rules")
        
        # Sort rules for combination
        sorted_rules = self.sort_rules_for_combination(valid_rules)
        
        # Choose combination strategy
        if params.COMBINATION_STRATEGY == "weighted_scoring":
            return self.weighted_scoring_combination(sorted_rules)
        elif params.COMBINATION_STRATEGY == "hierarchical":
            return self.hierarchical_combination(sorted_rules)
        elif params.COMBINATION_STRATEGY == "ensemble":
            return self.ensemble_combination(sorted_rules)
        else:  # simple_or (original approach)
            return self.simple_or_combination(sorted_rules)
    
    def weighted_scoring_combination(self, sorted_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine rules using weighted scoring with automatic threshold tuning.
        Each rule contributes a score based on its precision and weight.
        """
        logger.info(f"Using weighted scoring combination with automatic threshold tuning")
        
        calibrated_rules = []
        best_f1 = 0.0
        best_combination = None
        
        # Try different combinations of top rules
        for max_rules in [3, 5, 8, 10, 15, min(20, len(sorted_rules))]:
            current_rules = sorted_rules[:max_rules]
            
            # Calculate weighted scores for each test case
            weighted_scores = np.zeros(len(self.y_test))
            
            for rule in current_rules:
                rule_mask = self.get_rule_mask(rule)
                
                # Calculate rule contribution based on scoring method
                if params.SCORING_METHOD == "confidence_weighted":
                    # Weight by precision * causal weight
                    contribution = rule['best_precision'] * (rule['weight'] / 100)
                elif params.SCORING_METHOD == "precision_weighted":
                    # Weight purely by precision
                    contribution = rule['best_precision']
                else:  # simple_sum
                    # Equal weight for all rules
                    contribution = 0.05  # Base contribution per rule
                
                weighted_scores += rule_mask.astype(float) * contribution
            
            # Try different thresholds to find the best one
            score_values = sorted(set(weighted_scores[weighted_scores > 0]))
            
            # Add some fractional thresholds between the main score values
            extended_thresholds = []
            for i, score in enumerate(score_values):
                extended_thresholds.append(score)
                if i < len(score_values) - 1:
                    # Add intermediate thresholds
                    next_score = score_values[i + 1]
                    extended_thresholds.append((score + next_score) / 2)
                    extended_thresholds.append(score * 1.1)
            
            # Also try some absolute thresholds
            extended_thresholds.extend([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])
            extended_thresholds = sorted(set(extended_thresholds))
            
            for threshold in extended_thresholds:
                if threshold <= 0:
                    continue
                    
                # Apply threshold to get predictions
                predictions = (weighted_scores >= threshold)
                
                # Calculate metrics
                TP = int((predictions & (self.y_test == 1)).sum())
                FP = int((predictions & (self.y_test == 0)).sum())
                FN = int((~predictions & (self.y_test == 1)).sum())
                
                if (TP + FP) > 0 and TP > 0:
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Check if this combination meets our criteria and improves F1
                    if precision >= params.MIN_PRECISION and f1 > best_f1:
                        best_f1 = f1
                        best_combination = {
                            'rules': current_rules.copy(),
                            'threshold': threshold,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'support': TP,
                            'predictions': predictions.copy()
                        }
                        
                        logger.info(f"  New best: {max_rules} rules, threshold={threshold:.3f}, "
                                   f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={TP}")
        
        if best_combination:
            # Create calibrated rules from best combination
            calibrated_rules = []
            for i, rule in enumerate(best_combination['rules']):
                calibrated_rule = {
                    'feature': rule['feature'],
                    'feature_type': rule['feature_type'],
                    'outcome': rule['outcome'],
                    'weight': rule['weight'],
                    'threshold': rule['best_threshold'],
                    'rule_precision': rule['best_precision'],
                    'rule_support': rule['best_support'],
                    'combined_precision': best_combination['precision'],
                    'combined_support': best_combination['support'],
                    'combined_recall': best_combination['recall'],
                    'combined_f1': best_combination['f1'],
                    'scoring_threshold': best_combination['threshold']
                }
                
                if params.INCLUDE_DETAILED_METADATA:
                    calibrated_rule.update({
                        'rule_index': i,
                        'combined_false_positives': best_combination['support'] / best_combination['precision'] - best_combination['support'],
                        'rule_id': rule.get('rule_id', f"{rule['feature']}_{rule['outcome']}_{rule['weight']}")
                    })
                
                calibrated_rules.append(calibrated_rule)
            
            # Update final statistics
            self.stats['final_combined_precision'] = best_combination['precision']
            self.stats['final_combined_support'] = best_combination['support']
            self.stats['final_combined_recall'] = best_combination['recall']
            self.stats['final_combined_f1'] = best_combination['f1']
            self.stats['rules_used_in_combination'] = len(calibrated_rules)
            self.stats['target_achieved'] = best_combination['precision'] >= params.PRECISION_TARGET
            
            logger.info(f"🎯 Best weighted combination: {len(calibrated_rules)} rules")
            logger.info(f"   Scoring threshold: {best_combination['threshold']:.3f}")
            logger.info(f"   Final metrics: P={best_combination['precision']:.3f}, "
                       f"R={best_combination['recall']:.3f}, F1={best_combination['f1']:.3f}")
            logger.info(f"   Support: {best_combination['support']}/{(self.y_test == 1).sum()}")
        else:
            logger.warning("No weighted combination achieved minimum precision requirement")
        
        self.calibrated_rules = calibrated_rules
        return calibrated_rules
    
    def hierarchical_combination(self, sorted_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine rules using hierarchical tiers with different precision requirements.
        """
        logger.info("Using hierarchical combination with precision tiers")
        
        all_selected_rules = []
        cumulative_mask = np.zeros_like(self.y_test, dtype=bool)
        
        for tier_idx, tier in enumerate(params.HIERARCHICAL_TIERS):
            logger.info(f"Processing Tier {tier_idx + 1}: min_precision={tier['min_precision']:.1%}, max_rules={tier['max_rules']}")
            
            # Filter rules for this tier
            tier_rules = [r for r in sorted_rules if r['best_precision'] >= tier['min_precision']]
            tier_rules = tier_rules[:tier['max_rules']]
            
            if not tier_rules:
                logger.info(f"  No rules qualify for Tier {tier_idx + 1}")
                continue
            
            # Add tier rules and evaluate combined performance
            for rule in tier_rules:
                rule_mask = self.get_rule_mask(rule)
                new_cumulative_mask = cumulative_mask | rule_mask
                
                # Calculate metrics with new rule added
                TP = int((new_cumulative_mask & (self.y_test == 1)).sum())
                FP = int((new_cumulative_mask & (self.y_test == 0)).sum())
                
                if (TP + FP) > 0:
                    precision = TP / (TP + FP)
                    recall = TP / (self.y_test == 1).sum()
                    
                    # Only add rule if it doesn't hurt precision too much
                    if precision >= params.MIN_PRECISION:
                        cumulative_mask = new_cumulative_mask
                        all_selected_rules.append(rule)
                        logger.info(f"    Added {rule['feature']}: P={precision:.3f}, R={recall:.3f}, Support={TP}")
                    else:
                        logger.info(f"    Rejected {rule['feature']}: would reduce precision to {precision:.3f}")
            
            # Check if we've achieved target recall
            current_recall = cumulative_mask.sum() / (self.y_test == 1).sum()
            if current_recall >= params.TARGET_RECALL:
                logger.info(f"Target recall {params.TARGET_RECALL:.1%} achieved, stopping at Tier {tier_idx + 1}")
                break
        
        # Create final calibrated rules
        calibrated_rules = self.create_calibrated_rules_from_combination(all_selected_rules, cumulative_mask)
        
        self.calibrated_rules = calibrated_rules
        return calibrated_rules
    
    def ensemble_combination(self, sorted_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine rules using ensemble of different strategies.
        """
        logger.info("Using ensemble combination with multiple strategies")
        
        ensemble_results = []
        
        # Strategy 1: High Precision (top precision rules)
        high_precision_rules = sorted(sorted_rules, key=lambda r: r['best_precision'], reverse=True)[:5]
        hp_mask = self.get_combined_mask(high_precision_rules)
        ensemble_results.append(('high_precision', hp_mask))
        
        # Strategy 2: High Recall (rules that catch different positives)
        hr_mask = self.get_high_recall_combination(sorted_rules)
        ensemble_results.append(('high_recall', hr_mask))
        
        # Strategy 3: Balanced (weighted by F1 contribution)
        balanced_mask = self.get_balanced_combination(sorted_rules)
        ensemble_results.append(('balanced', balanced_mask))
        
        # Combine ensemble predictions
        if params.ENSEMBLE_VOTING == "majority":
            final_mask = sum(mask for _, mask in ensemble_results) >= 2
        elif params.ENSEMBLE_VOTING == "unanimous":
            final_mask = all(mask for _, mask in ensemble_results)
        else:  # weighted
            final_mask = self.get_weighted_ensemble_mask(ensemble_results)
        
        # Find which rules contribute to final result and create calibrated rules
        contributing_rules = self.find_contributing_rules(sorted_rules, final_mask)
        calibrated_rules = self.create_calibrated_rules_from_combination(contributing_rules, final_mask)
        
        self.calibrated_rules = calibrated_rules
        return calibrated_rules
    
    def get_rule_mask(self, rule: Dict[str, Any]) -> np.ndarray:
        """Helper method to get prediction mask for a single rule."""
        threshold = rule['best_threshold']
        feature = rule['feature']
        feature_type = rule['feature_type']
        
        if feature_type == 'continuous':
            return (self.X_test[feature] >= threshold)
        elif feature_type == 'ordinal':
            if isinstance(threshold, (int, float, np.integer, np.floating)):
                return (self.X_test[feature] >= threshold)
            else:
                return (self.X_test[feature] == threshold)
        elif feature_type == 'binary':
            return (self.X_test[feature] == True)
        elif feature_type == 'categorical':
            return (self.X_test[feature] == threshold)
        else:
            return np.zeros_like(self.y_test, dtype=bool)
    
    def get_combined_mask(self, rules: List[Dict[str, Any]]) -> np.ndarray:
        """Helper method to get combined OR mask for multiple rules."""
        if not rules:
            return np.zeros_like(self.y_test, dtype=bool)
        
        combined_mask = np.zeros_like(self.y_test, dtype=bool)
        for rule in rules:
            combined_mask |= self.get_rule_mask(rule)
        return combined_mask
    
    def get_high_recall_combination(self, sorted_rules: List[Dict[str, Any]]) -> np.ndarray:
        """Get combination optimized for recall."""
        selected_rules = []
        covered_positives = np.zeros_like(self.y_test, dtype=bool)
        
        for rule in sorted_rules[:15]:  # Consider top 15 rules
            rule_mask = self.get_rule_mask(rule)
            new_positives = rule_mask & (self.y_test == 1) & ~covered_positives
            
            if new_positives.sum() > 0:  # Rule adds new positive coverage
                selected_rules.append(rule)
                covered_positives |= rule_mask & (self.y_test == 1)
                
                # Stop if we've covered most positives
                recall = covered_positives.sum() / (self.y_test == 1).sum()
                if recall >= params.TARGET_RECALL:
                    break
        
        return self.get_combined_mask(selected_rules)
    
    def get_balanced_combination(self, sorted_rules: List[Dict[str, Any]]) -> np.ndarray:
        """Get combination optimized for F1 score."""
        best_f1 = 0
        best_mask = np.zeros_like(self.y_test, dtype=bool)
        
        for num_rules in range(1, min(16, len(sorted_rules) + 1)):
            mask = self.get_combined_mask(sorted_rules[:num_rules])
            
            TP = int((mask & (self.y_test == 1)).sum())
            FP = int((mask & (self.y_test == 0)).sum())
            FN = int((~mask & (self.y_test == 1)).sum())
            
            if TP + FP > 0:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1 and precision >= params.MIN_PRECISION:
                    best_f1 = f1
                    best_mask = mask
        
        return best_mask
    
    def create_calibrated_rules_from_combination(self, rules: List[Dict[str, Any]], final_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Create calibrated rules list from final combination."""
        TP = int((final_mask & (self.y_test == 1)).sum())
        FP = int((final_mask & (self.y_test == 0)).sum())
        FN = int((~final_mask & (self.y_test == 1)).sum())
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        calibrated_rules = []
        for i, rule in enumerate(rules):
            calibrated_rule = {
                'feature': rule['feature'],
                'feature_type': rule['feature_type'],
                'outcome': rule['outcome'],
                'weight': rule['weight'],
                'threshold': rule['best_threshold'],
                'rule_precision': rule['best_precision'],
                'rule_support': rule['best_support'],
                'combined_precision': precision,
                'combined_support': TP,
                'combined_recall': recall,
                'combined_f1': f1
            }
            
            if params.INCLUDE_DETAILED_METADATA:
                calibrated_rule.update({
                    'rule_index': i,
                    'combined_false_positives': FP,
                    'rule_id': rule.get('rule_id', f"{rule['feature']}_{rule['outcome']}_{rule['weight']}")
                })
            
            calibrated_rules.append(calibrated_rule)
        
        # Update statistics
        self.stats['final_combined_precision'] = precision
        self.stats['final_combined_support'] = TP
        self.stats['final_combined_recall'] = recall
        self.stats['final_combined_f1'] = f1
        self.stats['rules_used_in_combination'] = len(calibrated_rules)
        self.stats['target_achieved'] = precision >= params.PRECISION_TARGET
        
        return calibrated_rules
    
    def save_calibrated_rules(self, calibrated_rules: List[Dict[str, Any]]) -> None:
        """
        Save calibrated rules to JSON file.
        
        Args:
            calibrated_rules: List of calibrated rule dictionaries
        """
        logger.info(f"Saving {len(calibrated_rules)} calibrated rules to {params.CALIBRATED_RULES_PATH}")
        
        try:
            # Convert numpy types to Python native types for JSON serialization
            json_compatible_rules = []
            for rule in calibrated_rules:
                json_rule = {}
                for key, value in rule.items():
                    if isinstance(value, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                        json_rule[key] = int(value)
                    elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
                        json_rule[key] = float(value)
                    elif isinstance(value, np.bool_):
                        json_rule[key] = bool(value)
                    else:
                        json_rule[key] = value
                json_compatible_rules.append(json_rule)
            
            with open(params.CALIBRATED_RULES_PATH, 'w', encoding='utf-8') as f:
                json.dump(json_compatible_rules, f, indent=params.JSON_INDENT, 
                         ensure_ascii=params.JSON_ENSURE_ASCII)
            
            logger.info("Calibrated rules saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save calibrated rules: {e}")
            raise
    
    def save_threshold_metrics(self, calibrated_rules: List[Dict[str, Any]]) -> None:
        """
        Save threshold metrics to CSV file.
        
        Args:
            calibrated_rules: List of calibrated rule dictionaries
        """
        logger.info(f"Saving threshold metrics to {params.THRESHOLD_METRICS_PATH}")
        
        try:
            with open(params.THRESHOLD_METRICS_PATH, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                header = [
                    'feature', 'feature_type', 'outcome', 'weight',
                    'threshold', 'rule_precision', 'rule_support',
                    'combined_precision', 'combined_support'
                ]
                writer.writerow(header)
                
                # Write data rows
                for rule in calibrated_rules:
                    row = [
                        rule['feature'],
                        rule['feature_type'],
                        rule['outcome'],
                        rule['weight'],
                        rule['threshold'],
                        f"{rule['rule_precision']:.{params.CSV_FLOAT_PRECISION}f}",
                        rule['rule_support'],
                        f"{rule['combined_precision']:.{params.CSV_FLOAT_PRECISION}f}",
                        rule['combined_support']
                    ]
                    writer.writerow(row)
            
            logger.info("Threshold metrics CSV saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save threshold metrics: {e}")
            raise
    
    def save_calibration_summary(self) -> None:
        """
        Save calibration summary with statistics and metadata.
        """
        if not params.SAVE_INTERMEDIATE_RESULTS:
            return
        
        summary = {
            'calibration_summary': self.stats,
            'configuration': {
                'precision_target': params.PRECISION_TARGET,
                'min_support': params.MIN_SUPPORT,
                'max_rules_to_combine': params.MAX_RULES_TO_COMBINE,
                'rule_sorting_strategy': params.RULE_SORTING_STRATEGY,
                'stop_at_target': params.STOP_AT_TARGET
            },
            'test_set_info': {
                'size': len(self.test_data) if self.test_data is not None else 0,
                'positive_count': int(self.y_test.sum()) if self.y_test is not None else 0,
                'negative_count': int((self.y_test == 0).sum()) if self.y_test is not None else 0
            }
        }
        
        try:
            with open(params.CALIBRATION_SUMMARY_PATH, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=params.JSON_INDENT, ensure_ascii=params.JSON_ENSURE_ASCII)
            
            logger.info(f"Calibration summary saved to {params.CALIBRATION_SUMMARY_PATH}")
            
        except Exception as e:
            logger.warning(f"Failed to save calibration summary: {e}")
    
    def log_final_statistics(self) -> None:
        """
        Log comprehensive statistics about the calibration process.
        """
        logger.info("=" * 60)
        logger.info("THRESHOLD CALIBRATION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Combination strategy: {self.stats['combination_strategy']}")
        logger.info(f"Total rules processed: {self.stats['total_rules']}")
        logger.info(f"Rules with valid thresholds: {self.stats['rules_with_valid_thresholds']}")
        logger.info(f"Rules meeting minimum support: {self.stats['rules_meeting_min_support']}")
        logger.info(f"Rules used in final combination: {self.stats['rules_used_in_combination']}")
        logger.info(f"Final combined precision: {self.stats['final_combined_precision']:.3f}")
        logger.info(f"Final combined recall: {self.stats['final_combined_recall']:.3f}")
        logger.info(f"Final combined F1-score: {self.stats['final_combined_f1']:.3f}")
        logger.info(f"Final combined support: {self.stats['final_combined_support']}")
        logger.info(f"Target precision ({params.PRECISION_TARGET:.1%}): {'✅ ACHIEVED' if self.stats['target_achieved'] else '❌ NOT ACHIEVED'}")
        
        # Additional metrics
        total_positives = (self.y_test == 1).sum()
        recall_percentage = (self.stats['final_combined_support'] / total_positives) * 100 if total_positives > 0 else 0
        logger.info(f"Recall coverage: {self.stats['final_combined_support']}/{total_positives} ({recall_percentage:.1f}% of positive cases)")
        
        if self.calibrated_rules:
            logger.info("Top 5 rules by combination order:")
            for i, rule in enumerate(self.calibrated_rules[:5]):
                threshold_str = f"{rule['threshold']:.3f}" if isinstance(rule['threshold'], (int, float, np.number)) else str(rule['threshold'])
                logger.info(f"  {i+1}. {rule['feature']} (threshold: {threshold_str}, "
                           f"precision: {rule['rule_precision']:.3f})")
        
        logger.info("Threshold calibration completed!")
    
    def calibrate_thresholds(self) -> List[Dict[str, Any]]:
        """
        Main pipeline method to calibrate thresholds for all rules.
        
        Returns:
            List of calibrated rule dictionaries
            
        Raises:
            Exception: If any step in the pipeline fails critically
        """
        logger.info("Starting threshold calibration pipeline")
        
        # Step 1: Load inputs
        self.load_initial_rules()
        self.load_test_data()
        
        # Step 2: Evaluate all rules to find best thresholds
        evaluated_rules = self.evaluate_all_rules()
        
        # Step 3: Combine rules iteratively until target is met
        calibrated_rules = self.combine_rules_iteratively(evaluated_rules)
        
        # Step 4: Save outputs
        if calibrated_rules:
            self.save_calibrated_rules(calibrated_rules)
            self.save_threshold_metrics(calibrated_rules)
        else:
            logger.warning("No calibrated rules to save")
        
        self.save_calibration_summary()
        
        # Step 5: Log comprehensive statistics
        self.log_final_statistics()
        
        logger.info("Threshold calibration pipeline completed successfully!")
        
        return calibrated_rules
    
    def get_weighted_ensemble_mask(self, ensemble_results: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """Get weighted ensemble mask based on strategy performance."""
        # For simplicity, use equal weights for now
        # In a more sophisticated implementation, you could weight based on validation performance
        weights = [1/3, 1/3, 1/3]  # Equal weights for the three strategies
        
        weighted_sum = np.zeros_like(self.y_test, dtype=float)
        for (strategy, mask), weight in zip(ensemble_results, weights):
            weighted_sum += mask.astype(float) * weight
        
        return weighted_sum >= 0.5  # Majority threshold
    
    def find_contributing_rules(self, sorted_rules: List[Dict[str, Any]], final_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Find which rules contribute most to the final prediction mask."""
        contributing_rules = []
        remaining_mask = final_mask.copy()
        
        # Greedily select rules that contribute most to the final mask
        for rule in sorted_rules:
            rule_mask = self.get_rule_mask(rule)
            contribution = (rule_mask & remaining_mask).sum()
            
            if contribution > 0:
                contributing_rules.append(rule)
                remaining_mask = remaining_mask & ~rule_mask  # Remove covered cases
                
                # Stop when we've covered most of the final mask
                if remaining_mask.sum() < 0.1 * final_mask.sum():
                    break
        
        return contributing_rules
    
    def simple_or_combination(self, sorted_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Original simple OR combination strategy for backward compatibility.
        """
        logger.info("Using simple OR combination (original strategy)")
        
        # Determine maximum rules to combine
        max_rules = params.MAX_RULES_TO_COMBINE or len(sorted_rules)
        
        # Initialize combination tracking
        combined_mask = np.zeros_like(self.y_test, dtype=bool)
        calibrated_rules = []
        
        logger.info(f"Combining up to {max_rules} rules to achieve {params.PRECISION_TARGET:.1%} precision")
        
        for i, rule in enumerate(sorted_rules[:max_rules]):
            threshold = rule['best_threshold']
            feature = rule['feature']
            feature_type = rule['feature_type']
            
            if threshold is None:
                continue
            
            # Create mask for this rule
            rule_mask = self.get_rule_mask(rule)
            
            # Update combined mask (OR operation)
            combined_mask = combined_mask | rule_mask
            
            # Calculate combined metrics
            TP_combined = int((combined_mask & (self.y_test == 1)).sum())
            FP_combined = int((combined_mask & (self.y_test == 0)).sum())
            FN_combined = int((~combined_mask & (self.y_test == 1)).sum())
            
            if (TP_combined + FP_combined) > 0:
                precision_combined = TP_combined / (TP_combined + FP_combined)
                recall_combined = TP_combined / (TP_combined + FN_combined) if (TP_combined + FN_combined) > 0 else 0
                f1_combined = 2 * precision_combined * recall_combined / (precision_combined + recall_combined) if (precision_combined + recall_combined) > 0 else 0
            else:
                precision_combined = 0.0
                recall_combined = 0.0
                f1_combined = 0.0
            
            # Create calibrated rule entry
            calibrated_rule = {
                'feature': feature,
                'feature_type': feature_type,
                'outcome': rule['outcome'],
                'weight': rule['weight'],
                'threshold': threshold,
                'rule_precision': rule['best_precision'],
                'rule_support': rule['best_support'],
                'combined_precision': precision_combined,
                'combined_support': TP_combined,
                'combined_recall': recall_combined,
                'combined_f1': f1_combined
            }
            
            # Add detailed metadata if enabled
            if params.INCLUDE_DETAILED_METADATA:
                calibrated_rule.update({
                    'rule_index': i,
                    'combined_false_positives': FP_combined,
                    'rule_id': rule.get('rule_id', f"{feature}_{rule['outcome']}_{rule['weight']}")
                })
            
            calibrated_rules.append(calibrated_rule)
            
            if params.LOG_RULE_COMBINATIONS:
                threshold_str = f"{threshold:.3f}" if isinstance(threshold, (int, float, np.number)) else str(threshold)
                logger.info(f"Added rule {i+1}: {feature} (threshold: {threshold_str}, "
                           f"combined precision: {precision_combined:.3f}, support: {TP_combined})")
            
            # Check if target precision is reached
            if precision_combined >= params.PRECISION_TARGET:
                logger.info(f"🎯 Target precision {params.PRECISION_TARGET:.1%} achieved after {len(calibrated_rules)} rules!")
                self.stats['target_achieved'] = True
                if params.STOP_AT_TARGET:
                    break
        
        # Update final statistics
        if calibrated_rules:
            final_rule = calibrated_rules[-1]
            self.stats['final_combined_precision'] = final_rule['combined_precision']
            self.stats['final_combined_support'] = final_rule['combined_support']
            self.stats['final_combined_recall'] = final_rule['combined_recall']
            self.stats['final_combined_f1'] = final_rule['combined_f1']
            self.stats['rules_used_in_combination'] = len(calibrated_rules)
        
        logger.info(f"Rule combination complete: {len(calibrated_rules)} rules used")
        logger.info(f"Final combined precision: {self.stats['final_combined_precision']:.3f}")
        logger.info(f"Final combined support: {self.stats['final_combined_support']}")
        
        self.calibrated_rules = calibrated_rules
        return calibrated_rules


def run_threshold_calibration() -> List[Dict[str, Any]]:
    """
    Main entry point for Module 8: Threshold Calibration.
    
    Returns:
        List of calibrated rule dictionaries
    """
    # Initialize the calibrator
    calibrator = ThresholdCalibrator()
    
    # Calibrate thresholds
    calibrated_rules = calibrator.calibrate_thresholds()
    
    return calibrated_rules


if __name__ == "__main__":
    # For testing purposes
    run_threshold_calibration() 