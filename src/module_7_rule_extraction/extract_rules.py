#!/usr/bin/env python3

"""
Rule Extraction Module for Investment Decision Analysis

This module implements the rule extraction pipeline that:
1. Loads the weighted causal graph from Module 6
2. Extracts the top weighted edges as potential rules
3. Determines feature types (continuous, ordinal, binary, categorical)
4. Creates IF-THEN rule templates with placeholders
5. Serializes rule templates for Module 8 calibration

The module follows clean architecture principles with comprehensive error
handling and full traceability for explainable AI requirements.
"""

import logging
import os
import json
import pickle
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# Import module parameters
from . import parameters as params

# Configure logging
logger = logging.getLogger(__name__)


class RuleExtractor:
    """
    Extracts IF-THEN rule templates from weighted causal graphs.
    
    This class processes the causal graph to identify the strongest
    feature → outcome relationships and converts them into human-readable
    rule templates with placeholders for thresholds and confidence levels.
    """
    
    def __init__(self):
        """Initialize the rule extractor."""
        self.causal_graph = None
        self.selected_features = []
        self.train_data = None
        self.feature_types = {}
        self.edge_weights = []
        self.initial_rules = []
        
        # Statistics tracking
        self.stats = {
            'total_edges': 0,
            'valid_edges': 0,
            'rules_generated': 0,
            'rules_by_type': defaultdict(int),
            'rules_by_outcome': defaultdict(int),
            'skipped_features': []
        }
        
        # Ensure output directory exists
        os.makedirs(params.OUTPUT_DIR, exist_ok=True)
        
        logger.info("Initialized Rule Extractor")
        logger.info(f"Output directory: {params.OUTPUT_DIR}")
    
    def load_causal_graph(self) -> nx.DiGraph:
        """
        Load the weighted causal graph from Module 6.
        
        Returns:
            NetworkX DiGraph with weighted edges
            
        Raises:
            FileNotFoundError: If the graph file doesn't exist
            ValueError: If the graph format is invalid
        """
        logger.info(f"Loading causal graph from {params.CAUSAL_GRAPH_PATH}")
        
        if not os.path.exists(params.CAUSAL_GRAPH_PATH):
            raise FileNotFoundError(f"Causal graph not found: {params.CAUSAL_GRAPH_PATH}")
        
        try:
            with open(params.CAUSAL_GRAPH_PATH, 'rb') as f:
                graph = pickle.load(f)
            
            if not isinstance(graph, nx.DiGraph):
                raise ValueError("Loaded object is not a NetworkX DiGraph")
            
            logger.info(f"Loaded causal graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            
            self.causal_graph = graph
            return graph
            
        except Exception as e:
            raise ValueError(f"Failed to load causal graph: {e}")
    
    def load_selected_features(self) -> List[str]:
        """
        Load the list of selected features from Module 4.
        
        Returns:
            List of selected feature names
            
        Raises:
            FileNotFoundError: If the features file doesn't exist
            ValueError: If the features format is invalid
        """
        logger.info(f"Loading selected features from {params.SELECTED_FEATURES_PATH}")
        
        if not os.path.exists(params.SELECTED_FEATURES_PATH):
            raise FileNotFoundError(f"Selected features not found: {params.SELECTED_FEATURES_PATH}")
        
        try:
            with open(params.SELECTED_FEATURES_PATH, 'r', encoding='utf-8') as f:
                features = json.load(f)
            
            if not isinstance(features, list):
                raise ValueError("Expected selected features to be a list")
            
            logger.info(f"Loaded {len(features)} selected features")
            
            self.selected_features = features
            return features
            
        except Exception as e:
            raise ValueError(f"Failed to load selected features: {e}")
    
    def load_training_data(self) -> Optional[pd.DataFrame]:
        """
        Load training data for feature type detection.
        
        Returns:
            Training DataFrame or None if not available
        """
        if not os.path.exists(params.TRAIN_PREPARED_PATH):
            logger.warning(f"Training data not found: {params.TRAIN_PREPARED_PATH}")
            logger.warning("Will use manual feature type overrides only")
            return None
        
        try:
            with open(params.TRAIN_PREPARED_PATH, 'rb') as f:
                df = pickle.load(f)
            
            logger.info(f"Loaded training data with shape: {df.shape}")
            
            self.train_data = df
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load training data: {e}")
            return None
    
    def determine_feature_type(self, feature_name: str) -> str:
        """
        Determine the type of a feature (continuous, ordinal, binary, categorical).
        
        Args:
            feature_name: Name of the feature to analyze
            
        Returns:
            Feature type string
        """
        # First check manual overrides
        if feature_name in params.FEATURE_TYPE_OVERRIDES:
            feature_type = params.FEATURE_TYPE_OVERRIDES[feature_name]
            if params.LOG_FEATURE_TYPE_DETECTION:
                logger.debug(f"Feature '{feature_name}' type: {feature_type} (manual override)")
            return feature_type
        
        # If no training data available, default to continuous for numeric-sounding names
        if self.train_data is None:
            # Simple heuristics based on feature names
            if any(keyword in feature_name.lower() for keyword in ['number', 'count', 'yoe', 'roles']):
                feature_type = 'continuous'
            elif any(keyword in feature_name.lower() for keyword in ['experience', 'company', 'leadership']):
                feature_type = 'binary'
            else:
                feature_type = 'ordinal'
            
            if params.LOG_FEATURE_TYPE_DETECTION:
                logger.debug(f"Feature '{feature_name}' type: {feature_type} (heuristic)")
            return feature_type
        
        # Analyze the feature data if available
        if feature_name not in self.train_data.columns:
            logger.warning(f"Feature '{feature_name}' not found in training data, defaulting to continuous")
            return 'continuous'
        
        feature_data = self.train_data[feature_name]
        dtype_str = str(feature_data.dtype)
        unique_values = feature_data.nunique()
        
        # Determine type based on data characteristics
        if dtype_str in params.BINARY_DTYPES:
            feature_type = 'binary'
        elif dtype_str in params.CATEGORICAL_DTYPES:
            feature_type = 'categorical'
        elif unique_values <= 2:
            feature_type = 'binary'
        elif unique_values <= params.MAX_UNIQUE_FOR_ORDINAL:
            feature_type = 'ordinal'
        elif unique_values >= params.MIN_UNIQUE_FOR_CONTINUOUS:
            feature_type = 'continuous'
        else:
            feature_type = 'ordinal'  # Default fallback
        
        if params.LOG_FEATURE_TYPE_DETECTION:
            logger.debug(f"Feature '{feature_name}' type: {feature_type} (dtype: {dtype_str}, unique: {unique_values})")
        
        return feature_type
    
    def extract_edge_weights(self) -> List[Tuple[str, str, int]]:
        """
        Extract and sort edge weights from the causal graph.
        
        Returns:
            List of (feature, outcome, weight) tuples sorted by weight descending
        """
        logger.info("Extracting edge weights from causal graph")
        
        edge_weights = []
        
        for feature, outcome in self.causal_graph.edges():
            edge_data = self.causal_graph[feature][outcome]
            weight = edge_data.get('weight', 0)
            
            # Filter by minimum weight
            if weight >= params.MIN_EDGE_WEIGHT:
                edge_weights.append((feature, outcome, weight))
        
        # Sort by weight descending
        edge_weights.sort(key=lambda x: x[2], reverse=True)
        
        # Apply prioritization if needed
        if params.PRIORITIZE_SUCCESS_RULES:
            # Separate success and failure rules
            success_edges = [(f, o, w) for f, o, w in edge_weights if o == 'success']
            failure_edges = [(f, o, w) for f, o, w in edge_weights if o == 'failure']
            
            # Combine with success rules first
            edge_weights = success_edges + failure_edges
        
        self.stats['total_edges'] = len(edge_weights)
        
        logger.info(f"Extracted {len(edge_weights)} valid edges (weight >= {params.MIN_EDGE_WEIGHT})")
        
        self.edge_weights = edge_weights
        return edge_weights
    
    def create_rule_template(self, feature: str, outcome: str, weight: int) -> Dict[str, Any]:
        """
        Create an IF-THEN rule template for a given edge.
        
        Args:
            feature: Feature name
            outcome: Outcome name (success/failure)
            weight: Edge weight
            
        Returns:
            Rule template dictionary
        """
        # Determine feature type
        feature_type = self.determine_feature_type(feature)
        
        # Store feature type for statistics
        self.feature_types[feature] = feature_type
        
        # Get mechanisms if available
        mechanisms = []
        if params.INCLUDE_MECHANISMS:
            edge_data = self.causal_graph[feature][outcome]
            raw_mechanisms = edge_data.get('mechanisms', [])
            mechanisms = raw_mechanisms[:params.MAX_MECHANISMS_PER_RULE]
        
        # Create base rule dictionary
        rule = {
            'feature': feature,
            'feature_type': feature_type,
            'outcome': outcome,
            'weight': weight
        }
        
        # Add metadata if enabled
        if params.INCLUDE_RULE_METADATA:
            rule.update({
                'mechanisms': mechanisms,
                'mechanism_count': len(mechanisms),
                'rule_id': f"{feature}_{outcome}_{weight}"
            })
        
        # Create appropriate template based on feature type
        if feature_type == 'continuous':
            rule.update({
                'condition_template': params.RULE_TEMPLATES['continuous'].format(
                    feature=feature, outcome=outcome
                ),
                'threshold_placeholder': params.PLACEHOLDER_MAPPINGS['continuous']['threshold_placeholder'],
                'confidence_placeholder': params.PLACEHOLDER_MAPPINGS['continuous']['confidence_placeholder']
            })
        
        elif feature_type == 'ordinal':
            rule.update({
                'condition_template': params.RULE_TEMPLATES['ordinal'].format(
                    feature=feature, outcome=outcome
                ),
                'threshold_placeholder': params.PLACEHOLDER_MAPPINGS['ordinal']['threshold_placeholder'],
                'confidence_placeholder': params.PLACEHOLDER_MAPPINGS['ordinal']['confidence_placeholder']
            })
        
        elif feature_type == 'binary':
            # Choose appropriate binary template based on outcome
            if outcome == 'success':
                template_key = 'binary_success'
            else:
                template_key = 'binary_failure'
            
            rule.update({
                'condition_template': params.RULE_TEMPLATES[template_key].format(
                    feature=feature, outcome=outcome
                ),
                'confidence_placeholder': params.PLACEHOLDER_MAPPINGS['binary']['confidence_placeholder']
            })
        
        elif feature_type == 'categorical':
            rule.update({
                'condition_template': params.RULE_TEMPLATES['categorical'].format(
                    feature=feature, outcome=outcome
                ),
                'threshold_placeholder': params.PLACEHOLDER_MAPPINGS['categorical']['threshold_placeholder'],
                'confidence_placeholder': params.PLACEHOLDER_MAPPINGS['categorical']['confidence_placeholder']
            })
        
        else:
            logger.warning(f"Unknown feature type '{feature_type}' for feature '{feature}', using continuous template")
            rule.update({
                'condition_template': params.RULE_TEMPLATES['continuous'].format(
                    feature=feature, outcome=outcome
                ),
                'threshold_placeholder': params.PLACEHOLDER_MAPPINGS['continuous']['threshold_placeholder'],
                'confidence_placeholder': params.PLACEHOLDER_MAPPINGS['continuous']['confidence_placeholder']
            })
        
        if params.LOG_RULE_GENERATION:
            logger.debug(f"Created rule template: {rule['condition_template']}")
        
        return rule
    
    def generate_rule_templates(self) -> List[Dict[str, Any]]:
        """
        Generate IF-THEN rule templates from the top weighted edges.
        
        Returns:
            List of rule template dictionaries
        """
        logger.info(f"Generating rule templates for top {params.NUM_TOP_EDGES} edges")
        
        rules = []
        
        # Get top edges
        top_edges = self.edge_weights[:params.NUM_TOP_EDGES]
        
        for feature, outcome, weight in top_edges:
            # Validate feature if enabled
            if params.VALIDATE_FEATURE_NAMES and feature not in self.selected_features:
                if params.SKIP_INVALID_FEATURES:
                    logger.warning(f"Skipping feature '{feature}' - not in selected features")
                    self.stats['skipped_features'].append(feature)
                    continue
                else:
                    logger.warning(f"Feature '{feature}' not in selected features but proceeding")
            
            # Skip failure rules if not enabled
            if not params.INCLUDE_FAILURE_RULES and outcome == 'failure':
                continue
            
            # Create rule template
            rule = self.create_rule_template(feature, outcome, weight)
            rules.append(rule)
            
            # Update statistics
            self.stats['rules_generated'] += 1
            self.stats['rules_by_type'][rule['feature_type']] += 1
            self.stats['rules_by_outcome'][outcome] += 1
        
        logger.info(f"Generated {len(rules)} rule templates")
        
        self.initial_rules = rules
        return rules
    
    def save_rule_templates(self, rules: List[Dict[str, Any]]) -> None:
        """
        Save rule templates to JSON file.
        
        Args:
            rules: List of rule template dictionaries
        """
        logger.info(f"Saving {len(rules)} rule templates to {params.INITIAL_RULES_PATH}")
        
        try:
            with open(params.INITIAL_RULES_PATH, 'w', encoding='utf-8') as f:
                json.dump(rules, f, indent=params.JSON_INDENT, ensure_ascii=params.JSON_ENSURE_ASCII)
            
            logger.info("Rule templates saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save rule templates: {e}")
            raise
    
    def save_statistics(self) -> None:
        """
        Save rule extraction statistics to JSON file.
        """
        if not params.LOG_RULE_STATISTICS:
            return
        
        # Prepare statistics summary
        stats_summary = {
            'extraction_summary': {
                'total_edges_in_graph': self.stats['total_edges'],
                'valid_edges_processed': self.stats['valid_edges'],
                'rules_generated': self.stats['rules_generated'],
                'top_edges_requested': params.NUM_TOP_EDGES
            },
            'rules_by_feature_type': dict(self.stats['rules_by_type']),
            'rules_by_outcome': dict(self.stats['rules_by_outcome']),
            'feature_types_detected': self.feature_types,
            'skipped_features': self.stats['skipped_features'],
            'configuration': {
                'num_top_edges': params.NUM_TOP_EDGES,
                'min_edge_weight': params.MIN_EDGE_WEIGHT,
                'include_failure_rules': params.INCLUDE_FAILURE_RULES,
                'prioritize_success_rules': params.PRIORITIZE_SUCCESS_RULES
            }
        }
        
        try:
            with open(params.RULE_STATISTICS_PATH, 'w', encoding='utf-8') as f:
                json.dump(stats_summary, f, indent=params.JSON_INDENT, ensure_ascii=params.JSON_ENSURE_ASCII)
            
            logger.info(f"Rule extraction statistics saved to {params.RULE_STATISTICS_PATH}")
            
        except Exception as e:
            logger.warning(f"Failed to save statistics: {e}")
    
    def log_statistics(self) -> None:
        """
        Log comprehensive statistics about the rule extraction process.
        """
        logger.info("=" * 60)
        logger.info("RULE EXTRACTION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total edges in graph: {self.causal_graph.number_of_edges()}")
        logger.info(f"Valid edges processed: {self.stats['total_edges']}")
        logger.info(f"Top edges selected: {min(params.NUM_TOP_EDGES, self.stats['total_edges'])}")
        logger.info(f"Rules generated: {self.stats['rules_generated']}")
        
        if self.stats['rules_by_type']:
            logger.info("Rules by feature type:")
            for feature_type, count in self.stats['rules_by_type'].items():
                logger.info(f"  {feature_type}: {count}")
        
        if self.stats['rules_by_outcome']:
            logger.info("Rules by outcome:")
            for outcome, count in self.stats['rules_by_outcome'].items():
                logger.info(f"  {outcome}: {count}")
        
        if self.stats['skipped_features']:
            logger.warning(f"Skipped features ({len(self.stats['skipped_features'])}): {self.stats['skipped_features']}")
        
        logger.info("Rule extraction completed successfully!")
    
    def extract_rules(self) -> List[Dict[str, Any]]:
        """
        Main pipeline method to extract IF-THEN rules from the causal graph.
        
        Returns:
            List of rule template dictionaries
            
        Raises:
            Exception: If any step in the pipeline fails critically
        """
        logger.info("Starting rule extraction pipeline")
        
        # Step 1: Load inputs
        self.load_causal_graph()
        self.load_selected_features()
        self.load_training_data()  # Optional for feature type detection
        
        # Step 2: Extract and sort edge weights
        self.extract_edge_weights()
        
        # Step 3: Generate rule templates
        rules = self.generate_rule_templates()
        
        # Step 4: Save outputs
        self.save_rule_templates(rules)
        self.save_statistics()
        
        # Step 5: Log comprehensive statistics
        self.log_statistics()
        
        logger.info("Rule extraction pipeline completed successfully!")
        
        return rules


def run_rule_extraction() -> List[Dict[str, Any]]:
    """
    Main entry point for Module 7: IF-THEN Rule Extraction.
    
    Returns:
        List of rule template dictionaries
    """
    # Initialize the extractor
    extractor = RuleExtractor()
    
    # Extract rules
    rules = extractor.extract_rules()
    
    return rules


if __name__ == "__main__":
    # For testing purposes
    run_rule_extraction() 