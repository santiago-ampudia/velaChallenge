#!/usr/bin/env python3

"""
Causal Graph Construction Module for Investment Decision Analysis

This module implements the causal graph construction pipeline that:
1. Loads causal chain-of-thought explanations from Module 5
2. Parses "feature → mechanism → outcome" triples using regex patterns
3. Aggregates triples into weighted edges based on frequency
4. Builds a NetworkX directed graph with feature and outcome nodes
5. Serializes the graph and edge statistics for downstream rule extraction

The module follows clean architecture principles with comprehensive error
handling and full traceability for explainable AI requirements.
"""

import logging
import os
import json
import pickle
import csv
import re
import pandas as pd
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

# Import module parameters
from . import parameters as params

# Configure logging
logger = logging.getLogger(__name__)


class CausalGraphBuilder:
    """
    Builds weighted causal graphs from LLM chain-of-thought explanations.
    
    This class processes causal explanations to extract structured triples of
    (feature, mechanism, outcome), aggregates them into weighted directed edges,
    and constructs a NetworkX graph for downstream rule extraction and analysis.
    """
    
    def __init__(self):
        """Initialize the causal graph builder."""
        self.selected_features = []
        self.valid_features = None  
        self.causal_triples = []
        self.graph = None
        self.edge_data = defaultdict(lambda: {
            'weight': 0,
            'mechanisms': [],
            'confidences': []
        })
        
        # Statistics tracking (updated for new parsing logic)
        self.stats = {
            'total_explanations': 0,
            'parsed_explanations': 0,
            'total_causal_chains': 0,
            'extracted_triples': 0,
            'mapped_features': 0,
            'pattern_matches': defaultdict(int),
            'parsing_errors': []
        }
        
        # Ensure output directory exists
        os.makedirs(params.OUTPUT_DIR, exist_ok=True)
        
        logger.info("Initialized Causal Graph Builder")
        logger.info(f"Input: {params.INPUT_JSONL_PATH}")
        logger.info(f"Output directory: {params.OUTPUT_DIR}")
    
    def load_selected_features(self) -> List[str]:
        """
        Load the list of selected features from Module 4 for validation.
        
        Returns:
            List of selected feature names
            
        Raises:
            FileNotFoundError: If the selected features file doesn't exist
            ValueError: If the features file has unexpected structure
        """
        logger.info(f"Loading selected features from {params.SELECTED_FEATURES_PATH}")
        
        if not os.path.exists(params.SELECTED_FEATURES_PATH):
            raise FileNotFoundError(f"Selected features file not found: {params.SELECTED_FEATURES_PATH}")
        
        # Load the selected features JSON
        with open(params.SELECTED_FEATURES_PATH, 'r', encoding='utf-8') as f:
            features = json.load(f)
        
        # Validate structure
        if not isinstance(features, list):
            raise ValueError("Expected selected features to be a list")
        
        if len(features) == 0:
            raise ValueError("No features found in selected features file")
        
        logger.info(f"Loaded {len(features)} selected features")
        logger.info(f"Features: {features[:5]}..." if len(features) > 5 else f"Features: {features}")
        
        return features
    
    def _load_valid_features(self) -> Optional[Set[str]]:
        """
        Load valid features from Module 4 selected features file.
        
        Returns:
            Set of valid feature names, or None if validation disabled
        """
        if not params.VALIDATE_AGAINST_SELECTED_FEATURES:
            return None
        
        # Load selected features
        try:
            self.selected_features = self.load_selected_features()
            self.valid_features = set(self.selected_features)
            logger.info(f"Loaded {len(self.valid_features)} valid features for validation")
            return self.valid_features
        except Exception as e:
            logger.warning(f"Could not load selected features: {e}")
            if params.STRICT_FEATURE_VALIDATION:
                raise
            return None
    
    def load_causal_explanations(self) -> List[Dict[str, Any]]:
        """
        Load causal chain-of-thought explanations from Module 5 JSONL output.
        
        Returns:
            List of causal explanation objects
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the JSONL format is invalid
        """
        logger.info(f"Loading causal explanations from {params.INPUT_JSONL_PATH}")
        
        if not os.path.exists(params.INPUT_JSONL_PATH):
            raise FileNotFoundError(f"Causal explanations file not found: {params.INPUT_JSONL_PATH}")
        
        explanations = []
        
        with open(params.INPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    explanation = json.loads(line)
                    
                    # Validate required fields
                    for field in params.REQUIRED_JSONL_FIELDS:
                        if field not in explanation:
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Validate causal_cot length
                    causal_cot = explanation.get('causal_cot', '')
                    if len(causal_cot) < params.MIN_CAUSAL_COT_LENGTH:
                        logger.warning(f"Line {line_num}: causal_cot too short ({len(causal_cot)} chars), skipping")
                        continue
                    
                    explanations.append(explanation)
                    
                except json.JSONDecodeError as e:
                    error_msg = f"Line {line_num}: Invalid JSON - {str(e)}"
                    logger.error(error_msg)
                    if not params.SKIP_INVALID_TRIPLES:
                        raise ValueError(error_msg)
                except Exception as e:
                    error_msg = f"Line {line_num}: Validation error - {str(e)}"
                    logger.error(error_msg)
                    if not params.SKIP_INVALID_TRIPLES:
                        raise ValueError(error_msg)
        
        if len(explanations) == 0:
            raise ValueError("No valid causal explanations found in input file")
        
        logger.info(f"Loaded {len(explanations)} valid causal explanations")
        return explanations
    
    def _clean_mechanism_text(self, mechanism: str) -> str:
        """
        Clean and normalize mechanism text for consistent processing.
        
        Args:
            mechanism: Raw mechanism text from LLM output
            
        Returns:
            Cleaned mechanism text
        """
        if not params.CLEAN_MECHANISM_TEXT:
            return mechanism.strip()
        
        # Strip specified characters
        cleaned = mechanism.strip(params.MECHANISM_STRIP_CHARS)
        
        # Truncate if too long
        if len(cleaned) > params.MAX_MECHANISM_LENGTH:
            cleaned = cleaned[:params.MAX_MECHANISM_LENGTH].strip()
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def _normalize_outcome(self, outcome: str, success_label: int) -> str:
        """
        Normalize outcome text to standard "success" or "failure" labels.
        
        Args:
            outcome: Raw outcome text from pattern matching
            success_label: Numerical label (0 or 1) from the data
            
        Returns:
            Normalized outcome: "success" or "failure"
        """
        # Clean and lowercase the outcome text
        cleaned_outcome = outcome.lower().strip()
        
        # Use the mapping to normalize
        normalized = params.OUTCOME_NORMALIZATION.get(cleaned_outcome, cleaned_outcome)
        
        # Fall back to using the numerical label if mapping fails
        if normalized not in params.OUTCOME_NODES:
            normalized = "success" if success_label == 1 else "failure"
        
        return normalized
    
    def _extract_confidence(self, text: str) -> Optional[float]:
        """
        Extract confidence scores from causal explanation text.
        
        Args:
            text: Text to search for confidence indicators
            
        Returns:
            Confidence score as float (0-100), or None if not found
        """
        matches = params.CONFIDENCE_PATTERN.findall(text)
        if matches:
            try:
                # Return the first confidence score found
                return float(matches[0])
            except ValueError:
                pass
        return None
    
    def _validate_feature_name(self, feature: str) -> bool:
        """
        Validate that a feature name is in the list of selected features.
        
        Args:
            feature: Feature name to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not params.VALIDATE_AGAINST_SELECTED_FEATURES or self.valid_features is None:
            return True  # Validation disabled or no valid features loaded
        
        return feature in self.valid_features
    
    def parse_causal_triples(self, explanations: List[Dict[str, Any]]) -> List[Tuple[str, str, str, Optional[float]]]:
        """
        Parse causal explanations to extract (feature, outcome, mechanism, confidence) triples.
        
        Args:
            explanations: List of causal explanation objects from JSONL
            
        Returns:
            List of (feature, outcome, mechanism, confidence) tuples
            
        Raises:
            ValueError: If parsing fails critically
        """
        logger.info("Starting causal triple extraction with improved parsing...")
        logger.info(f"Processing {len(explanations)} causal explanations")
        
        all_triples = []
        
        for explanation in explanations:
            self.stats['total_explanations'] += 1
            
            try:
                # Extract causal text
                causal_text = explanation.get('causal_cot', '')
                if len(causal_text) < params.MIN_CAUSAL_COT_LENGTH:
                    logger.warning(f"Skipping explanation {explanation.get('index', '?')}: too short")
                    continue
                
                # Truncate if too long
                if len(causal_text) > params.MAX_CAUSAL_COT_LENGTH:
                    causal_text = causal_text[:params.MAX_CAUSAL_COT_LENGTH]
                
                # Split into individual causal chains
                causal_chains = self.split_causal_explanation(causal_text)
                self.stats['total_causal_chains'] += len(causal_chains)
                
                explanation_triples = []
                
                for chain_idx, chain_text in enumerate(causal_chains):
                    # Try each pattern to extract triples
                    chain_triples = []
                    extracted_features = set()  # Track features already extracted from this chain
                    
                    for pattern_name, pattern in params.CAUSAL_PATTERNS:
                        matches = pattern.findall(chain_text)
                        
                        for match in matches:
                            # Handle different match lengths based on pattern type
                            if len(match) >= 2:  # At least feature and mechanism
                                feature_text = match[0]
                                mechanism_text = match[1]
                                
                                # For incomplete pattern, mechanism might contain outcome info
                                if pattern_name == "incomplete":
                                    if len(match) >= 3 and match[2]:
                                        # Has outcome info embedded
                                        outcome_text = match[2]
                                    else:
                                        # No explicit outcome - use ground truth
                                        outcome_text = "success" if explanation.get('success_label', 0) == 1 else "failure"
                                elif len(match) >= 3:
                                    outcome_text = match[2]
                                else:
                                    continue  # Skip if not enough parts
                                
                                # Extract ALL features from compound descriptions
                                canonical_features = self.extract_all_features_from_compound(feature_text)
                                
                                # Create triples for each extracted feature
                                for canonical_feature in canonical_features:
                                    # Avoid duplicate features in same chain
                                    if canonical_feature and canonical_feature not in extracted_features:
                                        extracted_features.add(canonical_feature)
                                        
                                        # Handle outcome normalization
                                        if pattern_name in ["flexible", "incomplete"]:
                                            # Use semantic detection for descriptive/incomplete outcomes
                                            normalized_outcome = self.detect_semantic_outcome(outcome_text, explanation.get('success_label', 0))
                                        else:
                                            # Use direct mapping for explicit outcomes
                                            normalized_outcome = params.OUTCOME_NORMALIZATION.get(
                                                outcome_text.lower().strip(), 
                                                outcome_text.lower().strip()
                                            )
                                            # Fall back to ground truth if mapping fails
                                            if normalized_outcome not in params.OUTCOME_NODES:
                                                normalized_outcome = "success" if explanation.get('success_label', 0) == 1 else "failure"
                                        
                                        # Clean mechanism text
                                        cleaned_mechanism = mechanism_text.strip(params.MECHANISM_STRIP_CHARS)
                                        if len(cleaned_mechanism) > params.MAX_MECHANISM_LENGTH:
                                            cleaned_mechanism = cleaned_mechanism[:params.MAX_MECHANISM_LENGTH]
                                        
                                        # Extract confidence if present
                                        confidence = self.extract_confidence_score(chain_text)
                                        
                                        # Create triple (matching expected format)
                                        triple = (canonical_feature, normalized_outcome, cleaned_mechanism, confidence)
                                        
                                        chain_triples.append(triple)
                                        self.stats['extracted_triples'] += 1
                                        self.stats['mapped_features'] += 1
                                        self.stats['pattern_matches'][pattern_name] += 1
                                        
                                        if params.LOG_PATTERN_MATCHES:
                                            logger.debug(f"Extracted triple: {canonical_feature} → {cleaned_mechanism[:50]}... → {normalized_outcome}")
                                    else:
                                        if canonical_feature:
                                            logger.debug(f"Skipping duplicate feature in chain: '{canonical_feature}'")
                                        else:
                                            logger.debug(f"Failed to map feature: '{feature_text}'")
                    
                    # Add all triples from this chain
                    explanation_triples.extend(chain_triples)
                    
                if explanation_triples:
                    self.stats['parsed_explanations'] += 1
                    all_triples.extend(explanation_triples)
                    
                    if params.LOG_PARSING_DETAILS:
                        logger.debug(f"Explanation {explanation.get('index', '?')}: extracted {len(explanation_triples)} triples")
                else:
                    logger.warning(f"No triples extracted from explanation {explanation.get('index', '?')}")
                    
            except Exception as e:
                error_msg = f"Error parsing explanation {explanation.get('index', '?')}: {str(e)}"
                logger.error(error_msg)
                self.stats['parsing_errors'].append(error_msg)
                
                if len(self.stats['parsing_errors']) > params.MAX_PARSING_ERRORS:
                    raise Exception(f"Too many parsing errors ({len(self.stats['parsing_errors'])})")
        
        logger.info(f"Causal triple extraction completed")
        logger.info(f"Extracted {len(all_triples)} total triples from {self.stats['parsed_explanations']}/{self.stats['total_explanations']} explanations")
        logger.info(f"Total causal chains processed: {self.stats['total_causal_chains']}")
        logger.info(f"Pattern usage: {dict(self.stats['pattern_matches'])}")
        
        if params.LOG_PARSING_DETAILS and all_triples:
            logger.info("Sample triples:")
            for i, triple in enumerate(all_triples[:3]):
                logger.info(f"  {i+1}. {triple[0]} → {triple[2][:50]}... → {triple[1]}")
        
        if len(all_triples) == 0:
            logger.error("No valid causal triples could be extracted from the explanations")
            logger.error("This indicates a problem with the parsing patterns or feature mapping")
            
            # Log some examples for debugging
            if explanations:
                logger.error("Sample causal explanation for debugging:")
                sample_explanation = explanations[0]
                logger.error(f"Index: {sample_explanation.get('index', '?')}")
                logger.error(f"Causal CoT: {sample_explanation.get('causal_cot', '')[:200]}...")
            
            raise ValueError("No valid causal triples could be extracted from the explanations")
        
        return all_triples
    
    def aggregate_triples_to_edges(self, triples: List[Tuple[str, str, str, Optional[float]]]) -> None:
        """
        Aggregate causal triples into weighted edges for graph construction.
        
        Args:
            triples: List of (feature, outcome, mechanism, confidence) tuples
        """
        logger.info("Aggregating triples into weighted edges")
        
        # Clear existing data
        self.edge_data.clear()
        
        # Process each triple
        for feature, outcome, mechanism, confidence in triples:
            # Create edge key
            edge_key = (feature, outcome)
            
            # Update edge data
            edge_info = self.edge_data[edge_key]
            edge_info['weight'] += 1
            edge_info['mechanisms'].append(mechanism)
            
            if confidence is not None:
                edge_info['confidences'].append(confidence)
            
            # Limit mechanisms to prevent memory issues
            if len(edge_info['mechanisms']) > params.MAX_MECHANISMS_PER_EDGE:
                edge_info['mechanisms'] = edge_info['mechanisms'][-params.MAX_MECHANISMS_PER_EDGE:]
        
        # Filter edges by minimum weight
        filtered_edges = {
            edge_key: edge_info for edge_key, edge_info in self.edge_data.items()
            if edge_info['weight'] >= params.MIN_EDGE_WEIGHT
        }
        
        self.edge_data = defaultdict(lambda: {'weight': 0, 'mechanisms': [], 'confidences': []}, filtered_edges)
        
        logger.info(f"Aggregated {len(triples)} triples into {len(self.edge_data)} weighted edges")
        
        if params.LOG_GRAPH_STATISTICS and self.edge_data:
            logger.info("Top weighted edges:")
            sorted_edges = sorted(self.edge_data.items(), key=lambda x: x[1]['weight'], reverse=True)
            for i, ((feature, outcome), edge_info) in enumerate(sorted_edges[:5]):
                logger.info(f"  {i+1}. {feature} → {outcome}: weight={edge_info['weight']}")
        
        # Update statistics
        self.stats['valid_triples'] = sum(edge_info['weight'] for edge_info in self.edge_data.values())
    
    def build_directed_graph(self) -> nx.DiGraph:
        """
        Build the NetworkX directed graph from aggregated edge data.
        
        Returns:
            NetworkX DiGraph with features as nodes and causal relationships as edges
        """
        logger.info("Building NetworkX directed graph")
        
        # Initialize directed graph
        G = nx.DiGraph()
        
        # Add outcome nodes first
        for outcome in params.OUTCOME_NODES:
            G.add_node(outcome, node_type='outcome')
        
        # Add edges with attributes
        for (feature, outcome), edge_info in self.edge_data.items():
            # Add feature node if not already present
            if not G.has_node(feature):
                G.add_node(feature, node_type='feature')
            
            # Calculate confidence statistics
            confidences = edge_info['confidences']
            confidence_avg = sum(confidences) / len(confidences) if confidences else None
            confidence_sum = sum(confidences) if confidences else None
            
            # Get mechanisms for this edge
            mechanisms = edge_info['mechanisms']
            
            # Add directed edge with comprehensive attributes
            G.add_edge(
                feature, 
                outcome,
                weight=edge_info['weight'],
                mechanisms=mechanisms,
                mechanism_count=len(mechanisms),
                confidence_avg=confidence_avg,
                confidence_sum=confidence_sum
            )
        
        logger.info(f"Built directed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        if params.LOG_GRAPH_STATISTICS:
            # Log node type breakdown
            feature_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'feature']
            outcome_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'outcome']
            
            logger.info(f"Node breakdown: {len(feature_nodes)} features, {len(outcome_nodes)} outcomes")
            logger.info(f"Feature nodes: {feature_nodes[:10]}..." if len(feature_nodes) > 10 else f"Feature nodes: {feature_nodes}")
        
        self.graph = G
        return G
    
    def save_graph(self, graph: nx.DiGraph) -> None:
        """
        Save the NetworkX graph to a pickle file.
        
        Args:
            graph: NetworkX DiGraph to save
            
        Raises:
            IOError: If saving fails
        """
        logger.info(f"Saving graph to {params.CAUSAL_GRAPH_PKL_PATH}")
        
        # Ensure output directory exists
        os.makedirs(params.OUTPUT_DIR, exist_ok=True)
        
        try:
            with open(params.CAUSAL_GRAPH_PKL_PATH, 'wb') as f:
                pickle.dump(graph, f)
            
            # Get file size for logging
            file_size = os.path.getsize(params.CAUSAL_GRAPH_PKL_PATH)
            logger.info(f"Graph saved successfully: {file_size:,} bytes")
            
        except Exception as e:
            raise IOError(f"Failed to save graph: {e}")
    
    def save_edge_weights_csv(self) -> None:
        """
        Save edge weights and mechanisms to CSV file for analysis.
        """
        logger.info(f"Saving edge weights to {params.EDGE_WEIGHTS_CSV_PATH}")
        
        try:
            with open(params.EDGE_WEIGHTS_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers
                writer.writerow(params.CSV_HEADERS)
                
                # Write edge data
                for (feature, outcome), edge_info in self.edge_data.items():
                    # Get mechanisms and format for CSV
                    mechanisms = edge_info['mechanisms']
                    mechanisms_str = params.MECHANISM_SEPARATOR.join(mechanisms)
                    
                    # Truncate if too long
                    if len(mechanisms_str) > params.MAX_CSV_MECHANISMS_LENGTH:
                        mechanisms_str = mechanisms_str[:params.MAX_CSV_MECHANISMS_LENGTH] + "..."
                    
                    # Calculate confidence statistics
                    confidences = edge_info['confidences']
                    confidence_avg = sum(confidences) / len(confidences) if confidences else ""
                    confidence_sum = sum(confidences) if confidences else ""
                    
                    # Write row
                    writer.writerow([
                        feature,
                        outcome,
                        edge_info['weight'],
                        mechanisms_str,
                        f"{confidence_avg:.2f}" if confidence_avg else "",
                        f"{confidence_sum:.2f}" if confidence_sum else "",
                        len(mechanisms)
                    ])
            
            logger.info(f"Saved {len(self.edge_data)} edges to CSV file")
            
        except Exception as e:
            logger.error(f"Failed to save edge weights CSV: {e}")
            raise
    
    def generate_visualization(self, graph: nx.DiGraph) -> None:
        """
        Generate and save a visualization of the causal graph.
        
        Args:
            graph: NetworkX DiGraph to visualize
        """
        if not params.GENERATE_VISUALIZATION:
            return
        
        try:
            # Set up the plot
            plt.figure(figsize=params.FIGURE_SIZE)
            
            # Choose layout algorithm
            if params.LAYOUT_ALGORITHM == "spring":
                pos = nx.spring_layout(graph, k=2, iterations=50)
            elif params.LAYOUT_ALGORITHM == "circular":
                pos = nx.circular_layout(graph)
            elif params.LAYOUT_ALGORITHM == "kamada_kawai":
                pos = nx.kamada_kawai_layout(graph)
            else:
                pos = nx.random_layout(graph)
            
            # Separate nodes by type
            feature_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'feature']
            outcome_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'outcome']
            
            # Draw feature nodes
            nx.draw_networkx_nodes(
                graph, pos, 
                nodelist=feature_nodes,
                node_color='lightblue',
                node_size=params.NODE_SIZE_FACTOR,
                alpha=0.7
            )
            
            # Draw outcome nodes
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=outcome_nodes, 
                node_color='lightcoral',
                node_size=params.NODE_SIZE_FACTOR * 2,
                alpha=0.8
            )
            
            # Draw edges with weights
            edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]
            nx.draw_networkx_edges(
                graph, pos,
                width=[w * params.EDGE_WIDTH_FACTOR for w in edge_weights],
                alpha=0.6,
                edge_color='gray',
                arrows=True,
                arrowsize=20
            )
            
            # Draw labels
            nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
            
            # Draw edge labels if enabled
            if params.SHOW_EDGE_LABELS:
                edge_labels = {(u, v): str(d['weight']) for u, v, d in graph.edges(data=True)}
                nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=6)
            
            plt.title("Causal Graph: Feature → Outcome Relationships", fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig(params.GRAPH_VISUALIZATION_PATH, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Graph visualization saved successfully")
            
        except ImportError:
            logger.warning("Matplotlib not available - skipping visualization")
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
    
    def log_statistics(self) -> None:
        """
        Log comprehensive statistics about the causal graph construction process.
        """
        logger.info("=" * 60)
        logger.info("CAUSAL GRAPH CONSTRUCTION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total explanations processed: {self.stats['total_explanations']}")
        logger.info(f"Explanations with extracted triples: {self.stats['parsed_explanations']}")
        logger.info(f"Total causal chains processed: {self.stats['total_causal_chains']}")
        logger.info(f"Total triples extracted: {self.stats['extracted_triples']}")
        logger.info(f"Valid triples after aggregation: {self.stats.get('valid_triples', len(self.edge_data))}")
        logger.info(f"Unique edges created: {len(self.edge_data)}")
        logger.info(f"Parsing errors: {len(self.stats['parsing_errors'])}")
        
        if self.stats['pattern_matches']:
            logger.info("Pattern match breakdown:")
            for pattern_name, count in self.stats['pattern_matches'].items():
                logger.info(f"  {pattern_name}: {count} matches")
        
        if self.edge_data:
            weights = [edge_info['weight'] for edge_info in self.edge_data.values()]
            logger.info(f"Edge weight statistics:")
            logger.info(f"  Min weight: {min(weights)}")
            logger.info(f"  Max weight: {max(weights)}")
            logger.info(f"  Avg weight: {sum(weights) / len(weights):.2f}")
        
        if self.stats['parsing_errors']:
            logger.warning("Parsing errors encountered:")
            for error in self.stats['parsing_errors'][:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
            if len(self.stats['parsing_errors']) > 5:
                logger.warning(f"  ... and {len(self.stats['parsing_errors']) - 5} more errors")
    
    def map_feature_to_canonical(self, feature_text: str) -> Optional[str]:
        """
        Map LLM feature text to canonical feature name using multiple strategies.
        
        Args:
            feature_text: Feature text from LLM (e.g., "High educational qualifications and industry experience")
            
        Returns:
            Canonical feature name if found, None otherwise
        """
        # Normalize the input text
        normalized_text = self.normalize_feature_text(feature_text)
        
        # Strategy 1: Direct mapping lookup
        if normalized_text in params.FEATURE_NAME_MAPPING:
            mapped_feature = params.FEATURE_NAME_MAPPING[normalized_text]
            logger.debug(f"Direct mapping: '{feature_text}' → '{mapped_feature}'")
            return mapped_feature
        
        # Strategy 2: Partial text matching in mapping keys (improved for compound features)
        best_mapping_match = None
        best_mapping_score = 0
        
        for mapping_phrase, canonical_name in params.FEATURE_NAME_MAPPING.items():
            # Check if mapping phrase is contained in normalized text
            if mapping_phrase in normalized_text:
                # Calculate how much of the text this mapping covers
                coverage = len(mapping_phrase) / len(normalized_text)
                if coverage > best_mapping_score:
                    best_mapping_score = coverage
                    best_mapping_match = canonical_name
            
            # Check reverse containment for shorter feature descriptions
            elif normalized_text in mapping_phrase:
                coverage = len(normalized_text) / len(mapping_phrase)
                if coverage > best_mapping_score:
                    best_mapping_score = coverage
                    best_mapping_match = canonical_name
        
        if best_mapping_match and best_mapping_score > 0.3:  # Require at least 30% coverage
            logger.debug(f"Partial mapping: '{feature_text}' → '{best_mapping_match}' (coverage: {best_mapping_score:.2f})")
            return best_mapping_match
        
        # Strategy 3: Keyword-based fuzzy matching (enhanced for compound features)
        feature_scores = {}
        
        for feature_name, keywords in params.FEATURE_KEYWORDS.items():
            # Count how many keywords appear in the normalized text
            keyword_matches = sum(1 for keyword in keywords if keyword in normalized_text)
            
            if keyword_matches > 0:
                # Calculate keyword coverage score (more aggressive)
                keyword_score = keyword_matches / len(keywords)
                
                # Boost score for multiple keyword matches
                if keyword_matches > 1:
                    keyword_score = min(1.0, keyword_score * 1.5)
                
                # Also consider direct text similarity
                similarity = SequenceMatcher(None, normalized_text, feature_name).ratio()
                
                # Combine scores with emphasis on keyword matching for compound features
                combined_score = (keyword_score * 0.9) + (similarity * 0.1)
                feature_scores[feature_name] = combined_score
        
        # Find the best scoring feature with lower threshold
        if feature_scores:
            best_feature = max(feature_scores.items(), key=lambda x: x[1])
            feature_name, score = best_feature
            
            # Lower threshold for complex descriptions
            if score >= max(0.3, params.FUZZY_MATCH_THRESHOLD * 0.5):
                logger.debug(f"Keyword mapping: '{feature_text}' → '{feature_name}' (score: {score:.2f})")
                return feature_name
        
        # Strategy 4: Split compound features and try to map individual parts (improved)
        # Handle cases like "educational qualifications and industry experience"
        split_patterns = [r'\s+and\s+', r'\s+with\s+', r'\s*,\s*', r'\s+including\s+']
        
        for split_pattern in split_patterns:
            parts = [part.strip() for part in re.split(split_pattern, normalized_text)]
            
            if len(parts) > 1:
                logger.debug(f"Trying to map compound feature parts: {parts}")
                
                # Try to map each part separately using direct strategies (no recursion)
                best_part_mapping = None
                best_part_score = 0
                
                for part in parts:
                    if len(part) > 3:  # Skip very short parts
                        part_normalized = self.normalize_feature_text(part)
                        
                        # Try direct mapping for this part
                        if part_normalized in params.FEATURE_NAME_MAPPING:
                            part_mapping = params.FEATURE_NAME_MAPPING[part_normalized]
                            part_score = len(part) / len(normalized_text)
                            if part_score > best_part_score:
                                best_part_score = part_score
                                best_part_mapping = part_mapping
                        
                        # Try keyword matching for this part
                        part_scores = {}
                        for feature_name, keywords in params.FEATURE_KEYWORDS.items():
                            keyword_matches = sum(1 for keyword in keywords if keyword in part_normalized)
                            if keyword_matches > 0:
                                keyword_score = keyword_matches / len(keywords)
                                if keyword_matches > 1:
                                    keyword_score = min(1.0, keyword_score * 1.5)
                                part_scores[feature_name] = keyword_score
                        
                        if part_scores:
                            best_part_feature = max(part_scores.items(), key=lambda x: x[1])
                            feature_name, score = best_part_feature
                            if score >= 0.3:  # Lower threshold for parts
                                overall_score = score * (len(part) / len(normalized_text))
                                if overall_score > best_part_score:
                                    best_part_score = overall_score
                                    best_part_mapping = feature_name
                
                if best_part_mapping:
                    logger.debug(f"Compound mapping: '{feature_text}' → '{best_part_mapping}' (best part score: {best_part_score:.2f})")
                    return best_part_mapping
        
        # Strategy 5: Direct feature name check (in case LLM used exact names)
        if normalized_text in self.selected_features:
            logger.debug(f"Exact feature match: '{feature_text}' → '{normalized_text}'")
            return normalized_text
        
        # Strategy 6: Partial match with selected features (with better scoring)
        best_selected_feature = None
        best_selected_score = 0
        
        for selected_feature in self.selected_features:
            # Check containment in both directions
            if selected_feature in normalized_text:
                score = len(selected_feature) / len(normalized_text)
            elif normalized_text in selected_feature:
                score = len(normalized_text) / len(selected_feature)
            else:
                # Use sequence matching for similarity
                score = SequenceMatcher(None, normalized_text, selected_feature).ratio()
            
            if score > best_selected_score and score >= params.FUZZY_MATCH_THRESHOLD:
                best_selected_score = score
                best_selected_feature = selected_feature
        
        if best_selected_feature:
            logger.debug(f"Selected feature match: '{feature_text}' → '{best_selected_feature}' (score: {best_selected_score:.2f})")
            return best_selected_feature
        
        # No mapping found
        logger.debug(f"No mapping found for: '{feature_text}' (normalized: '{normalized_text}')")
        return None
    
    def normalize_feature_text(self, feature_text: str) -> str:
        """
        Normalize feature text for better matching by removing qualifiers and stop words.
        
        Args:
            feature_text: Raw feature text from LLM
            
        Returns:
            Normalized feature text
        """
        # Convert to lowercase
        normalized = feature_text.lower().strip()
        
        # Remove common qualifiers and stop words
        words = normalized.split()
        filtered_words = [word for word in words if word not in params.FEATURE_STOP_WORDS]
        
        # Rejoin words
        normalized = ' '.join(filtered_words)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def split_causal_explanation(self, causal_text: str) -> List[str]:
        """
        Split causal explanation into individual causal chains.
        
        Args:
            causal_text: Full causal explanation text
            
        Returns:
            List of individual causal chain texts
        """
        if not params.SPLIT_ON_SEMICOLONS:
            return [causal_text]
        
        # Split on semicolons and clean up
        chains = [chain.strip() for chain in causal_text.split(';') if chain.strip()]
        
        logger.debug(f"Split '{causal_text[:50]}...' into {len(chains)} chains")
        return chains
    
    def extract_confidence_score(self, text: str) -> Optional[float]:
        """
        Extract confidence score from text if present.
        
        Args:
            text: Text that may contain confidence information
            
        Returns:
            Confidence score (0.0-1.0) if found, None otherwise
        """
        match = params.CONFIDENCE_PATTERN.search(text)
        if match:
            confidence = float(match.group(1))
            # Convert percentage to decimal if needed
            if confidence > 1.0:
                confidence = confidence / 100.0
            return min(1.0, max(0.0, confidence))
        return None
    
    def detect_semantic_outcome(self, outcome_text: str, success_label: int) -> str:
        """
        Detect outcome sentiment from descriptive text using semantic keywords.
        
        Args:
            outcome_text: Descriptive outcome text from LLM
            success_label: Ground truth label (0=failure, 1=success)
            
        Returns:
            "success" or "failure" based on semantic analysis
        """
        # Normalize text for keyword matching
        text_lower = outcome_text.lower()
        words = set(text_lower.split())
        
        # Count keyword matches
        success_matches = len(words.intersection(params.SUCCESS_OUTCOME_KEYWORDS))
        failure_matches = len(words.intersection(params.FAILURE_OUTCOME_KEYWORDS))
        
        # Calculate confidence scores
        total_keywords = len(words)
        if total_keywords == 0:
            # Fall back to ground truth label
            return "success" if success_label == 1 else "failure"
        
        success_score = success_matches / total_keywords
        failure_score = failure_matches / total_keywords
        
        # Determine outcome based on keyword analysis
        if success_score > failure_score and success_score >= params.SEMANTIC_OUTCOME_THRESHOLD:
            detected_outcome = "success"
        elif failure_score > success_score and failure_score >= params.SEMANTIC_OUTCOME_THRESHOLD:
            detected_outcome = "failure"
        else:
            # Ambiguous text - fall back to ground truth
            detected_outcome = "success" if success_label == 1 else "failure"
        
        logger.debug(f"Semantic outcome detection: '{outcome_text[:50]}...' → {detected_outcome} (success_score={success_score:.2f}, failure_score={failure_score:.2f})")
        return detected_outcome
    
    def extract_all_features_from_compound(self, feature_text: str) -> List[str]:
        """
        Extract all features from a compound description.
        
        Args:
            feature_text: Compound feature description like "High educational qualification and NASDAQ leadership"
            
        Returns:
            List of canonical feature names found in the compound
        """
        # First check if this is actually a compound (contains separators)
        compound_indicators = [' and ', ' with ', ',', ' including ', ' plus ']
        is_compound = any(indicator in feature_text.lower() for indicator in compound_indicators)
        
        if not is_compound:
            # Try to map the entire text as-is (single feature)
            entire_mapping = self.map_feature_to_canonical_single(feature_text)
            if entire_mapping:
                return [entire_mapping]
        
        # Split on common compound separators
        split_patterns = [r'\s+and\s+', r'\s+with\s+', r'\s*,\s*', r'\s+including\s+', r'\s+plus\s+']
        
        all_features = []
        parts_to_process = [feature_text]
        
        # Apply each split pattern
        for pattern in split_patterns:
            new_parts = []
            for part in parts_to_process:
                split_parts = [p.strip() for p in re.split(pattern, part, flags=re.IGNORECASE)]
                if len(split_parts) > 1:
                    new_parts.extend(split_parts)
                else:
                    new_parts.append(part)
            parts_to_process = new_parts
        
        # Now try to map each part individually
        for part in parts_to_process:
            if len(part.strip()) > 3:  # Skip very short parts
                # Try direct mapping first
                canonical = self.map_feature_to_canonical_single(part.strip())
                if canonical and canonical not in all_features:
                    all_features.append(canonical)
                    logger.debug(f"Compound part mapped: '{part.strip()}' → '{canonical}'")
        
        # If no parts mapped, log for debugging
        if not all_features:
            logger.debug(f"No features extracted from compound: '{feature_text}'")
        
        return all_features
    
    def map_feature_to_canonical_single(self, feature_text: str) -> Optional[str]:
        """
        Map a single (non-compound) feature text to canonical name.
        This is the original mapping logic extracted to avoid recursion.
        
        Args:
            feature_text: Single feature description
            
        Returns:
            Canonical feature name if found, None otherwise
        """
        # Normalize the input text
        normalized_text = self.normalize_feature_text(feature_text)
        
        # Strategy 1: Direct mapping lookup
        if normalized_text in params.FEATURE_NAME_MAPPING:
            mapped_feature = params.FEATURE_NAME_MAPPING[normalized_text]
            logger.debug(f"Direct mapping: '{feature_text}' → '{mapped_feature}'")
            return mapped_feature
        
        # Strategy 2: Partial text matching in mapping keys
        best_mapping_match = None
        best_mapping_score = 0
        
        for mapping_phrase, canonical_name in params.FEATURE_NAME_MAPPING.items():
            # Check if mapping phrase is contained in normalized text
            if mapping_phrase in normalized_text:
                # Calculate how much of the text this mapping covers
                coverage = len(mapping_phrase) / len(normalized_text)
                if coverage > best_mapping_score:
                    best_mapping_score = coverage
                    best_mapping_match = canonical_name
            
            # Check reverse containment for shorter feature descriptions
            elif normalized_text in mapping_phrase:
                coverage = len(normalized_text) / len(mapping_phrase)
                if coverage > best_mapping_score:
                    best_mapping_score = coverage
                    best_mapping_match = canonical_name
        
        if best_mapping_match and best_mapping_score > 0.3:  # Require at least 30% coverage
            logger.debug(f"Partial mapping: '{feature_text}' → '{best_mapping_match}' (coverage: {best_mapping_score:.2f})")
            return best_mapping_match
        
        # Strategy 3: Keyword-based fuzzy matching
        feature_scores = {}
        
        for feature_name, keywords in params.FEATURE_KEYWORDS.items():
            # Count how many keywords appear in the normalized text
            keyword_matches = sum(1 for keyword in keywords if keyword in normalized_text)
            
            if keyword_matches > 0:
                # Calculate keyword coverage score
                keyword_score = keyword_matches / len(keywords)
                
                # Boost score for multiple keyword matches
                if keyword_matches > 1:
                    keyword_score = min(1.0, keyword_score * 1.5)
                
                # Also consider direct text similarity
                similarity = SequenceMatcher(None, normalized_text, feature_name).ratio()
                
                # Combine scores with emphasis on keyword matching
                combined_score = (keyword_score * 0.9) + (similarity * 0.1)
                feature_scores[feature_name] = combined_score
        
        # Find the best scoring feature with lower threshold
        if feature_scores:
            best_feature = max(feature_scores.items(), key=lambda x: x[1])
            feature_name, score = best_feature
            
            # Lower threshold for single features
            if score >= max(0.3, params.FUZZY_MATCH_THRESHOLD * 0.5):
                logger.debug(f"Keyword mapping: '{feature_text}' → '{feature_name}' (score: {score:.2f})")
                return feature_name
        
        # Strategy 4: Direct feature name check (in case LLM used exact names)
        if normalized_text in self.selected_features:
            logger.debug(f"Exact feature match: '{feature_text}' → '{normalized_text}'")
            return normalized_text
        
        # Strategy 5: Partial match with selected features (with better scoring)
        best_selected_feature = None
        best_selected_score = 0
        
        for selected_feature in self.selected_features:
            # Check containment in both directions
            if selected_feature in normalized_text:
                score = len(selected_feature) / len(normalized_text)
            elif normalized_text in selected_feature:
                score = len(normalized_text) / len(selected_feature)
            else:
                # Use sequence matching for similarity
                score = SequenceMatcher(None, normalized_text, selected_feature).ratio()
            
            if score > best_selected_score and score >= params.FUZZY_MATCH_THRESHOLD:
                best_selected_score = score
                best_selected_feature = selected_feature
        
        if best_selected_feature:
            logger.debug(f"Selected feature match: '{feature_text}' → '{best_selected_feature}' (score: {best_selected_score:.2f})")
            return best_selected_feature
        
        # No mapping found
        logger.debug(f"No mapping found for: '{feature_text}' (normalized: '{normalized_text}')")
        return None
    
    def build_causal_graph(self) -> nx.DiGraph:
        """
        Main pipeline method to build the complete causal graph.
        
        Returns:
            NetworkX DiGraph representing the causal relationships
            
        Raises:
            Exception: If any step in the pipeline fails critically
        """
        logger.info("Starting causal graph construction pipeline")
        
        # Step 0: Load valid features for validation
        if params.VALIDATE_AGAINST_SELECTED_FEATURES:
            self._load_valid_features()
        
        # Step 1: Load causal explanations
        explanations = self.load_causal_explanations()
        
        # Step 2: Parse causal triples
        triples = self.parse_causal_triples(explanations)
        
        # Step 3: Aggregate into weighted edges
        self.aggregate_triples_to_edges(triples)
        
        # Step 4: Build NetworkX graph
        graph = self.build_directed_graph()
        
        # Step 5: Save outputs
        self.save_graph(graph)
        self.save_edge_weights_csv()
        self.generate_visualization(graph)
        
        # Step 6: Log comprehensive statistics
        self.log_statistics()
        
        logger.info("Causal graph construction completed successfully!")
        
        return graph


def run_causal_graph_construction() -> nx.DiGraph:
    """
    Main entry point for Module 6: Causal Graph Construction.
    
    Returns:
        NetworkX DiGraph representing causal relationships
    """
    # Initialize the builder
    builder = CausalGraphBuilder()
    
    # Build the causal graph
    graph = builder.build_causal_graph()
    
    return graph


if __name__ == "__main__":
    # For testing purposes
    run_causal_graph_construction() 