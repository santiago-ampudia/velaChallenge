#!/usr/bin/env python3
"""
Module 9: Evaluation and Reporting

This module applies the calibrated IF-THEN rule set to the held-out test data,
computes final performance metrics (TP, FP, FN, TN, precision, recall, F1),
and generates a human-readable Markdown report with representative CoT excerpts
for full explainability.

The evaluation process:
1. Loads calibrated rules and test data
2. Applies rules directly to feature values (no CoT text for decisions)
3. Computes final performance metrics
4. Extracts representative CoT examples for each rule
5. Generates comprehensive Markdown report with explainable examples
"""

import json
import logging
import os
import random
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# Import parameters from the same module
from .parameters import (
    CALIBRATED_RULES_PATH,
    TEST_DATA_PATH,
    DESCRIPTIVE_COT_PATH,
    FINAL_METRICS_PATH,
    EVALUATION_REPORT_PATH,
    OUTPUT_DIR,
    TARGET_COLUMN,
    SUPPORTED_FEATURE_TYPES,
    MAX_COT_EXAMPLES_PER_RULE,
    METRICS_DECIMAL_PLACES,
    REQUIRED_RULE_FIELDS,
    EXPECTED_TEST_SET_SIZE,
    EXPECTED_POSITIVE_COUNT,
    EXPECTED_NEGATIVE_COUNT,
    LOG_MESSAGES,
    ERROR_MESSAGES,
    RANDOM_SEED,
    FILE_ENCODING,
    JSON_INDENT,
    MARKDOWN_COT_INDENT,
    REPORT_TITLE,
    RULES_TABLE_HEADERS,
    format_metric
)

# Configure module logger
logger = logging.getLogger(__name__)


class RuleEvaluator:
    """
    Evaluates calibrated IF-THEN rules on test data and generates comprehensive reports.
    
    This class handles:
    - Loading calibrated rules and test data
    - Applying rules to compute predictions and metrics
    - Extracting representative CoT examples
    - Generating detailed Markdown reports
    """
    
    def __init__(self):
        """Initialize the rule evaluator with configuration."""
        # Set random seed for reproducible example selection
        random.seed(RANDOM_SEED)
        
        # Initialize storage for loaded data
        self.calibrated_rules: List[Dict[str, Any]] = []
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.X_test: pd.DataFrame = pd.DataFrame()
        self.y_test: pd.Series = pd.Series()
        self.descriptive_cot_map: Dict[int, str] = {}
        
        # Initialize metrics storage
        self.final_metrics: Dict[str, Any] = {}
        self.rules_with_examples: List[Dict[str, Any]] = []
        
        logger.info(LOG_MESSAGES["start"])
    
    def load_calibrated_rules(self) -> bool:
        """
        Load calibrated rules from JSON file.
        
        Returns:
            bool: True if loading successful, False otherwise
            
        Raises:
            FileNotFoundError: If calibrated rules file doesn't exist
            ValueError: If rules format is invalid
        """
        try:
            logger.info(LOG_MESSAGES["load_rules"].format(path=CALIBRATED_RULES_PATH))
            
            if not os.path.exists(CALIBRATED_RULES_PATH):
                error_msg = ERROR_MESSAGES["missing_file"].format(path=CALIBRATED_RULES_PATH)
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            with open(CALIBRATED_RULES_PATH, "r", encoding=FILE_ENCODING) as f:
                self.calibrated_rules = json.load(f)
            
            # Validate rules format
            if not isinstance(self.calibrated_rules, list):
                raise ValueError("Calibrated rules must be a list")
            
            for i, rule in enumerate(self.calibrated_rules):
                if not isinstance(rule, dict):
                    raise ValueError(f"Rule {i} must be a dictionary")
                
                # Check required fields
                missing_fields = [field for field in REQUIRED_RULE_FIELDS if field not in rule]
                if missing_fields:
                    raise ValueError(f"Rule {i} missing required fields: {missing_fields}")
                
                # Validate feature type
                if rule["feature_type"] not in SUPPORTED_FEATURE_TYPES:
                    raise ValueError(f"Rule {i} has unsupported feature type: {rule['feature_type']}")
            
            logger.info(f"✅ Loaded {len(self.calibrated_rules)} calibrated rules")
            return True
            
        except Exception as e:
            error_msg = ERROR_MESSAGES["invalid_rules"].format(error=str(e))
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def load_test_data(self) -> bool:
        """
        Load test dataset from pickle file.
        
        Now uses the full original test set (1000 rows, 20 positives) for final evaluation.
        The training set was used for all threshold calibration, ensuring no data leakage.
        
        Returns:
            bool: True if loading successful, False otherwise
            
        Raises:
            FileNotFoundError: If test data file doesn't exist
            ValueError: If test data format is invalid
        """
        try:
            logger.info(LOG_MESSAGES["load_test"].format(path=TEST_DATA_PATH))
            logger.info("🎯 EVALUATION: Using full test set (1000 rows, 20 positives)")
            logger.info("📚 Training set was used for threshold calibration (no data leakage)")
            
            if not os.path.exists(TEST_DATA_PATH):
                error_msg = ERROR_MESSAGES["missing_file"].format(path=TEST_DATA_PATH)
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            self.test_df = pd.read_pickle(TEST_DATA_PATH)
            
            # Validate test data format
            if TARGET_COLUMN not in self.test_df.columns:
                error_msg = ERROR_MESSAGES["missing_target"].format(target=TARGET_COLUMN)
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Extract features and target
            self.y_test = self.test_df[TARGET_COLUMN].astype(int)
            self.X_test = self.test_df.drop(columns=[TARGET_COLUMN])
            
            # Validate test set size (warning only, not strict requirement)
            actual_size = len(self.test_df)
            if actual_size != EXPECTED_TEST_SET_SIZE:
                logger.warning(
                    ERROR_MESSAGES["wrong_test_size"].format(
                        actual=actual_size, 
                        expected=EXPECTED_TEST_SET_SIZE
                    )
                )
            
            # Log test set statistics
            positive_count = (self.y_test == 1).sum()
            negative_count = (self.y_test == 0).sum()
            
            logger.info(f"✅ Loaded full test dataset for final evaluation: {self.test_df.shape}")
            logger.info(f"   - Features: {self.X_test.shape[1]}")
            logger.info(f"   - Positive samples: {positive_count}")
            logger.info(f"   - Negative samples: {negative_count}")
            logger.info(f"   - Positive rate: {positive_count / len(self.test_df):.1%}")
            logger.info("✅ No data leakage: thresholds calibrated on training set only")
            
            return True
            
        except Exception as e:
            error_msg = ERROR_MESSAGES["invalid_test_data"].format(error=str(e))
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def load_descriptive_cot(self) -> bool:
        """
        Load descriptive chain-of-thought explanations from JSONL file.
        
        Returns:
            bool: True if loading successful, False otherwise
            
        Raises:
            FileNotFoundError: If CoT file doesn't exist
        """
        try:
            logger.info(LOG_MESSAGES["load_cot"].format(path=DESCRIPTIVE_COT_PATH))
            
            if not os.path.exists(DESCRIPTIVE_COT_PATH):
                error_msg = ERROR_MESSAGES["missing_file"].format(path=DESCRIPTIVE_COT_PATH)
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            self.descriptive_cot_map = {}
            
            with open(DESCRIPTIVE_COT_PATH, "r", encoding=FILE_ENCODING) as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            obj = json.loads(line.strip())
                            if "index" in obj and "descriptive_cot" in obj:
                                self.descriptive_cot_map[obj["index"]] = obj["descriptive_cot"]
                            else:
                                logger.warning(f"Line {line_num}: Missing 'index' or 'descriptive_cot' field")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Line {line_num}: Invalid JSON - {e}")
            
            logger.info(f"✅ Loaded {len(self.descriptive_cot_map)} CoT explanations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CoT explanations: {e}")
            raise
    
    def apply_rule(self, rule: Dict[str, Any], X: pd.DataFrame) -> pd.Series:
        """
        Apply a single rule to the feature data.
        
        Args:
            rule: Rule dictionary with feature, feature_type, and threshold
            X: Feature DataFrame
            
        Returns:
            pd.Series: Boolean mask where rule condition is satisfied
            
        Raises:
            ValueError: If feature type is unsupported or feature is missing
        """
        feature = rule["feature"]
        feature_type = rule["feature_type"]
        threshold = rule["threshold"]
        
        # Check if feature exists in data
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in test data")
        
        # Handle edge case: continuous feature with boolean threshold (should be binary)
        if feature_type == "continuous" and isinstance(threshold, bool):
            logger.debug(f"Feature '{feature}' marked as continuous but has boolean threshold - treating as binary")
            rule_mask = (X[feature] == threshold)
        # Apply rule based on feature type
        elif feature_type in ("continuous", "ordinal"):
            # For ordinal features with string thresholds, treat as categorical (exact match)
            if feature_type == "ordinal" and isinstance(threshold, str):
                logger.debug(f"Feature '{feature}' is ordinal with string threshold - using exact match")
                rule_mask = (X[feature] == threshold)
            else:
                # For continuous/numeric ordinal: feature >= threshold
                rule_mask = (X[feature] >= threshold)
        elif feature_type == "binary":
            # For binary: feature == True (threshold should be True/1)
            rule_mask = (X[feature] == True)
        elif feature_type == "categorical":
            # For categorical: feature == threshold (specific category)
            rule_mask = (X[feature] == threshold)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        return rule_mask
    
    def compute_final_metrics(self) -> Dict[str, Any]:
        """
        Apply all calibrated rules and compute final performance metrics.
        
        Uses the same weighted scoring approach as Module 8 threshold calibration.
        
        Returns:
            Dict containing TP, FP, FN, TN, precision, recall, F1
            
        Raises:
            ValueError: If rules cannot be applied or metrics cannot be computed
        """
        try:
            logger.info(LOG_MESSAGES["apply_rules"].format(count=len(self.calibrated_rules)))
            
            # Get the scoring threshold from the first rule (all rules should have the same)
            scoring_threshold = self.calibrated_rules[0].get("scoring_threshold", 0.5)
            logger.info(f"Using scoring threshold: {scoring_threshold:.6f}")
            
            # Initialize weighted scores for each test sample
            weighted_scores = pd.Series(0.0, index=self.X_test.index)
            
            # Apply each rule and accumulate weighted scores using Module 8's scoring method
            for i, rule in enumerate(self.calibrated_rules):
                try:
                    rule_mask = self.apply_rule(rule, self.X_test)
                    rule_weight = rule["weight"]
                    rule_precision = rule["rule_precision"]
                    
                    # Use the same scoring method as Module 8: precision_weighted
                    # contribution = rule['best_precision'] * (rule['weight'] / 100) for confidence_weighted
                    # contribution = rule['best_precision'] for precision_weighted
                    contribution = rule_precision
                    
                    # Add weighted contribution where rule fires
                    weighted_scores += rule_mask * contribution
                    
                    # Log rule application stats
                    rule_predictions = rule_mask.sum()
                    rule_true_positives = (rule_mask & (self.y_test == 1)).sum()
                    
                    logger.debug(
                        f"Rule {i+1} ({rule['feature']}): "
                        f"{rule_predictions} predictions, {rule_true_positives} TPs, "
                        f"weight={rule_weight}, precision={rule_precision:.3f}, contribution={contribution:.6f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to apply rule {i+1} ({rule.get('feature', 'unknown')}): {e}")
                    raise
            
            # Apply scoring threshold to get final predictions
            combined_mask = weighted_scores >= scoring_threshold
            
            logger.info(LOG_MESSAGES["compute_metrics"])
            logger.info(f"Weighted scores range: {weighted_scores.min():.6f} to {weighted_scores.max():.6f}")
            logger.info(f"Predictions above threshold: {combined_mask.sum()}")
            
            # Compute confusion matrix
            TP = int(((combined_mask) & (self.y_test == 1)).sum())
            FP = int(((combined_mask) & (self.y_test == 0)).sum())
            FN = int(((~combined_mask) & (self.y_test == 1)).sum())
            TN = int(((~combined_mask) & (self.y_test == 0)).sum())
            
            # Compute metrics with zero-division protection
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Store final metrics
            self.final_metrics = {
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "scoring_threshold": scoring_threshold,
                "total_predictions": TP + FP
            }
            
            # Log final metrics
            logger.info(f"✅ Final Metrics Computed:")
            logger.info(f"   - True Positives (TP): {TP}")
            logger.info(f"   - False Positives (FP): {FP}")
            logger.info(f"   - False Negatives (FN): {FN}")
            logger.info(f"   - True Negatives (TN): {TN}")
            logger.info(f"   - Precision: {precision:.3f}")
            logger.info(f"   - Recall: {recall:.3f}")
            logger.info(f"   - F1 Score: {f1_score:.3f}")
            
            return self.final_metrics
            
        except Exception as e:
            logger.error(f"Failed to compute final metrics: {e}")
            raise
    
    def extract_cot_examples_for_rules(self) -> List[Dict[str, Any]]:
        """
        Extract representative CoT examples for each rule.
        
        Returns:
            List of rule dictionaries with added example_indices and example_cots fields
            
        Raises:
            ValueError: If examples cannot be extracted
        """
        try:
            logger.info(LOG_MESSAGES["extract_examples"])
            
            self.rules_with_examples = []
            
            for i, rule in enumerate(self.calibrated_rules):
                # Create a copy of the rule to add examples
                rule_with_examples = rule.copy()
                
                try:
                    # Apply rule to find matching indices
                    rule_mask = self.apply_rule(rule, self.X_test)
                    
                    # Find successful indices where rule fired (rule condition true AND success == 1)
                    successful_indices = self.X_test[rule_mask & (self.y_test == 1)].index.tolist()
                    
                    if not successful_indices:
                        # No positive examples for this rule
                        logger.warning(
                            ERROR_MESSAGES["no_positive_examples"].format(
                                rule=f"{rule['feature']} (Rule {i+1})"
                            )
                        )
                        rule_with_examples["example_indices"] = []
                        rule_with_examples["example_cots"] = []
                    else:
                        # Select up to MAX_COT_EXAMPLES_PER_RULE examples
                        selected_indices = random.sample(
                            successful_indices, 
                            min(len(successful_indices), MAX_COT_EXAMPLES_PER_RULE)
                        )
                        
                        # Extract CoT explanations for selected indices
                        example_cots = []
                        valid_indices = []
                        
                        for idx in selected_indices:
                            if idx in self.descriptive_cot_map:
                                example_cots.append(self.descriptive_cot_map[idx])
                                valid_indices.append(idx)
                            else:
                                logger.warning(
                                    ERROR_MESSAGES["missing_cot"].format(index=idx)
                                )
                        
                        rule_with_examples["example_indices"] = valid_indices
                        rule_with_examples["example_cots"] = example_cots
                        
                        logger.debug(
                            f"Rule {i+1} ({rule['feature']}): "
                            f"Found {len(successful_indices)} positive examples, "
                            f"selected {len(valid_indices)} with CoT"
                        )
                
                except Exception as e:
                    logger.error(f"Failed to extract examples for rule {i+1}: {e}")
                    # Add empty examples to continue processing
                    rule_with_examples["example_indices"] = []
                    rule_with_examples["example_cots"] = []
                
                self.rules_with_examples.append(rule_with_examples)
            
            logger.info(f"✅ Extracted examples for {len(self.rules_with_examples)} rules")
            return self.rules_with_examples
            
        except Exception as e:
            logger.error(f"Failed to extract CoT examples: {e}")
            raise
    
    def generate_markdown_report(self) -> str:
        """
        Generate comprehensive Markdown evaluation report.
        
        Returns:
            str: Complete Markdown report content
            
        Raises:
            ValueError: If report cannot be generated
        """
        try:
            logger.info(LOG_MESSAGES["generate_report"])
            
            # Start building the report
            report_lines = []
            
            # Header
            report_lines.append(f"# {REPORT_TITLE}")
            report_lines.append("")
            
            # Test set summary
            positive_count = (self.y_test == 1).sum()
            negative_count = (self.y_test == 0).sum()
            total_count = len(self.y_test)
            
            report_lines.append(f"**Test Set Size:** {total_count:,}")
            report_lines.append(f"**Positives (Success):** {positive_count}")
            report_lines.append(f"**Negatives (Failure):** {negative_count}")
            report_lines.append("")
            
            # Overall metrics
            report_lines.append("## Overall Metrics")
            report_lines.append(f"- True Positives (TP):  {self.final_metrics['TP']}")
            report_lines.append(f"- False Positives (FP): {self.final_metrics['FP']}")
            report_lines.append(f"- False Negatives (FN): {self.final_metrics['FN']}")
            report_lines.append(f"- True Negatives (TN):  {self.final_metrics['TN']}")
            report_lines.append(f"- **Precision:** {format_metric(self.final_metrics['precision'])}")
            report_lines.append(f"- **Recall:**    {format_metric(self.final_metrics['recall'])}")
            report_lines.append(f"- **F1 Score:**  {format_metric(self.final_metrics['f1_score'])}")
            report_lines.append("")
            
            # Rules detail section
            report_lines.append("## Calibrated Rules Detail")
            report_lines.append("")
            
            # Rules table header
            header_row = "| " + " | ".join(RULES_TABLE_HEADERS) + " |"
            separator_row = "|" + "|".join(["---"] * len(RULES_TABLE_HEADERS)) + "|"
            
            report_lines.append(header_row)
            report_lines.append(separator_row)
            
            # Add each rule as a table row
            for i, rule in enumerate(self.rules_with_examples, start=1):
                # Format threshold based on feature type
                threshold = rule["threshold"]
                if rule["feature_type"] == "binary":
                    threshold_str = str(bool(threshold))
                elif isinstance(threshold, float):
                    threshold_str = f"{threshold:.3f}"
                else:
                    threshold_str = str(threshold)
                
                # Create table row
                row_data = [
                    str(i),
                    rule["feature"],
                    rule["feature_type"],
                    threshold_str,
                    format_metric(rule["rule_precision"]),
                    str(rule["rule_support"]),
                    format_metric(rule["combined_precision"]),
                    str(rule["combined_support"]),
                    ""  # Examples column - will be filled below
                ]
                
                table_row = "| " + " | ".join(row_data) + " |"
                report_lines.append(table_row)
                
                # Add CoT examples below the table row
                if rule["example_cots"]:
                    for cot in rule["example_cots"]:
                        # Clean and truncate CoT for readability
                        cleaned_cot = cot.replace("\n", " ").replace("|", "&#124;").strip()
                        if len(cleaned_cot) > 200:
                            cleaned_cot = cleaned_cot[:197] + "..."
                        
                        example_line = f"{MARKDOWN_COT_INDENT}- \"{cleaned_cot}\""
                        report_lines.append(example_line)
                else:
                    report_lines.append(f"{MARKDOWN_COT_INDENT}- *No positive examples found*")
                
                report_lines.append("")  # Add spacing between rules
            
            # Join all lines into final report
            report_content = "\n".join(report_lines)
            
            logger.info("✅ Generated comprehensive Markdown report")
            return report_content
            
        except Exception as e:
            logger.error(f"Failed to generate Markdown report: {e}")
            raise
    
    def save_final_metrics(self) -> bool:
        """
        Save final metrics to JSON file.
        
        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            logger.info(LOG_MESSAGES["save_metrics"].format(path=FINAL_METRICS_PATH))
            
            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            with open(FINAL_METRICS_PATH, "w", encoding=FILE_ENCODING) as f:
                json.dump(self.final_metrics, f, indent=JSON_INDENT)
            
            logger.info(f"✅ Saved final metrics to {FINAL_METRICS_PATH}")
            return True
            
        except Exception as e:
            error_msg = ERROR_MESSAGES["save_error"].format(path=FINAL_METRICS_PATH, error=str(e))
            logger.error(error_msg)
            raise
    
    def save_evaluation_report(self, report_content: str) -> bool:
        """
        Save evaluation report to Markdown file.
        
        Args:
            report_content: Complete Markdown report content
            
        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            logger.info(LOG_MESSAGES["save_report"].format(path=EVALUATION_REPORT_PATH))
            
            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            with open(EVALUATION_REPORT_PATH, "w", encoding=FILE_ENCODING) as f:
                f.write(report_content)
            
            logger.info(f"✅ Saved evaluation report to {EVALUATION_REPORT_PATH}")
            return True
            
        except Exception as e:
            error_msg = ERROR_MESSAGES["save_error"].format(path=EVALUATION_REPORT_PATH, error=str(e))
            logger.error(error_msg)
            raise
    
    def run_evaluation(self) -> Tuple[Dict[str, Any], str]:
        """
        Execute the complete evaluation pipeline.
        
        Returns:
            Tuple of (final_metrics_dict, report_content_string)
            
        Raises:
            Exception: If any step of the evaluation fails
        """
        try:
            # Step 1: Load all required data
            self.load_calibrated_rules()
            self.load_test_data()
            self.load_descriptive_cot()
            
            # Step 2: Apply rules and compute metrics
            final_metrics = self.compute_final_metrics()
            
            # Step 3: Extract CoT examples for each rule
            rules_with_examples = self.extract_cot_examples_for_rules()
            
            # Step 4: Generate Markdown report
            report_content = self.generate_markdown_report()
            
            # Step 5: Save outputs
            self.save_final_metrics()
            self.save_evaluation_report(report_content)
            
            logger.info(LOG_MESSAGES["complete"])
            
            return final_metrics, report_content
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


def run_evaluation() -> Tuple[Dict[str, Any], str]:
    """
    Main entry point for Module 9: Evaluation and Reporting.
    
    This function coordinates the complete evaluation pipeline:
    1. Loads calibrated rules and test data
    2. Applies rules to compute final metrics
    3. Extracts representative CoT examples
    4. Generates comprehensive Markdown report
    5. Saves all outputs to artifacts/eval/
    
    Returns:
        Tuple of (final_metrics_dict, report_content_string)
        
    Raises:
        Exception: If evaluation pipeline fails
    """
    evaluator = RuleEvaluator()
    return evaluator.run_evaluation()


if __name__ == "__main__":
    """Direct execution for testing purposes."""
    try:
        metrics, report = run_evaluation()
        print("Evaluation completed successfully!")
        print(f"Final precision: {metrics['precision']:.3f}")
        print(f"Report saved to: {EVALUATION_REPORT_PATH}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise 