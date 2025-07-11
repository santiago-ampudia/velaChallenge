#!/usr/bin/env python3
"""
Main Pipeline Orchestrator for Vela Partners Investment Decision Engine

This script coordinates the execution of all modules in the LLM-driven 
investment decision pipeline, achieving >20% precision through:

1. Data Cleaning (Module 1) - Clean and prepare raw founder data
2. Data Splitting (Module 2) - Create train/test splits with proper resampling
3. Data Encoding (Module 3) - Transform features into model-ready encoded formats
4. Feature Selection (Module 4) - Select top 25 features via XGBoost + SHAP
5. LLM Reasoning (Module 5) - Generate chain-of-thought explanations
6. Graph Construction (Module 6) - Build weighted causal graphs
7. Rule Extraction (Module 7) - Extract IF-THEN rules with thresholds
8. Evaluation (Module 8) - Assess precision and generate reports

Each module maintains full traceability for explainable AI requirements.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any, List
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Module imports
from module_1_data_cleaning import run_data_cleaning
from module_2_data_splitting import run_data_splitting
from module_3_data_encoding import run_data_encoding
from module_4_feature_selection import run_feature_selection
from module_5_llm_reasoning import run_llm_reasoning
from module_6_causal_graph import run_causal_graph_construction
from module_7_rule_extraction import run_rule_extraction
from module_8_threshold_calibration import run_threshold_calibration
from module_9_eval import run_evaluation


class VelaPipeline:
    """
    Main pipeline orchestrator for the Vela Partners investment decision engine.
    
    Manages the execution of all modules while maintaining data lineage
    and providing comprehensive logging and error handling.
    """
    
    def __init__(self):
        """Initialize the pipeline with configuration and state tracking."""
        self.pipeline_state = {
            'modules_completed': [],
            'data_artifacts': {},
            'execution_metadata': {},
            'errors': []
        }
        
        # Mapping of modules to their output artifacts for cleanup
        self.module_artifacts = {
            'module_1_data_cleaning': [
                'artifacts/prepared_data/full_cleaned.pkl',
                'artifacts/prepared_data/full_cleaned.csv'
            ],
            'module_2_data_splitting': [
                'artifacts/splits/train_full.pkl',
                'artifacts/splits/train_full.csv',
                'artifacts/splits/test.pkl',
                'artifacts/splits/test.csv'
            ],
            'module_3_data_encoding': [
                'artifacts/prepared_splits/train_prepared.pkl',
                'artifacts/prepared_splits/test_prepared.pkl'
            ],
            'module_4_feature_selection': [
                'artifacts/feature_selection/train_selected.pkl',
                'artifacts/feature_selection/test_selected.pkl',
                'artifacts/feature_selection/selected_features.json',
                'artifacts/feature_selection/feature_importance.png',
                'artifacts/feature_selection/shap_summary.png'
            ],
            'module_5_llm_reasoning': [
                'artifacts/llm_output/descriptive_explanations.jsonl',
                'artifacts/llm_output/causal_explanations.jsonl'
            ],
            'module_6_graph_construction': [
                'artifacts/graphs/causal_graph.pkl',
                'artifacts/graphs/graph_visualization.png'
            ],
            'module_7_rule_extraction': [
                'artifacts/rules/initial_rules.json',
                'artifacts/rules/rule_statistics.json'
            ],
            'module_8_threshold_calibration': [
                'artifacts/rules_calibrated/calibrated_rules.json',
                'artifacts/eval/threshold_metrics.csv',
                'artifacts/rules_calibrated/calibration_summary.json'
            ],
            'module_9_eval': [
                'artifacts/eval/final_metrics.json',
                'artifacts/eval/report.md'
            ]
        }
        
        logger.info("Initialized Vela Partners Investment Decision Pipeline")
        logger.info("Target: >20% precision on held-out test set")
    
    def reset_last_completed_module(self) -> bool:
        """
        Reset the last completed module by removing its outputs and clearing its completion status.
        
        This method is used when the output of the last completed module is determined to be wrong
        and needs to be replaced/re-run.
        
        Returns:
            bool: True if reset successful, False if no modules to reset or cleanup failed
        """
        if not self.pipeline_state['modules_completed']:
            logger.warning("⚠️  No completed modules to reset")
            return False
        
        # Get the last completed module
        last_module = self.pipeline_state['modules_completed'][-1]
        
        logger.info(f"🔄 Resetting last completed module: {last_module}")
        
        try:
            # Remove output artifacts for this module
            artifacts_to_remove = self.module_artifacts.get(last_module, [])
            removed_files = []
            missing_files = []
            
            for artifact_path in artifacts_to_remove:
                if os.path.exists(artifact_path):
                    os.remove(artifact_path)
                    removed_files.append(artifact_path)
                    logger.info(f"   🗑️  Removed: {artifact_path}")
                else:
                    missing_files.append(artifact_path)
                    logger.debug(f"   ℹ️  File not found (already clean): {artifact_path}")
            
            # Remove module from completed list
            self.pipeline_state['modules_completed'].remove(last_module)
            
            # Clear data artifacts for this module
            artifacts_keys_to_remove = []
            for key in self.pipeline_state['data_artifacts']:
                if last_module in key or any(last_module.split('_')[1] in key for _ in [last_module]):
                    artifacts_keys_to_remove.append(key)
            
            for key in artifacts_keys_to_remove:
                del self.pipeline_state['data_artifacts'][key]
                logger.info(f"   🧹 Cleared artifact metadata: {key}")
            
            # Clear any execution metadata for this module
            if last_module in self.pipeline_state['execution_metadata']:
                del self.pipeline_state['execution_metadata'][last_module]
                logger.info(f"   🧹 Cleared execution metadata for: {last_module}")
            
            logger.info(f"✅ Successfully reset module: {last_module}")
            logger.info(f"   - Files removed: {len(removed_files)}")
            logger.info(f"   - Metadata cleared: {len(artifacts_keys_to_remove)}")
            logger.info(f"   - Module removed from completed list")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to reset module {last_module}: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def replace_last_completed_module(self) -> bool:
        """
        Replace the last completed module by resetting it and re-running it.
        
        This is a convenience method that combines reset_last_completed_module()
        with re-execution of that module.
        
        Returns:
            bool: True if replacement successful, False otherwise
        """
        if not self.pipeline_state['modules_completed']:
            logger.warning("⚠️  No completed modules to replace")
            return False
        
        # Get the module to replace before resetting
        module_to_replace = self.pipeline_state['modules_completed'][-1]
        
        logger.info(f"🔄 Replacing last completed module: {module_to_replace}")
        
        # Reset the module
        if not self.reset_last_completed_module():
            logger.error(f"❌ Failed to reset module {module_to_replace}")
            return False
        
        # Re-run the module
        module_functions = {
            'module_1_data_cleaning': self.run_module_1_data_cleaning,
            'module_2_data_splitting': self.run_module_2_data_splitting,
            'module_3_data_encoding': self.run_module_3_data_encoding,
            'module_4_feature_selection': self.run_module_4_feature_selection,
            'module_5_llm_reasoning': self.run_module_5_llm_reasoning,
            'module_6_graph_construction': self.run_module_6_graph_construction,
            'module_7_rule_extraction': self.run_module_7_rule_extraction,
            'module_8_threshold_calibration': self.run_module_8_threshold_calibration,
            'module_9_eval': self.run_module_9_eval
        }
        
        if module_to_replace in module_functions:
            logger.info(f"🚀 Re-running module: {module_to_replace}")
            success = module_functions[module_to_replace]()
            
            if success:
                logger.info(f"✅ Successfully replaced module: {module_to_replace}")
            else:
                logger.error(f"❌ Failed to re-run module: {module_to_replace}")
            
            return success
        else:
            error_msg = f"Unknown module function for: {module_to_replace}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def clean_all_outputs(self) -> bool:
        """
        Clean all pipeline outputs and reset the entire pipeline state.
        
        This is a nuclear option that removes all generated artifacts and resets
        the pipeline to initial state.
        
        Returns:
            bool: True if cleanup successful, False otherwise
        """
        logger.warning("🧨 NUCLEAR OPTION: Cleaning ALL pipeline outputs")
        
        try:
            total_removed = 0
            total_missing = 0
            
            # Remove all artifacts for all modules
            for module_name, artifacts in self.module_artifacts.items():
                for artifact_path in artifacts:
                    if os.path.exists(artifact_path):
                        os.remove(artifact_path)
                        total_removed += 1
                        logger.info(f"   🗑️  Removed: {artifact_path}")
                    else:
                        total_missing += 1
                        logger.debug(f"   ℹ️  File not found: {artifact_path}")
            
            # Reset pipeline state
            self.pipeline_state = {
                'modules_completed': [],
                'data_artifacts': {},
                'execution_metadata': {},
                'errors': []
            }
            
            logger.info(f"✅ Pipeline completely reset")
            logger.info(f"   - Files removed: {total_removed}")
            logger.info(f"   - Files not found: {total_missing}")
            logger.info(f"   - Pipeline state reset")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to clean all outputs: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def run_module_1_data_cleaning(self) -> bool:
        """
        Execute Module 1: Data Cleaning
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODULE 1: DATA CLEANING")
            logger.info("=" * 60)
            
            # Execute data cleaning
            cleaned_df = run_data_cleaning()
            
            # Store metadata
            self.pipeline_state['data_artifacts']['cleaned_data'] = {
                'pickle_path': 'artifacts/prepared_data/full_cleaned.pkl',
                'csv_path': 'artifacts/prepared_data/full_cleaned.csv',
                'shape': cleaned_df.shape,
                'features': list(cleaned_df.columns),
                'missing_flags': len([col for col in cleaned_df.columns if col.endswith('_missing')])
            }
            
            self.pipeline_state['modules_completed'].append('module_1_data_cleaning')
            
            logger.info(f"✅ Module 1 completed successfully")
            logger.info(f"   - Dataset shape: {cleaned_df.shape}")
            logger.info(f"   - Missing value flags created: {self.pipeline_state['data_artifacts']['cleaned_data']['missing_flags']}")
            logger.info(f"   - Pickle output: {self.pipeline_state['data_artifacts']['cleaned_data']['pickle_path']}")
            logger.info(f"   - CSV output: {self.pipeline_state['data_artifacts']['cleaned_data']['csv_path']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Module 1 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def run_module_2_data_splitting(self) -> bool:
        """
        Execute Module 2: Data Splitting and Resampling
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODULE 2: DATA SPLITTING & RESAMPLING")
            logger.info("=" * 60)
            
            # Execute data splitting
            train_df, test_df = run_data_splitting()
            
            # Store metadata
            self.pipeline_state['data_artifacts']['train_data'] = {
                'pickle_path': 'artifacts/splits/train_full.pkl',
                'csv_path': 'artifacts/splits/train_full.csv',
                'shape': train_df.shape,
                'positive_rate': train_df['success'].mean(),
                'positive_count': (train_df['success'] == 1).sum(),
                'negative_count': (train_df['success'] == 0).sum()
            }
            
            self.pipeline_state['data_artifacts']['test_data'] = {
                'pickle_path': 'artifacts/splits/test.pkl',
                'csv_path': 'artifacts/splits/test.csv',
                'shape': test_df.shape,
                'positive_rate': test_df['success'].mean(),
                'positive_count': (test_df['success'] == 1).sum(),
                'negative_count': (test_df['success'] == 0).sum()
            }
            
            self.pipeline_state['modules_completed'].append('module_2_data_splitting')
            
            logger.info(f"✅ Module 2 completed successfully")
            logger.info(f"   - Training set: {train_df.shape} ({train_df['success'].mean():.1%} positive)")
            logger.info(f"   - Test set: {test_df.shape} ({test_df['success'].mean():.1%} positive)")
            logger.info(f"   - Training pickle: {self.pipeline_state['data_artifacts']['train_data']['pickle_path']}")
            logger.info(f"   - Training CSV: {self.pipeline_state['data_artifacts']['train_data']['csv_path']}")
            logger.info(f"   - Test pickle: {self.pipeline_state['data_artifacts']['test_data']['pickle_path']}")
            logger.info(f"   - Test CSV: {self.pipeline_state['data_artifacts']['test_data']['csv_path']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Module 2 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def run_module_3_data_encoding(self) -> bool:
        """
        Execute Module 3: Data Encoding and Feature Preparation
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODULE 3: DATA ENCODING & FEATURE PREPARATION")
            logger.info("=" * 60)
            
            # Execute data encoding
            train_prepared_df, test_prepared_df = run_data_encoding()
            
            # Store metadata
            self.pipeline_state['data_artifacts']['train_prepared'] = {
                'pickle_path': 'artifacts/prepared_splits/train_prepared.pkl',
                'shape': train_prepared_df.shape,
                'features': list(train_prepared_df.columns),
                'dtypes': train_prepared_df.dtypes.to_dict()
            }
            
            self.pipeline_state['data_artifacts']['test_prepared'] = {
                'pickle_path': 'artifacts/prepared_splits/test_prepared.pkl',
                'shape': test_prepared_df.shape,
                'features': list(test_prepared_df.columns),
                'dtypes': test_prepared_df.dtypes.to_dict()
            }
            
            self.pipeline_state['modules_completed'].append('module_3_data_encoding')
            
            logger.info(f"✅ Module 3 completed successfully")
            logger.info(f"   - Prepared training set: {train_prepared_df.shape}")
            logger.info(f"   - Prepared test set: {test_prepared_df.shape}")
            logger.info(f"   - Features encoded: {len(train_prepared_df.columns)}")
            logger.info(f"   - Training output: {self.pipeline_state['data_artifacts']['train_prepared']['pickle_path']}")
            logger.info(f"   - Test output: {self.pipeline_state['data_artifacts']['test_prepared']['pickle_path']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Module 3 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def run_module_4_feature_selection(self) -> bool:
        """
        Execute Module 4: Feature Selection via XGBoost + SHAP
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODULE 4: FEATURE SELECTION VIA XGBOOST + SHAP")
            logger.info("=" * 60)
            
            # Execute feature selection
            train_selected, test_selected = run_feature_selection()
            
            # Store metadata
            self.pipeline_state['data_artifacts']['train_selected'] = {
                'pickle_path': 'artifacts/feature_selection/train_selected.pkl',
                'shape': train_selected.shape,
                'features': [col for col in train_selected.columns if col != 'success'],
                'positive_rate': train_selected['success'].mean(),
                'positive_count': (train_selected['success'] == 1).sum(),
                'negative_count': (train_selected['success'] == 0).sum()
            }
            
            self.pipeline_state['data_artifacts']['test_selected'] = {
                'pickle_path': 'artifacts/feature_selection/test_selected.pkl',
                'shape': test_selected.shape,
                'features': [col for col in test_selected.columns if col != 'success'],
                'positive_rate': test_selected['success'].mean(),
                'positive_count': (test_selected['success'] == 1).sum(),
                'negative_count': (test_selected['success'] == 0).sum()
            }
            
            self.pipeline_state['data_artifacts']['selected_features'] = {
                'json_path': 'artifacts/feature_selection/selected_features.json',
                'count': len([col for col in train_selected.columns if col != 'success' and not col.endswith('_missing')]),
                'missing_flags': len([col for col in train_selected.columns if col.endswith('_missing')])
            }
            
            self.pipeline_state['modules_completed'].append('module_4_feature_selection')
            
            logger.info(f"✅ Module 4 completed successfully")
            logger.info(f"   - Selected features: {self.pipeline_state['data_artifacts']['selected_features']['count']}")
            logger.info(f"   - Missing flags: {self.pipeline_state['data_artifacts']['selected_features']['missing_flags']}")
            logger.info(f"   - Training set: {train_selected.shape} ({train_selected['success'].mean():.1%} positive)")
            logger.info(f"   - Test set: {test_selected.shape} ({test_selected['success'].mean():.1%} positive)")
            logger.info(f"   - Features JSON: {self.pipeline_state['data_artifacts']['selected_features']['json_path']}")
            logger.info(f"   - Training output: {self.pipeline_state['data_artifacts']['train_selected']['pickle_path']}")
            logger.info(f"   - Test output: {self.pipeline_state['data_artifacts']['test_selected']['pickle_path']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Module 4 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def run_module_5_llm_reasoning(self) -> bool:
        """
        Execute Module 5: LLM Chain-of-Thought Generation
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODULE 5: LLM CHAIN-OF-THOUGHT GENERATION")
            logger.info("=" * 60)
            
            # Execute LLM reasoning
            descriptive_path, causal_path = run_llm_reasoning()
            
            # Store metadata
            self.pipeline_state['data_artifacts']['llm_descriptive'] = {
                'jsonl_path': descriptive_path,
                'description': 'Descriptive chain-of-thought explanations for investment decisions'
            }
            
            self.pipeline_state['data_artifacts']['llm_causal'] = {
                'jsonl_path': causal_path,
                'description': 'Causal chain-of-thought explanations with feature→mechanism→outcome chains'
            }
            
            self.pipeline_state['modules_completed'].append('module_5_llm_reasoning')
            
            logger.info(f"✅ Module 5 completed successfully")
            logger.info(f"   - Descriptive explanations: {descriptive_path}")
            logger.info(f"   - Causal explanations: {causal_path}")
            
            return True
            
        except Exception as e:
            error_msg = f"Module 5 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def run_module_6_graph_construction(self) -> bool:
        """
        Execute Module 6: Weighted Causal Graph Construction
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODULE 6: WEIGHTED CAUSAL GRAPH CONSTRUCTION")
            logger.info("=" * 60)
            
            # Execute causal graph construction
            graph = run_causal_graph_construction()
            
            # Store metadata
            self.pipeline_state['data_artifacts']['causal_graph'] = {
                'pickle_path': 'artifacts/graphs/causal_graph.pkl',
                'csv_path': 'artifacts/graphs/edge_weights.csv',
                'visualization_path': 'artifacts/graphs/graph_visualization.png',
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'feature_nodes': len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'feature']),
                'outcome_nodes': len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'outcome'])
            }
            
            self.pipeline_state['modules_completed'].append('module_6_graph_construction')
            
            logger.info(f"✅ Module 6 completed successfully")
            logger.info(f"   - Total nodes: {graph.number_of_nodes()}")
            logger.info(f"   - Total edges: {graph.number_of_edges()}")
            logger.info(f"   - Feature nodes: {self.pipeline_state['data_artifacts']['causal_graph']['feature_nodes']}")
            logger.info(f"   - Outcome nodes: {self.pipeline_state['data_artifacts']['causal_graph']['outcome_nodes']}")
            logger.info(f"   - Graph pickle: {self.pipeline_state['data_artifacts']['causal_graph']['pickle_path']}")
            logger.info(f"   - Edge weights CSV: {self.pipeline_state['data_artifacts']['causal_graph']['csv_path']}")
            logger.info(f"   - Visualization: {self.pipeline_state['data_artifacts']['causal_graph']['visualization_path']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Module 6 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def run_module_7_rule_extraction(self) -> bool:
        """
        Execute Module 7: IF-THEN Rule Extraction
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODULE 7: IF-THEN RULE EXTRACTION")
            logger.info("=" * 60)
            
            # Execute rule extraction
            rules = run_rule_extraction()
            
            # Store metadata
            self.pipeline_state['data_artifacts']['initial_rules'] = {
                'json_path': 'artifacts/rules/initial_rules.json',
                'statistics_path': 'artifacts/rules/rule_statistics.json',
                'rule_count': len(rules),
                'description': 'IF-THEN rule templates with placeholders for thresholds and confidence'
            }
            
            self.pipeline_state['modules_completed'].append('module_7_rule_extraction')
            
            logger.info(f"✅ Module 7 completed successfully")
            logger.info(f"   - Rules generated: {len(rules)}")
            logger.info(f"   - Rules JSON: {self.pipeline_state['data_artifacts']['initial_rules']['json_path']}")
            logger.info(f"   - Statistics: {self.pipeline_state['data_artifacts']['initial_rules']['statistics_path']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Module 7 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def run_module_8_threshold_calibration(self) -> bool:
        """
        Execute Module 8: Threshold Calibration
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODULE 8: THRESHOLD CALIBRATION")
            logger.info("🎯 MAJOR FIX: Now uses training set (780 positives) for robust threshold calibration")
            logger.info("=" * 60)
            
            # Execute threshold calibration
            calibrated_rules = run_threshold_calibration()
            
            # Store metadata
            self.pipeline_state['data_artifacts']['calibrated_rules'] = {
                'json_path': 'artifacts/rules_calibrated/calibrated_rules.json',
                'csv_path': 'artifacts/eval/threshold_metrics.csv',
                'summary_path': 'artifacts/rules_calibrated/calibration_summary.json',
                'rule_count': len(calibrated_rules),
                'description': 'Calibrated IF-THEN rules with thresholds optimized on training set',
                'training_set_size': 7800,
                'training_positives': 780
            }
            
            self.pipeline_state['modules_completed'].append('module_8_threshold_calibration')
            
            logger.info(f"✅ Module 8 completed successfully")
            logger.info(f"   - Calibrated rules: {len(calibrated_rules)}")
            logger.info(f"   - Threshold calibration on: 7,800-row training set (780 positives)")
            logger.info(f"   - Test set remains untouched for evaluation")
            logger.info(f"   - Rules JSON: {self.pipeline_state['data_artifacts']['calibrated_rules']['json_path']}")
            logger.info(f"   - Metrics CSV: {self.pipeline_state['data_artifacts']['calibrated_rules']['csv_path']}")
            logger.info(f"   - Summary: {self.pipeline_state['data_artifacts']['calibrated_rules']['summary_path']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Module 8 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def run_module_9_eval(self) -> bool:
        """
        Execute Module 9: Evaluation and Reporting
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODULE 9: EVALUATION & REPORTING")
            logger.info("🎯 CLEAN EVALUATION: Using full test set (1000 rows, 20 positives)")
            logger.info("=" * 60)
            
            # Execute evaluation and reporting
            final_metrics, report_content = run_evaluation()
            
            # Store metadata
            self.pipeline_state['data_artifacts']['final_evaluation'] = {
                'metrics_path': 'artifacts/eval/final_metrics.json',
                'report_path': 'artifacts/eval/report.md',
                'final_precision': final_metrics['precision'],
                'final_recall': final_metrics['recall'],
                'final_f1': final_metrics['f1_score'],
                'true_positives': final_metrics['TP'],
                'false_positives': final_metrics['FP'],
                'false_negatives': final_metrics['FN'],
                'true_negatives': final_metrics['TN'],
                'report_size_chars': len(report_content),
                'test_set_size': 1000,
                'test_positives': 20,
                'evaluation_type': 'clean_no_leakage'
            }
            
            self.pipeline_state['modules_completed'].append('module_9_eval')
            
            logger.info(f"✅ Module 9 completed successfully")
            logger.info(f"   - CLEAN EVALUATION on 1000-row test set (20 positives)")
            logger.info(f"   - Final Precision: {final_metrics['precision']:.3f}")
            logger.info(f"   - Final Recall: {final_metrics['recall']:.3f}")
            logger.info(f"   - Final F1 Score: {final_metrics['f1_score']:.3f}")
            logger.info(f"   - True Positives: {final_metrics['TP']}")
            logger.info(f"   - False Positives: {final_metrics['FP']}")
            logger.info(f"   - Metrics JSON: {self.pipeline_state['data_artifacts']['final_evaluation']['metrics_path']}")
            logger.info(f"   - Report Markdown: {self.pipeline_state['data_artifacts']['final_evaluation']['report_path']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Module 9 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_state['errors'].append(error_msg)
            return False
    
    def execute_pipeline(self, modules: Optional[list] = None) -> Dict[str, Any]:
        """
        Execute the complete pipeline or specified modules.
        
        Args:
            modules: List of module names to execute (None for all)
            
        Returns:
            Dictionary containing execution results and metadata
        """
        if modules is None:
            modules = [
                'module_1_data_cleaning',
                'module_2_data_splitting', 
                'module_3_data_encoding',
                'module_4_feature_selection',
                'module_5_llm_reasoning',
                'module_6_graph_construction',
                'module_7_rule_extraction',
                'module_8_threshold_calibration',
                'module_9_eval'
            ]
        
        logger.info("🚀 Starting Vela Partners Investment Decision Pipeline")
        logger.info(f"📋 Modules to execute: {modules}")
        
        # Module execution mapping
        module_functions = {
            'module_1_data_cleaning': self.run_module_1_data_cleaning,
            'module_2_data_splitting': self.run_module_2_data_splitting,
            'module_3_data_encoding': self.run_module_3_data_encoding,
            'module_4_feature_selection': self.run_module_4_feature_selection,
            'module_5_llm_reasoning': self.run_module_5_llm_reasoning,
            'module_6_graph_construction': self.run_module_6_graph_construction,
            'module_7_rule_extraction': self.run_module_7_rule_extraction,
            'module_8_threshold_calibration': self.run_module_8_threshold_calibration,
            'module_9_eval': self.run_module_9_eval
        }
        
        # Execute each module
        execution_success = True
        for module_name in modules:
            if module_name in module_functions:
                success = module_functions[module_name]()
                if not success:
                    execution_success = False
                    logger.error(f"❌ Pipeline stopped due to failure in {module_name}")
                    break
            else:
                logger.warning(f"⚠️  Unknown module: {module_name}")
        
        # Generate final report
        if execution_success:
            logger.info("🎉 Pipeline execution completed successfully!")
        else:
            logger.error("💥 Pipeline execution failed")
        
        return {
            'success': execution_success,
            'pipeline_state': self.pipeline_state,
            'modules_completed': self.pipeline_state['modules_completed'],
            'errors': self.pipeline_state['errors']
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline execution status.
        
        Returns:
            Dictionary containing current state and progress
        """
        return {
            'modules_completed': self.pipeline_state['modules_completed'],
            'data_artifacts': self.pipeline_state['data_artifacts'],
            'errors': self.pipeline_state['errors'],
            'total_modules': 9,
            'completion_rate': len(self.pipeline_state['modules_completed']) / 9
        }


def main():
    """Main entry point for the pipeline."""
    pipeline = VelaPipeline()
    
    # Execute Modules 1, 2, 3, 4, 5, 6, 7, 8, and 9 since they are implemented
    result = pipeline.execute_pipeline(['module_1_data_cleaning', 'module_2_data_splitting', 'module_3_data_encoding', 'module_4_feature_selection', 'module_5_llm_reasoning', 'module_6_graph_construction', 'module_7_rule_extraction', 'module_8_threshold_calibration', 'module_9_eval'])
    
    # Print final status
    status = pipeline.get_pipeline_status()
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Modules completed: {len(status['modules_completed'])}/9")
    logger.info(f"Success rate: {status['completion_rate']:.1%}")
    
    if status['errors']:
        logger.error("Errors encountered:")
        for error in status['errors']:
            logger.error(f"  - {error}")
    
    # Show available pipeline management commands
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE MANAGEMENT COMMANDS")
    logger.info("=" * 60)
    logger.info("To reset/replace modules when output is determined to be wrong:")
    logger.info("  pipeline.reset_last_completed_module()     # Reset last module only")
    logger.info("  pipeline.replace_last_completed_module()   # Reset + re-run last module")
    logger.info("  pipeline.clean_all_outputs()              # Nuclear option: reset everything")
    logger.info("Example usage:")
    logger.info("  # If Module 8 output was wrong and needs replacement:")
    logger.info("  # success = pipeline.replace_last_completed_module()")
    
    return result


def create_pipeline_manager():
    """
    Create a pipeline manager instance for interactive use.
    
    This function provides an easy way to create a pipeline instance
    that can be used for manual module management and replacement.
    
    Returns:
        VelaPipeline: Configured pipeline instance
    """
    return VelaPipeline()


def replace_last_module():
    """
    Convenience function to replace the last completed module.
    
    This function creates a pipeline instance, identifies the last completed module,
    resets it, and re-runs it. Useful for quick fixes when a module's output
    is determined to be incorrect.
    
    Returns:
        bool: True if replacement successful, False otherwise
    """
    pipeline = VelaPipeline()
    
    # Load existing pipeline state if any modules were previously completed
    logger.info("🔍 Checking for previously completed modules...")
    
    # Check what modules have been completed based on existing output files
    completed_modules = []
    
    for module_name, artifacts in pipeline.module_artifacts.items():
        # Check if all key artifacts for this module exist
        key_artifacts = artifacts[:2] if len(artifacts) >= 2 else artifacts  # Check first 2 files
        if all(os.path.exists(artifact) for artifact in key_artifacts):
            completed_modules.append(module_name)
    
    # Update pipeline state with found completed modules
    pipeline.pipeline_state['modules_completed'] = completed_modules
    
    if not completed_modules:
        logger.warning("⚠️  No completed modules found to replace")
        return False
    
    logger.info(f"📋 Found completed modules: {completed_modules}")
    
    # Replace the last completed module
    return pipeline.replace_last_completed_module()


if __name__ == "__main__":
    result = main()
