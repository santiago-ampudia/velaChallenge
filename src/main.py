#!/usr/bin/env python3
"""
Main Pipeline Orchestrator for Vela Partners Investment Decision Engine

This script coordinates the execution of all modules in the LLM-driven 
investment decision pipeline, achieving >20% precision through:

1. Data Cleaning (Module 1) - Clean and prepare raw founder data
2. Data Splitting (Module 2) - Create train/test splits with proper resampling
3. Feature Selection (Module 3) - Identify most predictive features
4. LLM Reasoning (Module 4) - Generate chain-of-thought explanations
5. Graph Construction (Module 5) - Build weighted causal graphs
6. Rule Extraction (Module 6) - Extract IF-THEN rules with thresholds
7. Evaluation (Module 7) - Assess precision and generate reports

Each module maintains full traceability for explainable AI requirements.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any
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
        
        logger.info("Initialized Vela Partners Investment Decision Pipeline")
        logger.info("Target: >20% precision on held-out test set")
    
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
    
    def run_module_3_feature_selection(self) -> bool:
        """
        Execute Module 3: Feature Selection (Placeholder)
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("MODULE 3: FEATURE SELECTION - Not implemented yet")
        logger.info("=" * 60)
        return True
    
    def run_module_4_llm_reasoning(self) -> bool:
        """
        Execute Module 4: LLM Chain-of-Thought Generation (Placeholder)
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("MODULE 4: LLM REASONING - Not implemented yet")
        logger.info("=" * 60)
        return True
    
    def run_module_5_graph_construction(self) -> bool:
        """
        Execute Module 5: Weighted Causal Graph Construction (Placeholder)
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("MODULE 5: GRAPH CONSTRUCTION - Not implemented yet")
        logger.info("=" * 60)
        return True
    
    def run_module_6_rule_extraction(self) -> bool:
        """
        Execute Module 6: IF-THEN Rule Extraction (Placeholder)
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("MODULE 6: RULE EXTRACTION - Not implemented yet")
        logger.info("=" * 60)
        return True
    
    def run_module_7_evaluation(self) -> bool:
        """
        Execute Module 7: Model Evaluation and Reporting (Placeholder)
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("MODULE 7: EVALUATION - Not implemented yet")
        logger.info("=" * 60)
        return True
    
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
                'module_3_feature_selection',
                'module_4_llm_reasoning',
                'module_5_graph_construction',
                'module_6_rule_extraction',
                'module_7_evaluation'
            ]
        
        logger.info("🚀 Starting Vela Partners Investment Decision Pipeline")
        logger.info(f"📋 Modules to execute: {modules}")
        
        # Module execution mapping
        module_functions = {
            'module_1_data_cleaning': self.run_module_1_data_cleaning,
            'module_2_data_splitting': self.run_module_2_data_splitting,
            'module_3_feature_selection': self.run_module_3_feature_selection,
            'module_4_llm_reasoning': self.run_module_4_llm_reasoning,
            'module_5_graph_construction': self.run_module_5_graph_construction,
            'module_6_rule_extraction': self.run_module_6_rule_extraction,
            'module_7_evaluation': self.run_module_7_evaluation
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
            'total_modules': 7,
            'completion_rate': len(self.pipeline_state['modules_completed']) / 7
        }


def main():
    """Main entry point for the pipeline."""
    pipeline = VelaPipeline()
    
    # Execute Module 1 and Module 2 since they are implemented
    result = pipeline.execute_pipeline(['module_1_data_cleaning', 'module_2_data_splitting'])
    
    # Print final status
    status = pipeline.get_pipeline_status()
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Modules completed: {len(status['modules_completed'])}/7")
    logger.info(f"Success rate: {status['completion_rate']:.1%}")
    
    if status['errors']:
        logger.error("Errors encountered:")
        for error in status['errors']:
            logger.error(f"  - {error}")
    
    return result


if __name__ == "__main__":
    result = main()
