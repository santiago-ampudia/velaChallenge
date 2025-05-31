#!/usr/bin/env python3
"""
Feature Selection Module using XGBoost + SHAP

This module implements the feature selection pipeline that:
1. Loads prepared training and test datasets
2. Trains an XGBoost classifier with class weights reflecting 2% positive rate
3. Computes SHAP values to rank feature importance
4. Selects the top 25 most predictive features
5. Filters datasets to selected features plus missing flags
6. Saves the reduced datasets for downstream modules

The selection process ensures we focus on the strongest signals while maintaining
full explainability through SHAP values.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

# Machine learning imports
try:
    import xgboost as xgb
except ImportError:
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

try:
    import shap
except ImportError:
    raise ImportError("SHAP is required. Install with: pip install shap")

# Local imports
from . import parameters as params

# Configure logging
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    XGBoost + SHAP based feature selector for investment decision modeling.
    
    This class encapsulates the complete feature selection pipeline, from
    loading prepared data to saving the top-N selected features.
    """
    
    def __init__(self):
        """Initialize the feature selector with configuration parameters."""
        self.model = None
        self.explainer = None
        self.feature_importances = {}
        self.selected_features = []
        
        logger.info("Initialized FeatureSelector for investment decision modeling")
        logger.info(f"Target: Select top {params.NUM_TOP_FEATURES} features")
        logger.info(f"Class weighting for {params.TARGET_POSITIVE_RATE:.1%} positive rate")
    
    def load_prepared_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the prepared training and test datasets from Module 3.
        
        Returns:
            Tuple of (train_df, test_df) containing the prepared datasets
            
        Raises:
            FileNotFoundError: If prepared data files don't exist
            ValueError: If datasets have mismatched columns
        """
        logger.info("Loading prepared datasets from Module 3...")
        
        # Check if input files exist
        if not os.path.exists(params.TRAIN_PREPARED_PATH):
            raise FileNotFoundError(f"Training data not found: {params.TRAIN_PREPARED_PATH}")
        
        if not os.path.exists(params.TEST_PREPARED_PATH):
            raise FileNotFoundError(f"Test data not found: {params.TEST_PREPARED_PATH}")
        
        # Load datasets
        train_df = pd.read_pickle(params.TRAIN_PREPARED_PATH)
        test_df = pd.read_pickle(params.TEST_PREPARED_PATH)
        
        logger.info(f"✅ Loaded training data: {train_df.shape}")
        logger.info(f"✅ Loaded test data: {test_df.shape}")
        
        # Validate datasets have matching columns
        if set(train_df.columns) != set(test_df.columns):
            raise ValueError("Training and test datasets have mismatched columns")
        
        # Validate success column exists
        if 'success' not in train_df.columns:
            raise ValueError("'success' column not found in datasets")
        
        logger.info(f"📊 Dataset validation passed")
        logger.info(f"   - Total features: {len(train_df.columns) - 1}")  # Exclude 'success'
        logger.info(f"   - Training positives: {train_df['success'].sum()}/{len(train_df)} ({train_df['success'].mean():.1%})")
        logger.info(f"   - Test positives: {test_df['success'].sum()}/{len(test_df)} ({test_df['success'].mean():.1%})")
        
        return train_df, test_df
    
    def prepare_features_and_target(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Separate features and target variables from the datasets.
        
        Args:
            train_df: Training dataset
            test_df: Test dataset
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info("Separating features and target variables...")
        
        # Separate features and target
        X_train = train_df.drop(columns=params.EXCLUDE_COLUMNS)
        y_train = train_df['success'].astype('int8')
        X_test = test_df.drop(columns=params.EXCLUDE_COLUMNS)
        y_test = test_df['success'].astype('int8')
        
        # Ensure column alignment
        X_test = X_test[X_train.columns]  # Reorder test columns to match training
        
        # Convert categorical and object columns to numeric for XGBoost compatibility
        logger.info("Converting categorical and object columns to numeric...")
        
        # Handle categorical columns (convert to numeric codes)
        categorical_cols = X_train.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            # Convert to numeric codes
            X_train[col] = X_train[col].cat.codes.astype('int8')
            X_test[col] = X_test[col].cat.codes.astype('int8')
        
        # Handle object columns (label encode them)
        object_cols = X_train.select_dtypes(include=['object']).columns
        for col in object_cols:
            # Create a combined label encoder for both train and test
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            
            # Combine unique values from both train and test
            combined_values = pd.concat([X_train[col], X_test[col]]).unique()
            le.fit(combined_values)
            
            # Transform both datasets
            X_train[col] = le.transform(X_train[col]).astype('int8')
            X_test[col] = le.transform(X_test[col]).astype('int8')
        
        logger.info(f"✅ Feature separation completed")
        logger.info(f"   - Training features: {X_train.shape}")
        logger.info(f"   - Test features: {X_test.shape}")
        logger.info(f"   - Feature columns aligned: {list(X_train.columns) == list(X_test.columns)}")
        logger.info(f"   - Converted {len(categorical_cols)} categorical and {len(object_cols)} object columns")
        
        return X_train, y_train, X_test, y_test
    
    def calculate_class_weights(self, y_train: pd.Series) -> float:
        """
        Calculate XGBoost scale_pos_weight to reflect 2% positive rate.
        
        Args:
            y_train: Training target labels
            
        Returns:
            scale_pos_weight value for XGBoost
        """
        logger.info("Calculating class weights for 2% positive rate...")
        
        # Count positives and negatives
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        
        # Calculate scale_pos_weight for 2% positive rate
        # Formula: (num_neg / num_pos) * ((1 - target_rate) / target_rate)
        scale_pos_weight = (num_neg / num_pos) * ((1 - params.TARGET_POSITIVE_RATE) / params.TARGET_POSITIVE_RATE)
        
        logger.info(f"✅ Class weight calculation completed")
        logger.info(f"   - Training positives: {num_pos} ({num_pos/len(y_train):.1%})")
        logger.info(f"   - Training negatives: {num_neg} ({num_neg/len(y_train):.1%})")
        logger.info(f"   - Target positive rate: {params.TARGET_POSITIVE_RATE:.1%}")
        logger.info(f"   - Calculated scale_pos_weight: {scale_pos_weight:.2f}")
        
        return scale_pos_weight
    
    def train_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float) -> None:
        """
        Train XGBoost classifier for feature importance extraction.
        
        Args:
            X_train: Training features
            y_train: Training target
            scale_pos_weight: Class weight for positive class
        """
        logger.info("Training XGBoost model for feature importance...")
        
        # Prepare XGBoost parameters with calculated class weight
        xgb_params = params.XGBOOST_PARAMS.copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight
        
        # Initialize and train model
        self.model = xgb.XGBClassifier(**xgb_params)
        
        logger.info(f"🔥 Training XGBoost with {len(X_train.columns)} features...")
        self.model.fit(X_train, y_train)
        
        # Log training results
        train_score = self.model.score(X_train, y_train)
        logger.info(f"✅ XGBoost training completed")
        logger.info(f"   - Training accuracy: {train_score:.3f}")
        logger.info(f"   - Model parameters: {len(xgb_params)} configured")
        logger.info(f"   - Feature count: {len(X_train.columns)}")
    
    def compute_shap_importances(self, X_train: pd.DataFrame) -> Dict[str, float]:
        """
        Compute SHAP values and feature importances.
        
        Args:
            X_train: Training features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        logger.info("Computing SHAP values for feature importance ranking...")
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model, **params.SHAP_PARAMS)
        
        logger.info(f"🔍 Computing SHAP values for {len(X_train)} samples...")
        shap_values = self.explainer.shap_values(X_train)
        
        # Compute mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        self.feature_importances = dict(zip(X_train.columns, mean_abs_shap))
        
        # Sort by importance (descending)
        sorted_importances = dict(sorted(self.feature_importances.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        logger.info(f"✅ SHAP importance computation completed")
        logger.info(f"   - Features ranked: {len(sorted_importances)}")
        logger.info(f"   - Top feature: {list(sorted_importances.keys())[0]} (importance: {list(sorted_importances.values())[0]:.4f})")
        logger.info(f"   - Lowest feature: {list(sorted_importances.keys())[-1]} (importance: {list(sorted_importances.values())[-1]:.4f})")
        
        return sorted_importances
    
    def select_top_features(self, sorted_importances: Dict[str, float]) -> List[str]:
        """
        Select the top N features based on SHAP importance.
        
        Args:
            sorted_importances: Dictionary of features sorted by importance
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {params.NUM_TOP_FEATURES} features...")
        
        # Select top N features
        self.selected_features = list(sorted_importances.keys())[:params.NUM_TOP_FEATURES]
        
        # Validation checks
        if len(self.selected_features) < params.MIN_FEATURES_REQUIRED:
            raise ValueError(f"Only {len(self.selected_features)} features selected, minimum {params.MIN_FEATURES_REQUIRED} required")
        
        if len(self.selected_features) > params.MAX_FEATURES_ALLOWED:
            logger.warning(f"Selected {len(self.selected_features)} features, which exceeds maximum {params.MAX_FEATURES_ALLOWED}")
        
        logger.info(f"✅ Feature selection completed")
        logger.info(f"   - Selected features: {len(self.selected_features)}")
        logger.info(f"   - Top 5 features: {self.selected_features[:5]}")
        logger.info(f"   - Importance range: {sorted_importances[self.selected_features[0]]:.4f} to {sorted_importances[self.selected_features[-1]]:.4f}")
        
        return self.selected_features
    
    def filter_datasets_to_selected_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                           selected_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter datasets to selected features plus missing flags and target.
        
        Args:
            train_df: Original training dataset
            test_df: Original test dataset
            selected_features: List of selected feature names
            
        Returns:
            Tuple of (filtered_train, filtered_test)
        """
        logger.info("Filtering datasets to selected features...")
        
        # Start with selected features
        columns_to_keep = selected_features.copy()
        
        # Add corresponding missing flags
        missing_flags_added = 0
        for feature in selected_features:
            missing_col = feature + params.MISSING_FLAG_SUFFIX
            if missing_col in train_df.columns:
                columns_to_keep.append(missing_col)
                missing_flags_added += 1
        
        # Add target column
        columns_to_keep.append('success')
        
        # Filter datasets
        train_filtered = train_df[columns_to_keep].copy()
        test_filtered = test_df[columns_to_keep].copy()
        
        logger.info(f"✅ Dataset filtering completed")
        logger.info(f"   - Selected features: {len(selected_features)}")
        logger.info(f"   - Missing flags added: {missing_flags_added}")
        logger.info(f"   - Total columns kept: {len(columns_to_keep)}")
        logger.info(f"   - Filtered training shape: {train_filtered.shape}")
        logger.info(f"   - Filtered test shape: {test_filtered.shape}")
        
        return train_filtered, test_filtered
    
    def save_results(self, selected_features: List[str], train_filtered: pd.DataFrame, 
                    test_filtered: pd.DataFrame) -> None:
        """
        Save selected features list and filtered datasets to disk.
        
        Args:
            selected_features: List of selected feature names
            train_filtered: Filtered training dataset
            test_filtered: Filtered test dataset
        """
        logger.info("Saving feature selection results...")
        
        # Ensure output directory exists
        os.makedirs(params.OUTPUT_DIR, exist_ok=True)
        
        # Save selected features as JSON
        with open(params.SELECTED_FEATURES_JSON, 'w') as f:
            json.dump(selected_features, f, indent=2)
        
        # Save filtered datasets as pickle
        train_filtered.to_pickle(params.TRAIN_SELECTED_PKL)
        test_filtered.to_pickle(params.TEST_SELECTED_PKL)
        
        logger.info(f"✅ Results saved successfully")
        logger.info(f"   - Features list: {params.SELECTED_FEATURES_JSON}")
        logger.info(f"   - Training data: {params.TRAIN_SELECTED_PKL}")
        logger.info(f"   - Test data: {params.TEST_SELECTED_PKL}")
        
        # Log file sizes for reference
        try:
            features_size = os.path.getsize(params.SELECTED_FEATURES_JSON)
            train_size = os.path.getsize(params.TRAIN_SELECTED_PKL)
            test_size = os.path.getsize(params.TEST_SELECTED_PKL)
            
            logger.info(f"   - File sizes: features={features_size:,}B, train={train_size:,}B, test={test_size:,}B")
        except Exception as e:
            logger.warning(f"Could not determine file sizes: {e}")
    
    def run_feature_selection_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete feature selection pipeline.
        
        Returns:
            Tuple of (train_selected, test_selected) filtered datasets
        """
        logger.info("🚀 Starting feature selection pipeline...")
        
        try:
            # Step 1: Load prepared data
            train_df, test_df = self.load_prepared_data()
            
            # Step 2: Separate features and target
            X_train, y_train, X_test, y_test = self.prepare_features_and_target(train_df, test_df)
            
            # Step 3: Calculate class weights
            scale_pos_weight = self.calculate_class_weights(y_train)
            
            # Step 4: Train XGBoost model
            self.train_xgboost_model(X_train, y_train, scale_pos_weight)
            
            # Step 5: Compute SHAP importances
            sorted_importances = self.compute_shap_importances(X_train)
            
            # Step 6: Select top features
            selected_features = self.select_top_features(sorted_importances)
            
            # Step 7: Filter datasets
            train_filtered, test_filtered = self.filter_datasets_to_selected_features(
                train_df, test_df, selected_features)
            
            # Step 8: Save results
            self.save_results(selected_features, train_filtered, test_filtered)
            
            logger.info("🎉 Feature selection pipeline completed successfully!")
            
            return train_filtered, test_filtered
            
        except Exception as e:
            logger.error(f"❌ Feature selection pipeline failed: {str(e)}")
            raise


def run_feature_selection() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point for Module 4: Feature Selection.
    
    This function orchestrates the complete feature selection process:
    1. Loads prepared datasets from Module 3
    2. Trains XGBoost with appropriate class weights
    3. Computes SHAP feature importances
    4. Selects top 25 features
    5. Filters datasets and saves results
    
    Returns:
        Tuple of (train_selected, test_selected) containing filtered datasets
        
    Raises:
        Exception: If any step in the pipeline fails
    """
    logger.info("=" * 70)
    logger.info("MODULE 4: FEATURE SELECTION VIA XGBOOST + SHAP")
    logger.info("=" * 70)
    
    # Initialize feature selector
    selector = FeatureSelector()
    
    # Run the complete pipeline
    train_selected, test_selected = selector.run_feature_selection_pipeline()
    
    return train_selected, test_selected 