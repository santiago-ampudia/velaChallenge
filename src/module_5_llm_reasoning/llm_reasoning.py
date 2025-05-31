#!/usr/bin/env python3

"""
LLM Chain-of-Thought Generation Module using Batched Processing

This module implements the LLM-based chain-of-thought generation pipeline that:
1. Loads the feature-selected training dataset from Module 4
2. Processes founders in batches to minimize API calls and costs
3. Generates descriptive explanations for why each founder succeeded/failed
4. Generates causal explanations identifying feature→mechanism→outcome chains
5. Saves both explanation types to separate JSONL files for downstream modules

The module uses OpenAI's GPT models with carefully crafted prompts to ensure
high-quality, consistent explanations while maintaining cost efficiency through
batched processing.
"""

import logging
import os
import json
import time
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import module parameters
from . import parameters as params

# Configure logging
logger = logging.getLogger(__name__)


class LLMChainOfThoughtGenerator:
    """
    Generates chain-of-thought explanations for investment decisions using OpenAI's LLM.
    
    This class processes founder data in batches to generate both descriptive and causal
    explanations for investment outcomes. It implements robust error handling, retry logic,
    and validation to ensure high-quality outputs while minimizing API costs.
    """
    
    def __init__(self):
        """Initialize the LLM chain-of-thought generator."""
        self.client = openai.OpenAI()  # Initialize OpenAI client
        self.total_api_calls = 0
        self.total_tokens_used = 0
        
        # Ensure output directory exists
        os.makedirs(params.OUTPUT_DIR, exist_ok=True)
        
        logger.info("Initialized LLM Chain-of-Thought Generator")
        logger.info(f"Model: {params.LLM_MODEL}")
        logger.info(f"Batch size: {params.BATCH_SIZE}")
        logger.info(f"Output directory: {params.OUTPUT_DIR}")
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load the feature-selected training dataset from Module 4.
        
        Returns:
            pd.DataFrame: The training dataset with selected features
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the dataset has unexpected structure
        """
        logger.info(f"Loading training data from {params.TRAIN_SELECTED_PATH}")
        
        if not os.path.exists(params.TRAIN_SELECTED_PATH):
            raise FileNotFoundError(f"Training data not found: {params.TRAIN_SELECTED_PATH}")
        
        # Load the dataset
        train_df = pd.read_pickle(params.TRAIN_SELECTED_PATH)
        
        # Validate dataset structure
        if 'success' not in train_df.columns:
            raise ValueError("Dataset missing required 'success' column")
        
        if len(train_df) == 0:
            raise ValueError("Dataset is empty")
        
        logger.info(f"Loaded training data: {train_df.shape}")
        logger.info(f"Features: {len(train_df.columns) - 1}")  # Exclude 'success' column
        logger.info(f"Success rate: {train_df['success'].mean():.1%}")
        
        return train_df
    
    def create_batch_features(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert a batch of DataFrame rows into feature dictionaries for LLM processing.
        
        Args:
            batch_df: DataFrame containing the batch of rows to process
            
        Returns:
            List of dictionaries, each containing index, features, and success_label
        """
        batch_features = []
        
        for idx, row in batch_df.iterrows():
            # Extract all features except the target variable
            feature_dict = {}
            for col in batch_df.columns:
                if col not in params.EXCLUDE_FROM_LLM:
                    value = row[col]
                    # Convert numpy types to Python types for JSON serialization
                    if pd.isna(value):
                        feature_dict[col] = None
                    elif isinstance(value, (pd.Timestamp, pd.Period)):
                        feature_dict[col] = str(value)
                    elif hasattr(value, 'item'):  # numpy scalar
                        feature_dict[col] = value.item()
                    else:
                        feature_dict[col] = value
            
            # Create batch entry
            batch_entry = {
                "index": int(idx),
                "features": feature_dict,
                "success_label": int(row["success"])
            }
            batch_features.append(batch_entry)
        
        return batch_features
    
    def call_llm_with_retry(self, messages: List[Dict[str, str]], 
                           batch_type: str = "unknown") -> str:
        """
        Make an API call to the LLM with automatic retry logic.
        
        Args:
            messages: List of message dictionaries for the API call
            batch_type: Type of batch being processed (for logging)
            
        Returns:
            The response text from the LLM
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(params.MAX_RETRIES):
            try:
                if params.LOG_API_CALLS:
                    logger.info(f"Making API call (attempt {attempt + 1}/{params.MAX_RETRIES}) for {batch_type}")
                
                # Make the API call
                response = self.client.chat.completions.create(
                    model=params.LLM_MODEL,
                    messages=messages,
                    temperature=params.LLM_TEMPERATURE,
                    max_tokens=params.LLM_MAX_TOKENS
                )
                
                # Track usage statistics
                self.total_api_calls += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens_used += response.usage.total_tokens
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                if params.LOG_API_CALLS:
                    logger.info(f"API call successful for {batch_type}")
                    logger.info(f"Response length: {len(response_text)} characters")
                
                return response_text
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                if params.HANDLE_RATE_LIMITS and attempt < params.MAX_RETRIES - 1:
                    logger.info(f"Waiting {params.RATE_LIMIT_DELAY} seconds before retry...")
                    time.sleep(params.RATE_LIMIT_DELAY)
                else:
                    raise
                    
            except Exception as e:
                logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < params.MAX_RETRIES - 1:
                    logger.info(f"Waiting {params.RETRY_DELAY} seconds before retry...")
                    time.sleep(params.RETRY_DELAY)
                else:
                    raise
    
    def validate_batch_response(self, response_text: str, expected_indices: List[int],
                               required_fields: List[str]) -> List[Dict[str, Any]]:
        """
        Validate and parse the JSON response from the LLM.
        
        Args:
            response_text: Raw response text from the LLM
            expected_indices: List of expected index values
            required_fields: List of required fields in each response object
            
        Returns:
            Parsed and validated list of response objects
            
        Raises:
            ValueError: If the response is invalid or doesn't match expectations
        """
        try:
            # Parse JSON response
            response_objects = json.loads(response_text)
            
            # Validate it's a list
            if not isinstance(response_objects, list):
                raise ValueError("Response is not a JSON array")
            
            # Check we have the expected number of objects
            if len(response_objects) != len(expected_indices):
                raise ValueError(f"Expected {len(expected_indices)} objects, got {len(response_objects)}")
            
            # Validate each object
            validated_objects = []
            for obj in response_objects:
                # Check required fields
                for field in required_fields:
                    if field not in obj:
                        raise ValueError(f"Missing required field: {field}")
                
                # Validate index
                if obj["index"] not in expected_indices:
                    raise ValueError(f"Unexpected index: {obj['index']}")
                
                # Validate explanation length (if applicable)
                explanation_field = [f for f in required_fields if f.endswith('_cot')][0]
                explanation = obj[explanation_field]
                if len(explanation) < params.MIN_EXPLANATION_LENGTH:
                    raise ValueError(f"Explanation too short: {len(explanation)} characters")
                if len(explanation) > params.MAX_EXPLANATION_LENGTH:
                    raise ValueError(f"Explanation too long: {len(explanation)} characters")
                
                validated_objects.append(obj)
            
            return validated_objects
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
    
    def process_descriptive_batch(self, batch_features: List[Dict[str, Any]], 
                                 batch_index: int) -> List[Dict[str, Any]]:
        """
        Process a batch to generate descriptive chain-of-thought explanations.
        
        Args:
            batch_features: List of feature dictionaries for the batch
            batch_index: Index of the current batch
            
        Returns:
            List of validated explanation objects
            
        Raises:
            Exception: If the batch processing fails for any reason
        """
        # Extract expected indices for validation
        expected_indices = [entry["index"] for entry in batch_features]
        
        # Format the prompt
        batch_data_json = json.dumps(batch_features, indent=2)
        prompt = params.DESCRIPTIVE_TEMPLATE.format(batch_data=batch_data_json)
        
        # Prepare messages for API call
        messages = [
            {"role": "system", "content": params.SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}
        ]
        
        if params.LOG_BATCH_PROGRESS:
            logger.info(f"Processing descriptive batch {batch_index} ({len(batch_features)} founders)")
        
        # Make API call - will raise exception if it fails
        response_text = self.call_llm_with_retry(messages, f"descriptive batch {batch_index}")
        
        # Validate and parse response - will raise exception if validation fails
        validated_objects = self.validate_batch_response(
            response_text, expected_indices, params.REQUIRED_DESCRIPTIVE_FIELDS
        )
        
        if params.LOG_RESPONSE_SAMPLES and batch_index == 0:
            logger.info(f"Sample descriptive response: {validated_objects[0]['descriptive_cot'][:200]}...")
        
        return validated_objects
    
    def process_causal_batch(self, batch_descriptive: List[Dict[str, Any]], 
                            batch_features: List[Dict[str, Any]], 
                            batch_index: int) -> List[Dict[str, Any]]:
        """
        Process a batch to generate causal chain-of-thought explanations.
        
        Args:
            batch_descriptive: List of descriptive explanations for the batch
            batch_features: List of original feature dictionaries (for output)
            batch_index: Index of the current batch
            
        Returns:
            List of validated explanation objects
            
        Raises:
            Exception: If the batch processing fails for any reason
        """
        # Extract expected indices for validation
        expected_indices = [entry["index"] for entry in batch_descriptive]
        
        # Format the prompt with descriptive explanations
        batch_data_json = json.dumps(batch_descriptive, indent=2)
        prompt = params.CAUSAL_TEMPLATE.format(batch_data=batch_data_json)
        
        # Prepare messages for API call
        messages = [
            {"role": "system", "content": params.SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}
        ]
        
        if params.LOG_BATCH_PROGRESS:
            logger.info(f"Processing causal batch {batch_index} ({len(batch_descriptive)} founders)")
        
        # Make API call - will raise exception if it fails
        response_text = self.call_llm_with_retry(messages, f"causal batch {batch_index}")
        
        # Validate and parse response - will raise exception if validation fails
        validated_objects = self.validate_batch_response(
            response_text, expected_indices, params.REQUIRED_CAUSAL_FIELDS
        )
        
        if params.LOG_RESPONSE_SAMPLES and batch_index == 0:
            logger.info(f"Sample causal response: {validated_objects[0]['causal_cot'][:200]}...")
        
        return validated_objects
    
    def write_jsonl_output(self, output_path: str, explanations: List[Dict[str, Any]], 
                          batch_features_map: Dict[int, Dict[str, Any]], 
                          explanation_type: str) -> None:
        """
        Write explanations to a JSONL file, merging with original features.
        
        Args:
            output_path: Path to the output JSONL file
            explanations: List of explanation objects from LLM
            batch_features_map: Mapping from index to original features
            explanation_type: Type of explanation (for field naming)
        """
        with open(output_path, "a", encoding="utf-8") as fout:
            for explanation in explanations:
                idx = explanation["index"]
                original_features = batch_features_map[idx]
                
                # Create output object
                output_obj = {
                    "index": idx,
                    "features": original_features["features"],
                    "success_label": original_features["success_label"],
                    f"{explanation_type}_cot": explanation[f"{explanation_type}_cot"]
                }
                
                # Write as JSON line
                fout.write(json.dumps(output_obj) + "\n")
    
    def get_last_processed_index(self, file_path: str) -> int:
        """
        Get the last processed index from a JSONL output file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            The highest index found in the file, or -1 if file doesn't exist or is empty
        """
        if not os.path.exists(file_path):
            return -1
        
        try:
            max_index = -1
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        max_index = max(max_index, obj.get("index", -1))
            return max_index
        except Exception as e:
            logger.warning(f"Error reading {file_path} for resume: {e}")
            return -1
    
    def determine_resume_point(self) -> Tuple[int, int]:
        """
        Determine where to resume processing based on output files.
        
        When ENABLE_RESUME is True, this will resume at the last completed batch minus 1,
        effectively replacing the last batch when output is determined to be wrong.
        
        Returns:
            Tuple of (descriptive_last_index, causal_last_index)
        """
        if not params.ENABLE_RESUME or params.RESUME_MODE == "force_restart":
            logger.info("Resume disabled - starting from beginning")
            return -1, -1
        
        descriptive_last = self.get_last_processed_index(params.DESCRIPTIVE_JSONL_PATH)
        causal_last = self.get_last_processed_index(params.CAUSAL_JSONL_PATH)
        
        logger.info(f"Resume check (before adjustment): descriptive_last={descriptive_last}, causal_last={causal_last}")
        
        # REPLACE LAST BATCH LOGIC: When resuming, we want to redo the last completed batch
        # This is useful when the last batch output is determined to be wrong
        if descriptive_last >= 0:
            # Find the batch that contains the last completed index
            # We need to resume from the beginning of that batch to replace it entirely
            batch_size = params.BATCH_SIZE
            last_batch_start = (descriptive_last // batch_size) * batch_size
            
            # Move back one full batch to replace the last completed batch
            if last_batch_start >= batch_size:
                descriptive_last = last_batch_start - 1
                logger.info(f"Adjusted descriptive resume point: will replace last batch by resuming from index {descriptive_last}")
            else:
                # First batch completed, restart from beginning
                descriptive_last = -1
                logger.info("First batch completed - restarting from beginning to replace it")
        
        if causal_last >= 0:
            # Same logic for causal explanations
            batch_size = params.BATCH_SIZE
            last_batch_start = (causal_last // batch_size) * batch_size
            
            # Move back one full batch to replace the last completed batch
            if last_batch_start >= batch_size:
                causal_last = last_batch_start - 1
                logger.info(f"Adjusted causal resume point: will replace last batch by resuming from index {causal_last}")
            else:
                # First batch completed, restart from beginning
                causal_last = -1
                logger.info("First batch completed - restarting from beginning to replace it")
        
        logger.info(f"Final resume points: descriptive_last={descriptive_last}, causal_last={causal_last}")
        
        return descriptive_last, causal_last
    
    def should_process_batch(self, batch_indices: List[int], phase: str, last_index: int) -> bool:
        """
        Determine if a batch should be processed based on resume point.
        
        Args:
            batch_indices: List of indices in the batch
            phase: "descriptive" or "causal"
            last_index: Last completed index for this phase
            
        Returns:
            True if the batch should be processed
        """
        if last_index == -1:
            # No previous progress, process all batches
            return True
        
        # Check if any index in the batch is beyond our last completed index
        min_batch_index = min(batch_indices)
        max_batch_index = max(batch_indices)
        
        if max_batch_index <= last_index:
            # Entire batch already completed
            logger.info(f"Skipping {phase} batch with indices {min_batch_index}-{max_batch_index} (already completed)")
            return False
        elif min_batch_index <= last_index < max_batch_index:
            # Partial batch completion - need to be careful here
            logger.warning(f"Partial batch detected for {phase}: indices {min_batch_index}-{max_batch_index}, last completed: {last_index}")
            logger.warning(f"Will reprocess entire batch to ensure consistency")
            return True
        else:
            # Entire batch is new
            return True

    def initialize_output_files(self, descriptive_last: int, causal_last: int) -> None:
        """
        Initialize output files based on resume point.
        
        When resuming with adjusted points (to replace last batch), this will truncate
        the output files to remove entries beyond the resume point.
        
        Args:
            descriptive_last: Last completed descriptive index (-1 if starting fresh)
            causal_last: Last completed causal index (-1 if starting fresh)
        """
        # Handle descriptive output file
        if descriptive_last == -1:
            # Starting fresh - clear descriptive file
            with open(params.DESCRIPTIVE_JSONL_PATH, "w", encoding="utf-8") as f:
                pass  # Clear file
            logger.info("Starting descriptive phase from beginning - cleared output file")
        else:
            # Resuming - need to truncate file to remove entries beyond resume point
            logger.info(f"Resuming descriptive phase - truncating file to remove entries after index {descriptive_last}")
            self._truncate_jsonl_file(params.DESCRIPTIVE_JSONL_PATH, descriptive_last)
            logger.info(f"Resuming descriptive phase from index {descriptive_last + 1}")
        
        # Handle causal output file  
        if causal_last == -1:
            # Starting fresh - clear causal file  
            with open(params.CAUSAL_JSONL_PATH, "w", encoding="utf-8") as f:
                pass  # Clear file
            logger.info("Starting causal phase from beginning - cleared output file")
        else:
            # Resuming - need to truncate file to remove entries beyond resume point
            logger.info(f"Resuming causal phase - truncating file to remove entries after index {causal_last}")
            self._truncate_jsonl_file(params.CAUSAL_JSONL_PATH, causal_last)
            logger.info(f"Resuming causal phase from index {causal_last + 1}")
    
    def _truncate_jsonl_file(self, file_path: str, last_valid_index: int) -> None:
        """
        Truncate a JSONL file to only include entries up to and including last_valid_index.
        
        This method is used when replacing the last batch to remove entries that will be regenerated.
        
        Args:
            file_path: Path to the JSONL file to truncate
            last_valid_index: Last index to keep (inclusive)
        """
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist for truncation")
            return
        
        try:
            # Read all valid entries (up to and including last_valid_index)
            valid_entries = []
            entries_removed = 0
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if obj.get("index", -1) <= last_valid_index:
                            valid_entries.append(line)
                        else:
                            entries_removed += 1
            
            # Rewrite file with only valid entries
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(valid_entries)
            
            logger.info(f"Truncated {file_path}: kept {len(valid_entries)} entries, removed {entries_removed} entries")
            
        except Exception as e:
            logger.error(f"Failed to truncate {file_path}: {e}")
            # If truncation fails, clear the file to be safe
            with open(file_path, "w", encoding="utf-8") as f:
                pass
            logger.warning(f"Cleared {file_path} due to truncation failure - will restart from beginning")
    
    def generate_chain_of_thought_explanations(self) -> Tuple[str, str]:
        """
        Generate both descriptive and causal chain-of-thought explanations for all training data.
        
        Returns:
            Tuple of (descriptive_output_path, causal_output_path)
            
        Raises:
            Exception: If the generation process fails critically
        """
        logger.info("Starting chain-of-thought explanation generation")
        
        # Load training data
        train_df = self.load_training_data()
        num_rows = len(train_df)
        indices = train_df.index.tolist()
        
        # Determine resume point
        descriptive_last, causal_last = self.determine_resume_point()
        
        # Initialize output files based on resume point
        self.initialize_output_files(descriptive_last, causal_last)
        
        # Create batches
        batches = [
            indices[i:i + params.BATCH_SIZE]
            for i in range(0, num_rows, params.BATCH_SIZE)
        ]
        
        total_possible_batches = len(batches)
        
        # Apply batch limit if configured
        if params.BATCH_LIMIT is not None:
            batches = batches[:params.BATCH_LIMIT]
            logger.info(f"Batch limit applied: Processing {len(batches)} out of {total_possible_batches} total batches")
        
        logger.info(f"Processing {num_rows} rows in {len(batches)} batches")
        logger.info(f"Batch size: {params.BATCH_SIZE}")
        logger.info(f"Total founders to process: {len(batches) * params.BATCH_SIZE}")
        
        # Track all batch features for causal processing
        all_batch_features = {}
        descriptive_map = {}
        
        # Load existing descriptive explanations if resuming
        if descriptive_last >= 0:
            logger.info("Loading existing descriptive explanations for resume...")
            try:
                with open(params.DESCRIPTIVE_JSONL_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            descriptive_map[obj["index"]] = obj["descriptive_cot"]
                logger.info(f"Loaded {len(descriptive_map)} existing descriptive explanations")
            except Exception as e:
                logger.error(f"Failed to load existing descriptive explanations: {e}")
                raise
        
        # =============================================================================
        # PHASE 1: Generate Descriptive Chain-of-Thought Explanations
        # =============================================================================
        
        logger.info("Phase 1: Generating descriptive explanations...")
        
        descriptive_batches_processed = 0
        for batch_idx, batch_indices in enumerate(batches):
            # Check if this batch should be processed
            if not self.should_process_batch(batch_indices, "descriptive", descriptive_last):
                # Still need to load features for causal processing
                batch_df = train_df.loc[batch_indices]
                batch_features = self.create_batch_features(batch_df)
                for entry in batch_features:
                    all_batch_features[entry["index"]] = entry
                continue
            
            # Get batch data
            batch_df = train_df.loc[batch_indices]
            batch_features = self.create_batch_features(batch_df)
            
            # Store for later use in causal processing
            for entry in batch_features:
                all_batch_features[entry["index"]] = entry
            
            # Process descriptive batch - will raise exception if it fails
            explanations = self.process_descriptive_batch(batch_features, batch_idx)
            
            # Write to descriptive JSONL
            self.write_jsonl_output(
                params.DESCRIPTIVE_JSONL_PATH, 
                explanations, 
                {entry["index"]: entry for entry in batch_features},
                "descriptive"
            )
            
            # Store descriptive explanations for causal processing
            for explanation in explanations:
                descriptive_map[explanation["index"]] = explanation["descriptive_cot"]
            
            descriptive_batches_processed += 1
            if params.LOG_BATCH_PROGRESS:
                logger.info(f"✅ Descriptive batch {batch_idx} completed ({len(explanations)} founders)")
        
        logger.info(f"Descriptive phase completed: {descriptive_batches_processed} new batches processed")
        
        # =============================================================================
        # PHASE 2: Generate Causal Chain-of-Thought Explanations
        # =============================================================================
        
        logger.info("Phase 2: Generating causal explanations...")
        
        causal_batches_processed = 0
        for batch_idx, batch_indices in enumerate(batches):
            # Check if this batch should be processed
            if not self.should_process_batch(batch_indices, "causal", causal_last):
                continue
            
            # Prepare batch descriptive data
            batch_descriptive = []
            for idx in batch_indices:
                if idx not in descriptive_map:
                    raise Exception(f"Missing descriptive explanation for founder {idx} - this should never happen")
                batch_descriptive.append({
                    "index": idx,
                    "descriptive_cot": descriptive_map[idx]
                })
            
            # Get original batch features for output
            batch_features = [all_batch_features[idx] for idx in batch_indices]
            
            # Process causal batch - will raise exception if it fails
            explanations = self.process_causal_batch(batch_descriptive, batch_features, batch_idx)
            
            # Write to causal JSONL
            self.write_jsonl_output(
                params.CAUSAL_JSONL_PATH, 
                explanations, 
                {entry["index"]: entry for entry in batch_features},
                "causal"
            )
            
            causal_batches_processed += 1
            if params.LOG_BATCH_PROGRESS:
                logger.info(f"✅ Causal batch {batch_idx} completed ({len(explanations)} founders)")
        
        logger.info(f"Causal phase completed: {causal_batches_processed} new batches processed")
        
        # =============================================================================
        # Final Summary
        # =============================================================================
        
        logger.info("Chain-of-thought generation completed successfully!")
        logger.info(f"Total API calls: {self.total_api_calls}")
        logger.info(f"Total tokens used: {self.total_tokens_used:,}")
        logger.info(f"Descriptive output: {params.DESCRIPTIVE_JSONL_PATH}")
        logger.info(f"Causal output: {params.CAUSAL_JSONL_PATH}")
        
        return params.DESCRIPTIVE_JSONL_PATH, params.CAUSAL_JSONL_PATH


def run_llm_reasoning() -> Tuple[str, str]:
    """
    Main entry point for Module 5: LLM Chain-of-Thought Generation.
    
    Returns:
        Tuple of (descriptive_output_path, causal_output_path)
    """
    # Initialize the generator
    generator = LLMChainOfThoughtGenerator()
    
    # Generate explanations
    descriptive_path, causal_path = generator.generate_chain_of_thought_explanations()
    
    return descriptive_path, causal_path


if __name__ == "__main__":
    # For testing purposes
    run_llm_reasoning() 